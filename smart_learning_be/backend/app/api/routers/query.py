from fastapi import APIRouter, HTTPException, Request # <-- Thêm Request
from pydantic import BaseModel
# 1. Import StreamingResponse và AsyncGenerator
from fastapi.responses import StreamingResponse
from typing import Any, List, Optional, AsyncGenerator
# 2. Bỏ QueryResponse, giữ ContextChunk (hoặc định nghĩa lại nếu cần)
# from ...rag.chain import retrieve_answer (Giữ lại)
from ...rag.chain import retrieve_answer, dump_docs # <-- Import dump_docs nếu muốn log
import os
import time
import json # <-- Thêm json
from google.api_core.exceptions import ServiceUnavailable
from grpc import RpcError
# Bỏ JSONResponse nếu không dùng fallback kiểu cũ
# from fastapi.responses import JSONResponse
import logging

router = APIRouter(tags=["query"])

# ============================= #
# 📦 Models (Giữ lại Request, sửa Response)
# ============================= #
class QueryRequest(BaseModel):
    question: str
    top_k: int = 6 # Có thể bỏ top_k nếu Graph tự quản lý
    subject: Optional[str] = None
    course_id: Optional[str] = None
    # Thêm user_id nếu cần cho tool LMS
    # user_id: Optional[str] = "user_123" # Ví dụ

# Model cho source trả về (có thể nhúng vào stream hoặc gửi riêng)
class ContextChunk(BaseModel):
    page: Optional[int] = None
    text: str
    # Thêm các metadata khác nếu muốn hiển thị
    course_id: Optional[str] = None
    subject: Optional[str] = None
    score: Optional[float] = None # Ví dụ: rerank score

# BỎ QueryResponse cũ
# class QueryResponse(BaseModel): ...

# ============================= #
# 🚀 Query Endpoint (SỬA LẠI HOÀN TOÀN)
# ============================= #

# Bỏ response_model vì nó là StreamingResponse
@router.post("/query")
# Thêm Request để kiểm tra disconnect (nếu cần)
async def query(req: QueryRequest, request: Request):
    t_start_endpoint = time.perf_counter() # Đo thời gian tổng của endpoint

    # ---- Validate input (GIỮ NGUYÊN) ----
    q = (req.question or "").strip()
    if not q: raise HTTPException(status_code=400, detail="Question is required")
    subj = (req.subject or "").strip().lower()
    cid  = (req.course_id or "").strip().lower()
    if not subj: raise HTTPException(status_code=400, detail="Subject is required")
    if not cid: raise HTTPException(status_code=400, detail="Course ID is required")
    filters = {"subject": subj, "course_id": cid}
    # top_k không cần truyền vào retrieve_answer nữa
    # user_id = req.user_id # Lấy user_id nếu có
    # ---------------------------------

    logging.info(f"⚡ Nhận yêu cầu: '{q[:50]}...' (Subj: {subj}, Course: {cid})")

    try:
        # --- Gọi hàm retrieve_answer đã sửa ---
        # Nó trả về (async_generator, list_of_docs)
        answer_generator, docs = await retrieve_answer(
            question=q,
            subject=subj,
            course_id=cid,
            filters=filters,
            # k=top_k, # Không cần k ở đây nữa
            # user_id=user_id # Truyền user_id nếu cần
        )
        # ------------------------------------

        # --- Chuẩn bị sources (NGAY LẬP TỨC) ---
        # Chúng ta gửi sources SAU KHI stream text xong
        sources_data: List[Dict] = []
        if docs:
             # In log sources nếu muốn debug
             # dump_docs("Final Sources", docs)
             for d in docs:
                  meta = d.metadata or {}
                  text = (d.page_content or "")
                  # Chỉ lấy 300 ký tự đầu cho sources
                  sources_data.append(
                       ContextChunk(
                            page=meta.get("page"),
                            text=(text[:300] + "...") if len(text) > 300 else text,
                            course_id=meta.get("course_id"),
                            subject=meta.get("subject"),
                            score=meta.get("rerank_score") # Lấy rerank score nếu có
                       ).dict()
                  )
        # --------------------------------------

        # --- Định nghĩa Stream kết hợp Text và Metadata ---
        async def combined_stream_generator():
            full_answer_for_log = "" # Để log lại toàn bộ câu trả lời
            try:
                # 1. Stream phần text trả lời
                async for token in answer_generator:
                    # Kiểm tra client còn kết nối không
                    if await request.is_disconnected():
                         logging.warning("[Stream] Client disconnected during text stream.")
                         break # Dừng gửi nếu client ngắt kết nối
                    yield token
                    full_answer_for_log += token

                # 2. Gửi metadata (sources, elapsed) sau khi text kết thúc
                # Dùng một ký tự đặc biệt hoặc cấu trúc JSON để client nhận biết
                elapsed_total = int((time.perf_counter() - t_start_endpoint) * 1000)
                metadata_payload = {
                    "type": "metadata", # Đánh dấu đây là metadata
                    "sources": sources_data,
                    "elapsed_ms": elapsed_total
                }
                # Chuyển thành chuỗi JSON, thêm ký tự đặc biệt (ví dụ: null byte)
                # Hoặc dùng Server-Sent Events (SSE) format nếu client hỗ trợ
                metadata_str = "\n\n__METADATA__\n" + json.dumps(metadata_payload, ensure_ascii=False) + "\n__END_METADATA__\n"
                if not await request.is_disconnected():
                     yield metadata_str

                logging.info(f"✅ Streaming thành công. Total Endpoint Time: {elapsed_total} ms.")
                # Log câu trả lời đầy đủ (nếu cần)
                # logging.info(f"[Stream] Full Answer Sent:\n{full_answer_for_log}")

            except Exception as stream_exc:
                 logging.exception("[Stream] Lỗi trong quá trình combined_stream_generator")
                 # Gửi thông báo lỗi vào stream nếu có thể
                 try:
                      if not await request.is_disconnected():
                           yield f"\n[Lỗi Stream: {stream_exc}]"
                 except Exception: pass # Bỏ qua nếu không gửi được lỗi

            finally:
                 # Đảm bảo generator kết thúc sạch sẽ
                 logging.debug("[Stream] combined_stream_generator finished.")

        # --- Trả về StreamingResponse ---
        # media_type có thể là 'text/plain; charset=utf-8' hoặc 'text/event-stream' (cho SSE)
        return StreamingResponse(combined_stream_generator(), media_type="text/plain; charset=utf-8")
        # --------------------------------

    # --- Xử lý lỗi Endpoint (GIỮ NGUYÊN HOẶC SỬA ĐỂ STREAM LỖI) ---
    except (ServiceUnavailable, RpcError) as e:
        logging.error(f"[HTTP] LLM unavailable: {e}")
        # Có thể trả về stream lỗi thay vì JSONResponse
        async def error_llm_stream():
             yield "Hệ thống đang quá tải, vui lòng thử lại sau giây lát."
             metadata_payload = {"type": "error", "detail": "LLM unavailable"}
             yield "\n\n__METADATA__\n" + json.dumps(metadata_payload) + "\n__END_METADATA__\n"
        return StreamingResponse(error_llm_stream(), status_code=503, media_type="text/plain; charset=utf-8")

    except Exception as e:
        logging.exception("[HTTP] Unhandled error in /query endpoint")
        # Có thể trả về stream lỗi
        async def error_unhandled_stream():
             yield f"Lỗi hệ thống không xác định: {e}"
             metadata_payload = {"type": "error", "detail": "internal_error"}
             yield "\n\n__METADATA__\n" + json.dumps(metadata_payload) + "\n__END_METADATA__\n"
        return StreamingResponse(error_unhandled_stream(), status_code=500, media_type="text/plain; charset=utf-8")
    # --- Kết thúc xử lý lỗi ---
