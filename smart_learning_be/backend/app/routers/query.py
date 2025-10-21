from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, List, Optional
from ..services.chain import retrieve_answer
import os
import time
from google.api_core.exceptions import ServiceUnavailable
from grpc import RpcError
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from google.api_core.exceptions import ServiceUnavailable
from grpc import RpcError
import logging, time
from typing import List

router = APIRouter(tags=["query"])


# ============================= #
# 📦 Models
# ============================= #
class QueryRequest(BaseModel):
    question: str
    top_k: int = 6
    subject: Optional[str] = None
    course_id: Optional[str] = None


class ContextChunk(BaseModel):
    page: Optional[int] = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[ContextChunk]
    elapsed_ms: int


# ============================= #
# 🚀 Query Endpoint
# ============================= #

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    t0 = time.time()

    # ---- validate tối thiểu ----
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required")
    subj = (req.subject or "").strip().lower()
    cid  = (req.course_id or "").strip().lower()
    if not subj:
        raise HTTPException(status_code=400, detail="Subject is required")
    if not cid:
        raise HTTPException(status_code=400, detail="Course ID is required")

    # ---- filters (chỉ chứa các khóa đã chuẩn hóa) ----
    filters = {"subject": subj, "course_id": cid}

    # ---- clamp top_k an toàn ----
    top_k = int(req.top_k or 5)
    if top_k < 1: top_k = 1
    if top_k > 20: top_k = 20

    try:
        # ✅ Gọi async RAG chain (per-subq/no-intent)
        answer, docs = await retrieve_answer(
            question=q,
            subject=subj,           # truyền rõ ràng
            course_id=cid,          # truyền rõ ràng
            k=top_k,
            filters=filters,
        )

        # ---- Chuẩn hóa sources cho client ----
        sources: List[ContextChunk] = []
        for d in (docs or []):
            meta = d.metadata or {}
            text = (d.page_content or "")
            sources.append(
                ContextChunk(
                    page=meta.get("page"),
                    text=(text[:1000] + "...") if len(text) > 1000 else text,
                )
            )

        elapsed = int((time.time() - t0) * 1000)
        return QueryResponse(answer=answer or "", sources=sources, elapsed_ms=elapsed)

    # ---- LLM quá tải (503) → trả 200 với fallback ngắn, tránh 500 ----
    except (ServiceUnavailable, RpcError) as e:
        logging.error(f"[HTTP] LLM unavailable: {e}")
        elapsed = int((time.time() - t0) * 1000)
        # có thể trả câu ngắn gọn, tùy bạn muốn thông điệp hay để rỗng
        return JSONResponse(
            status_code=200,
            content=QueryResponse(
                answer="Hệ thống đang quá tải, mình đã trả phần có thể từ ngữ cảnh. Vui lòng thử lại.",
                sources=[],
                elapsed_ms=elapsed,
            ).dict(),
        )

    # ---- Lỗi khác: log và trả 500 gọn ----
    except Exception as e:
        logging.exception("[HTTP] Unhandled error in /query")
        raise HTTPException(status_code=500, detail="internal_error")
