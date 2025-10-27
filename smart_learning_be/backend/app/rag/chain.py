import logging
import os
from fastapi.responses import StreamingResponse
from typing import List, Tuple, AsyncGenerator, Dict, Any
from langchain_core.documents import Document

from .rag_graph import get_graph, run_graph
# 1. Sửa import: Bỏ build_context từ rag_state nếu không dùng trực tiếp ở đây nữa
# Lấy hàm build prompt MỚI
from .prompts.prompt_utils import build_structured_prompt_block, CACHED_PROMPT
from ..infrastructure.llm.llm_utils import async_stream_generate
from .rag_state import State # Vẫn cần State để type hint

from ..core.config import CFG
import time
import json

# --- Hàm dump_docs (Giữ nguyên) ---
def dump_docs(tag: str, docs: List[Document]): ...

# --- Sửa hàm retrieve_answer ---
async def retrieve_answer(
    question: str,
    subject: str = "General",
    course_id: str | None = None,
    filters: dict | None = None,
) -> Tuple[AsyncGenerator[str, None], List[Document]]:
    start_chain = time.perf_counter()
    logging.info(f"[Chain V3] Bắt đầu xử lý: '{question[:50]}...'")

    input_state: Dict[str, Any] = {
        "question": question or "", "subject": subject or "General",
        "course_id": course_id, "filters": filters or {}, "timing": {},
    }
    final_state: State = {}
    final_docs: List[Document] = [] # Docs để trả về sources metadata

    try:
        final_state = await run_graph(input_state)
        # --- Log Debug State (Giữ nguyên) ---
        print("\n--- DEBUG: Final State received in chain.py ---")
        print(f"  Keys: {list(final_state.keys())}")
        # In structured_context để kiểm tra
        structured_ctx_debug = final_state.get("structured_context_for_prompt")
        print(f"  structured_context_for_prompt (type: {type(structured_ctx_debug)}):")
        if isinstance(structured_ctx_debug, list):
             print(f"    Length: {len(structured_ctx_debug)}")
             if structured_ctx_debug:
                  print(f"    First item subq: {structured_ctx_debug[0].get('subq')}")
                  print(f"    First item docs count: {len(structured_ctx_debug[0].get('docs', []))}")
        else:
             print(f"    Value: {structured_ctx_debug}")
        # In final_context_docs để kiểm tra sources
        final_docs_debug = final_state.get("final_context_docs")
        print(f"  final_context_docs (type: {type(final_docs_debug)}):")
        if isinstance(final_docs_debug, list): print(f"    Length: {len(final_docs_debug)}")
        else: print(f"    Value: {final_docs_debug}")

        print("----------------------------------------------\n")

        # Lấy final_docs cho sources (Giữ nguyên)
        final_docs = final_state.get("final_context_docs") or []
        logging.info(f"[Chain V3] Graph hoàn thành. Thu được {len(final_docs)} final docs (sources).")

    except Exception as graph_error:
        # ... (Xử lý lỗi graph giữ nguyên) ...
        async def error_stream(err_msg: str): yield f"[Lỗi hệ thống: {err_msg}]"
        return error_stream(str(graph_error)), []

    # --- Trích xuất thông tin ---
    original_question = final_state.get("question_user") or question
    predefined_answer = final_state.get("answer") # Answer từ early exit
    status = final_state.get("analyze_meta", {}).get("status", "continue")
    # Lấy structured context từ state
    structured_context: List[Dict[str, Any]] = final_state.get("structured_context_for_prompt", [])

    # --- Xử lý dừng sớm (Giữ nguyên) ---
    if status == "stop" or predefined_answer is not None:
        logging.info("[Chain V3] Pipeline dừng sớm hoặc đã có câu trả lời.")
        final_answer_to_use = predefined_answer or "Lỗi: Không xác định được câu trả lời (trạng thái dừng)."
        async def predefined_stream(ans: str): yield ans
        return predefined_stream(final_answer_to_use), final_docs

    # --- Kiểm tra structured_context rỗng ---
    if not structured_context:
         logging.warning("[Chain V3] structured_context rỗng, không thể tạo prompt streaming.")
         # Có thể trả về câu hỏi lại user hoặc fallback
         async def no_context_stream(): yield "Xin lỗi, không có đủ thông tin để xử lý câu hỏi này."
         return no_context_stream(), final_docs # Trả về docs đã gộp (nếu có)

    # --- Chuẩn bị Prompt cuối cùng (DÙNG HÀM MỚI) ---
    # Lấy doc_source từ doc đầu tiên của sources cuối cùng
    final_doc_source = (final_docs[0].metadata.get("doc_source") if final_docs else "Tài liệu học tập")

    # Gọi hàm build prompt MỚI
    try:
        final_prompt_text = build_structured_prompt_block(
            subject=subject,
            structured_context=structured_context, # <-- Truyền cấu trúc mới vào đây
            original_full_question=original_question, # Câu hỏi gốc đầy đủ
            doc_source=final_doc_source
        )
        logging.info(f"[Chain V3] Chuẩn bị xong prompt cấu trúc (len={len(final_prompt_text)}) cho streaming.")
        # --- Log prompt chi tiết (Giữ nguyên) ---
        print("\n--- PROMPT ĐƯA VÀO LLM (ASYNC STREAM - CHAIN - STRUCTURED) ---")
        print(CACHED_PROMPT + "\n\n---\n\n" + final_prompt_text)
        print("----------------------------------------------------------\n")
    except Exception as prompt_error:
         logging.exception("[Chain V3] Lỗi khi xây dựng prompt cấu trúc.")
         async def prompt_error_stream(): yield f"[Lỗi hệ thống: Không thể tạo prompt - {prompt_error}]"
         return prompt_error_stream(), final_docs


    # --- Định nghĩa Async Generator (Giữ nguyên) ---
    async def stream_final_answer() -> AsyncGenerator[str, None]:
        logging.info("[Chain V3] Bắt đầu streaming câu trả lời cuối cùng...")
        stream_start_time = time.perf_counter()
        try:
            # Gọi async_stream_generate với prompt cấu trúc
            async for token in async_stream_generate(CACHED_PROMPT + "\n\n---\n\n" + final_prompt_text):
                yield token
            stream_duration = round((time.perf_counter() - stream_start_time) * 1000)
            logging.info(f"[Chain V3] ✅ Streaming hoàn thành sau {stream_duration} ms.")
        except Exception as stream_error:
            logging.exception("[Chain V3] Lỗi trong quá trình streaming LLM cuối cùng")
            yield f"[Lỗi streaming: {stream_error}]"


    # --- Trả về Generator và Docs ---
    chain_duration = round((time.perf_counter() - start_chain) * 1000)
    logging.info(f"⏱️ Chain `retrieve_answer` (V3 Structured) hoàn thành chuẩn bị sau {chain_duration} ms.")
    return stream_final_answer(), final_docs