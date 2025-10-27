import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List
from langchain_core.documents import Document
# Import State, timing, dedup, build_context
from ..rag_state import State, _update_timing, _dedup_by_sig, build_context
from ..rag_runtime import mmr_select
from ..retrieval.hybrid import hybrid_retrieve
from ...infrastructure.llm.reranker import rerank as heavy_rerank
# Bỏ import sync_stream_generate và prompts
# from ...infrastructure.llm.llm_utils import sync_stream_generate
# from ...services.prompts.prompt_utils import build_unified_block, CACHED_PROMPT
import os # <-- Giữ lại os nếu cần (ví dụ: replace linesep)

async def handle_academic_rag(sq_item: dict, state: State) -> Tuple[str, str, List[Document]]:
    """
    Tối ưu: Node này CHỈ retrieve và rerank context.
    KHÔNG gọi LLM tạo sinh ở đây.
    Trả về (original_subq, "", selected_docs).
    """
    t_start = time.perf_counter()
    original_sq = sq_item.get("original_subq", "")
    retrieval_sq = sq_item.get("retrieval_subq", original_sq)
    topic_sq = (sq_item.get("topic") or "").strip()
    ans_str = "" # LUÔN trả về chuỗi rỗng
    selected_docs: List[Document] = []

    print(f"  [RagHandler OPT] 🕵️  Đang Retrieve context cho: \"{original_sq[:50]}...\" (Dùng query: \"{retrieval_sq}\")")

    try:
        filters = state.get("filters", {})
        k_dense, k_sparse, top_after_rrf = 12, 0, 15
        rerank_top_n = 5
        context_max = 5

        print(f"  [RagHandler OPT] 1. Bắt đầu Hybrid Retrieve (k_dense={k_dense})...")
        docs = await hybrid_retrieve(retrieval_sq, k_dense, k_sparse, top_after_rrf, filters, False)
        print(f"  [RagHandler OPT] 2. Retrieve xong: {len(docs)} docs.")

        if len(docs) > top_after_rrf:
             docs = mmr_select(retrieval_sq, docs, k=top_after_rrf, lambda_mult=0.65)
             print(f"  [RagHandler OPT] 2.1 MMR selected: {len(docs)} docs.")

        print(f"  [RagHandler OPT] 3. Bắt đầu Rerank (top_n={rerank_top_n})...")
        reranked = await heavy_rerank(retrieval_sq, docs, top_n=rerank_top_n)
        print(f"  [RagHandler OPT] 4. Rerank xong: {len(reranked)} docs.")
            # --- Chuẩn hóa về List[Document] ---
        normalized_docs: List[Document] = []
        for i, d in enumerate(reranked):
            # TH1: đã là Document
            if isinstance(d, Document):
                normalized_docs.append(d)
                continue

            # TH2: reranker trả về dict với 'document' hoặc 'text'/'metadata'
            if isinstance(d, dict):
                if isinstance(d.get("document"), Document):
                    normalized_docs.append(d["document"])
                    continue
                if "text" in d:
                    normalized_docs.append(Document(
                        page_content=str(d.get("text", "")),
                        metadata=d.get("metadata", {}) or {}
                    ))
                    continue
                if "page_content" in d:
                    normalized_docs.append(Document(
                        page_content=str(d.get("page_content", "")),
                        metadata=d.get("metadata", {}) or {}
                    ))
                    continue

            # TH3: reranker trả về tuple (doc, score) hoặc tương tự
            if isinstance(d, (tuple, list)) and d:
                cand = d[0]
                if isinstance(cand, Document):
                    normalized_docs.append(cand)
                    continue
                if isinstance(cand, dict):
                    normalized_docs.append(Document(
                        page_content=str(cand.get("page_content") or cand.get("text") or ""),
                        metadata=cand.get("metadata", {}) or {}
                    ))
                    continue

            # Fallback cuối: ép về Document trống để không rơi item
            normalized_docs.append(Document(page_content=str(d), metadata={"_note": "normalized_fallback"}))

        # Cập nhật lại reranked và dedup
        reranked = normalized_docs
        # Giữ tối đa context_max
        selected_docs = _dedup_by_sig(reranked[:context_max])

        # Bổ sung log kiểm chứng kiểu dữ liệu
        try:
            t0 = type(selected_docs[0]).__name__ if selected_docs else "None"
            print(f"  [RagHandler OPT] 5. Thu thập xong {len(selected_docs)} context docs. First type={t0}")
        except Exception:
            print(f"  [RagHandler OPT] 5. Thu thập xong {len(selected_docs)} context docs.")
            
        selected_docs = _dedup_by_sig(reranked[:context_max])

        if not selected_docs:
            final_topic = topic_sq or original_sq
            print(f"  [RagHandler OPT] ❌ Không tìm thấy context cho '{final_topic}'.")
            # Vẫn trả về list rỗng, không cần gán ans_str lỗi
        else:
            # Chỉ log số lượng docs, không cần build context ở đây
             print(f"  [RagHandler OPT] 5. Thu thập xong {len(selected_docs)} context docs.")

        # --- BỎ HOÀN TOÀN PHẦN GỌI LLM SYNC ---
        # ctx = build_context(selected_docs)
        # print(f"  [RagHandler] 5. Context build xong...")
        # print("-" * 50) ...
        # soft_hints = ...
        # prompt_text = build_unified_block(...)
        # print("\n--- PROMPT ĐƯA VÀO LLM (SYNC - RAG HANDLER) ---") ...
        # print(f"  [RagHandler] 6. Bắt đầu gọi LLM (SYNC Tạo sinh)...")
        # ans_str = sync_stream_generate(...)
        # print(f"  [RagHandler] 7. LLM (SYNC Tạo sinh) hoàn thành...")
        # --- KẾT THÚC BỎ ---

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"  [RagHandler OPT] 💥 Lỗi khi xử lý RAG cho '{original_sq}': {err_msg}")
        logging.exception(f"[RagHandler OPT] error on '{original_sq}'")
        selected_docs = [] # Trả về list rỗng nếu lỗi

    finally:
        duration = round((time.perf_counter() - t_start) * 1000)
        print(f"⏱️  [Timing] Handler 'rag_handler' (Retrieve Only) cho '{original_sq[:30]}...' hoàn thành sau: {duration} ms")
        # Trả về chuỗi answer rỗng và selected_docs
        return original_sq, ans_str, selected_docs