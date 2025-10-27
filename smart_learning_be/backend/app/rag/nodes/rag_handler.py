import logging
import asyncio
import time
from typing import Tuple, List, Dict, Any # <-- Thêm Dict, Any
from langchain_core.documents import Document
# Import State, timing, dedup, build_context
from ..rag_state import State, _update_timing, _dedup_by_sig, build_context
from ..rag_runtime import mmr_select # Giữ lại MMR nếu muốn dùng sau RRF/Merge
# Import hybrid_retrieve
from ..retrieval.hybrid import hybrid_retrieve
from ...infrastructure.llm.reranker import rerank as heavy_rerank
# Bỏ import LLM và Prompt
import os

async def handle_academic_rag(sq_item: dict, state: State) -> Tuple[str, str, List[Document]]:
    """
    Tối ưu V3: Node này nhận list retrieval_queries, chạy retrieve song song,
    gộp kết quả, rerank bằng original_subq, và trả về context docs.
    KHÔNG gọi LLM tạo sinh.
    Trả về (original_subq, "", selected_docs).
    """
    t_start = time.perf_counter()
    original_sq = sq_item.get("original_subq", "")
    # Lấy list queries từ transform node
    retrieval_queries: List[str] = sq_item.get("retrieval_queries", [])
    topic_sq = (sq_item.get("topic") or "").strip()
    ans_str = "" # Luôn trả về chuỗi rỗng
    selected_docs: List[Document] = [] # Kết quả cuối cùng

    # Fallback nếu không có retrieval_queries
    if not retrieval_queries:
        print(f"  [RagHandler V3] ⚠️ Không có retrieval_queries cho '{original_sq[:30]}...', dùng câu gốc.")
        retrieval_queries = [original_sq] if original_sq else []

    if not retrieval_queries: # Nếu original_sq cũng rỗng
        print(f"  [RagHandler V3] ❌ Không có query nào để retrieve.")
        duration_empty = round((time.perf_counter() - t_start) * 1000)
        print(f"⏱️  [Timing] Handler 'rag_handler' (No Query) hoàn thành sau: {duration_empty} ms")
        return original_sq, ans_str, selected_docs # Trả về list rỗng

    print(f"  [RagHandler V3] 🕵️  Đang Retrieve context cho: \"{original_sq[:50]}...\"")
    print(f"    Queries: {retrieval_queries}")

    try:
        filters = state.get("filters", {})
        # Giữ nguyên các tham số K
        k_dense, k_sparse, top_after_rrf = 12, 0, 15 # top_after_rrf có thể tăng lên nếu dùng nhiều query
        rerank_top_n = 5
        context_max = 5 # Giới hạn context cuối cùng

        # --- CHẠY RETRIEVAL SONG SONG CHO TỪNG QUERY ---
        print(f"  [RagHandler V3] 1. Bắt đầu {len(retrieval_queries)} luồng Retrieve song song...")
        async def _retrieve_one(query: str):
            try:
                # Gọi hybrid_retrieve cho từng query
                # return_raw=False để chỉ lấy list docs đã merge RRF
                docs = await hybrid_retrieve(query, k_dense, k_sparse, top_after_rrf, filters, False)
                print(f"    [Retrieve Worker] Query \"{query[:30]}...\" -> {len(docs)} docs.")
                return docs
            except Exception as retrieve_e:
                print(f"    [Retrieve Worker] 💥 Lỗi khi retrieve cho query \"{query}\": {retrieve_e}")
                return [] # Trả về list rỗng nếu lỗi

        # Chạy các task retrieve song song
        retrieve_tasks = [_retrieve_one(q) for q in retrieval_queries]
        list_of_docs_lists = await asyncio.gather(*retrieve_tasks)

        # --- GỘP KẾT QUẢ RETRIEVAL ---
        all_retrieved_docs: List[Document] = []
        seen_sigs_gather = set() # Dedup ngay khi gộp
        for docs_list in list_of_docs_lists:
            if isinstance(docs_list, list):
                 for d in docs_list:
                      if isinstance(d, Document):
                           sig = (d.page_content or "")[:150]
                           if sig not in seen_sigs_gather:
                                all_retrieved_docs.append(d)
                                seen_sigs_gather.add(sig)

        print(f"  [RagHandler V3] 2. Gộp xong từ các luồng Retrieve: {len(all_retrieved_docs)} docs (đã dedup).")

        # (Tùy chọn) Có thể áp dụng MMR ở đây nếu muốn đa dạng hóa docs gộp
        # if len(all_retrieved_docs) > some_limit:
        #      all_retrieved_docs = mmr_select(original_sq, all_retrieved_docs, k=some_limit)

        # --- RERANK KẾT QUẢ ĐÃ GỘP BẰNG CÂU HỎI GỐC ---
        if not all_retrieved_docs:
             print(f"  [RagHandler V3] ❌ Không tìm thấy context nào sau khi gộp.")
             # selected_docs giữ nguyên là []
        else:
             print(f"  [RagHandler V3] 3. Bắt đầu Rerank {len(all_retrieved_docs)} docs gộp (top_n={rerank_top_n}) bằng câu hỏi GỐC: \"{original_sq[:30]}...\"")
             # DÙNG original_sq cho reranker
             reranked_final = await heavy_rerank(original_sq, all_retrieved_docs, top_n=rerank_top_n)
             print(f"  [RagHandler V3] 4. Rerank xong: {len(reranked_final)} docs.")

             # Chuẩn hóa output từ reranker (nếu cần, code cũ đã có)
             normalized_reranked: List[Document] = []
             for d in reranked_final:
                  if isinstance(d, Document): normalized_reranked.append(d)
                  # ... (Thêm logic chuẩn hóa nếu reranker trả về dict/tuple) ...
                  elif isinstance(d, dict) and "text" in d: # Ví dụ
                       normalized_reranked.append(Document(page_content=d["text"], metadata=d.get("metadata", {})))

             # Giới hạn context cuối cùng và dedup lại lần nữa (đảm bảo)
             selected_docs = _dedup_by_sig(normalized_reranked[:context_max])

             try: # Log kiểm tra
                  t0 = type(selected_docs[0]).__name__ if selected_docs else "None"
                  print(f"  [RagHandler V3] 5. Thu thập xong {len(selected_docs)} context docs cuối. First type={t0}")
             except Exception:
                  print(f"  [RagHandler V3] 5. Thu thập xong {len(selected_docs)} context docs cuối.")

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"  [RagHandler V3] 💥 Lỗi nghiêm trọng khi xử lý RAG cho '{original_sq}': {err_msg}")
        logging.exception(f"[RagHandler V3] error on '{original_sq}'")
        selected_docs = [] # Trả về list rỗng nếu lỗi

    finally:
        duration = round((time.perf_counter() - t_start) * 1000)
        print(f"⏱️  [Timing] Handler 'rag_handler' (Retrieve V3) cho '{original_sq[:30]}...' hoàn thành sau: {duration} ms")
        # Trả về chuỗi answer rỗng và selected_docs
        return original_sq, ans_str, selected_docs