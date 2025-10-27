import logging
import time
# Import State, timing, dedup
from ..rag_state import State, _update_timing, _dedup_by_sig, build_context
from typing import Dict, List, Any # <-- Thêm Any
from langchain_core.documents import Document

async def reflect_and_merge(state: State) -> State:
    """
    Tối ưu V3: Node này chuẩn bị dữ liệu có cấu trúc (subq + docs)
    cho prompt cuối cùng và gộp context tổng thể cho sources.
    KHÔNG tạo câu trả lời non-streamed ở đây nữa.
    """
    t_start = time.perf_counter()
    print("➡️  [Node] reflect_and_merge (V3 Structured): Bắt đầu chuẩn bị dữ liệu...")

    # --- Kiểm tra dừng sớm (GIỮ NGUYÊN) ---
    if state.get("answer") or (state.get("analyze_meta", {}).get("status") == "stop"):
        print("  [reflect V3] ⏭️ Bỏ qua...")
        return _update_timing(state, "reflect", t_start)
    # --- Kết thúc kiểm tra ---

    meta = state.get("analyze_meta") or {}
    per_subq_docs: Dict[str, List[Document]] = state.get("per_subq_docs", {})
    ordered_items = meta.get("sub_questions_with_intent") or []

    # --- CHUẨN BỊ STRUCTURED CONTEXT CHO PROMPT (DÙNG THẲNG ordered_blocks) ---
    ordered_blocks = state.get("ordered_blocks", []) or []
    structured_context_for_prompt: List[Dict[str, Any]] = ordered_blocks  # giữ nguyên thứ tự & đủ loại block

    # --- GỘP FINAL CONTEXT DOCS (CHO SOURCES METADATA) ---
    merged_ctx_docs: List[Document] = []
    seen_context_sigs = set()
    print(f"➡️  [Node] reflect_and_merge (V3 Structured): Gộp final context docs...")

    for b in ordered_blocks:
        if (b or {}).get("kind") == "academic":
            for d in (b.get("docs") or []):
                if not isinstance(d, Document):
                    continue
                sig = (d.page_content or "")[:150]
                if sig not in seen_context_sigs:
                    merged_ctx_docs.append(d); seen_context_sigs.add(sig)

    FINAL_SOURCES_LIMIT = 7
    final_merged_docs = _dedup_by_sig(merged_ctx_docs)[:FINAL_SOURCES_LIMIT]
    print(f"  [reflect V3] ✅ Gộp xong {len(final_merged_docs)} final context docs (sources).")


    # --- Cập nhật State ---
    state_diag = dict(state.get("diag") or {})
    state_diag["reflect"] = {
        "structured_prompt_items": len(structured_context_for_prompt),
        "final_docs_count": len(final_merged_docs)
    }

    new_state = {
        **state,
        "structured_context_for_prompt": structured_context_for_prompt, # <-- KEY MỚI
        "final_context_docs": final_merged_docs, # <-- Context để trả về sources
        "answer": None, # Xóa answer để báo hiệu cần stream
        "diag": state_diag,
        # Xóa các key không cần thiết nữa
        "sub_answers": None,
        "per_subq_docs": None,
        "answer_non_streamed": None, # Không còn dùng nữa
    }
    new_state = {k: v for k, v in new_state.items() if v is not None} # Dọn dẹp state

    print(f"✅ [Node] reflect_and_merge (V3 Structured): Hoàn thành.")
    return _update_timing(new_state, "reflect", t_start)

