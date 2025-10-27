import asyncio
import logging
import time # <-- Thêm time
from concurrent.futures import ThreadPoolExecutor
# Sửa import typing
from typing import Dict, List, Tuple, Optional, Any
from langchain_core.documents import Document # <-- Import Document

# Import handlers và router
from .system_handlers import handle_system
from .rag_handler import handle_academic_rag
# Sửa đường dẫn lms_router nếu cần
from ...services.lms.lms_router import route_lms_tool
# Import State và timing
from ..rag_state import State, _update_timing

# Import Semaphore và Config
from ...core.config import CFG
DEFAULT_CONCURRENCY = 5
sem = asyncio.Semaphore(getattr(CFG, 'CONCURRENCY', DEFAULT_CONCURRENCY))

async def generate_per_subq(state: State) -> State:
    """
    Node Dispatcher V3: Gọi handlers, thu thập kết quả (answer + docs).
    """
    t_start = time.perf_counter()
    print("➡️  [Node] generate_per_subq (Dispatcher): Bắt đầu...")

    # --- Kiểm tra dừng sớm (Giữ nguyên) ---
    if state.get("answer") or (state.get("analyze_meta", {}).get("status") == "stop"):
        print("  [Dispatcher] ⏭️ Bỏ qua...")
        return _update_timing(state, "generate_per_subq", t_start)
    # --- Kết thúc kiểm tra ---

    meta = state.get("analyze_meta") or {}
    sq_items = meta.get("sub_questions_with_intent") or []
    if not sq_items:
        print("  [Dispatcher] ⚠️ Không có sub-questions...")
        return _update_timing(state, "generate_per_subq", t_start)

    print(f"➡️  [Node] generate_per_subq (Dispatcher): Đang xử lý {len(sq_items)} sub-questions...")

    # --- Hàm _dispatch_one (Giữ nguyên) ---
    # Trả về: Optional[Tuple[str, str, List[Document]]]
    async def _dispatch_one(sq_item: dict) -> Optional[Tuple[str, str, List[Document]]]:
        """Xử lý một sub-question, trả về (original_sq, answer, context_docs)."""
        async with sem:
            original_sq = sq_item.get("original_subq", "")
            if not original_sq: return None

            intent_category = (sq_item.get("intent") or ["unknown"])[0]
            topic_for_sq = (sq_item.get("topic") or "").strip()
            print(f"  [Dispatcher] 💬 Bắt đầu SQ: \"{original_sq[:30]}...\" ({intent_category})")
            handler_start_time = time.perf_counter()

            ans_str: str = f"Lỗi mặc định"
            context_docs: List[Document] = []
            returned_sq: str = original_sq

            try:
                if intent_category == "academic_rag":
                    print(f"    [Dispatcher] ➡️  [rag_handler]...")
                    # rag_handler trả về (sq, ans, docs)
                    returned_sq, ans_str, context_docs = await handle_academic_rag(sq_item, state)

                elif intent_category == "lms_tools":
                    print(f"    [Dispatcher] ➡️  [lms_router]...")
                    # lms_router trả về (sq, ans)
                    returned_sq, ans_str = await route_lms_tool(original_sq, state, topic_for_sq)
                    # context_docs giữ nguyên là []

                elif intent_category == "system_handlers":
                    print(f"    [Dispatcher] ➡️  [system_handlers]...")
                    # system_handlers trả về (sq, ans)
                    returned_sq, ans_str = await handle_system(original_sq, state, topic_for_sq)
                    # context_docs giữ nguyên là []

                else:
                    print(f"    [Dispatcher] ⚠️ Handler không tồn tại: {intent_category}")
                    ans_str = f"Lỗi: Handler không tồn tại '{intent_category}'."

                # Kiểm tra subq trả về
                if returned_sq != original_sq:
                    print(f"  [Dispatcher] 💥 Lỗi Logic: Handler trả về subq không khớp!")
                    ans_str = "Lỗi logic: Handler trả về subq không khớp."
                    context_docs = [] # Reset docs

                handler_duration = round((time.perf_counter() - handler_start_time) * 1000)
                print(f"  [Dispatcher] ✅ HOÀN THÀNH SQ: \"{original_sq[:30]}...\" sau {handler_duration} ms")
                # Trả về cả 3 giá trị (docs có thể rỗng)
                return original_sq, ans_str, context_docs

            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                print(f"  [Dispatcher] 💥 Lỗi Exception khi xử lý SQ \"{original_sq[:30]}...\": {err_msg}")
                logging.exception(f"[Dispatcher] Exception trên subq '{original_sq}'")
                return (original_sq, f"Lỗi Exception khi xử lý.", []) # Trả về docs rỗng khi lỗi
    # --- Kết thúc _dispatch_one ---

    # Chạy song song (Giữ nguyên)
    results = await asyncio.gather(*[_dispatch_one(item) for item in sq_items], return_exceptions=True)

    # --- SỬA LỖI TẬP HỢP KẾT QUẢ ---
    sub_answers: Dict[str, str] = {}
    # KHỞI TẠO per_subq_docs là dict RỖNG
    per_subq_docs: Dict[str, List[Document]] = {}

    print(f"  [Dispatcher] ⚙️ Đang tập hợp kết quả từ {len(results)} tasks...")
    for idx, r in enumerate(results):
        # Lấy original_sq từ item gốc để đảm bảo khớp
        original_sq_for_item = sq_items[idx].get("original_subq", f"Item_{idx}_UnknownSQ")

        if isinstance(r, Exception):
            print(f"    [Dispatcher] 💥 Task {idx} (SQ: '{original_sq_for_item[:30]}...') bị lỗi Exception: {r}")
            sub_answers[original_sq_for_item] = f"Lỗi Exception (Gather): {r}"
            per_subq_docs[original_sq_for_item] = [] # Gán list rỗng

        # Kiểm tra tuple có đúng 3 phần tử
        elif isinstance(r, tuple) and len(r) == 3:
            sq, ans, docs = r
            # Kiểm tra lại lần nữa sq trả về có khớp không
            if sq == original_sq_for_item:
                 sub_answers[sq] = ans
                 # GÁN DOCS VÀO ĐÚNG KEY TRONG per_subq_docs
                 per_subq_docs[sq] = docs if isinstance(docs, list) else [] # Đảm bảo docs là list
                 print(f"    [Dispatcher] ✨ Thu thập kết quả SQ '{sq[:30]}...': Answer len={len(ans)}, Docs count={len(per_subq_docs[sq])}")
            else:
                 # Trường hợp này không nên xảy ra nếu _dispatch_one đã kiểm tra
                 print(f"    [Dispatcher] 💥 Lỗi Logic (Gather): Task {idx} trả về subq không khớp! Expected '{original_sq_for_item}', got '{sq}'")
                 sub_answers[original_sq_for_item] = "Lỗi logic (Gather): Subq không khớp."
                 per_subq_docs[original_sq_for_item] = []
        elif r is None:
             print(f"    [Dispatcher] ⚠️ Bỏ qua kết quả None cho item {idx} (SQ: '{original_sq_for_item[:30]}...').")
             # Không cần gán gì vào dict nếu là None
        else:
             print(f"    [Dispatcher] 💥 Lỗi lạ (Gather): Kết quả không hợp lệ cho SQ '{original_sq_for_item[:30]}...': {type(r)}")
             sub_answers[original_sq_for_item] = "Lỗi lạ: Kết quả không hợp lệ."
             per_subq_docs[original_sq_for_item] = []

    # --- Cập nhật state (Giữ nguyên) ---
    state_diag = dict(state.get("diag") or {})
    success_count = sum(1 for sq in sub_answers if not sub_answers[sq].startswith("Lỗi")) # Đếm câu trả lời không lỗi
    state_diag["dispatcher"] = {"total_sq": len(sq_items), "success": success_count}
    print(f"✅ [Node] generate_per_subq: Xử lý xong.")

    # In thử key và số lượng docs trong per_subq_docs để debug
    print(f"  [Dispatcher] DEBUG: per_subq_docs keys: {list(per_subq_docs.keys())}")
    for k, v in per_subq_docs.items():
         print(f"    - '{k[:30]}...': {len(v)} docs")

    # ✅ NEW: Dựng ordered_blocks theo đúng thứ tự user hỏi
    ordered_blocks = []
    for item in sq_items:  # giữ nguyên thứ tự user hỏi
        subq   = item.get("original_subq","")
        intent = (item.get("intent") or ["unknown"])[0]
        topic  = (item.get("topic") or "").strip()

        if intent == "system_handlers":
            ordered_blocks.append({
                "kind": "system",
                "role": topic,                 # meta | unclear | unsafe | toc
                "subq": subq,
                "text": (sub_answers.get(subq) or "").strip()
            })
        elif intent == "academic_rag":
            ordered_blocks.append({
                "kind": "academic",
                "role": None,
                "subq": subq,
                "docs": (per_subq_docs.get(subq) or [])
            })
        else:
            # nếu có loại khác (lms_tools...) thì vẫn đẩy như system
            ordered_blocks.append({
                "kind": "system",
                "role": intent,
                "subq": subq,
                "text": (sub_answers.get(subq) or "").strip()
            })

    new_state = {
        **state,
        "sub_answers": sub_answers,
        "per_subq_docs": per_subq_docs, # <-- LƯU DOCS ĐÃ THU THẬP ĐÚNG
        "ordered_blocks": ordered_blocks,
        "diag": state_diag,
        # Không cần 'context' ở đây nữa, reflect sẽ tự gộp 'final_context_docs'
    }
    return _update_timing(new_state, "generate_per_subq", t_start)

