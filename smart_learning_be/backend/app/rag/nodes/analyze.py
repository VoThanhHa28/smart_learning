import logging
import time
import re
from typing import Any, Dict, List
# 1. Sửa đường dẫn import cho đúng
from ...rag.analysis.analyze_query import analyze_question # <-- Hàm cốt lõi
from ...services.utils.toc_store import get_toc_by_course, get_toc_by_key, flatten_toc
# Bỏ import sync_stream_generate nếu chỉ dùng trong _format_toc_with_llm
# from ...infrastructure.llm.llm_utils import sync_stream_generate
from ..prompts.prompt_utils import build_toc_validation_block
from ..rag_state import State, _update_timing

# --- Các hàm Regex và Helper giữ nguyên ---
_TOC_RX = re.compile(r"(\bmục lục\b|\bmuc luc\b|\btoc\b|...)", re.IGNORECASE) # Giữ nguyên
_WS = re.compile(r"\s+")
def _looks_like_gibberish(s: str) -> bool: ... # Giữ nguyên
def _looks_like_toc_question(q: str) -> bool: ... # Giữ nguyên
def _format_toc_with_llm(toc_lines: list[str]) -> str: ... # Giữ nguyên
# --- Kết thúc helpers ---

# --- Node function `analyze` ---
async def analyze(state: State) -> State:
    """Node function chính cho bước analyze trong Graph."""
    t_start = time.perf_counter()
    print("➡️  [Node] analyze: Bắt đầu...")

    question = (state.get("question") or "").strip()

    # Xử lý câu hỏi rỗng (Giữ nguyên)
    if not question:
        # ... (Code xử lý câu hỏi rỗng giữ nguyên) ...
        print("  [Analyze] ⚠️ Câu hỏi rỗng.")
        analyze_meta = {"question_type": "unclear", "status": "stop", "normalized_question": ""}
        new_state = {**state, "answer": "Mình chưa hiểu câu hỏi của bạn.", "analyze_meta": analyze_meta}
        return _update_timing(new_state, "analyze", t_start)


    # Xử lý TOC bằng Regex (Giữ nguyên)
    if _looks_like_toc_question(question):
        # ... (Code xử lý TOC Regex giữ nguyên) ...
        print(f"  [Analyze] ⚡ Phát hiện TOC bằng Regex cho: '{question[:50]}...'")
        try:
            # ... (Lấy và format TOC) ...
            course_id = str(state.get("course_id") or "").strip()
            entry = get_toc_by_course(course_id)
            toc_lines = flatten_toc(entry)
            answer = _format_toc_with_llm(toc_lines)
            analyze_meta = {
                "status": "stop", "question_type": "academic", "task_type": "single",
                "normalized_question": question,
                "sub_questions_with_intent": [{"subq": question, "intent": ["system_handlers"], "topic": "toc"}],
                "subq_variants": {question: ["mục lục", "table of contents"]}
            }
            new_state = {**state, "answer": answer, "analyze_meta": analyze_meta}
            print(f"✅ [Node] analyze: Xử lý TOC bằng Regex xong.")
            return _update_timing(new_state, "analyze", t_start)
        except Exception as e:
            print(f"  [Analyze] 💥 Lỗi khi xử lý TOC Regex: {e}")
            # Tiếp tục chạy LLM

    # ===== SỬA CHỖ GỌI LLM Analyze =====
    print(f"  [Analyze] 🧠 Gọi hàm analyze_question cho: '{question[:50]}...'")
    result_meta: Dict[str, Any] = {} # Khởi tạo dict rỗng
    try:
        # TRUYỀN `question` (string) vào hàm `analyze_question`
        result_meta = await analyze_question(question) # <-- Sửa ở đây

    except Exception as e:
         print(f"  [Analyze] 💥 Lỗi nghiêm trọng khi gọi analyze_question: {e}")
         # Tạo fallback meta nếu analyze_question lỗi
         result_meta = {
             "normalized_question": question, "question_type": "unclear", "task_type": "single",
             "status": "stop", # Dừng lại nếu analyze lỗi
             "sub_questions_with_intent": [{"subq": question, "intent": ["system_handlers"], "topic": "unclear"}],
             "subq_variants": {question: [question]}
         }
         state["answer"] = "Lỗi: Đã xảy ra sự cố khi phân tích câu hỏi của bạn."
         # Gán fallback vào state và dừng
         new_state = {**state, "analyze_meta": result_meta}
         return _update_timing(new_state, "analyze", t_start)

    # ----- Xử lý kết quả `result_meta` (Giữ nguyên logic) -----
    # Ép gibberish thành unclear topic (Giữ nguyên)
    sqwi = result_meta.get("sub_questions_with_intent", [])
    sqwi_fix = []
    has_gibberish = False
    for item in sqwi:
         subq = item.get("subq", "")
         intent = item.get("intent", ["unknown"])[0]
         topic = item.get("topic", "")
         if _looks_like_gibberish(subq):
             print(f"  [Analyze] Phát hiện Gibberish: '{subq[:30]}...' -> Đổi thành system_handlers/unclear")
             intent = "system_handlers"
             topic = "unclear"
             has_gibberish = True
         sqwi_fix.append({"subq": subq, "intent": [intent], "topic": topic})
    result_meta["sub_questions_with_intent"] = sqwi_fix # Cập nhật lại meta

    # Xác định trạng thái cuối cùng (Giữ nguyên)
    final_status = "academic"
    all_intents = set(item["intent"][0] for item in sqwi_fix if item.get("intent"))
    all_topics = set(item.get("topic") for item in sqwi_fix if item.get("topic"))
    if "system_handlers" in all_intents:
        if "unsafe" in all_topics: final_status = "unsafe"
        elif "unclear" in all_topics: final_status = "unclear"
        elif all(item["intent"][0] == "system_handlers" and item["topic"] in ("meta", "toc") for item in sqwi_fix):
             final_status = "meta_or_toc_only"
    result_meta["status"] = final_status # Lưu trạng thái

    # Xử lý dừng sớm (Giữ nguyên)
    if final_status == "unsafe":
        print("  [Analyze] ⚠️ Phát hiện nội dung không an toàn (sẽ xử lý qua SYSTEM block).")
        result_meta["status"] = "continue"   # ⬅️ không stop, không gán answer
        # KHÔNG return ở đây

    if final_status == "unclear" or (has_gibberish and len(sqwi_fix) == 1):
        print("  [Analyze] ⚠️ Câu hỏi không rõ ràng (sẽ xử lý qua SYSTEM block).")
        result_meta["status"] = "continue"   # ⬅️ không stop, không gán answer
        # KHÔNG return ở đây

    if final_status == "meta_or_toc_only":
         print(f"  [Analyze] ⚡ Câu hỏi chỉ chứa Meta/TOC.")
         result_meta["status"] = "continue" # Để dispatcher xử lý

    # ===== Mặc định: Tiếp tục pipeline =====
    print(f"✅ [Node] analyze: Phân tích xong. Trạng thái: {final_status}")
    # Gán meta vào state
    new_state = {
        **state,
        "question_user": question,
        "question_norm": result_meta.get("normalized_question", question),
        # Đảm bảo gán đúng dict kết quả vào key "analyze_meta"
        "analyze_meta": result_meta,
    }
    return _update_timing(new_state, "analyze", t_start)

