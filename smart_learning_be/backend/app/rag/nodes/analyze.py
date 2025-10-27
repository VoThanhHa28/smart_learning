import logging
import time
import re
import json # <-- Thêm json
from typing import Any, Dict, List
# Import hàm cốt lõi và helper
from ...rag.analysis.analyze_query import analyze_question, _looks_like_gibberish
# Import utils
from ...services.utils.toc_store import get_toc_by_course, get_toc_by_key, flatten_toc
from ..prompts.prompt_utils import build_toc_validation_block
# Import State và timing
from ..rag_state import State, _update_timing

# --- Regex TOC và Helpers ---
# Mở rộng regex TOC
_TOC_KWS = (
    r"\bmục lục\b", r"\bmuc luc\b", r"\btoc\b",
    r"\btable of contents\b", r"\bcontents\b",
    r"\bdanh mục chương\b", r"\bcấu trúc chương\b",
    r"\blist of chapters\b", r"\boutline\b",
    r"\bmục lục (bài|bài học|bài giảng|giáo trình)\b"
)
_TOC_RX = re.compile("|".join(_TOC_KWS), re.IGNORECASE)
_WS = re.compile(r"\s+")
# Hàm _looks_like_gibberish đã được import từ analyze_query

def _looks_like_toc_question(q: str) -> bool:
    """Kiểm tra xem câu hỏi có phải là yêu cầu TOC không."""
    q_norm = (q or "").strip().lower()
    if not q_norm:
        return False
    # Kiểm tra độ dài để tránh câu dài vô tình chứa từ khóa
    return len(q_norm) < 100 and bool(_TOC_RX.search(q_norm))

def _format_toc_with_llm(toc_lines: list[str]) -> str:
    """Gọi LLM (Sync) để xác thực và định dạng TOC."""
    if not toc_lines:
        return "Tài liệu này hiện không có thông tin mục lục."

    prompt_text = build_toc_validation_block(toc_lines)
    
    # Import local để tránh circular dependency nếu llm_utils import ngược lại
    try:
        from ...infrastructure.llm.llm_utils import sync_stream_generate
        # Dùng sync_stream_generate vì đây là hàm sync blocking
        answer = sync_stream_generate(prompt_text)
        # LLM có thể trả về câu lỗi nếu input không hợp lệ
        if "mục lục tôi nhận được đang lỗi" in answer.lower():
             logging.warning(f"[Analyze] LLM TOC Validator báo lỗi.")
             return "Thông tin mục lục hiện đang được cập nhật, vui lòng thử lại sau."
        return answer # Trả về TOC đã được LLM format
    except ImportError:
         logging.error("[Analyze] Không thể import sync_stream_generate cho _format_toc_with_llm.")
         return "Lỗi: Không thể định dạng mục lục."
    except Exception as e:
         logging.exception(f"[Analyze] Lỗi khi gọi LLM format TOC: {e}")
         return "Lỗi: Đã xảy ra sự cố khi lấy mục lục."
# --- Kết thúc helpers ---

# --- Node function `analyze` ---
async def analyze(state: State) -> State:
    """Node function chính cho bước analyze trong Graph."""
    t_start = time.perf_counter()
    print("➡️  [Node] analyze: Bắt đầu...")

    question = (state.get("question") or "").strip()

    # Xử lý câu hỏi rỗng
    if not question:
        print("  [Analyze] ⚠️ Câu hỏi rỗng.")
        analyze_meta = {
            "question_type": "unclear", "status": "stop",
            "normalized_question": "",
            "sub_questions_with_intent": [{"subq": "", "intent": ["system_handlers"], "topic": "unclear"}]
        }
        new_state = {**state, "answer": "Mình chưa hiểu câu hỏi của bạn. Bạn có thể nói cụ thể hơn không?", "analyze_meta": analyze_meta}
        return _update_timing(new_state, "analyze", t_start)

    # Xử lý TOC bằng Regex (Fast Path)
    if _looks_like_toc_question(question):
        print(f"  [Analyze] ⚡ Phát hiện TOC bằng Regex cho: '{question[:50]}...'")
        try:
            course_id = str(state.get("course_id") or "").strip()
            # doc_key = state.get("doc_key") # Tạm thời chưa dùng
            entry = get_toc_by_course(course_id) # Chỉ dùng course_id
            toc_lines = flatten_toc(entry)
            answer = _format_toc_with_llm(toc_lines) # Gọi hàm sync

            analyze_meta = {
                "status": "stop", # Dừng pipeline
                "question_type": "academic",
                "task_type": "single",
                "normalized_question": question,
                "sub_questions_with_intent": [{"subq": question, "intent": ["system_handlers"], "topic": "toc"}],
            }
            new_state = {**state, "answer": answer, "analyze_meta": analyze_meta}
            print(f"✅ [Node] analyze: Xử lý TOC bằng Regex xong.")
            return _update_timing(new_state, "analyze", t_start)
        except Exception as e:
            print(f"  [Analyze] 💥 Lỗi xử lý TOC Regex: {e}. Tiếp tục LLM analyze...")
            # Không return, để LLM analyze xử lý tiếp

    # ===== GỌI LLM Analyze (Hàm analyze_question) =====
    print(f"  [Analyze] 🧠 Gọi hàm analyze_question cho: '{question[:50]}...'")
    result_meta: Dict[str, Any] = {} # Khởi tạo dict rỗng
    try:
        result_meta = await analyze_question(question) # Gọi hàm cốt lõi

        # --- LOG DEBUG CHI TIẾT `result_meta` ---
        print("\n--- DEBUG: Raw result_meta from analyze_question ---")
        try:
            print(json.dumps(result_meta, indent=2, ensure_ascii=False, default=str))
        except Exception as json_dump_err:
            print(f"  (Lỗi khi dump JSON result_meta: {json_dump_err})")
            print(f"  Raw value: {result_meta}")
        print("----------------------------------------------------\n")
        # --- KẾT THÚC LOG DEBUG ---

        if not isinstance(result_meta, dict) or not result_meta:
             print(f"  [Analyze] 💥 Lỗi: analyze_question trả về kiểu không hợp lệ: {type(result_meta)}")
             raise ValueError("analyze_question did not return a valid dictionary.")

    except Exception as e:
         print(f"  [Analyze] 💥 Lỗi nghiêm trọng khi gọi/xử lý analyze_question: {e}")
         logging.exception("Error during analyze_question call/processing")
         # Tạo fallback
         result_meta = {
             "normalized_question": question, "question_type": "unclear", "task_type": "single",
             "status": "stop", # Dừng lại nếu analyze lỗi
             "sub_questions_with_intent": [{"subq": question, "intent": ["system_handlers"], "topic": "unclear"}],
         }
         state["answer"] = "Lỗi: Đã xảy ra sự cố khi phân tích câu hỏi của bạn."
         new_state = {**state, "analyze_meta": result_meta}
         return _update_timing(new_state, "analyze", t_start)

    # ----- Xử lý kết quả `result_meta` (Logic an toàn) -----
    print("  [Analyze] ⚙️ Bắt đầu xử lý kết quả từ LLM analyze...")
    sqwi_fix = []
    has_gibberish = False
    sqwi = result_meta.get("sub_questions_with_intent", [])
    current_question_type = result_meta.get("question_type", "academic") # Lấy type từ LLM

    if isinstance(sqwi, list):
        for item in sqwi:
             if not isinstance(item, dict) or not item:
                  print(f"    [Analyze] ⚠️ Bỏ qua item không hợp lệ: {item}")
                  continue

             subq = item.get("subq", "")
             # Xử lý intent an toàn
             intent_list = item.get("intent")
             intent = "academic_rag" # Mặc định
             raw_intent_value = None
             if isinstance(intent_list, list) and intent_list:
                  raw_intent_value = str(intent_list[0]).lower().strip()
             elif isinstance(intent_list, str) and intent_list.strip():
                  raw_intent_value = intent_list.lower().strip()

             valid_intents = {"academic_rag", "lms_tools", "system_handlers"}
             if raw_intent_value in valid_intents:
                  intent = raw_intent_value
             elif raw_intent_value: # Có giá trị nhưng không hợp lệ
                  print(f"    [Analyze] ⚠️ Intent không hợp lệ '{raw_intent_value}', fallback sang academic_rag.")
                  # intent = "academic_rag" (Đã là mặc định)
             # else: (Nếu None hoặc list rỗng) intent = "academic_rag"

             topic = item.get("topic", "")

             # Kiểm tra Gibberish
             try:
                 if _looks_like_gibberish(subq):
                    print(f"    [Analyze] Phát hiện Gibberish: '{subq[:30]}...' -> Đổi thành system/unclear")
                    intent = "system_handlers"
                    topic = "unclear"
                    has_gibberish = True
                    current_question_type = "unclear"
             except NameError: pass # Bỏ qua nếu hàm chưa import (dù đã import ở trên)
             except Exception as gib_err:
                  print(f"    [Analyze] 💥 Lỗi kiểm tra gibberish: {gib_err}")

             if topic == "unsafe": # Cập nhật type nếu unsafe
                  current_question_type = "unsafe"

             if subq:
                  sqwi_fix.append({"subq": subq, "intent": [intent], "topic": topic})
             else:
                  print(f"    [Analyze] ⚠️ Bỏ qua sub-question rỗng.")

    else: # sqwi không phải list
        print(f"  [Analyze] ⚠️ 'sub_questions_with_intent' không phải list: {sqwi}")
        if question:
             sqwi_fix = [{"subq": question, "intent": ["system_handlers"], "topic": "unclear"}]
             current_question_type = "unclear"; has_gibberish = True

    # Cập nhật meta
    result_meta["sub_questions_with_intent"] = sqwi_fix
    result_meta["has_gibberish_subq"] = has_gibberish
    result_meta["question_type"] = current_question_type

    # Xác định trạng thái cuối cùng
    final_status = "continue" # Mặc định tiếp tục
    has_non_system_intent = any(
        (item.get("intent") or [""])[0] not in ("system_handlers", "unknown")
        for item in sqwi_fix
    )

    if current_question_type == "unsafe" and not has_non_system_intent:
         final_status = "stop"; state["answer"] = "Xin lỗi, mình không thể hỗ trợ với nội dung này."
         print(f"  [Analyze] ⚠️ TOÀN BỘ unsafe. Dừng pipeline.")
    elif current_question_type == "unclear" and not has_non_system_intent:
         final_status = "stop"; state["answer"] = "Mình chưa hiểu rõ câu hỏi, bạn có thể nói cụ thể hơn?"
         print(f"  [Analyze] ⚠️ TOÀN BỘ unclear. Dừng pipeline.")
    elif current_question_type == "meta" and not has_non_system_intent:
         print(f"  [Analyze] ⚡ Câu hỏi chỉ chứa Meta (sẽ được dispatcher xử lý).")
         # final_status = "continue" (Đã là mặc định)
    elif any(item.get("topic") == "toc" for item in sqwi_fix) and not has_non_system_intent:
         print(f"  [Analyze] ⚡ Câu hỏi chỉ chứa TOC (sẽ được dispatcher xử lý).")
         # final_status = "continue" (Đã là mặc định)

    result_meta["status"] = final_status

    # ===== Trả về State =====
    print(f"✅ [Node] analyze: Phân tích xong. Trạng thái cuối: {final_status}")
    new_state = {
        **state,
        "question_user": question,
        "question_norm": result_meta.get("normalized_question", question),
        "analyze_meta": result_meta,
        # Gán answer CHỈ KHI status là stop
        "answer": state.get("answer") if final_status == "stop" else None
    }
    return _update_timing(new_state, "analyze", t_start)