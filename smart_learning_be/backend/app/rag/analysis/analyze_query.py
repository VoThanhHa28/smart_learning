import json
import time
import logging
import re
from typing import Any, Dict, List
from ...infrastructure.llm.llm import get_llm
from langchain_core.prompts import PromptTemplate

# --- Helper functions (Giữ nguyên) ---
_ws = re.compile(r"\s+")
def _norm(s: Any) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    return _ws.sub(" ", s.strip())

def _looks_like_gibberish(s: str) -> bool:
    t = (s or "").strip().lower()
    t = _ws.sub(" ", t)
    if len(t) <= 3: return True
    alnum = sum(ch.isalnum() for ch in t)
    if alnum / max(1, len(t)) < 0.5: return True
    words = re.findall(r"[a-zA-ZÀ-ỹ]{2,}", t)
    if len(words) == 0: return True
    return False

def _to_list(x: Any) -> List[str]:
    if x is None: return []
    if isinstance(x, list): return [_norm(str(i)) for i in x if str(i).strip()]
    if isinstance(x, str):  return [_norm(i) for i in x.split(",") if i.strip()]
    try:
        return [_norm(str(i)) for i in x if str(i).strip()]
    except TypeError:
        return []

def _to_map(x: Any) -> Dict[str, List[str]]:
    if isinstance(x, dict):
        cleaned_dict = {}
        for k, v in x.items():
            key_str = _norm(str(k))
            if key_str:
                 cleaned_dict[key_str] = _to_list(v)
        return cleaned_dict
    return {}
# --- Kết thúc Helper ---

# --- RAW_ANALYZE_TMPL V5 (Bỏ "unclear" khỏi LLM) ---
RAW_ANALYZE_TMPL = r"""
Trả về CHỈ MỘT JSON hợp lệ (không markdown, không giải thích) với các khóa SAU:
- normalized_question (string)
- question_type ("academic"|"meta"|"unsafe")
- task_type ("single"|"comparison"|"multi_part")
- sub_questions_with_intent (mảng các đối tượng: {"subq":"<câu hỏi con gốc, đầy đủ>","intent":["<category>"],"topic":"<ngắn gọn>"})

QUY TẮC BẮT BUỘC:
- Intent hợp lệ (DANH MỤC, bắt buộc):
  ["academic_rag", "lms_tools", "system_handlers"]
- Mọi intent phải gán ở CẤP SUB-QUESTION.
- Mỗi sub question phải là tiếng Việt và giữ NGUYÊN Ý GỐC của user.
- "academic_rag": Bất kỳ câu hỏi nào cần TÌM KIẾM trong tài liệu học thuật (định nghĩa, so sánh, giải thích, ví dụ, tóm tắt, tìm số trang, kể cả các chủ đề chung chung như 'tìm hiểu bản thân').
- "lms_tools": Bất kỳ câu hỏi nào về HỆ THỐNG LMS (bài tập về nhà, điểm số, lịch học...).
- "system_handlers": Các câu hỏi về hệ thống (chào hỏi, meta, toc), hoặc câu hỏi không an toàn (unsafe).
- **KHÔNG** được gán intent là "system_handlers" với topic "unclear". Nếu câu hỏi vô nghĩa (gibberish), hãy gán nó là "academic_rag" với topic "gibberish".
- CHỈ JSON hợp lệ.

VÍ DỤ CHUẨN:

# 1) Academic (RAG)
Q: "int là gì?"
→ {
  "normalized_question": "int là gì?",
  "question_type": "academic",
  "task_type": "single",
  "sub_questions_with_intent": [
    {"subq":"int là gì?","intent":["academic_rag"],"topic":"int"}
  ]
}

# 2) Academic (RAG) - Câu hỏi phụ thuộc
Q: "Đoạn nào đề cập về chủ đề int? Đưa rõ số trang."
→ {
  "normalized_question": "Đoạn nào đề cập về chủ đề int? Đưa rõ số trang.",
  "question_type": "academic",
  "task_type": "single",
  "sub_questions_with_intent": [
    {"subq":"Đoạn nào đề cập về chủ đề int? Đưa rõ số trang.","intent":["academic_rag"],"topic":"int page number"}
  ]
}

# 3) Academic (RAG) + System (meta)
Q: "chào bạn, so sánh int và float"
→ {
  "normalized_question": "chào bạn, so sánh int và float",
  "question_type": "academic",
  "task_type": "multi_part",
  "sub_questions_with_intent": [
    {"subq":"chào bạn","intent":["system_handlers"],"topic":"meta"},
    {"subq":"so sánh int và float","intent":["academic_rag"],"topic":"int and float"}
  ]
}

# 4) Academic (RAG) + LMS (Tool)
Q: "int là gì và bài tập về nhà tuần này?"
→ {
  "normalized_question": "int là gì và bài tập về nhà tuần này?",
  "question_type": "academic",
  "task_type": "multi_part",
  "sub_questions_with_intent": [
    {"subq":"int là gì?","intent":["academic_rag"],"topic":"int"},
    {"subq":"Bài tập về nhà tuần này là gì?","intent":["lms_tools"],"topic":"lms_homework"}
  ]
}

# 5) System (unsafe) + LMS (Tool)
Q: "cách làm bom và điểm của tôi là bao nhiêu"
→ {
  "normalized_question": "cách làm bom và điểm của tôi là bao nhiêu",
  "question_type": "academic",
  "task_type": "multi_part",
  "sub_questions_with_intent": [
    {"subq":"cách làm bom","intent":["system_handlers"],"topic":"unsafe"},
    {"subq":"điểm của tôi là bao nhiêu","intent":["lms_tools"],"topic":"lms_grade"}
  ]
}

# 6) Academic (RAG) - Câu hỏi chung chung
Q: "tìm hiểu bản thân"
→ {
  "normalized_question": "tìm hiểu bản thân",
  "question_type": "academic",
  "task_type": "single",
  "sub_questions_with_intent": [
    {"subq":"tìm hiểu bản thân","intent":["academic_rag"],"topic":"tìm hiểu bản thân"}
  ]
}

# 7) Gibberish (Vô nghĩa)
Q: "asdfg zxcvb"
→ {
  "normalized_question": "asdfg zxcvb",
  "question_type": "academic",
  "task_type": "single",
  "sub_questions_with_intent": [
    {"subq":"asdfg zxcvb","intent":["academic_rag"],"topic":"gibberish"}
  ]
}

Now analyze this question:
{question}
"""
# --- KẾT THÚC RAW_ANALYZE_TMPL ---

# Escape template (Giữ nguyên)
SAFE_TMPL = (
    RAW_ANALYZE_TMPL
    .replace("{", "{{")
    .replace("}", "}}")
    .replace("{{question}}", "{question}")
)
ANALYZE_PROMPT = PromptTemplate.from_template(SAFE_TMPL)

# --- Hàm analyze_question chính (Giữ nguyên) ---
async def analyze_question(question: str) -> Dict[str, Any]:
    """
    Hàm cốt lõi V5: Gọi LLM phân tích (đã bỏ "unclear" khỏi prompt).
    Xử lý kết quả JSON an toàn.
    """
    t_start = time.perf_counter()
    print("      [analyze_question V5] 🧠 Bắt đầu gọi LLM...")
    llm = get_llm() # Sử dụng model nhanh
    fallback_data = {
        "normalized_question": question, "question_type": "academic", "task_type": "single",
        "sub_questions_with_intent": [{"subq": question, "intent": ["academic_rag"], "topic": "fallback_topic"}],
    }
    data: Dict[str, Any] = fallback_data

    try:
        # Gọi LLM
        res = await llm.ainvoke(ANALYZE_PROMPT.format(question=question))
        text = getattr(res, "content", str(res)).strip()
        # Trích xuất JSON
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
             text_clean = re.sub(r"^[^({\[]+", "", text)
             match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text_clean)
             if not match: raise ValueError("No valid JSON found")
        json_str = match.group(0)
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        parsed_data = json.loads(json_str)

        if not isinstance(parsed_data, dict):
             print(f"      [analyze_question V5] ⚠️ LLM JSON parsed không phải dict.")
             # data giữ nguyên fallback
        else:
             data = parsed_data
             print(f"      [analyze_question V5] ✅ LLM trả về JSON.")
    except Exception as e:
        print(f"      [analyze_question V5] 💥 Lỗi LLM/parse: {e}")
        # data giữ nguyên fallback (lần này fallback là academic_rag)

    # --- Normalize và Chuẩn hóa (Giữ nguyên) ---
    try:
        data["normalized_question"] = _norm(data.get("normalized_question", question))
        data["question_type"] = (data.get("question_type") or "academic").lower()
        data["task_type"] = (data.get("task_type") or "single").lower()

        sqwi = data.get("sub_questions_with_intent") or []
        norm_sqwi = []
        # Sửa: Bỏ "unclear" khỏi valid_intents
        valid_intents = {"academic_rag", "lms_tools", "system_handlers"}
        has_gibberish_subq = False # Cờ này sẽ được set bởi Python (node cha)
        if isinstance(sqwi, list):
            for item in sqwi:
                if not isinstance(item, dict) or not item:
                    print(f"        [SQW Processor] ⚠️ Bỏ qua item không hợp lệ: {item}")
                    continue
                # Xử lý subq an toàn
                raw_subq = item.get("subq")
                subq_str = _norm(raw_subq)
                if not subq_str:
                     print(f"        [SQW Processor] ⚠️ Bỏ qua item có subq rỗng.")
                     continue
                
                # Xử lý intent an toàn (Giữ nguyên)
                intent_list = item.get("intent")
                intent = "academic_rag" # Mặc định
                raw_intent_value = None
                if isinstance(intent_list, list) and intent_list:
                    raw_intent_value = str(intent_list[0]).lower().strip()
                elif isinstance(intent_list, str) and intent_list.strip():
                    raw_intent_value = intent_list.lower().strip()
                if raw_intent_value in valid_intents:
                    intent = raw_intent_value
                elif raw_intent_value:
                     print(f"        [SQW Processor] ⚠️ Intent '{raw_intent_value}' không hợp lệ, fallback 'academic_rag'.")
                
                # Xử lý topic
                topic = _norm(item.get("topic", "")) or _norm(subq_str[:50])
                
                # BỎ kiểm tra gibberish ở đây (để node cha `analyze.py` làm)
                # if _looks_like_gibberish(subq_str):
                #     ...

                norm_sqwi.append({"subq": subq_str, "intent": [intent], "topic": topic})

        if not norm_sqwi and question: # Fallback list subq
             q_type_fallback = data.get("question_type", "academic") # Mặc định academic
             norm_sqwi = [{"subq": question, "intent": ["academic_rag"], "topic": _norm(question[:50])}]
             data["question_type"] = q_type_fallback
             
        data["sub_questions_with_intent"] = norm_sqwi
        # Bỏ 'has_gibberish_subq' vì node cha sẽ quyết định
        if "has_gibberish_subq" in data: del data["has_gibberish_subq"]
        if "subq_variants" in data: del data["subq_variants"]

    except Exception as process_e:
         print(f"      [analyze_question V5] 💥 Lỗi khi xử lý data JSON: {process_e}")
         logging.exception("Error processing analyze_question result data")
         data = {
            "normalized_question": question, "question_type": "academic", "task_type": "single",
            "sub_questions_with_intent": [{"subq": question, "intent": ["academic_rag"], "topic": "fallback_process_error"}],
            "status": "continue", # Vẫn cho chạy RAG với câu gốc
            "processing_error": str(process_e)
         }

    duration = round((time.perf_counter() - t_start) * 1000)
    print(f"      [analyze_question V5] ⏱️ Hoàn thành sau {duration} ms.")
    return data


