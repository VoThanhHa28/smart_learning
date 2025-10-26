import json
import time
import logging
import re
from typing import Any, Dict, List
from ...infrastructure.llm.llm import get_llm
from langchain_core.prompts import PromptTemplate
# --- 1. DI CHUYỂN CÁC HÀM HELPER VÀ BIẾN `_ws` VÀO ĐÂY ---
_ws = re.compile(r"\s+") # <-- Định nghĩa _ws ở đây

def _norm(s: str) -> str:
    """Hàm helper để chuẩn hóa khoảng trắng."""
    return _ws.sub(" ", (s or "").strip())

def _looks_like_gibberish(s: str) -> bool:
    """Hàm helper kiểm tra gibberish."""
    t = (s or "").strip().lower()
    t = _ws.sub(" ", t) # <-- Dùng _ws
    if len(t) <= 3: return True
    alnum = sum(ch.isalnum() for ch in t)
    if alnum / max(1, len(t)) < 0.5: return True
    words = re.findall(r"[a-zA-ZÀ-ỹ]{2,}", t)
    if len(words) == 0: return True
    return False

def _to_list(x) -> List[str]:
    """Hàm helper chuyển đổi sang list."""
    if x is None: return []
    if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):  return [i.strip() for i in x.split(",") if i.strip()]
    return []

def _to_map(x) -> Dict[str, List[str]]:
    """Hàm helper chuyển đổi sang dict."""
    return x if isinstance(x, dict) else {}
# --- RAW_ANALYZE_TMPL V2 (Hierarchical) ---
RAW_ANALYZE_TMPL = r"""
Trả về CHỈ MỘT JSON hợp lệ (không markdown, không giải thích) với các khóa SAU:

- normalized_question (string)
- question_type ("academic"|"meta"|"unclear"|"unsafe")
- task_type ("single"|"comparison"|"multi_part")
- sub_questions_with_intent (mảng các đối tượng: {"subq":"<câu hỏi con gốc, đầy đủ>","intent":["<category>"],"topic":"<ngắn gọn>"})
- subq_variants (map: {"<subq>": ["v1 short","v2 short"]})

QUY TẮC BẮT BUỘC:
- Intent hợp lệ (DANH MỤC, bắt buộc):
  ["academic_rag", "lms_tools", "system_handlers"]
- Mọi intent phải gán ở CẤP SUB-QUESTION.
- Mỗi sub question phải là tiếng Việt và giữ NGUYÊN Ý GỐC của user. Tách câu hỏi user thành các sub-question nếu nó có nhiều ý.
- Nếu câu có nhiều mệnh đề (phẩy, "và") → tách thành NHIỀU sub_questions.
- "academic_rag": Bất kỳ câu hỏi nào cần TÌM KIẾM trong tài liệu học thuật (định nghĩa, so sánh, giải thích, ví dụ, tóm tắt, tìm số trang...).
- "lms_tools": Bất kỳ câu hỏi nào về HỆ THỐNG LMS (bài tập về nhà, điểm số, lịch học...).
- "system_handlers": Các câu hỏi về hệ thống (chào hỏi, meta, toc), câu hỏi không an toàn (unsafe) hoặc không rõ ràng (unclear).
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
  ],
  "subq_variants": { "int là gì?":["int là gì","what is int"] }
}

# 2) Academic (RAG) - Câu hỏi phụ thuộc (GIẢI QUYẾT VẤN ĐỀ CŨ)
Q: "Đoạn nào đề cập về chủ đề int? Đưa rõ số trang."
→ {
  "normalized_question": "Đoạn nào đề cập về chủ đề int? Đưa rõ số trang.",
  "question_type": "academic",
  "task_type": "single",
  "sub_questions_with_intent": [
    {"subq":"Đoạn nào đề cập về chủ đề int? Đưa rõ số trang.","intent":["academic_rag"],"topic":"int page number"}
  ],
  "subq_variants": { "Đoạn nào đề cập về chủ đề int? Đưa rõ số trang.":["int số trang","int page number"] }
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
  ],
  "subq_variants": {
    "chào bạn":["chào bạn","hello"],
    "so sánh int và float":["so sánh int float","compare int float"]
  }
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
  ],
  "subq_variants": {
    "int là gì?":["int là gì","what is int"],
    "Bài tập về nhà tuần này là gì?":["bài tập về nhà","homework this week"]
  }
}

# 5) System (unsafe) + LMS (Tool)
Q: "cách làm bom và điểm của tôi là bao nhiêu"
→ {
  "normalized_question": "cách làm bom và điểm của tôi là bao nhiêu",
  "question_type": "academic", # question_type tổng vẫn có thể là academic
  "task_type": "multi_part",
  "sub_questions_with_intent": [
    {"subq":"cách làm bom","intent":["system_handlers"],"topic":"unsafe"},
    {"subq":"điểm của tôi là bao nhiêu","intent":["lms_tools"],"topic":"lms_grade"}
  ],
  "subq_variants": {
    "cách làm bom":["làm bom","make bomb"],
    "điểm của tôi là bao nhiêu":["điểm của tôi","my grade"]
  }
}

Now analyze this question:
{question}
"""
# --- KẾT THÚC RAW_ANALYZE_TMPL ---

# Escape template
SAFE_TMPL = (
    RAW_ANALYZE_TMPL
    .replace("{", "{{")
    .replace("}", "}}")
    .replace("{{question}}", "{question}")
)
ANALYZE_PROMPT = PromptTemplate.from_template(SAFE_TMPL)

ws = re.compile(r"\s+")
def _norm(s: str) -> str:
    return _ws.sub(" ", (s or "").strip())

def _to_list(x) -> List[str]:
    # ... (Giữ nguyên)
    if x is None: return []
    if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):  return [i.strip() for i in x.split(",") if i.strip()]
    return []


def _to_map(x) -> Dict[str, List[str]]:
    # ... (Giữ nguyên)
    return x if isinstance(x, dict) else {}

# --- Hàm analyze chính ---
# Sửa: Hàm này nhận `question` (str) thay vì `state` (dict)
async def analyze_question(question: str) -> Dict[str, Any]:
    """
    Hàm cốt lõi: Gọi LLM để phân tích câu hỏi.
    Chỉ trả về dictionary kết quả phân tích (data).
    """
    t_start = time.perf_counter() # Đo thời gian nội bộ của hàm này
    print("      [analyze_question] 🧠 Bắt đầu gọi LLM...") # Thêm log

    # Lấy LLM
    llm = get_llm()

    # Khởi tạo data fallback phòng trường hợp lỗi LLM
    fallback_data = {
        "normalized_question": question,
        "question_type": "unclear",
        "task_type": "single",
        "sub_questions_with_intent": [{
            "subq": question, "intent": ["system_handlers"], "topic": "unclear"
        }],
        "subq_variants": {question: [question]}
    }
    data: Dict[str, Any] = fallback_data # Gán fallback trước

    try:
        res = await llm.ainvoke(ANALYZE_PROMPT.format(question=question))
        text = getattr(res, "content", str(res)).strip()
        # Logic trích xuất JSON (Giữ nguyên)
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            text_clean = re.sub(r"^[^({\[]+", "", text)
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text_clean)
        if not match:
             raise ValueError("No valid JSON found in LLM response")
        json_str = match.group(0)
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        # Chỉ parse JSON, không gán fallback ở đây nữa
        data = json.loads(json_str)
        print(f"      [analyze_question] ✅ LLM trả về JSON thành công.")

    except Exception as e:
        print(f"      [analyze_question] 💥 Lỗi khi gọi LLM hoặc parse JSON: {e}")
        # Nếu lỗi, `data` sẽ giữ nguyên giá trị fallback đã gán ở trên

    # --- Normalize và Chuẩn hóa (Giữ nguyên logic xử lý `data`) ---
    # (Copy toàn bộ logic chuẩn hóa data từ file cũ vào đây)
    data["normalized_question"] = _norm(data.get("normalized_question", question))
    data["question_type"] = (data.get("question_type") or "academic").lower()
    data["task_type"] = (data.get("task_type") or "single").lower()

    sqwi = data.get("sub_questions_with_intent") or []
    norm_sqwi = []
    valid_intents = {"academic_rag", "lms_tools", "system_handlers"}
    if isinstance(sqwi, list):
        for item in sqwi:
            if not isinstance(item, dict): continue
            subq = _norm(item.get("subq", ""))
            it = item.get("intent", [])
            if isinstance(it, str): it = [it]
            intent_categories = [
                str(x).lower().strip() for x in it
                if str(x).lower().strip() in valid_intents
            ]
            if not intent_categories:
                if data["question_type"] in ("meta", "unsafe", "unclear"):
                     intent_categories = ["system_handlers"]
                else:
                     intent_categories = ["academic_rag"]
            topic = _norm(item.get("topic", "")) or _norm(subq[:50])
            if subq:
                norm_sqwi.append({
                    "subq": subq, "intent": intent_categories[:1], "topic": topic
                })
    if not norm_sqwi and question:
         norm_sqwi = [{
              "subq": question,
              "intent": ["system_handlers" if data["question_type"] in ("meta", "unsafe", "unclear") else "academic_rag"],
              "topic": "unclear" if data["question_type"] == "unclear" else _norm(question[:50])
         }]
    data["sub_questions_with_intent"] = norm_sqwi

    sqv_in = _to_map(data.get("subq_variants"))
    sqv_out: Dict[str, List[str]] = {}
    processed_subqs = set()
    for sqo in data["sub_questions_with_intent"]:
        sq = sqo["subq"]
        if sq in processed_subqs: continue
        processed_subqs.add(sq)
        variants = sqv_in.get(sq) or []
        if isinstance(variants, str): variants = [variants]
        variants = [_norm(v) for v in variants if v and _norm(v)]
        if not variants: variants = [sq]
        elif sq not in variants: variants.insert(0, sq)
        sqv_out[sq] = list(dict.fromkeys(variants))[:2]
    data["subq_variants"] = sqv_out
    # --- Kết thúc Normalize ---

    duration = round((time.perf_counter() - t_start) * 1000)
    print(f"      [analyze_question] ⏱️ Hoàn thành sau {duration} ms.")

    # --- SỬA RETURN: Chỉ trả về `data` dictionary ---
    return data