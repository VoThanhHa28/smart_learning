# app/services/analyze_query.py
# -*- coding: utf-8 -*-

import json, time, logging, re
from typing import Any, Dict, List
from ...infrastructure.llm.llm import get_llm
from langchain_core.prompts import PromptTemplate
RAW_ANALYZE_TMPL = r"""
Trả về CHỈ MỘT JSON hợp lệ (không markdown, không giải thích) với các khóa SAU:

- normalized_question (string)
- question_type ("academic"|"meta"|"unclear"|"unsafe")
- task_type ("single"|"comparison"|"multi_part")
- sub_questions_with_intent (mảng các đối tượng: {"subq":"...","intent":["..."],"topic":"<ngắn gọn theo chính subq>"})
- subq_variants (đối tượng map: {"<subq>": ["v1 short","v2 short"]}, mỗi subq trong subq_variants có 2 biến thể (1 tiếng Việt + 1 tiếng Anh), ngắn, không stopwords)

QUY TẮC BẮT BUỘC:
- Intent hợp lệ (enum, bắt buộc): 
  ["definition","comparison","application","warning","exercise","paraphrase","socratic_question","example","quality","meta","toc","unsafe","unclear","router_fallback"].
- Không phát minh intent ngoài danh sách. Nếu không chắc ≥80% → dùng "other".
- Mọi intent phải gán ở CẤP SUB-QUESTION (mỗi subq có mảng intent riêng).
- Mỗi sub question là tiếng Việt và phải có ý nghĩa cho truy vấn (không gộp nhiều câu thành 1 sub question, mỗi sub question phải là 1 ý) -> Tách câu hỏi user thành các sub-question nếu nó có nhiều ý, tự tạo sub question rõ nghĩa mà vẫn đúng ý user.
- Nếu câu có nhiều mệnh đề cần tách (ngăn bởi dấu phẩy, chấm phẩy, xuống dòng, "và", "and") → tách thành NHIỀU sub_questions có ý nghĩa cho truy vấn, không gộp.
- Nếu phát hiện yêu cầu Mục lục/TOC/Outline → tạo subq với intent=["toc"], topic="toc". CHỈ ép intent cho đúng subq TOC, không ép các subq khác.
- Nếu phát hiện chào hỏi/giới thiệu hệ thống ("chào bạn", "bạn là ai", "who are you") → subq intent=["meta"], topic="meta".
- Nếu phát hiện nội dung cấm/độc hại (ví dụ làm bom, hack, chất nổ, tự hại...) → subq intent=["unsafe"], topic="unsafe".
- Không giới hạn cứng số sub_questions; nếu nhiều, liệt kê đầy đủ.
- Nếu có “X vs Y” → topic = "X and Y".
- Nếu “What is X?” → topic = "X".
- Nếu SUB-QUESTION là chuỗi VÔ NGHĨA (token đơn chỉ chữ dài ≥10 và thiếu nguyên âm/tỉ lệ nguyên âm <25%, chủ yếu ký hiệu/emoji, lặp vô nghĩa, hoặc có phụ âm liền ≥4) → intent=["unclear"], topic="unclear".
- CHỈ JSON hợp lệ.

VÍ DỤ CHUẨN:

# 1) TOC + academic (nhiều mệnh đề)
Q: "mục lục của bài học, int là gì?"
→ {
  "normalized_question": "mục lục của bài học, int là gì?",
  "question_type": "academic",
  "task_type": "multi_part",
  "sub_questions_with_intent": [
    {"subq":"Mục lục của bài học?","intent":["toc"],"topic":"toc"},
    {"subq":"What is int?","intent":["definition"],"topic":"int"}
  ],
  "subq_variants": {
    "Mục lục của bài học?":["mục lục bài học","table of contents"],
    "What is int?":["định nghĩa int","what is int"]
  }
}

# 2) Meta + academic
Q: "chào bạn, cho mình hỏi int là gì?"
→ {
  "normalized_question": "chào bạn, cho mình hỏi int là gì?",
  "question_type": "academic",
  "task_type": "multi_part",
  "sub_questions_with_intent": [
    {"subq":"Chào bạn?","intent":["meta"],"topic":"meta"},
    {"subq":"What is int?","intent":["definition"],"topic":"int"}
  ],
  "subq_variants": {
    "Chào bạn?":["chào bạn","hello"],
    "What is int?":["định nghĩa int","what is int"]
  }
}

# 3) Unsafe + academic
Q: "cách làm bom và float là gì"
→ {
  "normalized_question": "cách làm bom và float là gì",
  "question_type": "academic",
  "task_type": "multi_part",
  "sub_questions_with_intent": [
    {"subq":"Cách làm bom?","intent":["unsafe"],"topic":"unsafe"},
    {"subq":"What is float?","intent":["definition"],"topic":"float"}
  ],
  "subq_variants": {
    "Cách làm bom?":["bom cách làm","make bomb"],
    "What is float?":["định nghĩa float","what is float"]
  }
}

# 4) Nhiều học thuật + so sánh
Q: "int là gì, float là gì, so sánh int và float"
→ {
  "normalized_question": "int là gì, float là gì, so sánh int và float",
  "question_type": "academic",
  "task_type": "multi_part",
  "sub_questions_with_intent": [
    {"subq":"What is int?","intent":["definition"],"topic":"int"},
    {"subq":"What is float?","intent":["definition"],"topic":"float"},
    {"subq":"How do int and float differ?","intent":["comparison"],"topic":"int and float"}
  ],
  "subq_variants": {
    "What is int?":["định nghĩa int","what is int"],
    "What is float?":["định nghĩa float","what is float"],
    "How do int and float differ?":["so sánh int và float","compare int float"]
  }
}

# 5) Chỉ TOC đơn lẻ
Q: "cho mình mục lục giáo trình"
→ {
  "normalized_question": "cho mình mục lục giáo trình",
  "question_type": "academic",
  "task_type": "single",
  "sub_questions_with_intent": [
    {"subq":"Mục lục giáo trình?","intent":["toc"],"topic":"toc"}
  ],
  "subq_variants": {
    "Mục lục giáo trình?":["mục lục giáo trình","table of contents"]
  }
}

# 6) Gibberish + academic + unsafe
Q: "acsbsadcb, int là gì, cách làm bom"
→ {
  "normalized_question": "acsbsadcb, int là gì, cách làm bom",
  "question_type": "academic",
  "task_type": "multi_part",
  "sub_questions_with_intent": [
    {"subq":"acsbsadcb","intent":["unclear"],"topic":"unclear"},
    {"subq":"What is int?","intent":["definition"],"topic":"int"},
    {"subq":"Cách làm bom?","intent":["unsafe"],"topic":"unsafe"}
  ],
  "subq_variants": {
    "acsbsadcb":["acsbsadcb","gibberish"],
    "What is int?":["định nghĩa int","what is int"],
    "Cách làm bom?":["bom cách làm","make bomb"]
  }
}

# 7) Hai hành động khác nhau trong cùng chủ đề
Q: "Đoạn nào đề cập về chủ đề int? Đưa rõ số trang."
→ {
  "normalized_question": "Đoạn nào đề cập về chủ đề int? Đưa rõ số trang.",
  "question_type": "academic",
  "task_type": "multi_part",
  "sub_questions_with_intent": [
    {"subq":"Đoạn nào đề cập về chủ đề int?","intent":["locate"],"topic":"int"},
    {"subq":"Đưa rõ số trang nói về chủ đề int.","intent":["reference"],"topic":"int"}
  ],
  "subq_variants": {
    "Đoạn nào đề cập về chủ đề int?":["đoạn đề cập về chủ đề int","where int mentioned"],
    "Đưa rõ số trang nói về chủ đề int.":["số trang về chủ đề int","page numbers int"]
  }
}

Now analyze this question:
{question}
"""

# 🔒 Escape tất cả { } rồi khôi phục {question}
SAFE_TMPL = (
    RAW_ANALYZE_TMPL
    .replace("{", "{{")
    .replace("}", "}}")
    .replace("{{question}}", "{question}")
)

ANALYZE_PROMPT = PromptTemplate.from_template(SAFE_TMPL)


_ws = re.compile(r"\s+")
def _norm(s: str) -> str:
    return _ws.sub(" ", (s or "").strip())

def _to_list(x) -> List[str]:
    if x is None: return []
    if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):  return [i.strip() for i in x.split(",") if i.strip()]
    return []

def _to_map(x) -> Dict[str, List[str]]:
    return x if isinstance(x, dict) else {}

async def analyze_question(question: str) -> Dict[str, Any]:
    print("✅ ENTERING ANALYZE FUNCTION")
    t0 = time.time()
    llm = get_llm()  # dùng model nhanh cho Analyze

    try:
        res = await llm.ainvoke(ANALYZE_PROMPT.format(question=question))
        text = getattr(res, "content", str(res)).strip()
        text_clean = re.sub(r"^[^({\[]+", "", text)
        match = re.search(r"\{[\s\S]*\}", text_clean)
        if not match:
            raise ValueError("No JSON found")
        data: Dict[str, Any] = json.loads(match.group(0))
    except Exception as e:
        logging.warning(f"[Analyze] Parse failed: {e}")
        data = {
            "normalized_question": question,
            "question_type": "unclear",
            "task_type": "single",
            "intent": [],
            "sub_questions_with_intent": [],
            "subq_variants": {}
        }

    # --- Normalize tối thiểu ---
    data["normalized_question"] = _norm(data.get("normalized_question", question))
    data["question_type"] = (data.get("question_type") or "academic").lower()
    data["task_type"] = (data.get("task_type") or "single").lower()



    # sub_questions_with_intent
    sqwi = data.get("sub_questions_with_intent") or []
    norm_sqwi = []
    if isinstance(sqwi, list):
        for item in sqwi:
            subq = (item or {}).get("subq", "").strip()
            it = item.get("intent", [])
            if isinstance(it, str): it = [it]
            it = [str(x).strip().lower() for x in it if str(x).strip()]
            topic = (item.get("topic") or "").strip()
            if subq:
                norm_sqwi.append({
                    "subq": subq,
                    "intent": (it[:2] or ["definition"]),
                    "topic": topic
                })

    data["sub_questions_with_intent"] = norm_sqwi  # ❗️không hardcap số subq

    # subq_variants map (fallback 1 biến thể là chính subq nếu thiếu)
    sqv_in = _to_map(data.get("subq_variants"))
    sqv_out: Dict[str, List[str]] = {}
    for sqo in data["sub_questions_with_intent"]:
        sq = sqo["subq"]
        variants = sqv_in.get(sq) or []
        if isinstance(variants, str): variants = [variants]
        variants = [v.strip() for v in variants if v and v.strip()]
        if not variants:
            variants = [sq]  # fallback an toàn
        sqv_out[sq] = variants[:2]  # tối đa 2, ngắn
    data["subq_variants"] = sqv_out

    # ✅ Fallback: nếu LLM không sinh subq → tạo 1 subq cho single
    if not data.get("sub_questions_with_intent"):
        nq = data["normalized_question"] or (question or "").strip()
        if nq:
            data["sub_questions_with_intent"] = [{
                "subq": nq,
                "intent": ["definition"],
                "topic": ""
            }]
            data["subq_variants"] = {nq: [nq]}

    # ===== Ensure every subq has at least one intent (default 'definition')
    sq_list = data.get("sub_questions_with_intent") or []
    fixed_sq = []
    for it in sq_list:
        subq = (it.get("subq") or "").strip()
        intents_sq = it.get("intent") or []
        if isinstance(intents_sq, str):
            intents_sq = [intents_sq]
        intents_sq = [str(x).lower().strip() for x in intents_sq if str(x).strip()]
        if not intents_sq:
            intents_sq = ["definition"]  # default
        topic_sq = (it.get("topic") or "").strip()
        fixed_sq.append({"subq": subq, "intent": intents_sq[:2], "topic": topic_sq})
    data["sub_questions_with_intent"] = fixed_sq

    # ===== Backstop TOC purely at subq-level (no root intent)
    # Nếu câu hỏi/subq trông như TOC → ép intent subq = ["toc"], task_type = "single"
    import unicodedata
    def _strip_accents(s: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn")
    def _looks_toc(s: str) -> bool:
        kws = ("mục lục","muc luc","toc","table of contents","contents","outline","list of chapters")
        s_l = (s or "").lower().strip()
        s_no = _strip_accents(s_l)
        return any(k in s_l for k in kws) or any(k in s_no for k in kws)

    duration = round(time.time() - t0, 3)
    logging.info(f"🧩 [Analyze] Done in {duration}s → subq={len(data['sub_questions_with_intent'])}")

    status = data.get("question_type", "academic")
    if status in ["unsafe","unclear","meta"]:
        return {
            "status": status,
            "answer": (
                "Xin lỗi, mình không thể hỗ trợ với nội dung này." if status=="unsafe" else
                "Mình chưa hiểu rõ câu hỏi, bạn có thể nói cụ thể hơn không?" if status=="unclear" else
                "Đây là câu hỏi dạng meta về hệ thống; sẽ trả lời ngắn gọn hoặc dừng pipeline."
            ),
            **data
        }

    return {"status": status, **data}
