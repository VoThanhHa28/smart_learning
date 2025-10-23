# app/services/analyze_query.py
# -*- coding: utf-8 -*-

import json, time, logging, re
from typing import Any, Dict, List
from app.services.llm import get_llm
from langchain_core.prompts import PromptTemplate
RAW_ANALYZE_TMPL = r"""
Trả về CHỈ MỘT JSON hợp lệ (không markdown, không giải thích) với các khóa SAU:

- normalized_question (string)
- question_type ("academic"|"meta"|"unclear"|"unsafe")
- task_type ("single"|"comparison"|"multi_part")
- intent (mảng tối đa 2 mục, ví dụ: ["definition","comparison"])
- sub_questions_with_intent (mảng các đối tượng: {"subq":"...","intent":["..."],"topic":"<ngắn gọn theo chính subq>"})
- subq_variants (đối tượng map: {"<subq>": ["v1 short","v2 short"]}, mỗi subq trong subq_variants có 2 biến thể (1 tiếng Việt + 1 tiếng Anh), ngắn, không stopwords)

Quy tắc:
- Không giới hạn cứng số sub_questions; nếu rất nhiều, vẫn liệt kê đầy đủ.
- Nếu có “X vs Y” → topic của subq đó là "X and Y".
- Nếu “What is X?” → topic = "X".
- CHỈ JSON hợp lệ.

Ví dụ:
Q: int là gì, int vs float, string immutable
→ {
  "normalized_question":"int là gì, so sánh int và float, string immutable là gì?",
  "question_type":"academic",
  "task_type":"multi_part",
  "intent":["definition","comparison"],
  "sub_questions_with_intent":[
    {"subq":"What is int?","intent":["definition"],"topic":"int"},
    {"subq":"What is float?","intent":["definition"],"topic":"float"},
    {"subq":"How do int and float differ?","intent":["comparison"],"topic":"int and float"},
    {"subq":"Is string immutable?","intent":["definition"],"topic":"string immutability"}
  ],
  "subq_variants":{
    "What is int?":["định nghĩa int","what is int"],
    "What is float?":["định nghĩa float","what is float"],
    "How do int and float differ?":["so sánh int và float","compare int float"],
    "Is string immutable?":["định nghĩa tính bất biến của chuỗi","immutable string"]
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

    intents = data.get("intent") or []
    if isinstance(intents, str): intents = [intents]
    data["intent"] = [i.strip().lower() for i in intents][:2]

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
                    "intent": (it[:2] or data["intent"][:1] or ["definition"]),
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
                "intent": (data.get("intent")[:1] or ["definition"]),
                "topic": ""
            }]
            data["subq_variants"] = {nq: [nq]}


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
