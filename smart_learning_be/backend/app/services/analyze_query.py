# app/services/analyze_query.py
# -*- coding: utf-8 -*-

import json, time, logging, re
from typing import Any, Dict, List
from app.services.llm import get_llm
from langchain_core.prompts import PromptTemplate

RAW_ANALYZE_TMPL = r"""
Trả về CHỈ MỘT JSON hợp lệ cho phân tích câu hỏi sau.

Các khóa (CHỈ các khóa này):
- normalized_question
- intent (tập con của ["definition","example","comparison","exercise","application","theory","summary","warning","socratic_question","quick_check","paraphrase","meta"])
- question_type ("academic","meta","unclear","unsafe")
- main_topic (trọng tâm của câu hỏi CHÍNH; là một khái niệm cụ thể, không quá rộng)
- difficulty ("basic","intermediate","advanced")
- task_type ("single","comparison","multi_part")
- sub_questions (danh sách)
- query_variants (1–2 truy vấn ngắn)
- hyde_query (<=100 ký tự; bám ý định/chủ đề)
- entities (danh sách)
- focus_entities (danh sách)
- per_entity_queries (dict: entity -> 1–2 truy vấn ngắn)
- sub_questions_with_intent (danh sách các đối tượng {"subq":"...","intent":["..."],"topic":"..."})
-> "topic" trong các sub question là CHỦ ĐỀ RIÊNG của từng sub-question, KHÔNG phải chủ đề chính.

Quy tắc:
- CHỈ JSON, KHÔNG markdown, KHÔNG giải thích thêm.
- Truy vấn ngắn, tránh stopwords.
- Trường rỗng dùng "" hoặc [].
- Thêm sub_questions nếu là câu đa phần hoặc so sánh (vd. "X vs Y").
- Mỗi focus_entity phải có ≥ 2 truy vấn ngắn trong per_entity_queries.

STRICT sub-question topic rules:
- Tuyệt đối KHÔNG sao chép hoặc diễn giải lại main_topic vào sub_questions_with_intent[].topic.
- Xác định topic của từng sub-question CHỈ từ chính văn bản của sub-question, với các dạng CHÍNH XÁC sau:
  * “What is X?” / “X là gì?” → topic = "X"
  * “X vs Y” / “Compare X and Y” → topic = "X and Y"
  * “Is String immutable?” → topic = "String immutability"
- Nếu topic của sub-question chứa dấu phẩy, chữ "and" ghép nhiều thực thể, hoặc gồm các thực thể không xuất hiện rõ ràng trong sub-question → SAI.
- Ở JSON cuối, mỗi sub_questions_with_intent[i].topic PHẢI khớp chính tả với cụm được suy ra trực tiếp từ subq như các dạng trên.

Ví dụ:
Q: hello
→ {"normalized_question":"","intent":["meta"],"question_type":"meta","main_topic":"","difficulty":"basic","task_type":"single","sub_questions":[],"query_variants":[],"hyde_query":"","entities":[],"focus_entities":[],"per_entity_queries":{},"sub_questions_with_intent":[]}

Q: Compare int and float in Python
→ {
  "normalized_question":"Compare int and float in Python",
  "intent":["comparison"],
  "question_type":"academic",
  "main_topic":"int and float in Python",
  "difficulty":"basic",
  "task_type":"comparison",
  "sub_questions":["What is int?","What is float?","How do they differ?"],
  "query_variants":["int vs float python","compare int float"],
  "hyde_query":"Compare Python int and float: precision, range, usage.",
  "entities":["int","float","Python"],
  "focus_entities":["int","float"],
  "per_entity_queries":{
    "int":["python int type","int data type"],
    "float":["python float type","float precision"]
  },
  "sub_questions_with_intent":[
    {"subq":"What is int?","intent":["definition"],"topic":"int"},
    {"subq":"What is float?","intent":["definition"],"topic":"float"},
    {"subq":"How do they differ?","intent":["comparison"],"topic":"int and float"}
  ]
}

Q: int là gì, int vs float, string immutable
→ {
  "normalized_question":"int là gì, so sánh int và float, string immutable là gì?",
  "intent":["definition","comparison","definition"],
  "question_type":"academic",
  "main_topic":"int, float and string immutability",
  "difficulty":"basic",
  "task_type":"multi_part",
  "sub_questions":["What is int?","What is float?","How do int and float differ?","Is string immutable?"],
  "query_variants":["int vs float","string immutability definition"],
  "hyde_query":"Explain int, compare int and float, define string immutability.",
  "entities":["int","float","string immutability"],
  "focus_entities":["int","float","string immutability"],
  "per_entity_queries":{
    "int":["int definition","python int type"],
    "float":["float definition","python float type"],
    "string immutability":["string immutability definition","immutable string"]
  },
  "sub_questions_with_intent":[
    {"subq":"What is int?","intent":["definition"],"topic":"int"},
    {"subq":"What is float?","intent":["definition"],"topic":"float"},
    {"subq":"How do int and float differ?","intent":["comparison"],"topic":"int and float"},
    {"subq":"Is string immutable?","intent":["definition"],"topic":"string immutability"}
  ]
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

async def analyze_question(question: str) -> Dict[str, Any]:
    print("✅ ENTERING ANALYZE FUNCTION")
    t0 = time.time()
    llm = get_llm()  # set this to a fast model in get_llm()

    try:
        res = await llm.ainvoke(ANALYZE_PROMPT.format(question=question))
        text = getattr(res, "content", str(res)).strip()
        text_clean = re.sub(r"^[^({\[]+", "", text)
        match = re.search(r"\{[\s\S]*\}", text_clean)
        if not match: raise ValueError("No JSON found")
        data: Dict[str, Any] = json.loads(match.group(0))
    except Exception as e:
      logging.warning(f"[Analyze] Parse failed: {e}")
      data = {
          "normalized_question": question, "intent": [], "question_type": "unclear",
          "topic": "", "difficulty": "basic", "validity": "unclear",
          "language": "mixed", "sub_questions": [],
          "query_variants": [], "hyde_query": "",
          "task_type": "single",
          "entities": [],
          "focus_entities": [],
          "per_entity_queries": {},
          "sub_questions_with_intent": []  # 👈 THÊM DÒNG NÀY
      }


    # normalize fields
    data["normalized_question"] = _norm(data.get("normalized_question", question))
    data["intent"]          = _to_list(data.get("intent"))
    data["sub_questions"]   = _to_list(data.get("sub_questions"))
    data["query_variants"]  = _to_list(data.get("query_variants"))
    data["language"]        = (data.get("language") or "mixed").lower()
    data["hyde_query"]      = _norm(data.get("hyde_query",""))
    data["task_type"] = (data.get("task_type") or "single").lower()
    data["entities"]  = _to_list(data.get("entities"))
    data["focus_entities"] = _to_list(data.get("focus_entities"))

    sqwi = data.get("sub_questions_with_intent") or []
    norm_sqwi = []
    if isinstance(sqwi, list):
        for item in sqwi:
            subq = (item or {}).get("subq", "")
            it = (item or {}).get("intent", [])
            if isinstance(it, str):
                it = [it]
            it = [str(x).strip().lower() for x in it if str(x).strip()]
            if subq.strip():
                norm_sqwi.append({"subq": subq.strip(), "intent": it})
    data["sub_questions_with_intent"] = norm_sqwi
    if data.get("sub_questions") and not data.get("sub_questions_with_intent"):
    # dùng intent đầu tiên làm mặc định cho từng subq
      fallback_intent = (data.get("intent")[:1] or ["definition"])
      data["sub_questions_with_intent"] = [
          {"subq": s, "intent": fallback_intent} for s in data["sub_questions"]
    ]

    # Chuẩn hóa per_entity_queries, ưu tiên focus_entities
    peq = data.get("per_entity_queries") or {}
    if not isinstance(peq, dict):
        peq = {}

    targets = data["focus_entities"] or data.get("entities") or []
    # Bảo đảm mỗi target có >= 2 query (LLM sinh; nếu thiếu, backfill tối thiểu không hardcode)
    for e in targets:
        cur = peq.get(e) or []
        seen, out = set(), []
        for q in cur:
            q = (q or "").strip()
            if q and q.lower() not in seen:
                seen.add(q.lower()); out.append(q)
        peq[e] = out[:3]
        while len(peq[e]) < 2:
            # backfill tối giản: dùng "e" hoặc "e + topic" (nếu có)
            key = (data.get("topic") or "").strip()
            hint = f"{e} {key}".strip() if key else e
            if hint and hint.lower() not in seen:
                peq[e].append(hint); seen.add(hint.lower())
            else:
                break

    data["per_entity_queries"] = peq


    # 🧩 Ensure each sub-question has a topic
    main_topic = data.get("topic", "")
    entities = [e.lower() for e in data.get("entities", [])]
    focus_ents = [e.lower() for e in data.get("focus_entities", [])]
    for sq_meta in data.get("sub_questions_with_intent", []):
        sq = sq_meta.get("subq", "")
        topic_val = sq_meta.get("topic", "").strip()
        if not topic_val:
            sq_low = sq.lower()
            if any(k in sq_low for k in ["compare", "vs", "khác", "so sánh"]):
                sq_meta["topic"] = " and ".join(data.get("focus_entities", [])) or main_topic
            elif any(e in sq_low for e in entities):
                sq_meta["topic"] = " and ".join([e for e in data["entities"] if e.lower() in sq_low]) or main_topic
            else:
                sq_meta["topic"] = main_topic

    duration = round(time.time() - t0, 3)
    logging.info(f"🧩 [Analyze] Done in {duration}s → intents={data['intent']} | qvars={len(data['query_variants'])}")

    status = data.get("question_type", "unclear")

    # Early exit on unsafe/unclear
    if status in ["unclear","unsafe"]:
        return {
            "status": status,
            "answer": ("Mình chưa hiểu rõ câu hỏi, bạn có thể nói cụ thể hơn không?"
                       if status=="unclear" else "Xin lỗi, mình không thể hỗ trợ với nội dung này."),
            **data
        }

    # Ensure we have at least the normalized question as a variant
    if not data["query_variants"]:
        data["query_variants"] = [data["normalized_question"]] if data["normalized_question"] else []

    return {"status": status, **data}
