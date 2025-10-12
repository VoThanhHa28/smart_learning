# app/services/analyze_query.py
import json
import time
import logging
import re
from app.services.llm import get_llm
from app.services.meta_responder import get_meta_response   # ✅ Thêm dòng này
from langchain_core.prompts import PromptTemplate

# ======================= #
# ⚙️ Prompt Template
# ======================= #
ANALYZE_PROMPT = PromptTemplate.from_template("""
You are a **question analyzer** for an educational chatbot.

Read the student's question carefully and return a **strict JSON object** with these 8 fields:

- "normalized_question": rewritten question (clear and grammatically correct) or "" if meaningless.
- "topic": main concept of the question ("" if none).
- "difficulty": one of ["basic", "intermediate", "advanced"].
- "doc_type": one of ["definition", "example", "exercise", "code", "theory", "comparison", "application"].
- "intent": one of ["definition", "example", "compare", "problem", "application", "theory", "reasoning", "meta"].
- "question_type": classify as one of ["academic", "meta", "unclear", "unsafe"].
- "meta_type": if question_type="meta", choose one of ["greeting", "identity", "gratitude", "goodbye", "other"], otherwise "".
- "validity": one of:
  - "ok" → question is meaningful and safe.
  - "unclear" → question is incomplete, random, or meaningless (e.g., "abc", "what??", "???").
  - "unsafe" → question contains offensive, inappropriate, or dangerous content (e.g., sexual, violent, political, etc.).

Return **ONLY valid JSON**, no text or markdown.

---

### Examples

Question: hello
Output:
{{
  "normalized_question": "",
  "topic": "",
  "difficulty": "basic",
  "doc_type": "",
  "intent": "meta",
  "question_type": "meta",
  "meta_type": "greeting",
  "validity": "ok"
}}

Question: who are you?
Output:
{{
  "normalized_question": "",
  "topic": "",
  "difficulty": "basic",
  "doc_type": "",
  "intent": "meta",
  "question_type": "meta",
  "meta_type": "identity",
  "validity": "ok"
}}

Question: What is an int in Python?
Output:
{{
  "normalized_question": "What is the int data type in Python?",
  "topic": "data types in Python",
  "difficulty": "basic",
  "doc_type": "definition",
  "intent": "definition",
  "question_type": "academic",
  "meta_type": "",
  "validity": "ok"
}}

Question: asd????
Output:
{{
  "normalized_question": "",
  "topic": "",
  "difficulty": "",
  "doc_type": "",
  "intent": "",
  "question_type": "unclear",
  "meta_type": "",
  "validity": "unclear"
}}

---

Analyze this question and output JSON ONLY:

Question: {question}
-> MAKE QUICK AND EXACT OUTPUT
""")

# ======================= #
# 🧠 Analyze Question
# ======================= #
async def analyze_question(question: str):
    print("✅ ENTERING ANALYZE FUNCTION")
    t0 = time.time()
    llm = get_llm()

    try:
        res = await llm.ainvoke(ANALYZE_PROMPT.format(question=question))
        text = getattr(res, "content", str(res)).strip()

        print("\n================ RAW LLM OUTPUT ================")
        print(text[:2000])
        print("================================================\n")

        text_clean = re.sub(r"^[^({\[]+", "", text.strip())
        match = re.search(r"\{[\s\S]*\}", text_clean)
        if not match:
            raise ValueError("❌ No JSON found in output.")
        json_text = match.group(0)
        data = json.loads(json_text)

    except Exception as e:
        logging.warning(f"[Analyze] ❌ Parse failed: {e}")
        data = {
            "normalized_question": question,
            "topic": "",
            "difficulty": "basic",
            "doc_type": "",
            "intent": "",
            "question_type": "unclear",
            "meta_type": "",
            "validity": "unclear"
        }

    duration = round(time.time() - t0, 3)
    logging.info(f"🧩 [Analyze] Done in {duration}s → {data}")

    # 🧱 CASE 1: Unsafe
    if data.get("validity") == "unsafe":
        return {
            "status": "stop",
            "answer": "Xin lỗi, mình không thể hỗ trợ với nội dung này.",
            **data
        }

    # 🧱 CASE 2: Unclear
    if data.get("question_type") == "unclear" or data.get("validity") == "unclear":
        return {
            "status": "stop",
            "answer": "Mình chưa hiểu rõ câu hỏi của bạn, bạn có thể nói cụ thể hơn không?",
            **data
        }

    # 🧱 CASE 3: Meta question
    if data.get("question_type") == "meta":
        meta_type = data.get("meta_type", "other")
        meta_answer = get_meta_response(meta_type)  # ✅ Gọi hàm từ file meta_responder.py
        return {
            "status": "stop",
            "answer": meta_answer,
            **data
        }

    # ✅ Default (academic)
    logging.info(
        f"→ Normalized: {data.get('normalized_question', question)} | "
        f"Intent: {data.get('intent', '')} | Type: {data.get('question_type', '')}"
    )

    return {"status": "ok", **data}
