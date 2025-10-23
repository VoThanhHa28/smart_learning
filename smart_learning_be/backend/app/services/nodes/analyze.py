# app/services/nodes/analyze.py
import logging, time
from typing import Any, Dict
from app.services.analyze_query import analyze_question

async def analyze(state: dict):
    t0 = time.time()
    question = (state.get("question") or "").strip()

    if not question:
        dur = round(time.time() - t0, 3)
        return {
            **state,
            "answer": "Mình chưa hiểu câu hỏi của bạn. Bạn có thể nói cụ thể hơn không?",
            "analyze_meta": {"question_type": "unclear", "status": "unclear", "normalized_question": ""},
            "timing": {"analyze": dur},
        }

    result: Dict[str, Any] = await analyze_question(question)
    logging.info(f"🔎 [Analyze Result] {result}")

    status = (result.get("question_type") or "unclear").lower()
    dur = round(time.time() - t0, 3)

    # UNSAFE
    if status == "unsafe":
        # ✅ chặn pipeline
        result["status"] = "stop"
        result["sub_questions_with_intent"] = []
        result["subq_variants"] = {}
        return {
            **state,
            "answer": "Xin lỗi, mình không thể hỗ trợ với nội dung này.",
            "analyze_meta": result,
            "timing": {"analyze": dur},
        }

    # UNCLEAR
    if status == "unclear":
        result["status"] = "stop"
        result["sub_questions_with_intent"] = []
        result["subq_variants"] = {}
        return {
            **state,
            "answer": "Mình chưa hiểu rõ câu hỏi của bạn, bạn có thể nói cụ thể hơn không?",
            "analyze_meta": result,
            "timing": {"analyze": dur},
        }

    # META
    if status == "meta":
        try:
            from app.services.meta_responder import search_meta
            meta_ans = search_meta(question) or "Xin lỗi, mình chưa có câu trả lời cho nội dung này."
        except Exception as e:
            logging.error(f"[Analyze][meta] handler error: {e}")
            meta_ans = "Xin lỗi, mình chưa có câu trả lời cho nội dung này."
        result["status"] = "stop"
        result["sub_questions_with_intent"] = []
        result["subq_variants"] = {}
        return {
            **state,
            "answer": meta_ans,
            "analyze_meta": result,
            "timing": {"analyze": dur},
        }


    return {
        **state,
        "question_user": state.get("question") or question,
        "question_norm": result.get("normalized_question") or question,
        "analyze_meta": result,
        "timing": {"analyze": dur},
    }
