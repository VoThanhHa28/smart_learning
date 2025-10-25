# app/services/nodes/analyze.py
import logging, time, re
from typing import Any, Dict
from ...rag.analysis.analyze_query import analyze_question
from ...services.utils.toc_store import get_toc_by_course, get_toc_by_key, flatten_toc
from ...infrastructure.llm.llm_utils import sync_stream_generate
from ..prompts.prompt_utils import CACHED_PROMPT, build_toc_validation_block

# 🔎 Regex TOC phủ rộng (không dấu + có dấu)
_TOC_KWS = (
    r"\bmục lục\b", r"\bmuc luc\b", r"\btoc\b",
    r"\btable of contents\b", r"\bcontents\b",
    r"\bdanh mục chương\b", r"\bcấu trúc chương\b",
    r"\blist of chapters\b", r"\boutline\b",
    r"\bmục lục (bài|bài học|bài giảng|giáo trình)\b"
)
_TOC_RX = re.compile("|".join(_TOC_KWS), re.IGNORECASE)

import re

_WS = re.compile(r"\s+")
def _looks_like_gibberish(s: str) -> bool:
    t = (s or "").strip().lower()
    t = _WS.sub(" ", t)
    # rất ngắn + không có từ điển cơ bản → coi là gibberish
    if len(t) <= 3:
        return True
    # tỷ lệ ký tự chữ/số quá thấp
    alnum = sum(ch.isalnum() for ch in t)
    if alnum / max(1, len(t)) < 0.5:
        return True
    # không có từ “thật” (>= 2 ký tự chữ) nào
    words = re.findall(r"[a-zA-ZÀ-ỹ]{2,}", t)
    if len(words) == 0:
        return True
    return False


def _looks_like_toc_question(q: str) -> bool:
    q = (q or "").strip()
    return bool(_TOC_RX.search(q))

def _format_toc_with_llm(toc_lines: list[str]) -> str:
    """
    LLM chỉ format/validate. Không bịa. Nếu thấy input vô nghĩa → phải trả lời thất bại.
    """
    if not toc_lines:
        return "Tài liệu không có mục lục hoặc không nhận dạng được mục lục."

    prompt_text = build_toc_validation_block(toc_lines)
    return sync_stream_generate(prompt_text)

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

    # ===== 0) Regex-first intent TOC (bypass toàn bộ nếu trúng)
    if _looks_like_toc_question(question):
        course_id = str(state.get("course_id") or "").strip()
        doc_key = state.get("doc_key")  # nếu bạn có lưu doc hiện hành
        entry = get_toc_by_key(doc_key) if doc_key else get_toc_by_course(course_id)
        toc_lines = flatten_toc(entry)
        answer = _format_toc_with_llm(toc_lines)

        return {
            **state,
            "answer": answer,
            "analyze_meta": {
                "status": "stop",
                "question_type": "academic",
                "task_type": "single",
                "normalized_question": question,
                "sub_questions_with_intent": [{"subq": question, "intent": ["toc"], "topic": "toc"}],
                "subq_variants": {question: ["mục lục", "table of contents"]}
            },
            "timing": {"analyze": round(time.time() - t0, 3)},
        }

    # ===== 1) Không trúng regex → chạy LLM analyze như trước
    result: Dict[str, Any] = await analyze_question(question)
    logging.info(f"🔎 [Analyze Result] {result}")

    status = (result.get("question_type") or "unclear").lower()
    dur = round(time.time() - t0, 3)

    # ép các sub-question gibberish thành 'unclear'
    sqwi_fix = []
    for it in (result.get("sub_questions_with_intent") or []):
        subq = (it.get("subq") or "").strip()
        intents = [str(x).lower().strip() for x in (it.get("intent") or []) if x]
        if _looks_like_gibberish(subq):
            intents = ["unclear"]
        sqwi_fix.append({"subq": subq, "intent": intents or ["unclear"], "topic": (it.get("topic") or "").strip()})
    result["sub_questions_with_intent"] = sqwi_fix

    # lấy intent hiệu dụng từ sub_questions_with_intent
    sqwi = result.get("sub_questions_with_intent") or []
    intents_eff = []
    for sq in sqwi:
        for it in (sq.get("intent") or []):
            if it:
                intents_eff.append(str(it).lower().strip())

    # ===== meta-only → stop tại analyze và trả lời ngay
    def _is_special(intents):
        s = {str(x).lower().strip() for x in (intents or [])}
        return any(k in s for k in ["meta", "toc", "unsafe", "unclear"])

    has_academic = any(not _is_special(sq.get("intent")) for sq in sqwi)
    meta_only = (status == "meta") and not has_academic

    if meta_only:
        try:
            from ...services.meta.meta_responder import search_meta as _meta_search
            meta_ans = (_meta_search(result.get("normalized_question") or question)
                        or result.get("answer")
                        or "Chào bạn! Mình là trợ lý của hệ thống. Bạn cần hỗ trợ gì?")
        except Exception:
            meta_ans = result.get("answer") or "Chào bạn! Mình là trợ lý của hệ thống. Bạn cần hỗ trợ gì?"

        result["status"] = "stop"
        return {
            **state,
            "answer": meta_ans,       # propagate ngay tại analyze
            "analyze_meta": result,
            "timing": {"analyze": dur},
        }

    if "toc" in intents_eff and len(sqwi) == 1:
        course_id = str(state.get("course_id") or "").strip()
        doc_key = state.get("doc_key")
        entry = get_toc_by_key(doc_key) if doc_key else get_toc_by_course(course_id)
        toc_lines = flatten_toc(entry)
        answer = _format_toc_with_llm(toc_lines)
        result["status"] = "stop"
        result["task_type"] = "single"
        return {
            **state,
            "answer": answer,
            "analyze_meta": result,
            "timing": {"analyze": dur},
        }

    # ===== 3) Các nhánh cũ: unsafe/unclear/meta → giữ nguyên
    if status == "unsafe":
        result["status"] = "stop"
        result["sub_questions_with_intent"] = []
        result["subq_variants"] = {}
        return {**state, "answer":"Xin lỗi, mình không thể hỗ trợ với nội dung này.",
                "analyze_meta": result, "timing": {"analyze": dur}}

    if status == "unclear":
        result["status"] = "stop"
        result["sub_questions_with_intent"] = []
        result["subq_variants"] = {}
        return {**state, "answer":"Mình chưa hiểu rõ câu hỏi của bạn, bạn có thể nói cụ thể hơn không?",
                "analyze_meta": result, "timing": {"analyze": dur}}

    # ===== 4) Mặc định → tiếp tục pipeline như cũ
    return {
        **state,
        "question_user": state.get("question") or question,
        "question_norm": result.get("normalized_question") or question,
        "analyze_meta": result,
        "timing": {"analyze": dur},
    }