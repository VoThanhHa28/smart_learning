# app/services/nodes/generate.py
import os, asyncio, time
from concurrent.futures import ThreadPoolExecutor
from ..rag_utils import build_context
from ..prompts.prompt_utils import normalize_and_order_intents, get_soft_hints, build_unified_block, CACHED_PROMPT
from ...infrastructure.llm.llm_utils import sync_stream_generate

async def generate(state: dict):
    if state.get("answer"):
        return {"answer": state["answer"], "timing": {"generate": 0.0}}

    t0 = time.time()
    docs_in = state.get("context", [])
    docs_in_sorted = docs_in

    course_id = str(state.get("course_id", "")).lower()
    same_course = [d for d in docs_in_sorted if str(d.metadata.get("course_id","")).lower() == course_id]
    other = [d for d in docs_in_sorted if d not in same_course]

    analyze_meta = state.get("analyze_meta", {}) or {}
    sqwi = analyze_meta.get("sub_questions_with_intent") or []
    subq_intents = []
    for sq in sqwi:
        for it in (sq.get("intent") or []):
            if it:
                subq_intents.append(str(it).lower().strip())
    intents_in = normalize_and_order_intents(subq_intents) or ["router_fallback"]
    is_definition = "definition" in intents_in

    # Guard: nếu (lỡ) có 'toc' thì generate không xử lý (đã bypass ở analyze)
    if "toc" in intents_in:
        duration = round(time.time() - t0, 3)
        return {"answer": "Mục lục (TOC) đã được xử lý ở bước analyze.", "timing": {"generate": duration}}


    limit_main = 4 if is_definition else 6
    limit_supp = 2 if is_definition else 4

    main_pick = (same_course or docs_in_sorted)[:limit_main]
    supp_pick = other[:limit_supp]

    main_context = build_context(main_pick) if main_pick else "Không có ngữ cảnh nào được tìm thấy."
    supp_context = build_context(supp_pick) if supp_pick else ""
    context = main_context + ("\n\n## Supplementary\n" + supp_context if supp_context else "")

    m0 = (main_pick[0].metadata if main_pick else (docs_in_sorted[0].metadata if docs_in_sorted else {})) or {}
    enrich_meta = {"heading": m0.get("heading", ""), "headings": " > ".join(m0.get("headings", []) or []), "doc_source": m0.get("doc_source", "")}

    meta = {
        "subject": state.get("subject", "General"),
        "context": context or "Không có ngữ cảnh phù hợp.",
        "topic": analyze_meta.get("topic") or enrich_meta.get("headings", ""),
        "difficulty": analyze_meta.get("difficulty", "medium"),
        "doc_source": enrich_meta.get("doc_source", "Tài liệu học tập"),
        "doc_type": analyze_meta.get("doc_type", ""),
        "intent": intents_in,
        "question": state.get("question_user") or state.get("question") or "",
        "sub_questions": analyze_meta.get("sub_questions", []),
    }

    # soft hints từ intent hiệu dụng (lấy từ subqs theo patch trước); tối đa 1–2 intent
    soft_hints = get_soft_hints(intents_in[:2])
    delimiter = "\n\n---\n\n"

    prompt_text = (
        f"{CACHED_PROMPT}{delimiter}" +
        build_unified_block(
            subject=state.get("subject", "General"),
            context=context or "Không có ngữ cảnh phù hợp.",
            question=state.get("question_user") or state.get("question") or "",
            topic=analyze_meta.get("topic") or enrich_meta.get("headings", ""),
            doc_source=enrich_meta.get("doc_source", "Tài liệu học tập"),
            soft_hints=soft_hints
        )
    )

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        answer = await loop.run_in_executor(pool, sync_stream_generate, prompt_text)
    duration = round(time.time() - t0, 3)
    return {"answer": answer, "timing": {"generate": duration}}

