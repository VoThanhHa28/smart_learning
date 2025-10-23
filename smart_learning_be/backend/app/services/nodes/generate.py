# app/services/nodes/generate.py
import os, asyncio, time
from concurrent.futures import ThreadPoolExecutor
from app.services.rag_utils import build_context
from app.services.prompt_utils import normalize_and_order_intents, CACHED_PROMPT
from app.services.llm_utils import sync_stream_generate

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
    intents_in = analyze_meta.get("intent", []) or []
    if isinstance(intents_in, str): intents_in = [intents_in]
    intents_in = [i.lower() for i in intents_in]
    is_definition = "definition" in intents_in

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

    intents_ord = normalize_and_order_intents(intents_in)
    MAX_INTENTS = int(os.getenv("MAX_INTENTS", "2"))
    intents_use = intents_ord[:MAX_INTENTS] or ["router_fallback"]

    delimiter = "\n\n---\n\n"
    def build_task_prompt(it_name: str) -> str:
        return get_prompt_by_intent(it_name, meta.get("doc_type","")).format(**meta)

    task_prompts = [build_task_prompt(it) for it in intents_use]
    if analyze_meta.get("task_type") == "multi_part" or analyze_meta.get("sub_questions"):
        task_prompts.append(get_prompt_by_intent("multi_part").format(**meta))

    if len(task_prompts) <= 2:
        prompt_text = f"{CACHED_PROMPT}{delimiter}" + delimiter.join(task_prompts)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            answer = await loop.run_in_executor(pool, sync_stream_generate, prompt_text)
        duration = round(time.time() - t0, 3)
        return {"answer": answer, "timing": {"generate": duration}}

    def stream_one(block_text: str) -> str:
        pt = f"{CACHED_PROMPT}{delimiter}{block_text}"
        return sync_stream_generate(pt)

    parts = [stream_one(tp) for tp in task_prompts]
    answer = "\n\n".join(parts)
    duration = round(time.time() - t0, 3)
    return {"answer": answer, "timing": {"generate": duration}}
