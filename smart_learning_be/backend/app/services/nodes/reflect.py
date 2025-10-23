# app/services/nodes/reflect.py
import logging

async def reflect_and_merge(state):
    if state.get("answer") or (state.get("analyze_meta", {}).get("status") == "stop"):
        return state
    logging.info("[reflect] keys=%s", list(state.keys()))
    meta = state.get("analyze_meta") or {}
    sub_answers = state.get("sub_answers") or {}
    per_docs = state.get("per_subq_docs") or {}

    if not sub_answers:
        return {**state, "answer": "Không có câu trả lời nào được tạo."}

    ordered_items = meta.get("sub_questions_with_intent") or [{"subq": s, "intent": ["definition"]} for s in sub_answers.keys()]

    sections = []
    for idx, item in enumerate(ordered_items, 1):
        sq = item["subq"]
        intent = (item.get("intent") or ["definition"])[0]
        part_ans = (sub_answers.get(sq) or "").strip()
        docs_sq = per_docs.get(sq) or []

        src_lines = []
        for d in docs_sq[:3]:
            m = d.metadata or {}
            p = m.get("page")
            snippet = (d.page_content or "").strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:160] + "…"
            src_lines.append(f"- p.{p}: {snippet}" if p else f"- {snippet}")

        src_block = "\n".join(src_lines)
        part = f"### {idx}. {sq} ({intent})\n{part_ans}"
        if src_block:
            part += f"\n\n**Nguồn tham khảo:**\n{src_block}"
        sections.append(part)

    final_answer = "\n\n---\n\n".join(sections).strip()
    state_diag = dict(state.get("diag") or {})
    state_diag["reflect"] = {"sections": len(sections), "final_len": len(final_answer)}

    return {**state, "answer": final_answer, "diag": state_diag}
