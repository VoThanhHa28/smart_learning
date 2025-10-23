# app/services/nodes/generate_per_subq.py
import asyncio, logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from app.services.rag_state import State
from app.services.config import CONCURRENCY
from app.services.intent_router import get_routing
from app.services.rag_runtime import mmr_select
from app.services.rag_utils import build_context
from app.services.hybrid import hybrid_retrieve
from app.services.reranker import rerank as heavy_rerank
from app.services.prompt_utils import get_soft_hints, build_unified_block, CACHED_PROMPT
from app.services.llm_utils import sync_stream_generate

def _dedup_by_sig(docs):
    seen, out = set(), []
    for d in docs:
        sig = (d.metadata or {}).get("_sig") or (d.page_content or "")[:120]
        if sig in seen: 
            continue
        seen.add(sig); out.append(d)
    return out

async def generate_per_subq(state: State):
    if state.get("answer") or (state.get("analyze_meta", {}).get("status") == "stop"):
        return state
    meta = state.get("analyze_meta") or {}
    sq_items = meta.get("sub_questions_with_intent") or []
    subqs = [item.get("subq","").strip() for item in sq_items if item.get("subq")]
    if not subqs:
        return state

    intent_map = {i.get("subq"): (i.get("intent") or ["definition"]) for i in sq_items}
    topic_map  = {i.get("subq"): (i.get("topic") or "").strip() for i in sq_items}

    answers, diag_per, per_docs = {}, [], {}
    filters = {k: v for k, v in {"subject": state.get("subject"), "course_id": state.get("course_id")}.items() if v}
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(sq: str):
        async with sem:
            err_msg = ""
            try:
                sq_meta = next((m for m in (meta.get("sub_questions_with_intent") or []) if m.get("subq") == sq), {})
                intents_for_sq = intent_map.get(sq, ["definition"])
                primary_intent = intents_for_sq[0] if intents_for_sq else "definition"
                routing = get_routing(primary_intent)

                topic_sq = (sq_meta.get("topic") or meta.get("topic") or "").strip()

                filters_sq = dict(filters)
                if topic_sq:
                    filters_sq["topic"] = topic_sq

                docs = await hybrid_retrieve(
                    sq,
                    k_dense=routing["retrieve"]["k_dense"],
                    k_sparse=routing["retrieve"]["k_sparse"],
                    top_after_rrf=routing["retrieve"]["top_after_rrf"],
                    filters=filters_sq,
                    return_raw=False
                )
                before_docs = len(docs)
                if len(docs) > routing["retrieve"]["top_after_rrf"]:
                    docs = mmr_select(sq, docs, k=routing["retrieve"]["top_after_rrf"], lambda_mult=0.65)

                reranked = await heavy_rerank(sq, docs, top_n=routing["rerank"]["top_n"])
                selected = _dedup_by_sig(reranked[:routing["context"]["max_main"] + routing["context"]["max_supp"]])
                per_docs[sq] = selected[:]

                ctx = build_context(selected)
                # --- soft hints cho CHÍNH sub-question này: 1 asked_aspect + 1 soft hint ---
                primary_intent = intents_for_sq[0] if intents_for_sq else "definition"
                topic_sq_final = (topic_map.get(sq) or topic_sq or sq)

                # 1 câu bắt buộc: ép trọng tâm vào phần CHÍNH
                focus_line = (
                    f"Phần chính chỉ tập trung vào '{topic_sq_final}'; ví dụ phải cùng loại với '{topic_sq_final}'. "
                    f"Tránh mọi chi tiết không phục vụ trực tiếp cho trọng tâm. Khía cạnh: {primary_intent}. "
                    f"Các nội dung liên quan như hàm/công cụ phải tách riêng ở mục 'Ngoài ra' (nếu NGỮ CẢNH có)."
                )

                # 1 câu cho 'Ngoài ra' (nếu context có)
                extra_line = (
                    "Nếu NGỮ CẢNH có thông tin liên quan (ví dụ: công cụ/hàm phụ trợ), đặt vào mục 'Ngoài ra' riêng; "
                    "ví dụ của mục này phải minh hoạ đúng phần liên quan, không chèn vào phần chính."
                )

                # Lấy hint theo intent và rút còn 1 câu đầu (giữ gọn)
                _base_hints = get_soft_hints(intents_for_sq) or ""
                base_hint = _base_hints.split(". ")[0].rstrip(".") if _base_hints else ""

                # Gộp: 1 câu focus + 1 câu 'Ngoài ra' + (tuỳ) 1 câu intent
                soft_hints = f"{focus_line} {extra_line}" if not base_hint else f"{focus_line} {extra_line} {base_hint}."

                # Log để kiểm
                logging.info("[per-subq][%s] soft_hints=%s", sq, soft_hints)

                prompt_text = build_unified_block(
                    subject=state.get("subject", "General"),
                    context=ctx,
                    question=sq,
                    topic=topic_sq_final,
                    doc_source="Tài liệu học tập",
                    soft_hints=soft_hints
                )

                logging.info("[per-subq][%s] prompt_len=%d | ctx_len=%d | soft_hints_len=%d", sq, len(prompt_text), len(ctx), len(soft_hints))
                logging.debug("[per-subq][%s] prompt_preview=%s", sq, prompt_text[:400].replace("\n", " ⏎ "))

                loop = asyncio.get_event_loop() 
                with ThreadPoolExecutor() as pool:
                    ans = await loop.run_in_executor(pool, sync_stream_generate, CACHED_PROMPT + "\n\n---\n\n" + prompt_text)

                ans_len = len(ans or "")
                diag_per.append({"subq": sq, "intent": intents_for_sq, "docs_in": before_docs, "ctx_len": len(ctx), "ans_len": ans_len, "err": err_msg})
                return sq, ans

            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                logging.exception(f"[per-subq] error on '{sq}': {err_msg}")
                return sq, f"Lỗi khi xử lý sub-question: {err_msg}"

    results = await asyncio.gather(*[_one(sq) for sq in subqs], return_exceptions=True)
    for r in results:
        if isinstance(r, tuple) and len(r) == 2:
            sq, ans = r
            answers[sq] = ans

    merged_ctx_docs = _dedup_by_sig([d for docs in per_docs.values() for d in docs])
    state_diag = dict(state.get("diag") or {})
    state_diag["per_subq"] = diag_per

    return {**state, "sub_answers": answers, "per_subq_docs": per_docs, "context": merged_ctx_docs, "diag": state_diag}
