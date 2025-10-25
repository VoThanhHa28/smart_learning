# app/services/nodes/retrieve.py
import asyncio, json, logging, time
from ..rag_state import debug_state
from ...rag.retrieval.hybrid import hybrid_retrieve
from ...rag.retrieval.rrf import rrf_merge
from ...rag.rag_runtime import doc_sig, mark_doc, mmr_select
from ...core.config import CFG

async def retrieve(state: dict):
    debug_state("retrieve", state)

    if state.get("answer") or (state.get("analyze_meta", {}).get("status") == "stop"):
        return {**state, "context": [], "timing": {"retrieve": 0.0}}

    t0 = time.time()
    query = (state.get("question_norm") or "").strip()
    if not query:
        return {**state, "context": [], "timing": {"retrieve": 0.0}}

    # filters
    raw_filters = state.get("filters", {}) or {}
    raw_filters.setdefault("subject", state.get("subject"))
    raw_filters.setdefault("course_id", state.get("course_id"))
    filters = {}
    if raw_filters.get("subject"):
        filters["subject"] = str(raw_filters.get("subject")).lower().strip()
    if raw_filters.get("course_id"):
        filters["course_id"] = str(raw_filters.get("course_id")).lower().strip()

    meta = state.get("analyze_meta") or {}

    # params theo task_type
    if (meta.get("task_type") or "").lower() == "single":
        k_dense = 16; k_sparse = 16; top_after_rrf = 36
    else:
        k_dense = 14; k_sparse = 12; top_after_rrf = 24

    # subq-first
    sq_items = meta.get("sub_questions_with_intent") or []
    sq_variants = meta.get("subq_variants") or {}

    def build_queries_for_subq(sq: str) -> list[str]:
        base = [sq]
        for v in (sq_variants.get(sq) or [])[:2]:
            v = (v or "").strip()
            if v and v not in base: base.append(v)
        return base

    groups = {}
    for i, item in enumerate(sq_items, 1):
        sq = (item.get("subq") or "").strip()
        if sq:
            groups[f"sq::{i}"] = build_queries_for_subq(sq)

    async def _one(q: str, group_name: str):
        docs = await hybrid_retrieve(q, k_dense=k_dense, k_sparse=k_sparse, top_after_rrf=top_after_rrf, filters=filters or None, return_raw=False)
        out = []
        for d in docs:
            d.metadata = dict(d.metadata or {})
            d.metadata["_sig"] = d.metadata.get("_sig") or doc_sig(d)
            mark_doc(d, source_group=group_name, source_subq=q, stage="hybrid")
            out.append(d)
        return out

    async def run_group(qs: list[str], group_name: str):
        sem = asyncio.Semaphore(CFG.CONCURRENCY)
        async def wrapped(q: str):
            async with sem:
                try:
                    return await _one(q, group_name)
                except Exception as e:
                    logging.warning(f"⚠️ subQ error: {e}")
                    return []
        results = await asyncio.gather(*[wrapped(q) for q in qs])
        rrf_docs = rrf_merge(results, k=CFG.RRF_INTERQUERY_K, top_n=CFG.RRF_INTERQUERY_TOPN)
        for d in rrf_docs:
            mark_doc(d, source_group="INTRA_RRF", source_subq=group_name, stage="rrf_intra")
        return rrf_docs

    bags = {}
    for name, qs in groups.items():
        bags[name] = await run_group(qs, name)

    picked = []
    for i, _ in enumerate(sq_items, 1):
        picked += (bags.get(f"sq::{i}", [])[:CFG.SQ_MIN_PER_GROUP])

    lists_all = [bags.get(name, []) for name in bags.keys()]
    fused_all = rrf_merge(lists_all, k=CFG.RRF_INTERQUERY_K, top_n=CFG.RRF_INTERQUERY_TOPN)

    seen = set((d.page_content or "").strip() for d in picked)
    fused = picked[:]
    for d in fused_all:
        sig = (d.page_content or "").strip()
        if sig and sig not in seen:
            seen.add(sig)
            fused.append(d)
    for d in fused_all:
        mark_doc(d, source_group="INTER_RRF", source_subq="*all*", stage="rrf_inter")

    # MMR
    mmr_query = state.get("question_norm") or state.get("question") or ""
    if len(fused) > top_after_rrf:
        fused = mmr_select(mmr_query, fused, k=top_after_rrf, lambda_mult=0.65)

    duration = round(time.time() - t0, 3)
    return {**state, "context": fused, "timing": {"retrieve": duration}, "search_params": {"top_after_rrf": top_after_rrf}}
