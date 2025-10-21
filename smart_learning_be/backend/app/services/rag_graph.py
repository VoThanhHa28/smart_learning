# ======================= #
# 🧠 Simple RAG Pipeline (Stable)
# ======================= #
# --- imports (đặt đầu file rag_graph.py) ---
import os, time, json, asyncio, traceback
from typing import Any, Dict, List
import torch
import numpy as np
from langgraph.graph import StateGraph, START
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

from app.services.hybrid import hybrid_retrieve
from app.services.llm import get_llm
from app.services.reranker import rerank as heavy_rerank
from app.services.config import (
    RERANKER_MODE, FINAL_TOP_N,
    CONCURRENCY, RRF_INTERQUERY_K, RRF_INTERQUERY_TOPN,
    CMP_MIN_PER_ENTITY, CMP_MIN_BOTH, SQ_MIN_PER_GROUP,
    MAX_SUBQS_BOTH, MAX_SUBQS_ENTITY, MAX_SUBQS_SQ,
)
from app.services.prompt_utils import (
    CACHED_PROMPT, get_prompt_by_intent, normalize_and_order_intents
)
from app.services.rag_utils import build_context, fuse_chunks_by_heading
from app.services.rrf import rrf_merge

from google.api_core.exceptions import ServiceUnavailable
from grpc import RpcError
import hashlib

from app.services.analyze_query import analyze_question

import logging, sys



_mmre5 = None
def _get_mmre5():
    global _mmre5
    if _mmre5 is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"[MMR] loading E5 on device={device}")
        _mmre5 = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    return _mmre5

def doc_sig(d: Document) -> str:
    meta = d.metadata or {}
    head = meta.get("heading") or meta.get("headings") or ""
    page = str(meta.get("page", ""))
    key = f"{(d.page_content or '')[:200]}|{head}|{page}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

def mark_doc(d: Document, *, source_group: str, source_subq: str, stage: str):
    d.metadata = dict(d.metadata or {})
    trace = d.metadata.get("_trace") or {}
    sources = trace.get("sources") or []
    sources.append({
        "stage": stage,
        "group": source_group,
        "subq": source_subq,
        "ts": time.time()
    })
    trace["sources"] = sources
    d.metadata["_trace"] = trace

def mmr_select(query: str, docs: list[Document], k: int = 24, lambda_mult: float = 0.65):
    if not docs:
        return []
    model = _get_mmre5()

    # ✅ encode có inference_mode khi có CUDA; giữ nguyên normalize + cosine
    if torch.cuda.is_available():
        with torch.inference_mode():
            q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
            d_texts = [d.page_content or "" for d in docs]
            d_embs = model.encode(d_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    else:
        # CPU path: tối ưu batch & tắt progress (không đổi logic)
        q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
        d_texts = [d.page_content or "" for d in docs]
        d_embs = model.encode(d_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, show_progress_bar=False)

    sims_q = (q_emb @ d_embs.T).ravel()  # cosine

    selected = []
    rep = np.zeros_like(sims_q)
    while len(selected) < min(k, len(docs)):
        if selected:
            rep = np.max((d_embs[selected] @ d_embs.T), axis=0)
        scores = lambda_mult * sims_q - (1 - lambda_mult) * rep
        scores[selected] = -1e9
        pick = int(np.argmax(scores))
        selected.append(pick)
    return [docs[i] for i in selected]



from app.services.config import (
    CONCURRENCY,
    RRF_INTERQUERY_K,
    RRF_INTERQUERY_TOPN,
    CMP_MIN_PER_ENTITY,
    CMP_MIN_BOTH,
    SQ_MIN_PER_GROUP,
    MAX_SUBQS_BOTH,
    MAX_SUBQS_ENTITY,
    MAX_SUBQS_SQ,
)

from typing_extensions import TypedDict, List
from typing import Dict

# ======================= #
# 1️⃣ Define State
# ======================= #
class State(TypedDict, total=False):
    question: str
    question_user: str
    question_norm: str
    subject: str
    course_id: str
    context: List[Document]
    analyze_meta: dict
    answer: str
    timing: dict

    # 👇 THÊM MỚI (bắt buộc để giữ qua các node)
    sub_answers: Dict[str, str]   # <— KHAI BÁO KEY NÀY
    filters: dict                 # (đã truyền trong retrieve/generate_per_subq)
    search_params: dict           # (đang set trong retrieve)
    per_subq_docs: Dict[str, List[Document]]  # các chunk đã chọn theo từng subq (để debug)
    diag: dict                                # chẩn đoán pipeline (đã dùng ở chain.py)

def debug_state(node_name, state):
    cid = state.get("course_id")
    print(f"🐛 [DEBUG] Node={node_name} | course_id={cid} | keys={list(state.keys())}")

PER_SUBQ_CONTEXT_N = 2   # số doc tốt nhất/ subq sẽ “bong” ra context cho client
RERANK_THRESHOLD = 0.50    # chỉ lấy chunk có score >= 0.5
MIN_PER_SUBQ     = 1       # đảm bảo tối thiểu 1 chunk / subq (backfill)
MAX_PER_SUBQ     = 2       # giới hạn tối đa 2 chunk / subq

def _score_of(d: Document) -> float:
    m = d.metadata or {}
    # tuỳ model rerank, một số trả "score", một số "rerank_score"
    return float(m.get("score") or m.get("rerank_score") or 0.0)

def _dedup_by_sig(docs):
    seen, out = set(), []
    for d in docs:
        sig = (d.metadata or {}).get("_sig") or (d.page_content or "")[:120]
        if sig in seen: 
            continue
        seen.add(sig)
        out.append(d)
    return out

from app.services.intent_router import get_routing, get_prompt_path

def plan_groups(analyze_meta):
    groups = []
    for i, item in enumerate(analyze_meta.get("sub_questions_with_intent", [])):
        subq = item.get("subq")
        intent = (item.get("intent") or ["definition"])[0]
        routing = get_routing(intent)
        prompt_path = get_prompt_path(intent)
        groups.append({
            "group_id": f"{intent}_{i}",
            "subq": subq,
            "intent": intent,
            "routing": routing,
            "prompt_path": prompt_path,
        })
    return groups

async def generate_per_subq(state: State):
    meta = state.get("analyze_meta") or {}
    sq_list = meta.get("sub_questions") or []
    sqwi = meta.get("sub_questions_with_intent") or []

    from app.services.prompt_utils import normalize_and_order_intents, get_prompt_by_intent
    global_intents = normalize_and_order_intents(meta.get("intent") or [])
    default_intent = (global_intents[0] if global_intents else "definition")

    intent_map = {}
    for i, sq in enumerate(sq_list):
        it = (sqwi[i]["intent"] if i < len(sqwi) and isinstance(sqwi[i], dict) else []) or []
        primary = (it[0] if it else default_intent)
        intent_map[sq] = primary

    subqs = meta.get("sub_questions") or []
    if not subqs:
        return state

    answers, diag_per, per_docs = {}, [], {}
    filters = {k: v for k, v in {"subject": state.get("subject"), "course_id": state.get("course_id")}.items() if v}

    # ✅ Thêm semaphore để điều phối GPU/IO; KHÔNG ảnh hưởng kết quả
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(sq: str):
        async with sem:
            sq_meta = next((m for m in meta.get("sub_questions_with_intent", []) if m.get("subq") == sq), {})
            intent_for_sq = intent_map.get(sq, default_intent)
            routing = get_routing(intent_for_sq)

            # ✅ Topic theo từng sub-question (ưu tiên sq_meta.topic)
            topic_sq = (sq_meta.get("topic") or meta.get("topic") or "").strip()

            # ✅ Focus theo từng sub-question (nếu analyzer có trường riêng), fallback hợp lý
            focus_list_sq = sq_meta.get("focus_entities") or meta.get("focus_entities") or []
            focus_sq = (focus_list_sq[0] if isinstance(focus_list_sq, list) and focus_list_sq else "") or topic_sq or sq

            # ---- Filters cho truy vấn theo sub-question ----
            filters_sq = dict(filters)
            if topic_sq:
                filters_sq["topic"] = topic_sq 

            logging.info(
                "🔎 [Retrieve Input] q='%s' | topic_sq=%s | intent=%s",
                sq, topic_sq, sq_meta.get("intent")
            )
            err_msg = ""
            try:
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
                    # ⚡ MMR đã tối ưu encode; giữ nguyên k/logic
                    docs = mmr_select(sq, docs, k=routing["retrieve"]["top_after_rrf"], lambda_mult=0.65)

                reranked = await heavy_rerank(sq, docs, top_n=routing["rerank"]["top_n"])
                selected = _dedup_by_sig(reranked[:routing["context"]["max_main"] + routing["context"]["max_supp"]])
                per_docs[sq] = selected[:]

                ctx = build_context(selected)
                prompt_tpl = get_prompt_by_intent(intent_for_sq)

                focus_entity = (sq_meta.get("focus_entities") or meta.get("focus_entities") or [topic_sq or sq])[0]
                prompt_text = prompt_tpl.format(
                    subject=state.get("subject", "General"),
                    context=ctx,
                    topic=topic_sq or focus_sq or sq,
                    difficulty=meta.get("difficulty", "basic"),
                    doc_source="Tài liệu học tập",
                    intent=[intent_for_sq],
                    question=sq,
                )

                logging.info(
                    "[Prompt Build] subject=%s | topic=%s | focus_entity=%s | used_topic=%s | subq=%s",
                    state.get("subject"),
                    topic_sq or (meta.get("topic") or ""),
                    focus_sq,
                    topic_sq or focus_sq or sq,
                    sq
                )

                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as pool:
                    logging.info("[per-subq] q=%s | intent=%s | ctx_chars=%d", sq, intent_for_sq, len(ctx))
                    ans = await loop.run_in_executor(pool, sync_stream_generate, CACHED_PROMPT + "\n\n---\n\n" + prompt_text)

                ans_len = len(ans or "")
                diag_per.append({
                    "subq": sq, "intent": intent_for_sq,
                    "docs_in": before_docs, "ctx_len": len(ctx),
                    "ans_len": ans_len, "err": err_msg
                })
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






async def reflect_and_merge(state: State):
    logging.info("[reflect] keys=%s", list(state.keys()))
    meta = state.get("analyze_meta") or {}
    sub_answers = state.get("sub_answers") or {}
    per_docs = state.get("per_subq_docs") or {}

    if not sub_answers:
        return {**state, "answer": "Không có câu trả lời nào được tạo."}

    ordered_items = meta.get("sub_questions_with_intent") or [
        {"subq": s, "intent": ["definition"]} for s in sub_answers.keys()
    ]

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
    logging.info("[reflect] merged_answer_len=%d", len(final_answer))
    state_diag = dict(state.get("diag") or {})
    state_diag["reflect"] = {"sections": len(sections), "final_len": len(final_answer)}

    return {**state, "answer": final_answer, "diag": state_diag}





# ---------------- META HANDLER ---------------- #
async def meta_handler(state):
    from app.services.meta_responder import search_meta
    q = state["question"]
    ans = search_meta(q)
    if not ans:
        ans = "Xin lỗi, mình chưa có câu trả lời cho nội dung này."
    return {"answer": ans, "timing": {"meta": 0.1}}


# ======================= #
# 2️⃣ Retrieve (Mono-call analyze → Multi-query + Entities Coverage)
# ======================= #
async def retrieve(state: dict):
    debug_state("retrieve", state)

    if state.get("answer") or (state.get("analyze_meta", {}).get("status") == "stop"):
        print("⏭️ [Retrieve] Skipped because analyze() already produced an answer or requested stop.")
        return {**state, "context": [], "timing": {"retrieve": 0.0}}

    t0 = time.time()
    query = (state.get("question_norm") or "").strip()
    if not query:
        print("🔕 [Retrieve] Empty normalized query -> skipping search to avoid irrelevant results.")
        return {**state, "context": [], "timing": {"retrieve": 0.0}}

    print(f"🔎 [Retrieve] Query='{query}'")

    # ---- Build filters (subject/course_id) ----
    raw_filters = state.get("filters", {}) or {}
    raw_filters.setdefault("subject", state.get("subject"))
    raw_filters.setdefault("course_id", state.get("course_id"))
    filters = {}
    if raw_filters.get("subject"):
        filters["subject"] = str(raw_filters.get("subject")).lower().strip()
    if raw_filters.get("course_id"):
        filters["course_id"] = str(raw_filters.get("course_id")).lower().strip()

    # ---- Analyze meta ----
    meta = state.get("analyze_meta") or {}

    # ---- Intent-aware search params (đặt TRƯỚC khi dùng) ----
    intent_list = meta.get("intent") or []
    if isinstance(intent_list, str):
        intent_list = [intent_list]
    intent_list = [i.lower() for i in intent_list]

    use_hyde = bool(meta.get("hyde_query"))
    k_dense = 12
    k_sparse = 24
    top_after_rrf = 50

    # ---- Build sub-queries ----
    subqs_both = (meta.get("query_variants") or [])[:]
    hyde_q = (meta.get("hyde_query") or "").strip()
    if hyde_q and len(hyde_q) > 220:
        hyde_q = hyde_q[:220]
    if use_hyde and hyde_q and hyde_q not in subqs_both:
        subqs_both.append(hyde_q)
    if not subqs_both:
        subqs_both = [query]

    is_cmp = (meta.get("task_type") == "comparison")
    focus_entities = meta.get("focus_entities") or []
    per_entity = meta.get("per_entity_queries") or {}

    def _dedup_keep_order(items: list[str], limit: int) -> list[str]:
        seen = set(); out = []
        for s in items:
            s = (s or "").strip()
            if s and s.lower() not in seen:
                seen.add(s.lower()); out.append(s)
            if len(out) >= limit:
                break
        return out

    subqs_both = _dedup_keep_order(subqs_both, MAX_SUBQS_BOTH)
    print(f"🧩 Sub-queries (both={len(subqs_both)}): {subqs_both}")
    if is_cmp:
        print(f"🧭 Comparison mode with focus_entities={focus_entities}")

    groups: dict[str, list[str]] = {"both": subqs_both}
    if is_cmp and focus_entities:
        for e in focus_entities:
            qs = _dedup_keep_order((per_entity.get(e) or []), MAX_SUBQS_ENTITY)
            if qs:
                groups[f"ent::{e}"] = qs

    def make_queries_for_subq(sq: str, meta: dict) -> list[str]:
        base = []
        sq = (sq or "").strip()
        if sq:
            base.append(sq)
        for q in (meta.get("query_variants") or [])[:2]:
            if q not in base:
                base.append(q)
        hyde = (meta.get("hyde_query") or "").strip()
        if use_hyde and hyde and hyde not in base:
            base.append(hyde)
        return base[:4]

    for i, sq in enumerate(meta.get("sub_questions") or []):
        qsq = _dedup_keep_order(make_queries_for_subq(sq, meta), MAX_SUBQS_SQ)
        if qsq:
            groups[f"sq::{i+1}"] = qsq

    # ---- Fan-out hybrid_retrieve ----
    # ---- Fan-out hybrid_retrieve ----
    async def _one(q: str, group_name: str):
        docs = await hybrid_retrieve(
            q,
            k_dense=k_dense,
            k_sparse=k_sparse,
            top_after_rrf=top_after_rrf,
            filters=filters or None,
            return_raw=False
            # , use_hyde=use_hyde
        )
        out = []
        for d in docs:
            d.metadata = dict(d.metadata or {})
            d.metadata["_sig"] = d.metadata.get("_sig") or doc_sig(d)
            mark_doc(d, source_group=group_name, source_subq=q, stage="hybrid")
            out.append(d)
        return out


    async def run_group(qs: list[str], group_name: str):
        sem = asyncio.Semaphore(CONCURRENCY)
        async def wrapped(q: str):
            async with sem:
                try:
                    res = await _one(q, group_name)
                    print(f"⏱ subQ got {len(res)} docs for: {q[:80]}")
                    return res
                except Exception as e:
                    print(f"⚠️ subQ error: {e}")
                    return []
        results = await asyncio.gather(*[wrapped(q) for q in qs])
        rrf_docs = rrf_merge(results, k=RRF_INTERQUERY_K, top_n=RRF_INTERQUERY_TOPN)
        for d in rrf_docs:
            mark_doc(d, source_group="INTRA_RRF", source_subq=group_name, stage="rrf_intra")
        return rrf_docs



    bags: dict[str, list] = {}
    for name, qs in groups.items():
        print(f"🔎 Group {name}: {len(qs)} subqs")
        bags[name] = await run_group(qs, name)


    picked = []
    picked += bags.get("both", [])[:CMP_MIN_BOTH]
    if is_cmp and focus_entities:
        for e in focus_entities:
            picked += bags.get(f"ent::{e}", [])[:CMP_MIN_PER_ENTITY]
    for i, _ in enumerate(meta.get("sub_questions") or []):
        picked += bags.get(f"sq::{i+1}", [])[:SQ_MIN_PER_GROUP]

    lists_all = [bags.get("both", [])]
    if is_cmp and focus_entities:
        lists_all += [bags.get(f"ent::{e}", []) for e in focus_entities]
    for i, _ in enumerate(meta.get("sub_questions") or []):
        lists_all.append(bags.get(f"sq::{i+1}", []))

    fused_all = rrf_merge(lists_all, k=RRF_INTERQUERY_K, top_n=RRF_INTERQUERY_TOPN)

    seen = set((d.page_content or "").strip() for d in picked)
    fused = picked[:]
    for d in fused_all:
        sig = (d.page_content or "").strip()
        if sig and sig not in seen:
            seen.add(sig)
            fused.append(d)

    for d in fused_all:
        mark_doc(d, source_group="INTER_RRF", source_subq="*all*", stage="rrf_inter")

    # Log pre-MMR mix & focus hit
    focus = (meta.get("focus_entities") or meta.get("entities") or [])
    focus_main = str(focus[0]).lower().strip() if focus else ""
    def _hit(d):
        return int(f" {focus_main} " in f" {(d.page_content or '').lower()} ")
    mix = {}
    for d in fused:
        for src in d.metadata.get("_trace", {}).get("sources", []):
            if src.get("stage") == "hybrid":
                g = src.get("group")
                mix[g] = mix.get(g, 0) + 1
    logging.info("[TRACE] pre_mmr mix_by_group=%s focus_hits=%d/%d",
                json.dumps(mix, ensure_ascii=False),
                sum(_hit(d) for d in fused), len(fused))


    for name, docs in bags.items():
        print(f"📦 Group {name}: kept={len(docs)}")
    print(f"📦 Picked pre-fuse: {len(picked)} | After fuse: {len(fused)}")

    duration = round(time.time() - t0, 3)
    total_in = sum(len(lst) for lst in lists_all if lst)
    print(f"✅ [Retrieve] Fused {total_in} → {len(fused)} docs in {duration:.2f}s")


    mmr_start = time.time()

    # 🔁 MMR sau RRF (giảm overview) — dùng câu hỏi đã chuẩn hoá, KHÔNG phụ thuộc intent
    mmr_query = state.get("question_norm") or state.get("question") or ""

    if len(fused) > top_after_rrf:
        before = [x.metadata.get("_sig") for x in fused]
        mmr_start = time.time()
        fused = mmr_select(mmr_query, fused, k=top_after_rrf, lambda_mult=0.65)
        mmr_dur = round(time.time() - mmr_start, 3)
        after = [x.metadata.get("_sig") for x in fused]
        logging.info("[TRACE] mmr kept=%d changed=%d | ⏱ %.3fs",
                    len(fused), len(set(before)-set(after)), mmr_dur)
        for d in fused:
            mark_doc(d, source_group="MMR", source_subq=mmr_query, stage="mmr")
    else:
        mmr_dur = 0.0


    duration = round(time.time() - t0, 3)
    print(f"✅ [Retrieve+MMR] kept={len(fused)} | mmr={mmr_dur}s | total={duration:.2f}s")

    # 👉 LƯU search params vào state để node sau dùng
    search_params = {"top_after_rrf": top_after_rrf}
    logging.info(f"[retrieve] fused_docs={len(fused)} | top_after_rrf={top_after_rrf}")
    return {
        **state,
        "context": fused,
        "timing": {"retrieve": duration},
        "search_params": {
            "top_after_rrf": top_after_rrf
        }
    }



# ======================= #
# 3️⃣ Rerank (optional)
# ======================= #
async def rerank_node(state: State):
    debug_state("rerank", state)
    t0 = time.time()

    TOP_FOR_RERANK = int((state.get("search_params") or {}).get("top_after_rrf", 24))
    docs = (state.get("context") or [])[:TOP_FOR_RERANK]
    if not docs:
        return {"context": [], "timing": state.get("timing", {})}

    # log vào rerank input theo group (trace)
    dist = {}
    for d in docs:
        for src in d.metadata.get("_trace", {}).get("sources", []):
            if src.get("stage") == "hybrid":
                g = src.get("group")
                dist[g] = dist.get(g, 0) + 1
    logging.info("[TRACE] rerank_in dist_by_group=%s", json.dumps(dist, ensure_ascii=False))

    analyze_meta = state.get("analyze_meta") or {}
    intents = analyze_meta.get("intent") or []
    if isinstance(intents, str):
        intents = [intents]
    intents = [i.lower() for i in intents]

    query_text = state.get("question_norm") or state.get("question") or ""
    # Nếu muốn “định nghĩa” ngắn gọn vẫn bám sát nguyên câu hỏi:
    # (tuỳ chọn) có thể prefex nhẹ nhưng không domain-specific
    if "definition" in intents:
        query_text = f"Concise definition: {query_text}"

    if RERANKER_MODE == "off":
        reranked = docs[:FINAL_TOP_N]
    else:
        reranked = await heavy_rerank(query_text, docs, top_n=FINAL_TOP_N)

    def code_signal(d) -> int:
        t_raw = (d.page_content or "")
        t_cln = (d.metadata.get("clean_text") or "")
        t = (t_raw + " " + t_cln).lower()
        # normalize quote variants
        t = t.replace("‘","'").replace("’","'").replace("“","\"").replace("”","\"")
        return int(("<class 'int'>" in t) or ("type(" in t) or ("int(" in t))

    top_preview = []
    for i, d in enumerate(reranked[:5]):
        txt = (d.metadata.get("clean_text") or d.page_content or "")[:160].replace("\n"," ")
        top_preview.append({
            "i": i,
            "sig": d.metadata.get("_sig"),
            "code": code_signal(d),  # ← thay vì code_signal(txt)
            "srcs": d.metadata.get("_trace",{}).get("sources", [])[-2:]
        })
    logging.info("[TRACE] rerank_top=%s", json.dumps(top_preview, ensure_ascii=False))

    duration = round(time.time() - t0, 3)
    print(f"🏅 [Reranker] Selected {len(reranked)} docs in {duration}s")
    return {"context": reranked, "timing": {"rerank": duration}}





# ======================= #
# 4️⃣ Generate (LLM)
# ======================= #
def sync_stream_generate(prompt_text: str) -> str:
    stage = "stream_generate"
    max_attempts = 2
    delay = 0.8
    last_err = None

    # 1) thử streaming
    for attempt in range(1, max_attempts + 1):
        try:
            llm = get_llm(streaming=True)
            out = ""
            for chunk in llm.stream(prompt_text):
                token = getattr(chunk, "content", "") or ""
                out += token
            logging.info(f"[LLM:{stage}] streaming ok | len={len(out)} chars")
            print("\n✅ [Stream Completed]")
            return out
        except (ServiceUnavailable, RpcError) as e:
            last_err = e
            logging.exception(f"[LLM:{stage}] attempt={attempt} ServiceUnavailable/RpcError: {e}")
            time.sleep(delay)
            delay *= 1.6
        except Exception as e:
            last_err = e
            logging.exception(f"[LLM:{stage}] attempt={attempt} unknown error: {e}")
            time.sleep(delay)
            delay *= 1.6

    # 2) fallback: non-streaming
    logging.warning(f"[LLM:{stage}] Falling back to non-streaming invoke after failures. last_err={last_err}")
    try:
        llm2 = get_llm(streaming=False)
        res = llm2.invoke(prompt_text)
        text = getattr(res, "content", str(res)) or ""
        logging.info(f"[LLM:{stage}] invoke ok | len={len(text)} chars")
        print("\n✅ [Invoke Completed]")
        return text
    except Exception as e2:
        logging.exception(f"[LLM:{stage}] fallback invoke failed: {e2} | original={last_err}")
        return ""  # caller sẽ log tiếp khi thấy rỗng

async def generate(state: State):
    debug_state("generate", state)
    if state.get("answer"):
        print("⏭️ [Generate] Skipped because answer present in state (from analyze()).")
        return {"answer": state["answer"], "timing": {"generate": 0.0}}

    t0 = time.time()

    # ==== Context debug ====
    print("\n==== 🔍 DEBUG CONTEXT PIPELINE ====")
    docs_in = state.get("context", [])
    print(f"[1] Retrieved docs (input to rerank/generate): {len(docs_in)}")
    for i, d in enumerate(docs_in[:5]):
        txt = getattr(d, "page_content", "")[:120].replace("\n", " ")
        print(f"→ DOC[{i}] {txt}")
    print("===================================\n")

    # ĐÃ RERANK xong ở node trước, giữ nguyên thứ tự
    docs_in_sorted = docs_in

    # Ưu tiên doc đúng course_id
    course_id = str(state.get("course_id", "")).lower()
    same_course = [d for d in docs_in_sorted if str(d.metadata.get("course_id","")).lower() == course_id]
    other = [d for d in docs_in_sorted if d not in same_course]

    # Giới hạn context theo intent (định nghĩa → mỏng)
    analyze_meta = state.get("analyze_meta", {}) or {}
    intents_in = analyze_meta.get("intent", []) or []
    if isinstance(intents_in, str):
        intents_in = [intents_in]
    intents_in = [i.lower() for i in intents_in]
    is_definition = "definition" in intents_in

    limit_main = 4 if is_definition else 6
    limit_supp = 2 if is_definition else 4

    main_pick = (same_course or docs_in_sorted)[:limit_main]
    supp_pick = other[:limit_supp]

    # Build dual contexts (KHÔNG dùng biến không tồn tại)
    main_context = build_context(main_pick) if main_pick else "Không có ngữ cảnh nào được tìm thấy."
    supp_context = build_context(supp_pick) if supp_pick else ""

    if supp_context:
        context = main_context + "\n\n## Supplementary\n" + supp_context
    else:
        context = main_context

    print(f"📚 [DEBUG] Context length = {len(context)} chars")
    print(f"📚 [DEBUG] Context preview:\n{context[:800]}")

    # ==== Enrichment meta from chunk ====
    if main_pick:
        m0 = main_pick[0].metadata or {}
    elif docs_in_sorted:
        m0 = docs_in_sorted[0].metadata or {}
    else:
        m0 = {}

    enrich_meta = {
        "heading": m0.get("heading", ""),
        "headings": " > ".join(m0.get("headings", []) or []),
        "doc_source": m0.get("doc_source", "")
    }

    # ==== Analyze meta ====
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

    focus_entities = analyze_meta.get("focus_entities") or []
    if focus_entities:
        meta["topic"] = (analyze_meta.get("focus_entities") or [state.get("question_user")])[0]


    # ==== Intent selection once (no duplicates) ====
    intents_ord = normalize_and_order_intents(intents_in)
    MAX_INTENTS = int(os.getenv("MAX_INTENTS", "2"))
    intents_use = intents_ord[:MAX_INTENTS] or ["router_fallback"]

    # ==== Compose prompts ====
    delimiter = "\n\n---\n\n"
    def build_task_prompt(it_name: str) -> str:
        return get_prompt_by_intent(it_name, meta.get("doc_type","")).format(**meta)

    task_prompts = [build_task_prompt(it) for it in intents_use]
    if analyze_meta.get("task_type") == "multi_part" or analyze_meta.get("sub_questions"):
        task_prompts.append(get_prompt_by_intent("multi_part").format(**meta))

    use_streaming = len(task_prompts) > 2
    if not use_streaming:
        prompt_text = f"{CACHED_PROMPT}{delimiter}" + delimiter.join(task_prompts)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            answer = await loop.run_in_executor(pool, sync_stream_generate, prompt_text)
        duration = round(time.time() - t0, 3)
        print(f"💬 [Generate] Done in {duration}s")
        return {"answer": answer, "timing": {"generate": duration}}

    def stream_one(block_text: str) -> str:
        pt = f"{CACHED_PROMPT}{delimiter}{block_text}"
        return sync_stream_generate(pt)

    parts = [stream_one(tp) for tp in task_prompts]
    answer = "\n\n".join(parts)
    duration = round(time.time() - t0, 3)
    print(f"💬 [Generate] Done in {duration}s")
    return {"answer": answer, "timing": {"generate": duration}}


async def analyze(state: dict):
    """Node phân tích câu hỏi → chuẩn hoá + meta cho các bước sau.
    Không tách thô, không hardcode; chỉ dùng kết quả từ analyze_question().
    """
    t0 = time.time()
    question = (state.get("question") or "").strip()

    # Empty input → trả lời nhẹ nhàng và dừng
    if not question:
        dur = round(time.time() - t0, 3)
        return {
            **state,
            "answer": "Mình chưa hiểu câu hỏi của bạn. Bạn có thể nói cụ thể hơn không?",
            "analyze_meta": {
                "question_type": "unclear",
                "status": "unclear",
                "normalized_question": "",
                "intent": [],
                "sub_questions": [],
                "query_variants": [],
                "hyde_query": "",
                "task_type": "single",
                "entities": [],
                "focus_entities": [],
                "per_entity_queries": {},
            },
            "timing": {"analyze": dur},
        }

    # Gọi LLM analyzer (bạn đã cấu hình get_llm phía trong analyze_question)
    result: Dict[str, Any] = await analyze_question(question)
    logging.info(f"🔎 [Analyze Result] {result}")

    status = (result.get("question_type") or "unclear").lower()
    dur = round(time.time() - t0, 3)

    # Nhánh đặc biệt: unsafe/unclear/meta → trả luôn answer ở đây
    if status == "unsafe":
        return {
            **state,
            "answer": "Xin lỗi, mình không thể hỗ trợ với nội dung này.",
            "analyze_meta": result,
            "timing": {"analyze": dur},
        }

    if status == "unclear":
        return {
            **state,
            "answer": "Mình chưa hiểu rõ câu hỏi của bạn, bạn có thể nói cụ thể hơn không?",
            "analyze_meta": result,
            "timing": {"analyze": dur},
        }

    if status == "meta":
        try:
            from app.services.meta_responder import search_meta
            meta_ans = search_meta(question) or "Xin lỗi, mình chưa có câu trả lời cho nội dung này."
        except Exception as e:
            logging.error(f"[Analyze][meta] handler error: {e}")
            meta_ans = "Xin lỗi, mình chưa có câu trả lời cho nội dung này."
        return {
            **state,
            "answer": meta_ans,
            "analyze_meta": result,
            "timing": {"analyze": dur},
        }

    # Academic bình thường → đi tiếp
    subqs = result.get("sub_questions") or []
    if subqs:
        logging.info("🧭 Router[after_analyze] Sub-question details:")
        for sq_meta in result.get("sub_questions_with_intent", []):
            subq = sq_meta.get("subq")
            intents = sq_meta.get("intent", [])
            topic = sq_meta.get("topic", result.get("topic"))
            logging.info(f"   • {subq} | intent={intents} | topic={topic}")

    return {
        **state,
        "question_user": state.get("question") or question,
        "question_norm": result.get("normalized_question") or question,
        "analyze_meta": result,
        "timing": {"analyze": dur},
    }

# ======================= #
# 5️⃣ Build Graph
# ======================= #
# ... (imports, State, tất cả hàm analyze/retrieve/rerank_node/generate_per_subq/reflect_and_merge/generate)
_graph = None  # cache singleton

def get_graph():
    global _graph
    # 👉 Tuỳ chọn: set NO_GRAPH_CACHE=true trong .env khi debug
    if _graph is not None and os.getenv("NO_GRAPH_CACHE", "false").lower() != "true":
        return _graph

    graph_builder = StateGraph(State)

    # --- nodes ---
    graph_builder.add_node("analyze", analyze)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("rerank", rerank_node)
    graph_builder.add_node("generate_per_subq", generate_per_subq)
    graph_builder.add_node("reflect", reflect_and_merge)
    graph_builder.add_node("generate", generate)

    # --- start ---
    graph_builder.add_edge(START, "analyze")

    # --- routers ---
    def _route_after_analyze(state: State) -> str:
        meta = state.get("analyze_meta") or {}
        has_subqs = bool(meta.get("sub_questions"))
        return "generate_per_subq" if has_subqs else "retrieve"

    graph_builder.add_conditional_edges(
        "analyze",
        _route_after_analyze,
        {
            "generate_per_subq": "generate_per_subq",
            "retrieve": "retrieve",
        }
    )

    graph_builder.add_edge("retrieve", "rerank")

    def _route_after_rerank(state: State) -> str:
        meta = state.get("analyze_meta") or {}
        task_type = (meta.get("task_type") or "").lower()
        has_subqs = bool(meta.get("sub_questions"))
        if has_subqs or task_type in ("comparison", "multi_part"):
            return "generate_per_subq"
        return "generate"

    graph_builder.add_conditional_edges(
        "rerank",
        _route_after_rerank,
        {
            "generate_per_subq": "generate_per_subq",
            "generate": "generate"
        }
    )

    graph_builder.add_edge("generate_per_subq", "reflect")

    _graph = graph_builder.compile()
    return _graph


# Đặt ở cuối file (trước run_graph hoặc cạnh get_graph)
async def warmup_rag():
    """
    Preload các thành phần nặng để giảm TTFB:
    - E5 (MMR)
    - Reranker (GPU) qua heavy_rerank(...)
    - Tạo một encode/dummy pass ngắn
    Không đổi logic runtime; chỉ nạp model trước.
    """
    try:
        _ = _get_mmre5()
        # dummy encode ngắn (không ảnh hưởng state chung)
        if torch.cuda.is_available():
            with torch.inference_mode():
                _._first_module().to("cuda") if hasattr(_, "_first_module") else None
                _.encode(["warmup"], normalize_embeddings=True, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
        else:
            _.encode(["warmup"], normalize_embeddings=True, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
    except Exception as e:
        logging.warning(f"[warmup_rag] MMR warmup skipped: {e}")
    try:
        # gọi reranker một lần thông qua hàm trong services/reranker
        from app.services.reranker import get_reranker
        get_reranker()
    except Exception as e:
        logging.warning(f"[warmup_rag] Reranker warmup skipped: {e}")



# ======================= #
# 6️⃣ Runner
# ======================= #
async def run_graph(input_state: dict):
    start_total = time.time()
    final_state = await get_graph().ainvoke(input_state)  # type: ignore
    total_time = round(time.time() - start_total, 3)

    print("\n⏱️ [RAG Pipeline Summary]")
    print(json.dumps(final_state.get("timing", {}), indent=2))
    print(f"🚀 Total time: {total_time}s")
    return final_state
