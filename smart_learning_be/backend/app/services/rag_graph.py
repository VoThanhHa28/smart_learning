# ======================= #
# 🧠 Simple RAG Pipeline (Stable)
# ======================= #

import os
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
import asyncio, json, time
from concurrent.futures import ThreadPoolExecutor
import logging
logging.basicConfig(level=logging.INFO)

from app.services.hybrid import hybrid_retrieve
from app.services.llm import get_llm
from app.services.reranker import rerank as heavy_rerank
from app.services.config import RERANKER_MODE, FINAL_TOP_N
from app.services.prompt_utils import CACHED_PROMPT
from app.services.rag_utils import build_context, fuse_chunks_by_heading
from app.services.analyze_query import analyze_question

# ======================= #
# 1️⃣ Define State
# ======================= #
class State(TypedDict, total=False):  # total=False giúp cho phép key optional
    question: str
    question_user: str
    question_norm: str
    subject: str
    course_id: str
    context: List[Document]
    analyze_meta: dict
    answer: str
    timing: dict

def debug_state(node_name, state):
    cid = state.get("course_id")
    print(f"🐛 [DEBUG] Node={node_name} | course_id={cid} | keys={list(state.keys())}")


async def analyze(state: dict):
    question = state["question"]
    result = await analyze_question(question)

    if result.get("status") == "stop":
        print("🚫 [Analyze] stop -> returning answer and short-circuit downstream")
        return {
            **state,
            "question_user": question,
            "question_norm": "",         # tránh KeyError downstream
            "analyze_meta": result,
            "answer": result["answer"],  # đã có answer
            "timing": {"analyze": 0.0}
        }

    return {
        **state,
        "question_user": question,
        "question_norm": result.get("normalized_question", question),
        "analyze_meta": result,
    }


# ======================= #
# 2️⃣ Retrieve
# ======================= #
async def retrieve(state: dict):
    debug_state("retrieve", state)

    # Early-exit: nếu analyze đã dừng pipeline hoặc đã có answer, skip heavy work
    if state.get("answer") or (state.get("analyze_meta", {}).get("status") == "stop"):
        print("⏭️ [Retrieve] Skipped because analyze() already produced an answer or requested stop.")
        return {**state, "context": [], "timing": {"retrieve": 0.0}}

    t0 = time.time()
    query = state.get("question_norm", "") or ""  # safe get

    # If normalized query is empty, skip search (avoid BM25 returning top-k on empty query)
    if not query.strip():
        print("🔕 [Retrieve] Empty normalized query -> skipping search to avoid irrelevant results.")
        return {**state, "context": [], "timing": {"retrieve": 0.0}}

    print(f"🔎 [Retrieve] Query='{query}'")

    # build filters safely and lower-case (Milvus/BM25 stored lowercase)
    raw_filters = state.get("filters", {}) or {}
    # also accept subject/course_id passed in top-level state
    raw_filters.setdefault("subject", state.get("subject"))
    raw_filters.setdefault("course_id", state.get("course_id"))

    filters = {}
    if raw_filters.get("subject"):
        filters["subject"] = str(raw_filters.get("subject")).lower().strip()
    if raw_filters.get("course_id"):
        filters["course_id"] = str(raw_filters.get("course_id")).lower().strip()

    results = await hybrid_retrieve(
        query,
        k_dense=12,
        k_sparse=24,
        top_after_rrf=50,
        filters=filters or None,
        return_raw=False
    )

    duration = round(time.time() - t0, 3)
    print(f"✅ [Retrieve] Got {len(results)} docs in {duration:.2f}s")
    return {**state, "context": results, "timing": {"retrieve": duration}}


# ======================= #
# 3️⃣ Rerank (optional)
# ======================= #
async def rerank_node(state: State):
    debug_state("rerank", state)
    t0 = time.time()
    docs = state["context"][:12]

    if not docs:
        return {"context": [], "timing": state.get("timing", {})}

    # 🧩 Dedup duplicate docs by content
    docs = list({d.page_content.strip(): d for d in docs if d.page_content.strip()}.values())
    print(f"🧹 Dedup after retrieve: {len(docs)} unique docs remain for rerank.")

    if RERANKER_MODE == "off":
        reranked = docs[:FINAL_TOP_N]
    else:
        reranked = await heavy_rerank(state["question"], docs, top_n=FINAL_TOP_N)

    duration = round(time.time() - t0, 3)
    print(f"🏅 [Reranker] Selected {len(reranked)} docs in {duration}s")
    return {"context": reranked, "timing": {"rerank": duration}}


# ======================= #
# 4️⃣ Generate (LLM)
# ======================= #
def sync_stream_generate(prompt_text: str) -> str:
    llm = get_llm(streaming=True)
    output = ""
    for chunk in llm.stream(prompt_text):
        token = getattr(chunk, "content", "") or ""
        print(token, end="", flush=True)
        output += token
    print("\n✅ [Stream Completed]")
    return output


async def generate(state: State):
    debug_state("generate", state)

    # Early-exit: if analyze already produced an answer (stop or quick reply), return it
    if state.get("answer"):
        print("⏭️ [Generate] Skipped because answer present in state (from analyze()).")
        return {"answer": state["answer"], "timing": {"generate": 0.0}}

    t0 = time.time()

    # 🧩 DEBUG context flow
    print("\n==== 🔍 DEBUG CONTEXT PIPELINE ====")
    docs_in = state.get("context", [])
    print(f"[1] Retrieved docs (input to rerank/generate): {len(docs_in)}")
    for i, d in enumerate(docs_in[:5]):
        txt = getattr(d, "page_content", "")[:120].replace("\n", " ")
        print(f"→ DOC[{i}] {txt}")
    print("===================================\n")

    fused_docs = fuse_chunks_by_heading(docs_in or [])
    # Prefer course_id match (case-insensitive), but fallback to top N fused docs
    selected_docs = [
        d for d in fused_docs
        if str(d.metadata.get("course_id", "")).lower() == str(state.get("course_id", "")).lower()
    ]
    if not selected_docs:
        selected_docs = fused_docs[:5]

    context = build_context(selected_docs or [])

    print(f"📚 [DEBUG] Context length = {len(context)} chars")
    print(f"📚 [DEBUG] Context preview:\n{context[:800]}")

    # 🧩 Enrichment metadata từ chunk
    enrich_meta = {}
    if selected_docs:
        first_meta = selected_docs[0].metadata or {}
        enrich_meta = {
            "heading": first_meta.get("heading", ""),
            "headings": " > ".join(first_meta.get("headings", []) or []),
            "doc_source": first_meta.get("doc_source", "")
        }

    # 🧠 Analyze metadata
    analyze_meta = state.get("analyze_meta", {})

    # Intent-aware fallback cho câu hỏi liên quan TOC / chương
    q_lower = (state.get("question_norm") or "").lower()
    toc_keywords = ["chương", "chapter", "mục lục", "bài học", "lesson", "phần"]
    need_toc = any(k in q_lower for k in toc_keywords)

    is_toc_context = False
    # Nếu câu hỏi liên quan TOC hoặc context rỗng
    if need_toc or not context.strip():
        toc_index_path = os.path.join("uploads", "toc_index.json")
        toc_data = []
        course_id = state.get("course_id") or state.get("course") or ""
        if not course_id:
            print("⚠️ [WARN] course_id bị thiếu — không thể fallback TOC theo môn học.")

        print(f"🧭 Câu hỏi có vẻ liên quan TOC hoặc context rỗng, sẽ fallback dùng TOC nếu có (course_id={course_id})")

        if os.path.exists(toc_index_path):
            with open(toc_index_path, "r", encoding="utf-8") as f:
                toc_index = json.load(f)

            # Ưu tiên match theo course_id (case-insensitive)
            if course_id:
                for key, val in toc_index.items():
                    print(f"🔍 So khớp: {key} ? {course_id}")
                    if key.lower().startswith(f"{course_id.lower()}_") or val.get("course_id", "").lower() == course_id.lower():
                        toc_data = val.get("toc", [])
                        break
                print(f"🧭 Tìm TOC theo course_id='{course_id}' → {len(toc_data)} mục")

        if toc_data:
            context = "\n".join([f"- {t}" for t in toc_data])
            is_toc_context = True
            print(f"🧭 Fallback dùng TOC ({len(toc_data)} mục, match course_id={course_id})")
        else:
            print(f"⚠️ Không tìm thấy TOC cho course_id={course_id}")

    # ✅ Merge có fallback an toàn
    meta = {
        "subject": state.get("subject", "General"),
        "context": context or "Không có ngữ cảnh phù hợp.",
        "topic": analyze_meta.get("topic") or enrich_meta.get("headings", ""),
        "difficulty": analyze_meta.get("difficulty", "medium"),
        "doc_source": enrich_meta.get("doc_source", "Tài liệu học tập"),
        "doc_type": "TOC" if is_toc_context else analyze_meta.get("doc_type", ""),
        "intent": analyze_meta.get("intent", ""),
        "question": state.get("question_user")  # LLM dùng câu hỏi thật của user
    }

    # 🧩 chọn prompt phù hợp
    prompt = CACHED_PROMPT.partial(subject=meta["subject"])
    prompt_text = prompt.format(**meta)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        answer = await loop.run_in_executor(pool, sync_stream_generate, prompt_text)

    duration = round(time.time() - t0, 3)
    print(f"💬 [Generate] Done in {duration}s")
    return {"answer": answer, "timing": {"generate": duration}}



# ======================= #
# 5️⃣ Build Graph
# ======================= #
graph_builder = StateGraph(State)
graph_builder.add_node("analyze", analyze)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("rerank", rerank_node)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "analyze")
graph_builder.add_edge("analyze", "retrieve")
graph_builder.add_edge("retrieve", "rerank")
graph_builder.add_edge("rerank", "generate")
graph = graph_builder.compile()


# ======================= #
# 6️⃣ Runner
# ======================= #
async def run_graph(input_state: dict):
    start_total = time.time()
    final_state = await graph.ainvoke(input_state)
    total_time = round(time.time() - start_total, 3)

    print("\n⏱️ [RAG Pipeline Summary]")
    print(json.dumps(final_state.get("timing", {}), indent=2))
    print(f"🚀 Total time: {total_time}s")
    return final_state
