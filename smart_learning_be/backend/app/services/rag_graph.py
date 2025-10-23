# app/services/rag_graph.py
import os, time, json, logging, torch
from langgraph.graph import StateGraph, START
from app.services.rag_state import State
from app.services.nodes import analyze, retrieve, generate_per_subq, reflect_and_merge, generate

_graph = None

def get_graph():
    global _graph
    if _graph is not None and os.getenv("NO_GRAPH_CACHE", "false").lower() != "true":
        return _graph

    graph_builder = StateGraph(State)
    graph_builder.add_node("analyze", analyze)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate_per_subq", generate_per_subq)
    graph_builder.add_node("reflect", reflect_and_merge)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "analyze")

    def _route_after_analyze(state: State) -> str:
        meta = state.get("analyze_meta") or {}

        # ✅ nếu đã có answer hoặc analyzer yêu cầu dừng → nhảy thẳng generate
        if state.get("answer") or meta.get("status") == "stop" \
        or (meta.get("question_type") in ("unsafe", "unclear", "meta")):
            return "generate"


        # còn lại: route theo sub-questions
        has_subqs = bool(meta.get("sub_questions_with_intent"))
        return "generate_per_subq" if has_subqs else "retrieve"

    graph_builder.add_conditional_edges(
        "analyze",
        _route_after_analyze,
        {
            "generate_per_subq": "generate_per_subq",
            "retrieve": "retrieve",
            "generate": "generate",          # ✅ thêm nhánh này
        }
    )

    def _route_after_rerank(state: State) -> str:
        meta = state.get("analyze_meta") or {}
        has_subqs = bool(meta.get("sub_questions_with_intent"))
        task_type = (meta.get("task_type") or "").lower()
        if has_subqs or task_type in ("comparison", "multi_part"):
            return "generate_per_subq"
        return "generate"

    # nếu bạn vẫn muốn có node rerank riêng, thêm ở đây (hiện tại pipeline đã dùng rerank trong node riêng khác)
    # graph_builder.add_node("rerank", rerank_node)
    # graph_builder.add_edge("retrieve", "rerank")
    # graph_builder.add_conditional_edges("rerank", _route_after_rerank, {...})

    # hoặc đi thẳng (retrieve -> generate_per_subq/generate)
    graph_builder.add_conditional_edges("retrieve", _route_after_rerank, {
        "generate_per_subq": "generate_per_subq",
        "generate": "generate"
    })

    graph_builder.add_edge("generate_per_subq", "reflect")
    _graph = graph_builder.compile()
    return _graph

async def warmup_rag():
    try:
        from app.services.rag_runtime import _get_mmre5
        _ = _get_mmre5()
        if torch.cuda.is_available():
            import torch as _t
            with _t.inference_mode():
                _.encode(["warmup"], normalize_embeddings=True, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
        else:
            _.encode(["warmup"], normalize_embeddings=True, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
    except Exception as e:
        logging.warning(f"[warmup_rag] MMR warmup skipped: {e}")
    try:
        from app.services.reranker import get_reranker
        get_reranker()
    except Exception as e:
        logging.warning(f"[warmup_rag] Reranker warmup skipped: {e}")

async def run_graph(input_state: dict):
    start_total = time.time()
    final_state = await get_graph().ainvoke(input_state)  # type: ignore
    total_time = round(time.time() - start_total, 3)
    print("\n⏱️ [RAG Pipeline Summary]")
    print(json.dumps(final_state.get("timing", {}), indent=2))
    print(f"🚀 Total time: {total_time}s")
    return final_state
