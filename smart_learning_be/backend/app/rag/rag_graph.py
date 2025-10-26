import os, time, json, logging, torch
from langgraph.graph import StateGraph, START
# 1. Sửa import State và _update_timing (đảm bảo đường dẫn đúng)
from .rag_state import State, _update_timing
# Import các node (đảm bảo đường dẫn đúng)
from .nodes.analyze import analyze
from .nodes.transform_query import transform_query_node
from .nodes.retrieve import retrieve
from .nodes.generate_per_subq import generate_per_subq
from .nodes.reflect import reflect_and_merge
from .nodes.generate import generate as generate_simple # Đổi tên generate cũ
# 2. Sửa import reranker (đảm bảo đường dẫn đúng)
from ..infrastructure.llm.reranker import get_reranker # Giả sử import đúng

_graph = None

def get_graph():
    global _graph
    if _graph is not None and os.getenv("NO_GRAPH_CACHE", "false").lower() != "true":
        return _graph

    print("🏁 [Graph] Đang khởi tạo StateGraph...")
    graph_builder = StateGraph(State)

    # Thêm các node (Giữ nguyên)
    graph_builder.add_node("analyze", analyze)
    graph_builder.add_node("transform_query", transform_query_node)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate_per_subq", generate_per_subq)
    graph_builder.add_node("reflect", reflect_and_merge)
    graph_builder.add_node("generate_simple", generate_simple)

    # Định nghĩa luồng (flow) (Giữ nguyên)
    graph_builder.add_edge(START, "analyze")
    graph_builder.add_edge("analyze", "transform_query")

    # Logic rẽ nhánh sau transform (Giữ nguyên logic)
    def _route_after_transform(state: State) -> str:
        print("➡️  [Node] _route_after_transform: Đang kiểm tra kết quả 'transform'...")
        meta = state.get("analyze_meta") or {}
        sqwi = meta.get("sub_questions_with_intent", [])
        # Xử lý dừng sớm hoặc đặc biệt
        if len(sqwi) == 1 and sqwi[0].get("intent", [""])[0] == "system_handlers":
             topic = sqwi[0].get("topic", "")
             if topic in ("unsafe", "unclear", "meta", "toc"):
                  print(f"✅ [Router] V2: Phát hiện 1 sub-question ({topic}). Định tuyến đến [generate_per_subq]")
                  return "generate_per_subq"
        # Logic chính
        if sqwi:
             print(f"✅ [Router] V2: Phát hiện {len(sqwi)} sub-questions. Định tuyến đến [generate_per_subq]")
             return "generate_per_subq"
        else:
             print(f"✅ [Router] V2: Không có sub-questions. Định tuyến đến [retrieve]")
             return "retrieve"

    graph_builder.add_conditional_edges(
        "transform_query", _route_after_transform,
        {"generate_per_subq": "generate_per_subq", "retrieve": "retrieve"}
    )

    # Các cạnh còn lại (Giữ nguyên)
    graph_builder.add_edge("retrieve", "generate_simple")
    graph_builder.add_edge("generate_per_subq", "reflect")

    _graph = graph_builder.compile()
    print("✅ [Graph] StateGraph đã compile xong.")
    return _graph

# --- Hàm warmup giữ nguyên ---
async def warmup_rag():
    # ... (code warmup của bạn, đảm bảo gọi get_reranker nếu cần) ...
    try:
        get_reranker() # Gọi để preload reranker
        logging.info("✅ Reranker warmup initiated.")
        # Thêm warmup cho embedding nếu cần
        # from ..infrastructure.llm.embedding import get_embeddings
        # get_embeddings().embed_query("warmup")
        # logging.info("✅ Embedding warmup initiated.")
    except Exception as e:
        logging.warning(f"⚠️ Warmup component failed: {e}")
    # Giữ lại pass nếu không có gì khác
    pass


# --- Sửa hàm run_graph (ĐÃ BAO GỒM LOG DEBUG) ---
async def run_graph(input_state: dict) -> dict:
    t_start_total = time.perf_counter()
    print(f"\n🚀 [Graph] Bắt đầu chạy Graph cho câu hỏi: \"{input_state.get('question', '')[:50]}...\"")

    current_timing = input_state.get("timing", {})
    # Đảm bảo input state có kiểu đúng theo State TypedDict nếu cần ép kiểu
    input_state_with_timing = {**input_state, "timing": current_timing}

    # Chạy Graph
    final_state: State = {} # Khởi tạo để mypy không báo lỗi
    try:
        # Sử dụng get_graph() để lấy instance đã compile
        compiled_graph = get_graph()
        # Chạy Graph với input state
        final_state = await compiled_graph.ainvoke(input_state_with_timing)

    except Exception as graph_exec_error:
         logging.exception(f"💥 Lỗi nghiêm trọng khi thực thi Graph (ainvoke): {graph_exec_error}")
         # Trả về state lỗi để chain.py xử lý
         final_state = {
             **input_state_with_timing,
             "answer": f"[Lỗi Graph: {graph_exec_error}]",
             "status": "stop",
             "diag": {"graph_error": str(graph_exec_error)}
         }
         # Vẫn ghi lại thời gian lỗi
         total_duration_ms = round((time.perf_counter() - t_start_total) * 1000)
         print(f"💥 [Graph] Thực thi thất bại sau {total_duration_ms} ms.")
         return final_state # Trả về state lỗi


    # --- LOG DEBUG CHI TIẾT FINAL_STATE SAU KHI INVOKE ---
    print("\n--- DEBUG: Final State TRỰC TIẾP TỪ ainvoke ---")
    print(f"  Type: {type(final_state)}")
    if isinstance(final_state, dict):
        print(f"  Keys: {list(final_state.keys())}")
        # Kiểm tra key 'final_context_docs'
        final_docs_debug = final_state.get("final_context_docs")
        print(f"  final_context_docs (type: {type(final_docs_debug)}):")
        if isinstance(final_docs_debug, list):
            print(f"    Length: {len(final_docs_debug)}")
            if final_docs_debug:
                try:
                    # Truy cập metadata an toàn hơn
                    first_doc = final_docs_debug[0]
                    metadata = getattr(first_doc, 'metadata', {})
                    print(f"    First doc metadata sample: {{'page': {metadata.get('page')}, 'subject': '{metadata.get('subject')}'}}")
                except Exception as e:
                    print(f"    Error accessing first doc metadata: {e}")
            else:
                 print("    List is empty.")
        else:
            print(f"    Value: {final_docs_debug}")

        # Kiểm tra key 'timing'
        timing_debug = final_state.get("timing")
        print(f"  timing (type: {type(timing_debug)}): {timing_debug}")
    else:
        print(f"  Value: {final_state}") # In giá trị nếu không phải dict
    print("-------------------------------------------------\n")
    # --- KẾT THÚC LOG DEBUG ---

    # Tính tổng thời gian
    total_duration_ms = round((time.perf_counter() - t_start_total) * 1000)

    # --- IN TỔNG KẾT THỜI GIAN ---
    print("\n⏱️ --- RAG Pipeline Timing Summary --- ⏱️")
    timing_info = final_state.get("timing", {})
    node_order = ["analyze", "transform_query", "retrieve", "generate_per_subq", "generate_simple", "reflect"]
    if timing_info:
        printed_nodes = set()
        for node_name in node_order:
            if node_name in timing_info:
                duration_ms = timing_info[node_name]
                print(f"   - {node_name:<20}: {duration_ms:>7} ms")
                printed_nodes.add(node_name)
        # In các node khác không theo thứ tự nếu có
        for node_name, duration_ms in timing_info.items():
            if node_name not in printed_nodes:
                 print(f"   - {node_name:<20}: {duration_ms:>7} ms")
    else:
        print("   (Không có thông tin timing chi tiết)")
    print(f"------------------------------------------")
    print(f"⏱️ --- Tổng cộng:{total_duration_ms:>22} ms --- ⏱️")
    print(f"🏁 [Graph] Hoàn thành.")
    # --- KẾT THÚC IN TIMING ---

    # --- IN KẾT QUẢ ANALYZE VÀ TRANSFORM ---
    analyze_res = final_state.get("analyze_meta", {})
    if analyze_res:
         print("\n--- Final Analyze & Transform Meta ---")
         try:
              # In sub_questions_with_intent nếu có
              sqwi_to_print = analyze_res.get("sub_questions_with_intent", "Not found")
              print(json.dumps(sqwi_to_print, indent=2, ensure_ascii=False))
         except Exception as json_err:
              print(f"   (Error printing analyze_meta: {json_err})")
              print(f"   Raw analyze_meta: {analyze_res}") # In raw nếu không dump được
         print("------------------------------------\n")
    # --- KẾT THÚC IN META ---

    return final_state # Trả về final_state nhận được từ ainvoke
