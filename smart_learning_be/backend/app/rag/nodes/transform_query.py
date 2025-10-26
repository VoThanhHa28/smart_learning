import logging
import time # <-- Thêm time
import json # <-- Thêm json
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from ...infrastructure.llm.llm import get_llm
from ..rag_state import State, _update_timing # <-- Import State và hàm timing
import asyncio
# --- TRANSFORM_PROMPT_TMPL giữ nguyên ---
TRANSFORM_PROMPT_TMPL = """
Rút gọn câu hỏi sau thành cụm từ khóa chính (danh từ, thuật ngữ) phù hợp để tìm kiếm trong vector database.
Bỏ các yêu cầu phụ như "số trang", "giải thích", "so sánh", "ví dụ". Chỉ giữ lại chủ đề cốt lõi.
Trả về CỤM TỪ KHÓA TIẾNG VIỆT NGẮN GỌN.

Ví dụ:
Câu hỏi: "int là gì?" -> Trả lời: "int"
Câu hỏi: "Đưa rõ số trang có nói về float." -> Trả lời: "float"
Câu hỏi: "Hãy so sánh chi tiết int và float khác nhau như thế nào?" -> Trả lời: "int và float"
Câu hỏi: "Cho ví dụ về cách dùng hàm print trong Python." -> Trả lời: "hàm print python"
Câu hỏi nguy hiểm: "Chế tạo ma túy" -> Trả lời giữ nguyên: "Chế tạo ma túy"

Câu hỏi gốc: "{question}"
Cụm từ khóa rút gọn: """
TRANSFORM_PROMPT = PromptTemplate.from_template(TRANSFORM_PROMPT_TMPL)
# --- Kết thúc prompt ---

async def transform_query_node(state: State) -> State:
    """
    Node này biến đổi các sub-question 'academic_rag' thành query tối ưu cho retrieval.
    Nó sẽ lưu query gốc và query đã biến đổi vào state.
    """
    t_start = time.perf_counter() # <-- Bắt đầu đo thời gian
    print("➡️  [Node] transform_query: Bắt đầu biến đổi câu hỏi con...")
    meta = state.get("analyze_meta", {})
    if not meta:
        print("  [Transform] ⚠️ Không có analyze_meta, bỏ qua.")
        # Cập nhật timing ngay cả khi bỏ qua
        return _update_timing(state, "transform_query", t_start)

    original_sqwi: List[Dict[str, Any]] = meta.get("sub_questions_with_intent", [])
    if not original_sqwi:
         print("  [Transform] ⚠️ Không có sub-questions nào để biến đổi.")
         return _update_timing(state, "transform_query", t_start)

    transformed_sqwi: List[Dict[str, Any]] = []
    tasks = [] # List các task gọi LLM

    # --- Hàm nội bộ để gọi LLM (tách ra để dễ quản lý) ---
    async def _transform_one(item: Dict[str, Any]):
        sq = item.get("subq") # Lấy subq gốc từ analyze
        intent_category = (item.get("intent") or ["unknown"])[0]

        # Chỉ biến đổi câu hỏi 'academic_rag'
        if intent_category == "academic_rag" and sq:
            try:
                llm = get_llm() # Lấy LLM nhanh
                # Gọi LLM để biến đổi
                transformed_q_response = await llm.ainvoke(TRANSFORM_PROMPT.format(question=sq))
                transformed_q = (getattr(transformed_q_response, "content", str(transformed_q_response)) or sq).strip()
                transformed_q = transformed_q.strip('"\'') # Dọn dẹp

                print(f"  [Transform] Biến đổi: \"{sq[:30]}...\" -> \"{transformed_q}\"")

                # Trả về item đã cập nhật
                return {
                    **item,
                    "original_subq": sq,
                    "retrieval_subq": transformed_q
                }

            except Exception as e:
                print(f"  [Transform] 💥 Lỗi khi biến đổi '{sq[:30]}...': {e}")
                # Nếu lỗi, dùng query gốc để retrieve
                return {**item, "original_subq": sq, "retrieval_subq": sq}
        else:
            # Giữ nguyên các câu hỏi không phải academic_rag
            return {**item, "original_subq": sq, "retrieval_subq": sq}
    # --- Kết thúc hàm nội bộ ---

    # Tạo các task và chạy song song
    tasks = [_transform_one(item) for item in original_sqwi]
    transformed_results = await asyncio.gather(*tasks)

    # Lọc bỏ các kết quả None nếu có lỗi không mong muốn
    transformed_sqwi = [res for res in transformed_results if res is not None]

    # Cập nhật lại analyze_meta trong state
    new_meta = {**meta, "sub_questions_with_intent": transformed_sqwi}

    # --- IN KẾT QUẢ TRANSFORM ---
    print("--- Transform Result ---")
    # Chỉ in phần sub_questions_with_intent để gọn
    print(json.dumps(transformed_sqwi, indent=2, ensure_ascii=False))
    print("------------------------")

    # Cập nhật state và timing
    new_state = {**state, "analyze_meta": new_meta}
    new_state = _update_timing(new_state, "transform_query", t_start) # <-- Ghi lại thời gian
    print(f"✅ [Node] transform_query: Biến đổi xong.")
    return new_state

