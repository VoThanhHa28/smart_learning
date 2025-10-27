import logging
import time
import json
import re
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
# Sửa import: Lấy get_llm từ đường dẫn đúng
from ...infrastructure.llm.llm import get_llm
# Sửa import: Lấy State và _update_timing từ đường dẫn đúng
from ..rag_state import State, _update_timing
import asyncio

# --- PROMPT MỚI V4: LLM tự phân tích và quyết định ---
TRANSFORM_V4_PROMPT_TMPL = """
Bạn là chuyên gia tối ưu hóa truy vấn tìm kiếm vector. Nhiệm vụ của bạn là phân tích câu hỏi gốc và tạo ra các truy vấn (queries) hiệu quả nhất để tìm tài liệu liên quan trong cơ sở dữ liệu vector.

**PHÂN TÍCH CÂU HỎI GỐC:**
Đầu tiên, xác định mục đích chính của câu hỏi:
1.  **Loại A (Vị trí/Tham chiếu):** Câu hỏi chủ yếu hỏi *ở đâu* thông tin xuất hiện (ví dụ: "số trang", "đoạn nào", "trích dẫn", "liệt kê", "chương mấy").
2.  **Loại B (Giải thích/So sánh/Ví dụ):** Câu hỏi chủ yếu hỏi *cái gì*, *như thế nào*, *tại sao* (ví dụ: "là gì", "giải thích", "so sánh", "ví dụ", "làm thế nào", "tại sao").

**QUY TẮC TẠO TRUY VẤN:**
* **Nếu là Loại A:** Chỉ tạo **MỘT** truy vấn duy nhất. Truy vấn này là **từ khóa cốt lõi** (core keywords) của chủ đề, loại bỏ tất cả các từ ngữ liên quan đến vị trí/tham chiếu.
* **Nếu là Loại B:** Tạo **BA** truy vấn:
    * Truy vấn 1: **Từ khóa cốt lõi** (giống như Loại A).
    * Truy vấn 2 & 3: Hai **biến thể ngữ nghĩa** (semantic variations) của từ khóa cốt lõi, sử dụng từ đồng nghĩa hoặc cách diễn đạt khác, nhưng vẫn giữ trọng tâm chủ đề. Các biến thể phải ngắn gọn, phù hợp tìm kiếm.

**ĐỊNH DẠNG ĐẦU RA:**
Chỉ trả về một **JSON list chứa các chuỗi truy vấn** đã tạo. KHÔNG markdown, KHÔNG giải thích.

**VÍ DỤ:**

1.  **Câu hỏi gốc:** "int là gì?"
    * **Phân tích:** Loại B (Giải thích).
    * **Đầu ra:** `["int", "số nguyên", "kiểu dữ liệu integer"]`

2.  **Câu hỏi gốc:** "Đưa rõ số trang có nói về float."
    * **Phân tích:** Loại A (Vị trí).
    * **Đầu ra:** `["float"]`

3.  **Câu hỏi gốc:** "So sánh chi tiết int và float khác nhau như thế nào?"
    * **Phân tích:** Loại B (So sánh).
    * **Đầu ra:** `["int và float", "khác biệt giữa integer và floating-point", "so sánh kiểu số nguyên và số thực"]`

4.  **Câu hỏi gốc:** "Cho ví dụ về cách dùng hàm print trong Python."
    * **Phân tích:** Loại B (Ví dụ).
    * **Đầu ra:** `["hàm print python", "cách sử dụng lệnh print", "ví dụ code print python"]`

5.  **Câu hỏi gốc:** "Liệt kê các chương nói về lập trình hướng đối tượng."
    * **Phân tích:** Loại A (Tham chiếu/Liệt kê).
    * **Đầu ra:** `["lập trình hướng đối tượng"]`

**BÂY GIỜ, HÃY XỬ LÝ CÂU HỎI SAU:**

**Câu hỏi gốc:** "{question}"
**JSON Output:**
"""
TRANSFORM_V4_PROMPT = PromptTemplate.from_template(TRANSFORM_V4_PROMPT_TMPL)

# --- Bỏ hàm is_location_query ---
# def is_location_query(question: str) -> bool: ...

async def transform_query_node(state: State) -> State:
    """
    Node V4: Dùng LLM để phân tích loại câu hỏi và tạo query phù hợp
    (keyword-only hoặc keyword + expansion).
    Trả về 'retrieval_queries' (list).
    """
    t_start = time.perf_counter()
    print("➡️  [Node] transform_query (V4 LLM Decides): Bắt đầu...")
    meta = state.get("analyze_meta", {})
    if not meta:
        print("  [Transform V4] ⚠️ Không có analyze_meta.")
        return _update_timing(state, "transform_query", t_start)

    original_sqwi: List[Dict[str, Any]] = meta.get("sub_questions_with_intent", [])
    if not original_sqwi:
        print("  [Transform V4] ⚠️ Không có sub-questions.")
        return _update_timing(state, "transform_query", t_start)

    transformed_sqwi: List[Dict[str, Any]] = []
    tasks = []

    # --- Hàm nội bộ _transform_one (Sửa lại hoàn toàn) ---
    async def _transform_one(item: Dict[str, Any]):
        # Lấy subq gốc từ analyze
        original_sq = item.get("subq")
        intent_category = (item.get("intent") or ["unknown"])[0]
        # Khởi tạo retrieval_queries mặc định là câu gốc
        retrieval_queries = [original_sq] if original_sq else []

        # Chỉ xử lý 'academic_rag'
        if intent_category == "academic_rag" and original_sq:
            print(f"  [Transform V4] ⏳ Bắt đầu xử lý LLM Transform cho: \"{original_sq[:30]}...\"")
            try:
                # Dùng model nhanh (đã sửa ở llm.py)
                llm_fast = get_llm()

                # Gọi LLM một lần duy nhất với prompt V4
                response = await llm_fast.ainvoke(TRANSFORM_V4_PROMPT.format(question=original_sq))
                response_content = getattr(response, "content", str(response) or "[]").strip()

                # Cố gắng parse JSON list trả về
                try:
                    # Trích xuất phần JSON nếu có text thừa
                    json_match = re.search(r"\[[\s\S]*\]", response_content)
                    if json_match:
                        json_str = json_match.group(0)
                        # Sửa lỗi phẩy thừa cuối list (nếu có)
                        json_str = re.sub(r",\s*\]", "]", json_str)
                        parsed_queries = json.loads(json_str)
                        # Đảm bảo kết quả là list các string không rỗng
                        if isinstance(parsed_queries, list):
                             retrieval_queries = [str(q).strip() for q in parsed_queries if isinstance(q, str) and str(q).strip()]
                        else:
                             print(f"    [Transform V4] ⚠️ LLM không trả về JSON list hợp lệ. Dùng câu gốc.")
                             retrieval_queries = [original_sq]
                    else:
                         print(f"    [Transform V4] ⚠️ Không tìm thấy JSON list trong phản hồi LLM. Dùng câu gốc.")
                         retrieval_queries = [original_sq]

                except json.JSONDecodeError as json_e:
                    print(f"    [Transform V4] 💥 Lỗi parse JSON từ LLM: {json_e}. Phản hồi rå: '{response_content[:100]}...' Dùng câu gốc.")
                    retrieval_queries = [original_sq]
                except Exception as parse_e:
                     print(f"    [Transform V4] 💥 Lỗi không xác định khi xử lý JSON: {parse_e}. Dùng câu gốc.")
                     retrieval_queries = [original_sq]

                # Đảm bảo luôn có ít nhất 1 query
                if not retrieval_queries:
                     retrieval_queries = [original_sq]

                print(f"  [Transform V4] ✅ Xong \"{original_sq[:30]}...\". Final Queries: {retrieval_queries}")

            except Exception as e:
                print(f"  [Transform V4] 💥 Lỗi nghiêm trọng khi gọi LLM Transform '{original_sq[:30]}...': {e}")
                retrieval_queries = [original_sq] # Fallback dùng câu gốc

            # Trả về item với list các query
            return {
                **item,
                "original_subq": original_sq,
                "retrieval_queries": retrieval_queries # <-- KEY chứa list query
            }
        else:
            # Giữ nguyên các câu hỏi không phải academic_rag
            print(f"  [Transform V4] ⏩ Giữ nguyên non-academic: \"{original_sq[:30]}...\"")
            return {**item, "original_subq": original_sq, "retrieval_queries": [original_sq]}
    # --- Kết thúc hàm nội bộ ---

    tasks = [_transform_one(item) for item in original_sqwi]
    transformed_results = await asyncio.gather(*tasks)
    transformed_sqwi = [res for res in transformed_results if res is not None]

    new_meta = {**meta, "sub_questions_with_intent": transformed_sqwi}

    print("--- Transform Result (V4 LLM Decides) ---")
    print(json.dumps(transformed_sqwi, indent=2, ensure_ascii=False))
    print("-----------------------------------------")

    new_state = {**state, "analyze_meta": new_meta}
    new_state = _update_timing(new_state, "transform_query", t_start)
    print(f"✅ [Node] transform_query (V4 LLM Decides): Biến đổi xong.")
    return new_state