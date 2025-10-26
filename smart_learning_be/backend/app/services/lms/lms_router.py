import logging
from typing import List, Tuple, Dict, Any
from langchain_core.tools import tool
from pymilvus import Collection, connections, utility, DataType, FieldSchema, CollectionSchema
import numpy as np
import os

# 1. IMPORT CÁC HÀM "CHUẨN" TỪ FILE CỦA BẠN
from ...rag.retrieval.vectorstore import get_embeddings, _connect_once, EMBED_DIM, METRIC_TYPE

# --- 2. Định nghĩa "Nhân viên" (Tools) ---
# (Đây là nơi bạn thêm 1000 tool của mình)
@tool
def get_student_grade(course_id: str, student_id: str) -> str:
    """
    Dùng tool này khi người dùng hỏi về ĐIỂM SỐ, BẢNG ĐIỂM, KẾT QUẢ HỌC TẬP.
    """
    logging.info(f"[LMS Tool] Đang chạy get_student_grade cho C:{course_id} S:{student_id}")
    # (Giả lập)
    if student_id == "user_123":
        return "Điểm của bạn cho khóa học này là 9.5"
    return "Không tìm thấy thông tin điểm số."

@tool
def get_homework(course_id: str) -> str:
    """
    Dùng tool này khi người dùng hỏi về BÀI TẬP VỀ NHÀ, ASSIGNMENT, HẠN CHÓT nộp bài.
    """
    logging.info(f"[LMS Tool] Đang chạy get_homework cho C:{course_id}")
    # (Giả lập)
    if course_id: 
        return "Bài tập về nhà tuần này là: 'Trắc nghiệm Chương 2', hạn chót Thứ Sáu."
    return "Không tìm thấy thông tin bài tập."

# --- 3. Sổ đăng ký 1000 Tools ---
LMS_TOOLS = [get_student_grade, get_homework] 
LMS_TOOL_MAP: Dict[str, Any] = {tool.name: tool for tool in LMS_TOOLS}

# --- 4. Cấu hình Collection (RIÊNG) cho Tools ---
LMS_TOOL_COLLECTION_NAME = "lms_tools"
_tool_collection = None

def get_tool_collection() -> Collection:
    """
    Lấy (hoặc tạo) collection Milvus để lưu trữ mô tả tool.
    """
    global _tool_collection
    if _tool_collection:
        return _tool_collection

    _connect_once() # <-- Dùng hàm kết nối "chuẩn" của bạn
    
    if not utility.has_collection(LMS_TOOL_COLLECTION_NAME):
        logging.info(f"Đang tạo collection cho tool: {LMS_TOOL_COLLECTION_NAME}")
        # Định nghĩa schema (cấu trúc) cho collection tool
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="tool_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM)
        ]
        schema = CollectionSchema(fields, "Collection for LMS tool retrieval")
        _tool_collection = Collection(LMS_TOOL_COLLECTION_NAME, schema)
        
        # Tạo index
        _tool_collection.create_index(
            "embedding",
            {"index_type": "HNSW", "metric_type": METRIC_TYPE, "params": {"M": 16, "efConstruction": 64}}
        )
    else:
        _tool_collection = Collection(LMS_TOOL_COLLECTION_NAME)
        
    _tool_collection.load()
    logging.info(f"✅ Collection tool '{LMS_TOOL_COLLECTION_NAME}' đã sẵn sàng.")
    return _tool_collection

def initialize_lms_retriever():
    """
    (Hàm này được `main.py` gọi khi khởi động)
    "Nhúng" (embed) và "Chèn" (upsert) mô tả tool vào Milvus.
    """
    try:
        logging.info(f"Đang khởi tạo LMS Tool Retriever (Milvus)...")
        col = get_tool_collection()
        
        # Chỉ upsert nếu tool chưa có trong collection (đơn giản)
        existing_tools = col.query("tool_name != ''", output_fields=["tool_name"], limit=2000)
        existing_tool_names = set(item['tool_name'] for item in existing_tools)
        
        new_tools_to_insert = [t for t in LMS_TOOLS if t.name not in existing_tool_names]
        
        if not new_tools_to_insert:
            logging.info(f"Không có tool LMS mới nào cần thêm vào collection.")
            return

        logging.info(f"Tìm thấy {len(new_tools_to_insert)} tool LMS mới, đang embed và upsert...")
        
        embedder = get_embeddings() # <-- Dùng hàm embedding "chuẩn" của bạn
        
        tool_names = [t.name for t in new_tools_to_insert]
        tool_descs = [t.description for t in new_tools_to_insert]
        
        # Nhúng mô tả
        embeddings = embedder.embed_documents(tool_descs)
        
        # Chuẩn bị dữ liệu để insert
        data = [
            tool_names,
            tool_descs,
            embeddings
        ]
        
        col.insert(data)
        col.flush()
        logging.info(f"✅ Đã upsert {len(new_tools_to_insert)} tool LMS vào Milvus.")
        
    except Exception as e:
        logging.exception(f"Lỗi nghiêm trọng khi khởi tạo LMS Tool Retriever: {e}")
        # Bạn có thể `raise e` ở đây nếu muốn app dừng lại nếu không khởi tạo được tool

async def route_lms_tool(sq: str, state: dict, topic: str) -> Tuple[str, str]:
    """
    (Hàm này được 'generate_per_subq' gọi)
    Tìm và chạy tool LMS tốt nhất cho sub-question.
    """
    logging.info(f"[LMSRouter] Đang định tuyến: sq='{sq}', topic='{topic}'")
    try:
        col = get_tool_collection()
        embedder = get_embeddings()
        
        # 1. Nhúng câu hỏi con (sub-question)
        query_embedding = embedder.embed_query(sq)
        
        # 2. Tìm kiếm (Vector Search) trong collection tool
        search_params = {"metric_type": METRIC_TYPE, "params": {"ef": 64}}
        results = col.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=1, # Luôn lấy 1 tool tốt nhất
            output_fields=["tool_name"]
        )
        
        if not results or not results[0]:
            logging.warning(f"[LMSRouter] Không tìm thấy tool nào cho: {sq}")
            return sq, "Xin lỗi, mình không tìm thấy công cụ phù hợp cho yêu cầu LMS của bạn."
            
        best_tool_name = results[0][0].entity.get("tool_name")
        best_tool_score = results[0][0].distance
        
        if best_tool_name not in LMS_TOOL_MAP:
            logging.error(f"[LMSRouter] Tool '{best_tool_name}' có trong Milvus nhưng không có trong Code!")
            return sq, "Lỗi: Tool hệ thống không đồng bộ."
            
        logging.info(f"[LMSRouter] Tìm thấy tool: {best_tool_name} (Score: {best_tool_score:.4f})")
        
        # 3. Lấy tool từ "Sổ đăng ký"
        best_tool = LMS_TOOL_MAP[best_tool_name]
        
        # 4. Chuẩn bị tham số và Chạy tool
        # (Đây là phần giả lập, bạn cần logic thật để lấy `student_id`)
        tool_input = {
            "course_id": state.get("course_id", ""),
            "student_id": state.get("user_id", "user_123") # (Giả lập)
        }
        
        # Lọc các tham số mà tool cần
        valid_args = {k: v for k, v in tool_input.items() if k in best_tool.args}
        
        # Chạy tool (tool của Langchain là sync, nên chạy trong thread)
        loop = asyncio.get_event_loop()
        result_str = await loop.run_in_executor(None, best_tool.run, valid_args)
        
        return sq, str(result_str)
        
    except Exception as e:
        logging.exception(f"[LMSRouter] Lỗi khi định tuyến tool: {e}")
        return sq, "Lỗi: Đã xảy ra sự cố khi mình tìm thông tin LMS."