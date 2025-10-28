# services/ingestion/ingestion.py
# (File này gọi DoclingPipeline)
import logging # Thêm logging
from .docling_pipeline import DoclingPipeline # Đảm bảo import đúng

# Khởi tạo pipeline (chỉ chạy 1 lần khi import)
try:
    _pipeline = DoclingPipeline(
        lang="vi",
        enable_ocr=False,
        enable_code=False,
        enable_formula=False,
        enable_table=False,
        fast_mode=True  # Bật chế độ nhanh (ít enrichment hơn)
    )
    logging.info("DoclingPipeline initialized in ingestion service.")
except Exception as e:
    logging.error(f"Failed to initialize DoclingPipeline: {e}", exc_info=True)
    # Có thể raise lỗi ở đây nếu pipeline là bắt buộc
    _pipeline = None # Đặt là None để kiểm tra sau

async def ingest_file(file_path: str, course_id: str, subject: str, user_id: str):
    """
    Hàm này nhận user_id từ API và truyền nó vào pipeline.
    Pipeline chịu trách nhiệm gán user_id vào metadata của mỗi chunk.
    """
    if _pipeline is None:
        raise RuntimeError("DoclingPipeline is not available.")
        
    logging.info(f"Starting ingestion for single file: {file_path}, user: {user_id}, course: {course_id}")
    try:
        result = await _pipeline.process_file(
            file_path, 
            course_id=course_id, 
            subject=subject, 
            user_id=user_id # Truyền user_id vào
        )
        logging.info(f"Ingestion successful for single file: {file_path}")
        # Đảm bảo trả về dict hợp lệ ngay cả khi result không đầy đủ
        return {
            "course_id": course_id,
            "subject": subject,
            "chunks_indexed": result.get("chunks_indexed", 0),
            "enrichment_file": result.get("enrichment_file", "")
        }
    except Exception as e:
        logging.error(f"Error during ingest_file for {file_path}, user {user_id}: {e}", exc_info=True)
        # Ném lại lỗi để API xử lý và trả về 500
        raise e 

async def ingest_many(files: list[str], course_id: str, subject: str, user_id: str):
    """
    Hàm này nhận user_id từ API và truyền nó vào pipeline (cho batch).
    """
    if _pipeline is None:
        raise RuntimeError("DoclingPipeline is not available.")
        
    logging.info(f"Starting ingestion for batch of {len(files)} files, user: {user_id}, course: {course_id}")
    try:
        results = await _pipeline.process_many(
            files, 
            course_id=course_id, 
            subject=subject,
            user_id=user_id # Truyền user_id vào
        )
        logging.info(f"Ingestion successful for batch, course: {course_id}")
        return results # Trả về list các kết quả
    except Exception as e:
        logging.error(f"Error during ingest_many for course {course_id}, user {user_id}: {e}", exc_info=True)
        # Ném lại lỗi để API xử lý
        raise e