# api/index.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import uuid
import time
import os
import logging 

from ...services.ingestion.ingestion import ingest_file, ingest_many
from ...rag.retrieval.vectorstore import get_courses_by_user 


# --- Bỏ hoàn toàn Firebase Auth ---

router = APIRouter(prefix="/index", tags=["index"])

def slugify(s: str) -> str:
    """Chuẩn hóa chuỗi thành dạng slug (an toàn cho ID)."""
    import re, unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s if s else "untitled" # Tránh trả về chuỗi rỗng

# ==================================================
# === API LẤY KHÓA HỌC (Nhận user_id từ FE) ===
# ==================================================
@router.get("/my_courses", summary="Lấy danh sách khóa học theo user_id")
async def get_my_courses_simple(user_id: str): # Nhận user_id từ query param (vd: ?user_id=user_123)
    """
    Endpoint này lấy user_id TỪ FE (query param)
    và gọi Milvus để truy vấn danh sách khóa học.
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="Thiếu tham số user_id")
        
    logging.info(f"API: Fetching courses for user_id received from FE: {user_id}")
    courses_list = await get_courses_by_user(user_id)
    
    return {
        "user_id": user_id,
        "courses": courses_list
    }

# ==================================================
# === API UPLOAD (Nhận user_id từ FE) ===
# ==================================================
@router.post("/upload", summary="Tải lên file, nhận user_id từ FE")
async def upload_simple(
    file: UploadFile = File(..., description="File tài liệu"), 
    subject: str = Form(..., description="Tên môn học"), 
    
    # === NHẬN user_id và course_id TỪ FE ===
    user_id: str = Form(..., description="ID của người dùng"), 
    course_id: str | None = Form(None, description="ID khóa học (bỏ trống để tạo mới)")
    # ======================================
):
    filename = (file.filename or "uploaded_file")
    allowed_extensions = (".pdf", ".docx", ".pptx", ".html")
    if not filename.lower().endswith(allowed_extensions):
        logging.warning(f"Upload rejected: Invalid file type '{filename}' for user {user_id}")
        raise HTTPException(
            status_code=400, 
            detail=f"Chỉ hỗ trợ: {', '.join(ext.upper() for ext in allowed_extensions)}"
            )

    if not user_id: # Kiểm tra cơ bản
        raise HTTPException(status_code=400, detail="Thiếu thông tin user_id")

    # Lưu file tạm
    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"{uuid.uuid4()}_{filename}"
    temp_file_path = os.path.join(temp_dir, temp_filename)
    try:
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logging.info(f"API: File '{filename}' saved temporarily to '{temp_file_path}' for user {user_id}")
    except Exception as e:
        logging.error(f"API: Failed to save uploaded file '{filename}' for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Không thể lưu file tải lên: {e}")

    # Tạo course_id nếu FE không gửi
    final_course_id = course_id if course_id else f"{slugify(subject)}_{uuid.uuid4().hex[:8]}"

    start_time = time.time()
    
    # Gọi service ingestion, truyền user_id và course_id từ FE
    try:
        result = await ingest_file(
            file_path=temp_file_path, 
            course_id=final_course_id, 
            subject=subject,
            user_id=user_id # user_id từ FE
        )
    except Exception as e:
        logging.error(f"API: Ingestion failed for file '{filename}', user {user_id}, course {final_course_id}: {e}", exc_info=True)
        try: # Dọn dẹp file tạm khi lỗi
            if os.path.exists(temp_file_path): os.remove(temp_file_path)
        except Exception: pass
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý tài liệu: {e}")

    # Dọn dẹp file tạm
    try:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
    except Exception as e:
        logging.warning(f"API: Cleanup skipped for temp file '{temp_file_path}': {e}")

    elapsed_ms = int((time.time() - start_time) * 1000)
    chunks_indexed = result.get('chunks_indexed', 0)
    logging.info(f"API: ✅ Upload success User '{user_id}', File '{filename}', Course '{final_course_id}', Chunks={chunks_indexed}, Time={elapsed_ms}ms")

    return {
        "doc_id": filename,
        "course_id": final_course_id,
        "subject": subject,
        "user_id": user_id, 
        "chunks": chunks_indexed,
        "elapsed_ms": elapsed_ms,
        "enrichment_file": result.get("enrichment_file", "") 
    }

# ==================================================
# === API UPLOAD BATCH (Nhận user_id từ FE) ===
# ==================================================
@router.post("/upload_batch", summary="Tải lên nhiều file, nhận user_id từ FE")
async def upload_batch_simple(
    files: list[UploadFile] = File(..., description="Danh sách file"), 
    subject: str = Form(..., description="Tên môn học"), 
    # === NHẬN user_id và course_id TỪ FE ===
    user_id: str = Form(..., description="ID của người dùng"),
    course_id: str | None = Form(None, description="ID khóa học (bỏ trống để tạo mới)")
    # ======================================
):
    if not user_id:
        raise HTTPException(status_code=400, detail="Thiếu thông tin user_id")

    final_course_id = course_id if course_id else f"{slugify(subject)}_{uuid.uuid4().hex[:8]}"

    # Lưu file tạm
    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_paths = []
    processed_files_info = [] 
    for file in files:
        filename = (file.filename or "uploaded_file")
        allowed_extensions = (".pdf", ".docx", ".pptx", ".html")
        if not filename.lower().endswith(allowed_extensions):
            logging.warning(f"API Batch: Skipping invalid file type '{filename}' for user {user_id}")
            continue 
        
        temp_filename = f"{uuid.uuid4()}_{filename}"
        temp_file_path = os.path.join(temp_dir, temp_filename)
        try:
            with open(temp_file_path, "wb") as out:
                content = await file.read()
                out.write(content)
            temp_file_paths.append(temp_file_path)
            processed_files_info.append({"original_name": filename, "temp_path": temp_file_path})
        except Exception as e:
            logging.error(f"API Batch: Failed to save file '{filename}' for user {user_id}: {e}", exc_info=True)
            # Dọn dẹp các file đã lưu nếu lỗi
            for info in processed_files_info:
                try:
                    if os.path.exists(info["temp_path"]): os.remove(info["temp_path"])
                except Exception: pass
            raise HTTPException(status_code=500, detail=f"Không thể lưu file '{filename}': {e}")

    if not temp_file_paths:
         raise HTTPException(status_code=400, detail="Không có file hợp lệ nào trong batch.")

    start_time = time.time()
    
    # Gọi service ingestion
    try:
        results = await ingest_many(
            files=temp_file_paths, 
            course_id=final_course_id, 
            subject=subject,
            user_id=user_id # user_id từ FE
        )
    except Exception as e:
        logging.error(f"API Batch: Ingestion failed, user {user_id}, course {final_course_id}: {e}", exc_info=True)
        # Dọn dẹp file tạm
        for path in temp_file_paths:
             try:
                 if os.path.exists(path): os.remove(path)
             except Exception: pass
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý batch tài liệu: {e}")

    # Dọn dẹp file tạm
    for path in temp_file_paths:
        try:
            if os.path.exists(path): os.remove(path)
        except Exception as e:
            logging.warning(f"API Batch: Cleanup skipped for temp file '{path}': {e}")

    # Tính toán kết quả
    total_chunks = 0
    detailed_results = [] 
    if isinstance(results, list):
        total_chunks = sum(r.get("chunks_indexed", 0) for r in results)
        if len(results) == len(processed_files_info):
             for i, res in enumerate(results):
                 detailed_results.append({
                     "doc_id": processed_files_info[i]["original_name"],
                     "chunks_indexed": res.get("chunks_indexed", 0),
                     "enrichment_file": res.get("enrichment_file", ""),
                     "error": res.get("error") 
                 })
        else:
             detailed_results = results 

    elapsed_ms = int((time.time() - start_time) * 1000)
    logging.info(f"API: ✅ Batch Upload success User '{user_id}', Course '{final_course_id}', Files={len(processed_files_info)}, Chunks={total_chunks}, Time={elapsed_ms}ms")

    return {
        "course_id": final_course_id,
        "subject": subject,
        "user_id": user_id, 
        "files_processed": len(processed_files_info),
        "results_detail": detailed_results, 
        "total_chunks": total_chunks,
        "elapsed_ms": elapsed_ms
    }