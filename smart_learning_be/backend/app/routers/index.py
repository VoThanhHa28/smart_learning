from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from ..services.docling_pipeline import fast_parse_pdf
from ..services.vectorstore import upsert_chunks
import uuid, time, os

router = APIRouter(prefix="/index", tags=["index"])

def slugify(s: str) -> str:
    import re, unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s

@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    subject: str = Form(...),          # <-- thêm dòng này
    course_id: str | None = Form(None) # <-- tuỳ chọn
):
    filename = (file.filename or "uploaded.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Chỉ hỗ trợ PDF")

    os.makedirs("tmp", exist_ok=True)
    tmp_path = os.path.join("tmp", f"{uuid.uuid4()}_{filename}")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

     # Nếu không có course_id → tự sinh
    if not course_id:
        course_id = f"{subject}_{uuid.uuid4().hex[:8]}"  # ví dụ: programming_a1b2c3d4

    t0 = time.time()
    docs, meta = fast_parse_pdf(tmp_path, subject=subject, course_id=course_id)


    # 👉 log kiểm tra số docs
    print(f"DEBUG: fast_parse_pdf trả về {len(docs)} docs, meta={meta}")

    # Gắn metadata chung
    s_subject = slugify(subject)
    for i, d in enumerate(docs):
        d.metadata["subject"] = s_subject
        d.metadata["course_id"] = course_id
        d.metadata["source"] = filename
        d.metadata["chunk_id"] = d.metadata.get("chunk_id", f"{filename}__{i}")

    added = upsert_chunks(docs)

    # 👉 log kiểm tra số docs đã add
    # print(f"DEBUG: upsert_chunks thêm {added} docs vào Db")

    # 👉 chỉ xóa file nếu chắc chắn đã add thành công
    if added > 0:
        os.remove(tmp_path)
    else:
        print(f"⚠️ Không thêm được doc nào, giữ lại file tạm: {tmp_path}")

    return {
        "doc_id": os.path.basename(filename),
        "course_id": course_id,
        "pages": meta["pages"],
        "chunks": added,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "warnings": [] if added > 0 else ["Không thêm được doc nào"],
    }