from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.services.ingestion.ingestion import ingest_file, ingest_many
import uuid, time, os

router = APIRouter(prefix="/index", tags=["index"])

def slugify(s: str) -> str:
    import re, unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s


@router.post("/upload")
async def upload(file: UploadFile = File(...), subject: str = Form(...), course_id: str | None = Form(None)):
    filename = (file.filename or "uploaded.pdf")
    if not filename.lower().endswith((".pdf", ".docx", ".pptx", ".html")):
        raise HTTPException(400, "Chỉ hỗ trợ PDF/DOCX/PPTX/HTML")

    os.makedirs("tmp", exist_ok=True)
    tmp_path = os.path.join("tmp", f"{uuid.uuid4()}_{filename}")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    if not course_id:
        course_id = f"{slugify(subject)}_{uuid.uuid4().hex[:8]}"

    t0 = time.time()
    result = await ingest_file(tmp_path, course_id, subject)

    try:
        os.remove(tmp_path)
    except Exception as e:
        print(f"⚠️ Cleanup skipped: {e}")

    print(f"✅ [INDEX] Uploaded {filename}, subject={subject}, course={course_id}, chunks={result['chunks_indexed']}")

    return {
        "doc_id": filename,
        "course_id": course_id,
        "subject": subject,
        "chunks": result["chunks_indexed"],
        "elapsed_ms": int((time.time() - t0) * 1000),
        "enrichment_file": result["enrichment_file"]
    }


@router.post("/upload_batch")
async def upload_batch(files: list[UploadFile] = File(...), subject: str = Form(...), course_id: str | None = Form(None)):
    if not course_id:
        course_id = f"{slugify(subject)}_{uuid.uuid4().hex[:8]}"

    os.makedirs("tmp", exist_ok=True)
    tmp_paths = []
    for f in files:
        filename = f.filename or "uploaded"
        tmp_path = os.path.join("tmp", f"{uuid.uuid4()}_{filename}")
        with open(tmp_path, "wb") as out:
            out.write(await f.read())
        tmp_paths.append(tmp_path)

    t0 = time.time()
    results = await ingest_many(tmp_paths, course_id, subject)

    try:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)
    except Exception as e:
        print(f"⚠️ Cleanup skipped: {e}")

    total_chunks = sum(r["chunks_indexed"] for r in results)
    print(f"✅ [INDEX-BATCH] Uploaded {len(files)} files, subject={subject}, total_chunks={total_chunks}")

    return {
        "course_id": course_id,
        "subject": subject,
        "files": len(files),
        "results": results,
        "total_chunks": total_chunks,
        "elapsed_ms": int((time.time() - t0) * 1000)
    }
