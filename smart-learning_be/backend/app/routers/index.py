import os, uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.docling_pipeline import convert_pdf_to_markdown
from app.services.vectorstore import upsert_chunks
from app.services.vectorstore import get_client, collection_name

router = APIRouter(prefix="/index", tags=["index"])

def simple_markdown_chunk(md: str, max_chars=2000, overlap=300):
    chunks = []
    i = 0
    while i < len(md):
        j = min(i + max_chars, len(md))
        chunk = md[i:j]
        chunks.append(chunk)
        if j >= len(md): break
        i = max(0, j - overlap)
    return [c for c in chunks if c.strip()]

@router.post("/upload")
async def upload_doc(
    docId: str,
    file: UploadFile = File(...),
    ocr: str = Query("auto", enum=["off", "auto", "force"])
):
    try:
        tmp_dir = "./tmp"; os.makedirs(tmp_dir, exist_ok=True)
        local_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_{file.filename}")
        with open(local_path, "wb") as f:
            f.write(await file.read())

        # 1) Chuẩn hoá PDF → Markdown bằng Docling
        md = convert_pdf_to_markdown(local_path, ocr_mode=ocr, ocr_langs=["vi","en"], do_tables=True)
        if not md.strip():
            raise HTTPException(status_code=400, detail="Doc rỗng hoặc không đọc được.")

        # 2) Chunk markdown (đơn giản, ổn định). Có thể nâng cấp sang Docling HybridChunker sau.
        chunks = simple_markdown_chunk(md, max_chars=2000, overlap=300)

        # 3) Lưu Vector DB (Chroma)
        metas = [{"docId": docId, "chunk_id": i} for i in range(len(chunks))]
        upsert_chunks(docId, chunks, metas)

        try: os.remove(local_path)
        except: pass

        return {"msg": f"Indexed {len(chunks)} chunks for {docId}", "ocr": ocr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload/Index failed: {e}")



@router.get("/list")
def list_chunks(docId: str, limit: int = 5):
    client = get_client()
    coll = client.get_or_create_collection(collection_name(docId))
    res = coll.get(limit=limit)  # lấy vài item đầu
    return {
        "count": len(res["ids"]),
        "documents": res["documents"],
        "metadatas": res["metadatas"],
        "ids": res["ids"]
    }

