from fastapi import APIRouter, UploadFile, File
from app.services.ocr_client import call_ocr
import tempfile

router = APIRouter()

@router.post("/test-ocr")
async def test_ocr(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    text = call_ocr(tmp_path)
    return {"ocr_text": text}
