# services/utils/ocr_client.py
import fitz  # PyMuPDF
import httpx, asyncio

OCR_URL = "http://ocr:9000/ocr_page"  # giả sử bạn có endpoint OCR từng trang ảnh

async def ocr_page_async(img_bytes: bytes) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(OCR_URL, files={"file": ("page.png", img_bytes, "image/png")})
        res.raise_for_status()
        return res.json().get("text", "")

async def parse_pdf_async(file_path: str, dpi: int = 200) -> str:
    """Song song OCR PDF scan bằng asyncio"""
    doc = fitz.open(file_path)
    tasks = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi) #type: ignore
        img_bytes = pix.tobytes("png")
        tasks.append(ocr_page_async(img_bytes))

    texts = await asyncio.gather(*tasks, return_exceptions=False)

    # Ghép lại văn bản theo thứ tự trang
    result = []
    for i, txt in enumerate(texts, start=1):
        result.append(f"--- Page {i} ---\n{txt}")
    return "\n\n".join(result)
