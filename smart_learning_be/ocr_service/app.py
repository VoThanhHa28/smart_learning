# app.py
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn
from paddleocr import PaddleOCR
import fitz  # PyMuPDF

app = FastAPI()

# --- Init PaddleOCR ---
paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang="vi"   # đổi "vi" nếu cần tiếng Việt
)

def ocr_page_image(page, dpi=200) -> str:
    """OCR một trang PDF scan bằng PaddleOCR"""
    pix = page.get_pixmap(dpi=dpi)         # render page thành ảnh
    img_bytes = pix.tobytes("png")

    import PIL.Image
    img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)

    result = paddle_ocr.ocr(img)
    texts = []
    if result and result != [None]:
        for block in result:
            if block:
                for line in block:
                    if line and len(line) > 1:
                        texts.append(line[1][0])
    return "\n".join(texts)

def parse_pdf(pdf_bytes: bytes) -> str:
    """Xử lý PDF: text layer → lấy text; nếu scan → OCR fallback"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []

    for i, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        if raw.strip():  
            # Có text layer → dùng pymupdf4llm để giữ layout
            md = pymupdf4llm.to_markdown(doc, pages=[i-1])
            all_text.append(f"--- Page {i} ---\n{md}")
        else:
            # Không có text → OCR
            print(f"[DEBUG] OCR page {i}")
            page_text = ocr_page_image(page)
            all_text.append(f"--- Page {i} ---\n{page_text}")

    return "\n\n".join(all_text)

# --- API endpoint ---
@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Chỉ hỗ trợ PDF"}

    contents = await file.read()
    text = parse_pdf(contents)
    return {"text": text}


@app.post("/ocr_page")
async def run_ocr_page(file: UploadFile = File(...)):
    contents = await file.read()
    import PIL.Image, io, numpy as np
    img = PIL.Image.open(io.BytesIO(contents)).convert("RGB")
    img = np.array(img)

    result = paddle_ocr.ocr(img)
    texts = []
    if result and result != [None]:
        for block in result:
            if block:
                for line in block:
                    if line and len(line) > 1:
                        texts.append(line[1][0])
    return {"text": "\n".join(texts)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
