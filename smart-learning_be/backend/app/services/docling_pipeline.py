import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from .ocr_service import ocr_image_bytes
from .text_clean import clean_text, remove_headers_footers

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

def _extract_page_text(page: "fitz.Page") -> str:
    # 'blocks' giữ cấu trúc tốt hơn 'text'; nếu cần tự sắp xếp tự nhiên thì dùng rawdict.
    try:
        blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,"text",block_no,...)
        lines = [b[4] for b in blocks if isinstance(b, (list, tuple)) and isinstance(b[4], str)]
        text = "\n".join(lines).strip()
    except Exception:
        text = page.get_text("text")
    return text

def pdf_to_pages_text(file_path: str) -> Tuple[List[Tuple[int, str, bool]], int]:
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        txt = _extract_page_text(page)
        is_ocr = False
        if not txt.strip():
            pix = page.get_pixmap(dpi=200)
            txt = ocr_image_bytes(pix.tobytes("png"))
            if txt.strip():
                txt = f"[OCR]\n{txt}"
                is_ocr = True
        pages.append((i+1, txt, is_ocr))  # i+1 để trang bắt đầu từ 1
    return pages, len(doc)


def split_to_docs(full_text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(full_text)
    return [Document(page_content=c, metadata={}) for c in chunks]

def parse_pdf(file_path: str, subject: str, course_id: str | None = None):
    pages, n_pages = pdf_to_pages_text(file_path)

    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    for page_num, text, is_ocr in pages:
        clean = clean_text(text)
        # split từng trang
        for i, chunk in enumerate(splitter.split_text(clean)):
            meta = {
                "source": os.path.basename(file_path),
                "chunk_id": f"{os.path.basename(file_path)}__p{page_num}_{i}",
                "page": page_num,
                "ocr": is_ocr,
                "subject": subject,
            }
            if course_id:
                meta["course_id"] = course_id
            doc = Document(page_content=chunk, metadata=meta)
            print("INDEX DEBUG:", doc.metadata)   # 👈 log metadata
            docs.append(doc)
    return docs, {"pages": n_pages, "chunks": len(docs)}

