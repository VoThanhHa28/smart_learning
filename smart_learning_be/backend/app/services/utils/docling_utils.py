import re
from docling_core.types import DoclingDocument

def wrap_ocr_text_as_docling(text: str) -> DoclingDocument:
    """
    Chuyển văn bản OCR thành DoclingDocument giữ đúng số trang.
    Hỗ trợ format như "--- Page 1 ---".
    """
    doc = DoclingDocument()
    # tách theo định dạng OCR trả về
    pages = re.split(r"---\s*Page\s+(\d+)\s*---", text)

    # pages = ["", "1", "page1_text", "2", "page2_text", ...]
    for i in range(1, len(pages), 2):
        try:
            page_num = int(pages[i])
        except ValueError:
            page_num = -1
        page_text = pages[i + 1].strip() if i + 1 < len(pages) else ""
        if not page_text:
            continue
        doc.add_text(page_text, page_num=page_num)

    # Nếu OCR không có định dạng trang → fallback
    if not doc.blocks:
        doc.add_text(text, page_num=-1)

    return doc
