from docling.document_converter import DocumentConverter
from .text_clean import clean_text, remove_headers_footers

def pdf_to_pages_text(file_path: str):
    """
    Parse PDF bằng Docling, tự động OCR nếu cần.
    Trả về: [(page_num, text, is_ocr), ...], tổng số trang
    """
    converter = DocumentConverter()
    dl_doc = converter.convert(file_path)   # ❌ bỏ fmt=DocumentFormat.PDF

    pages = []
    raw_texts = []

    for i, page in enumerate(dl_doc.pages):
        txt = page.to_text().strip()
        is_ocr = getattr(page, "has_ocr_layer", False)  # fallback an toàn
        raw_texts.append(txt)
        pages.append((i + 1, txt, is_ocr))

    # 🔥 Bỏ header/footer
    cleaned_pages = remove_headers_footers(raw_texts)

    # Clean regex
    final_pages = []
    for i, txt in enumerate(cleaned_pages):
        clean = clean_text(txt)
        final_pages.append((i + 1, clean, pages[i][2]))

    return final_pages, len(dl_doc.pages)
