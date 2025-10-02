# -*- coding: utf-8 -*-
from typing import Tuple, List
from langchain_core.documents import Document
import pymupdf
import pymupdf4llm
from .ocr_client import call_ocr
from cleantext import clean
from .chunk import split_to_docs
from .postprocess import process_text_chunks
import fitz  # PyMuPDF



def extract_text(file_path: str) -> list[tuple[int, str]]:
    """
    Extract text từ PDF:
      - Nếu PDF có text layer → dùng PyMuPDF4LLM.
      - Nếu PDF scan (không có text) → render ảnh + OCR.
    Trả về [(page_number, text), ...]
    """
    results = []
    doc = fitz.open(file_path)

    # 1. Check text layer trước
    for i, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        if raw.strip():
            md = pymupdf4llm.to_markdown(file_path, pages=[i-1])
            results.append((i, md))

    if results:  # có text layer
        return results

    # 2. Nếu không có text layer → OCR từng trang
    for i, page in enumerate(doc, start=1):
        print(f"[DEBUG] Rendering page {i} ...")
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        print(f"[DEBUG] OCR page {i} ...")
        text = call_ocr(img_bytes)
        print(f"[DEBUG] Done OCR page {i}, length={len(text)}")
        results.append((i, text))

    return results



def clean_extracted_text(text: str) -> str:
    """
    B2. Clean text bằng clean-text library (free, mạnh).
    """
    return clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_line_breaks=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
    )


def fast_parse_pdf(
    file_path: str, subject: str, course_id: str | None = None
) -> Tuple[List[Document], dict]:
    """
    Parse PDF nhanh gọn (text-based hoặc scan).
    Trả về docs (list Document) + stats (dict).
    """

    # --- 1. Extract PDF text ---
    page_texts = extract_text(file_path)   # [(page, text), ...]

    processed_chunks = []
    toc_meta = []

    for page, raw_text in page_texts:
        # --- 2. Clean text ---
        cleaned_text = clean_extracted_text(raw_text)

        # --- 3. Postprocess: detect toc, code, formula, remove url ---
        chunks, toc = process_text_chunks(cleaned_text, page=page)
        processed_chunks.extend(chunks)
        toc_meta.extend(toc)

    # --- 4. Chunking + attach metadata ---
    docs = split_to_docs(processed_chunks, subject, course_id)

    # Gắn TOC vào metadata (nếu có)
    # for d in docs:
    #     if toc_meta:
    #         d.metadata["toc"] = "\n".join(toc_meta)

    # --- 5. Stats ---
    stats = {
        "pages": len(page_texts),
        "chunks": len(docs),
    }
    return docs, stats
