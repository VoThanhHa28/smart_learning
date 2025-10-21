from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .sentence_splitter import split_sentences

# services/docling_split_to_docs.py
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from docling_core.types import DoclingDocument

def docling_split_to_docs(
    dl_doc: DoclingDocument,
    subject: str,
    course_id: str = None,
    chunk_size: int = 1000,
    overlap: int = 120
) -> List[Document]:
    """
    Chuyển DoclingDocument → list[Document] để đưa vào embedding
    - Text: NLP sentence split + recursive chunk
    - Code/Formula/Table: giữ nguyên block
    - Heading: giữ riêng để tăng recall
    """

    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    # --- Xử lý text blocks ---
    for i, block in enumerate(dl_doc.blocks):
        if not block.text or not block.text.strip():
            continue

        heading = getattr(block, "heading_text", None) or getattr(block, "role", None) or "unknown"

        if block.type == "paragraph":
            sentences = split_sentences(block.text)
            for j, sent in enumerate(sentences):
                for k, sub_chunk in enumerate(splitter.split_text(sent)):
                    meta = {
                        "page": getattr(block, "page_idx", None),
                        "heading": heading,
                        "subject": subject,
                        "course_id": course_id,
                        "chunk_id": f"text_{i}_{j}_{k}",
                        "type": "text"
                    }
                    docs.append(Document(page_content=sub_chunk, metadata=meta))

        elif block.type == "heading":
            meta = {
                "page": getattr(block, "page_idx", None),
                "heading": block.text,
                "subject": subject,
                "course_id": course_id,
                "chunk_id": f"heading_{i}",
                "type": "heading"
            }
            docs.append(Document(page_content=block.text, metadata=meta))

    # --- Code blocks ---
    for k, code in enumerate(getattr(dl_doc, "codes", [])):
        meta = {
            "page": getattr(code, "page_idx", None),
            "heading": getattr(code, "heading_text", None),
            "subject": subject,
            "course_id": course_id,
            "chunk_id": f"code_{k}",
            "type": "code"
        }
        docs.append(Document(page_content=code.text, metadata=meta))

    # --- Formulas ---
    for k, formula in enumerate(getattr(dl_doc, "formulas", [])):
        meta = {
            "page": getattr(formula, "page_idx", None),
            "heading": getattr(formula, "heading_text", None),
            "subject": subject,
            "course_id": course_id,
            "chunk_id": f"formula_{k}",
            "type": "formula"
        }
        docs.append(Document(
            page_content=getattr(formula, "latex", str(formula)),
            metadata=meta
        ))

    # --- Tables ---
    for k, table in enumerate(getattr(dl_doc, "tables", [])):
        meta = {
            "page": getattr(table, "page_idx", None),
            "heading": getattr(table, "heading_text", None),
            "subject": subject,
            "course_id": course_id,
            "chunk_id": f"table_{k}",
            "type": "table"
        }
        # Xuất table dưới dạng markdown để giữ cấu trúc
        docs.append(Document(page_content=table.to_markdown(), metadata=meta))

    return docs
