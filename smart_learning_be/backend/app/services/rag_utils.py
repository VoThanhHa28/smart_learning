# app/services/rag_utils.py
import os
import json
import logging
from langchain_core.documents import Document
from typing import List
from tiktoken import get_encoding

def build_context(docs: List[Document], limit_chars: int = 5000) -> str:
    if not docs:
        return "Không có ngữ cảnh nào được tìm thấy."

    def safe_sort_key(d):
        meta = d.metadata or {}
        rr = float(meta.get("rerank_score", 0.0))
        sr = float(meta.get("semantic_score", 0.0))  # nếu sau này bạn dùng fallback light
        lvl = meta.get("heading_level")
        pg = meta.get("page")
        try:
            lvl_num = int(lvl) if lvl is not None else 99
        except (TypeError, ValueError):
            lvl_num = 99
        pg_num = pg if isinstance(pg, int) else 9999
        return (-rr, -sr, lvl_num, pg_num)

    sorted_docs = sorted(docs, key=safe_sort_key)
    # Optional: debug nếu thiếu heading_level
    for d in docs:
        if d.metadata.get("heading_level") is None:
            logging.debug(f"⚠️ Missing heading_level in doc: {d.metadata}")

    parts = []
    total_chars = 0

    for i, d in enumerate(sorted_docs, start=1):
        meta = d.metadata or {}
        pg = meta.get("page", "-")
        heading = meta.get("heading") or meta.get("headings", ["-"])
        heading_str = heading[0] if isinstance(heading, list) else heading

        section_pos = meta.get("section_position", "")
        content_type = meta.get("content_type", "text")
        tags = ", ".join(meta.get("semantic_tags", []))
        diff = meta.get("difficulty", "")
        src = meta.get("doc_source", "")

        header_info = (
            f"[Doc {i} | p:{pg} | {heading_str} | {content_type} | "
            f"{section_pos} | diff:{diff} | src:{src} | tags:{tags}]"
        )

        text = d.page_content.strip()
        block = f"{header_info}\n{text}\n"

        parts.append(block)
        total_chars += len(block)
        if total_chars >= limit_chars:
            break

    context = "\n\n".join(parts)
    
    # Kiểm tra số token (nếu dùng tiktoken)
    enc = get_encoding("cl100k_base")
    n_tokens = len(enc.encode(context))
    print(f"🧮 Context tokens: {n_tokens}")

    return context[:limit_chars]


def expand_context_by_heading(top_doc, all_docs, n_neighbors=1):
    """Lấy thêm context cùng heading hoặc ±1 trang"""
    base_heading = top_doc.metadata.get("headings")
    base_page = top_doc.metadata.get("page", -1)

    expanded = []
    for d in all_docs:
        heading = d.metadata.get("headings")
        page = d.metadata.get("page", -1)
        if heading == base_heading or abs(page - base_page) <= n_neighbors:
            expanded.append(d)

    expanded = [d for d in expanded if d.metadata != top_doc.metadata]
    return expanded[: n_neighbors * 2 + 1]


def fuse_chunks_by_heading(docs: List[Document]) -> List[Document]:
    """
    Gom các chunks có cùng heading lại thành 1 Document duy nhất.
    - Hữu ích khi các đoạn cùng heading bị chia nhỏ sau chunking.
    - Giữ metadata của đoạn đầu tiên.
    """
    if not docs:
        return []

    fused = []
    current_heading = None
    current_texts = []
    current_meta = None

    for d in docs:
        heading = d.metadata.get("heading") or str(d.metadata.get("headings", [""])) or ""
        if heading != current_heading and current_texts:
            fused.append(
                Document(page_content="\n".join(current_texts), metadata=current_meta)
            )
            current_texts = []

        current_heading = heading
        # ✅ Giữ metadata cũ, chỉ cập nhật nếu có thêm key mới
        if current_meta:
            current_meta = {**current_meta, **{k: v for k, v in d.metadata.items() if v not in [None, ""]}}
        else:
            current_meta = d.metadata

        current_texts.append(d.page_content.strip())

    if current_texts:
        fused.append(Document(page_content="\n".join(current_texts), metadata=current_meta))

    return fused
