# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Tuple

def process_text_chunks(text: str, page: int) -> Tuple[List[Dict], List[str]]:
    """
    Postprocess text trước khi chunk:
    - Loại bỏ URL
    - Tách TOC (Table of Contents)
    - Gom code block / formula block riêng biệt
    - Chuẩn hóa schema: mỗi chunk chỉ thuộc 1 loại (text / code / formula)
    Trả về:
        chunks: list[dict]
        toc_lines: list[str]
    """
    lines = text.split("\n")

    toc_lines: List[str] = []
    chunks: List[Dict] = []

    buffer: List[str] = []
    mode = "text"  # text | code | formula

    def flush_buffer(buf: List[str], mode: str):
        """Đẩy buffer ra chunks theo mode"""
        if not buf:
            return
        chunk_text = "\n".join(buf).strip()
        if not chunk_text:
            return

        flags = {
            "is_text": mode == "text",
            "is_code": mode == "code",
            "is_formula": mode == "formula",
        }
        chunks.append({
            "page": page,
            "text": chunk_text,
            **flags
        })

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            # gặp dòng trống → flush buffer
            flush_buffer(buffer, mode)
            buffer, mode = [], "text"
            continue

        # --- 1. Detect TOC ---
        if (
            re.search(r'\.{3,}', line_strip)
            or re.match(r'^\d+(\.\d+)*\s', line_strip)
            or re.match(r'^(Chương|Chapter|Phần|Appendix)\s+', line_strip, re.I)
        ):
            if len(line_strip) < 120:
                toc_lines.append(line_strip)
                continue

        # --- 2. Remove URLs ---
        if re.search(r'http[s]?://\S+', line_strip):
            continue

        # --- 3. Detect code line ---
        symbols = re.findall(r'[;{}=<>+\-*/()\[\]]', line_strip)
        is_code_line = (
            (re.match(r'^\s{2,}', line) and len(symbols) >= 2)
            or re.search(r'(def |class |#include|public |private )', line_strip)
        )

        # --- 4. Detect formula line ---
        is_formula_line = (
            re.search(r'\\(frac|sum|int|alpha|beta|sqrt)', line_strip)
            or re.search(r'\$\$.*\$\$', line_strip)
            or re.search(r'\\\[.*\\\]', line_strip)
            or re.search(r'[∑√∫≤≥≈]', line_strip)
        )

        # --- 5. Push theo mode ---
        if is_code_line:
            if mode != "code":
                flush_buffer(buffer, mode)
                buffer, mode = [], "code"
            buffer.append(line_strip)
        elif is_formula_line:
            if mode != "formula":
                flush_buffer(buffer, mode)
                buffer, mode = [], "formula"
            buffer.append(line_strip)
        else:
            if mode != "text":
                flush_buffer(buffer, mode)
                buffer, mode = [], "text"
            buffer.append(line_strip)

    # flush cuối
    flush_buffer(buffer, mode)

    return chunks, toc_lines
