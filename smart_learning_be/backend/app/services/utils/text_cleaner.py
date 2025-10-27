# -*- coding: utf-8 -*-
"""
Text cleaning utilities cho DoclingPipeline.
Mục tiêu: văn bản sạch, chuẩn Unicode, giảm noise trước khi embed.
"""

import unicodedata
import regex as re
from typing import List

P_END = r"[\.!\?:…»”)]"  # dấu kết câu phổ biến (VN/EN)


def normalize_unicode_nfc(s: str) -> str:
    """Chuẩn hóa Unicode theo NFC (tiếng Việt chuẩn nhất)."""
    return unicodedata.normalize("NFC", s)


def remove_zero_width_and_ctrl(s: str) -> str:
    """Xóa ký tự zero-width và control (ẩn)."""
    s = re.sub(r"[\u200B-\u200F\uFEFF]", "", s)   # zero-width
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)         # control chars
    return s


def dehyphenate_eol(s: str) -> str:
    """Ghép từ bị tách bởi gạch nối ở cuối dòng: 'học-\n sinh' -> 'học sinh'."""
    return re.sub(r"(\p{L})-\n(\p{L})", r"\1\2", s)


def merge_soft_linebreaks(s: str) -> str:
    """Nối dòng nếu dòng trước không kết thúc bằng dấu câu."""
    return re.sub(rf"(?<!{P_END})\n(?!\n)", " ", s)


def normalize_whitespace(s: str) -> str:
    """Chuẩn hóa khoảng trắng và xuống dòng."""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_text(text: str) -> str:
    """Pipeline cleaning chính cho text."""
    if not text:
        return ""

    # 1. Unicode NFC
    text = normalize_unicode_nfc(text)

    # 2. Bỏ ký tự vô hình / control
    text = remove_zero_width_and_ctrl(text)

    # 3. Bỏ footnote, DOI, ISBN, URL
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"(doi:|https?://\S+|ISBN\s*\d+[-\d]*)", "", text)

    # 4. Xử lý gạch nối cuối dòng
    text = dehyphenate_eol(text)
    text = re.sub(r"-\n", "", text)

    # 5. Ghép dòng gãy
    text = merge_soft_linebreaks(text)

    # 6. Chuẩn hóa whitespace
    text = normalize_whitespace(text)

    return text.strip()


def remove_headers_footers(pages: List[str], min_ratio: float = 0.6) -> List[str]:
    """
    Loại bỏ header/footer lặp lại trên nhiều trang.
    min_ratio = tỷ lệ xuất hiện tối thiểu để coi là header/footer.
    """
    from collections import Counter

    lines = [p.split("\n")[0] for p in pages if p.strip()]
    counter = Counter(lines)
    headers = {h for h, c in counter.items() if c / len(pages) > min_ratio}

    return [
        "\n".join(l for l in page.split("\n") if l not in headers)
        for page in pages
    ]
