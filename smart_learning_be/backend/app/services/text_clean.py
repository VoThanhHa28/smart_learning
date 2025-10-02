# -*- coding: utf-8 -*-
import unicodedata, regex as re
from typing import List

P_END = r"[\.!\?:…»”)]"  # dấu kết câu phổ biến (VN/EN)

def normalize_unicode_nfc(s: str) -> str:
    # NFC là chuẩn khuyến nghị cho văn bản nói chung (giữ dấu tiếng Việt đúng). 
    # (Unicode FAQ / UAX#15 khuyên dùng NFC cho general text)
    return unicodedata.normalize("NFC", s)

def remove_zero_width_and_ctrl(s: str) -> str:
    s = re.sub(r"[\u200B-\u200F\uFEFF]", "", s)               # zero-width
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)                     # control chars
    return s

def dehyphenate_eol(s: str) -> str:
    # ghép các từ bị tách bởi gạch nối ở cuối dòng: "học-\n sinh" -> "học sinh"
    s = re.sub(r"(\p{L})-\n(\p{L})", r"\1\2", s)
    return s

def merge_soft_linebreaks(s: str) -> str:
    # nối dòng nếu dòng trước không kết thúc bằng dấu câu
    s = re.sub(rf"(?<!{P_END})\n(?!\n)", " ", s)
    return s

def normalize_whitespace(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def clean_text(text: str) -> str:
    if not text: return ""
    # chuẩn hóa unicode
    text = unicodedata.normalize("NFC", text)
    # bỏ ký tự vô hình / zero-width
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    # bỏ footnote [123], doi, isbn, url
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"(doi:|https?://\S+|ISBN\s*\d+[-\d]*)", "", text)
    # bỏ gạch nối cuối dòng
    text = re.sub(r"-\n", "", text)
    # ghép dòng gãy
    text = re.sub(r"\n+", " ", text)
    return text.strip()

def remove_headers_footers(pages: list[str], min_ratio: float = 0.6) -> list[str]:
    from collections import Counter
    lines = [p.split("\n")[0] for p in pages if p.strip()]
    counter = Counter(lines)
    headers = {h for h, c in counter.items() if c/len(pages) > min_ratio}
    return [
        "\n".join(l for l in page.split("\n") if l not in headers)
        for page in pages
    ]