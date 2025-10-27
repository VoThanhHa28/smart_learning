# -*- coding: utf-8 -*-
import re, os, json, time, hashlib
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple

# =========================
# ⚙️ CONFIG
# =========================
MAX_SCAN_PAGES = 30          # quét tối đa trang đầu
GAP_TOLERANCE = 4            # số dòng "lệch pattern" cho phép trước khi dừng
DENSITY_WINDOW = 20          # cửa sổ kiểm tra mật độ
DENSITY_MIN_HITS = 6         # tối thiểu số dòng khớp trong cửa sổ
MAX_LEVELS = 6               # tối đa 6 cấp mục
ALLOW_EQUAL_PAGE = True      # cho phép số trang không giảm (subheading cùng trang)
TOC_KEYWORDS = ("mục lục", "contents", "table of contents")
TITLE_PREFIXES = r"(Chương|Chapter|Phần|Section|Bài)"

# pattern chính (số mục + tiêu đề + dot leaders + số trang)
# hỗ trợ: 
#  - 1 / 1.2 / 1.2.3...
#  - hoặc Chương/Chapter/Phần/Bài + (I/II/1/2/…)
#  - dot leaders tuỳ chọn
MAIN_PATTERN = re.compile(
    rf"""^\s*(?:
            (?P<num>\d+(?:\.\d+){{0,{MAX_LEVELS-1}}})  # 1 hoặc 1.2.3...
            |
            (?P<titleprefix>{TITLE_PREFIXES})\s+(?P<titleidx>[IVXLCDM]+|\d+)  # Chương I / Chapter 2
        )\s+
        (?P<title>.+?)                                       # tiêu đề
        \s*(?:\.{{2,}}|\s)\s*                                # dot leaders hoặc khoảng trắng
        (?P<page>\d{{1,4}})\s*$                              # số trang
    """,
    re.IGNORECASE | re.VERBOSE,
)

# pattern phụ: dòng chỉ có số trang (để ghép với dòng tiêu đề phía trước)
TRAILING_PAGE_PATTERN = re.compile(r"^\s*(\d{1,4})\s*$")

# pattern đánh dấu “khu vực TOC” (tăng precision)
TOC_MARK_PATTERN = re.compile("|".join([re.escape(k) for k in TOC_KEYWORDS]), re.IGNORECASE)

ROMAN_MAP = {
    'M':1000,'CM':900,'D':500,'CD':400,'C':100,'XC':90,'L':50,'XL':40,'X':10,'IX':9,'V':5,'IV':4,'I':1
}

def roman_to_int(s: str) -> Optional[int]:
    s = s.upper().strip()
    i = 0
    res = 0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] in ROMAN_MAP:
            res += ROMAN_MAP[s[i:i+2]]; i += 2
        elif s[i] in ROMAN_MAP:
            res += ROMAN_MAP[s[i]]; i += 1
        else:
            return None
    return res if res > 0 else None

def normalize_numbering(num: Optional[str],
                        titleprefix: Optional[str],
                        titleidx: Optional[str]) -> Tuple[str, int]:
    """
    Chuẩn hoá ký hiệu mục & cấp độ:
    - Nếu có num: "1.2.3" → level = 3
    - Nếu dạng Chương/Chapter/Phần/Bài + idx: level = 1, num = "<idx>"
      (idx có thể là La Mã → chuyển sang Ả Rập; giữ nguyên hiển thị gốc trong title)
    """
    if num:
        parts = num.split(".")
        level = min(len(parts), MAX_LEVELS)
        return ".".join(parts[:level]), level

    if titleprefix and titleidx:
        # ưu tiên chuyển sang int nếu có thể (Roman/Arabic)
        idx_num = roman_to_int(titleidx) if re.match(r"^[IVXLCDM]+$", titleidx, re.I) else None
        if idx_num is None:
            # dạng Ả Rập
            try:
                idx_num = int(titleidx)
            except ValueError:
                idx_num = None
        canonical = str(idx_num) if idx_num is not None else titleidx
        return canonical, 1

    return "", 0

def find_toc_region(lines: List[str]) -> Tuple[int, int]:
    """
    Xác định vùng scan khả nghi là TOC:
    - Ưu tiên trang có "Mục lục/Contents" → quét cửa sổ ± N dòng.
    - Nếu không thấy, quét từ đầu đến MAX_SCAN_PAGES.
    """
    idxs = [i for i, ln in enumerate(lines) if TOC_MARK_PATTERN.search(ln)]
    if idxs:
        start = max(0, idxs[0] - 30)
        end = min(len(lines), idxs[0] + 300)  # đủ rộng cho TOC dài
        return start, end
    # fallback: đầu tài liệu
    return 0, min(len(lines), 1200)  # ~30 trang đầu (giả định ~40 dòng/trang)

def read_text_lines(file_path: str, ocr_text: Optional[str], max_pages: int) -> Tuple[List[str], int]:
    if ocr_text:
        text = ocr_text
        total_pages = None
        try:
            with fitz.open(file_path) as doc:
                total_pages = doc.page_count
        except Exception:
            total_pages = 10_000  # fallback an toàn
    else:
        with fitz.open(file_path) as doc:
            total_pages = doc.page_count
            pages = [doc[i].get_text("text") for i in range(min(doc.page_count, max_pages))]
        text = "\n".join(pages)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines, (total_pages or 10_000)

def density_ok(hits_mask: List[bool]) -> bool:
    if not hits_mask:
        return False
    w = DENSITY_WINDOW
    need = DENSITY_MIN_HITS
    for i in range(0, len(hits_mask), max(1, w//2)):
        window = hits_mask[i:i+w]
        if window and sum(window) >= need:
            return True
    return False

def validate_sequence(items: List[Dict[str, Any]], pages_total: int) -> List[Dict[str, Any]]:
    """Loại bỏ mục bất thường theo quy tắc đơn giản & nhanh."""
    out = []
    prev_page = 0
    prev_levels = [0]  # stack giả cho cấu trúc
    for it in items:
        p = it["page"]
        lv = it["level"]
        if not (1 <= p <= pages_total):
            continue
        if not ALLOW_EQUAL_PAGE and p < prev_page:
            continue
        # chấp nhận = nếu là subheading; nếu giảm thì chỉ cho giảm về cấp thấp hơn phù hợp
        if p + (0 if ALLOW_EQUAL_PAGE else 1) < prev_page:
            # Giảm trang quá mức → loại
            continue
        # mức cấp hợp lý
        if lv < 1 or lv > MAX_LEVELS:
            continue
        prev_page = max(prev_page, p)
        out.append(it)
    return out

def try_outline(file_path: str) -> List[Dict[str, Any]]:
    try:
        with fitz.open(file_path) as doc:
            toc = doc.get_toc(simple=True) or []
            total = doc.page_count
    except Exception:
        return []
    out = []
    for lvl, title, page in toc:
        if not isinstance(page, int) or page < 1 or page > total:
            continue
        out.append({
            "num": "",  # outline không luôn có numbering
            "title": re.sub(r"\s+", " ", title).strip(),
            "page": page,
            "level": int(lvl) if lvl and lvl > 0 else 1,
            "source": "outline",
            "confidence": 0.99,
        })
    return out

def try_regex(lines: List[str], pages_total: int) -> List[Dict[str, Any]]:
    start, end = find_toc_region(lines)
    region = lines[start:end]
    items: List[Dict[str, Any]] = []
    hits_mask: List[bool] = []

    skip_count = 0
    prev_idx = -1

    i = 0
    while i < len(region):
        ln = region[i]
        m = MAIN_PATTERN.match(ln)
        if m:
            num_raw = (m.group("num") or "").strip()
            titleprefix = m.group("titleprefix")
            titleidx = m.group("titleidx")
            title = re.sub(r"\s+", " ", m.group("title")).strip(" .·•-")
            page = int(m.group("page"))

            num_norm, level = normalize_numbering(num_raw, titleprefix, titleidx)
            if level == 0:
                # dự phòng: đoán level theo số lượng dấu chấm
                level = num_raw.count(".") + 1 if num_raw else 1

            items.append({
                "num": num_norm,
                "title": title,
                "page": page,
                "level": min(level, MAX_LEVELS),
                "source": "regex",
                "confidence": 0.85 + min(0.1, 0.02 * num_norm.count(".")),
            })
            hits_mask.append(True)
            skip_count = 0
            prev_idx = i
            i += 1
            continue

        # Thử case dòng bọc: dòng hiện tại là tiêu đề, dòng sau chỉ là số trang
        if (i + 1) < len(region):
            m2_page = TRAILING_PAGE_PATTERN.match(region[i+1])
            m2_title = re.match(
                rf"""^\s*(?:
                        (?P<num>\d+(?:\.\d+){{0,{MAX_LEVELS-1}}})
                        |
                        (?P<titleprefix>{TITLE_PREFIXES})\s+(?P<titleidx>[IVXLCDM]+|\d+)
                    )\s+(?P<title>.+?)\s*$""",
                ln, re.IGNORECASE | re.VERBOSE,
            )
            if m2_page and m2_title:
                page = int(m2_page.group(1))
                num_raw = (m2_title.group("num") or "").strip()
                titleprefix = m2_title.group("titleprefix")
                titleidx = m2_title.group("titleidx")
                title = re.sub(r"\s+", " ", m2_title.group("title")).strip(" .·•-")
                num_norm, level = normalize_numbering(num_raw, titleprefix, titleidx)
                if level == 0:
                    level = num_raw.count(".") + 1 if num_raw else 1
                items.append({
                    "num": num_norm,
                    "title": title,
                    "page": page,
                    "level": min(level, MAX_LEVELS),
                    "source": "regex-wrap",
                    "confidence": 0.80,
                })
                hits_mask.append(True)
                skip_count = 0
                prev_idx = i+1
                i += 2
                continue

        # không khớp
        hits_mask.append(False)
        skip_count += 1
        if skip_count >= GAP_TOLERANCE and len(items) > 5:
            # dừng sớm nếu đã có vài mục & liên tiếp lệch nhiều dòng
            break
        i += 1

    # Kiểm tra mật độ
    if not density_ok(hits_mask):
        return []

    # Loại bất thường & đảm bảo số trang hợp lệ
    items = validate_sequence(items, pages_total)
    return items

def extract_toc_hybrid(file_path: str,
                       ocr_text: Optional[str] = None,
                       max_scan_pages: int = MAX_SCAN_PAGES) -> List[Dict[str, Any]]:
    """
    Trả về list[dict]:
      { num, title, page, level, source, confidence }
    Chiến lược: outline → regex (text) → (nếu ocr_text có) regex(ocr)
    """
    # 1) Outline-first
    outline_items = try_outline(file_path)
    if outline_items:
        return outline_items

    # 2) Regex trên text thật
    lines, pages_total = read_text_lines(file_path, None, max_scan_pages)
    items = try_regex(lines, pages_total)
    if items:
        return items

    # 3) Nếu có OCR text, thử lại
    if ocr_text:
        lines_ocr, _ = read_text_lines(file_path, ocr_text, max_scan_pages)
        items = try_regex(lines_ocr, pages_total)
        if items:
            return items

    return []

# =========================
# 🔐 Lưu chỉ mục TOC (an toàn)
# =========================
def update_toc_index_safe(file_path: str, course_id: str, subject: str, toc_items: List[Dict[str, Any]]) -> str:
    """
    - Không xoá tài liệu khác cùng course_id
    - Key: <slug_course>__<slug_filename>__<sha1(path+mtime)>
    - Lưu: pages_total, mtime, source, confidence_avg
    """
    from app.services.utils.common_utils import slugify_filename

    this_file = Path(__file__).resolve()
    backend_dir = this_file.parents[3]
    data_dir = Path(os.getenv("DATA_DIR", backend_dir / "data"))
    uploads_dir = Path(os.getenv("UPLOADS_DIR", data_dir / "uploads"))
    uploads_dir.mkdir(parents=True, exist_ok=True)

    index_path = uploads_dir / "toc_index.json"

    try:
        toc_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    except json.JSONDecodeError:
        toc_index = {}

    stat = os.stat(file_path)
    mtime = int(stat.st_mtime)
    base = os.path.basename(file_path)
    slug_base = slugify_filename(file_path, course_id)  # vẫn giữ để tương thích
    sha = hashlib.sha1(f"{file_path}|{mtime}".encode("utf-8")).hexdigest()[:12]
    key = f"{slug_base}__{sha}"

    confidence_avg = round(sum(it.get("confidence", 0.8) for it in toc_items) / max(1, len(toc_items)), 3) if toc_items else 0.0

    entry = {
        "course_id": course_id,
        "subject": subject,
        "file": base,
        "path": file_path,
        "mtime": mtime,
        "key": key,
        "toc": toc_items,
        "confidence_avg": confidence_avg,
        "pages_total": None,
        "source": "unknown",
        "created_at": int(time.time()),
    }

    # cố lấy tổng trang
    try:
        with fitz.open(file_path) as doc:
            entry["pages_total"] = doc.page_count
    except Exception:
        entry["pages_total"] = None

    if toc_items:
        entry["source"] = toc_items[0].get("source", "unknown")

    toc_index[key] = entry
    index_path.write_text(json.dumps(toc_index, ensure_ascii=False, indent=2), encoding="utf-8")
    return key
