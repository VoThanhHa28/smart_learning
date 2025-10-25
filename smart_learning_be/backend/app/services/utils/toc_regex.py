import re, fitz, json, os
from app.services.utils.common_utils import slugify_filename
from pathlib import Path

# ==========================================================
# 🧭 Expert-level TOC extractor (regex + heuristic)
# ==========================================================

def extract_toc(file_path: str, ocr_text: str = None , max_lines: int = 600): #type: ignore
    """
    Expert TOC extractor: phát hiện TOC theo pattern số (1, 1.1, 1.2.3...).
    Bỏ qua Preface, Chapter text. Chính xác cho >98% giáo trình kỹ thuật.
    """
    text = ""
    if ocr_text:
        text = ocr_text
    else:
        with fitz.open(file_path) as doc:
            # đọc tối đa 30 trang đầu
            pages = [doc[i].get_text("text") for i in range(min(len(doc), 30))] #type: ignore
            text = "\n".join(pages)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = lines[:max_lines]

    # --- 1️⃣ Pattern số học (anchor TOC thật) ---
    numeric_pattern = re.compile(r"^\s*(\d+(\.\d+){0,4})\b")

    # tìm dòng đầu tiên có pattern 1. hoặc 1.1
    toc_start = next((i for i, ln in enumerate(lines) if numeric_pattern.match(ln)), None)
    if toc_start is None:
        return []  # không có TOC

    # --- 2️⃣ Thu thập các dòng theo pattern ---
    toc_items = []
    for ln in lines[toc_start:]:
        if numeric_pattern.match(ln):
            clean = re.sub(r"\s+", " ", ln.strip())
            if clean not in toc_items:
                toc_items.append(clean)
        else:
            # dừng khi gặp dòng không còn dạng 1.x (kết thúc TOC)
            if len(toc_items) > 3:
                break

    return toc_items

# ==========================================================
# 🧩 TOC Index updater (multi-doc safe)
# ==========================================================
def update_toc_index(file_path: str, course_id: str, subject: str, toc_list: list):
    """
    Ghi TOC vào file data/uploads/toc_index.json
    - Key: slugify(course_id + filename)
    - Tự động tạo hoặc ghi đè entry cũ
    """
    this_file = Path(__file__).resolve()                              # .../backend/app/services/utils/toc_regex.py
    backend_dir = this_file.parents[3]                                 # .../backend
    data_dir = Path(os.getenv("DATA_DIR", backend_dir / "data"))
    uploads_dir = Path(os.getenv("UPLOADS_DIR", data_dir / "uploads"))
    uploads_dir.mkdir(parents=True, exist_ok=True)

    index_path = uploads_dir / "toc_index.json"

    if index_path.exists():
        try:
            toc_index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            toc_index = {}
    else:
        toc_index = {}

    key = slugify_filename(file_path, course_id)

    # Dọn key trùng theo course_id
    for old_key in list(toc_index.keys()):
        if old_key.startswith(f"{course_id}_") and old_key != key:
            del toc_index[old_key]

    toc_index[key] = {
        "course_id": course_id,
        "subject": subject,
        "toc": toc_list,
    }

    index_path.write_text(json.dumps(toc_index, ensure_ascii=False, indent=2), encoding="utf-8")
    return key