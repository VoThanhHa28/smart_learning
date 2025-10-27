from pathlib import Path
import os, json

def _index_path():
    this = Path(__file__).resolve()
    backend = this.parents[3]
    data_dir = Path(os.getenv("DATA_DIR", backend / "data"))
    uploads = Path(os.getenv("UPLOADS_DIR", data_dir / "uploads"))
    uploads.mkdir(parents=True, exist_ok=True)
    return uploads / "toc_index.json"

def load_toc_index() -> dict:
    p = _index_path()
    if not p.exists(): return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def get_toc_by_course(course_id: str):
    idx = load_toc_index()
    cid = (course_id or "").lower()
    for k, v in idx.items():
        if str(v.get("course_id","")).lower() == cid:
            return v
    return None

def get_toc_by_key(doc_key: str):
    idx = load_toc_index()
    return idx.get(doc_key)

def flatten_toc(entry: dict) -> list[str]:
    if not entry: return []
    struct = entry.get("toc_struct") or []
    if struct:
        out = []
        for it in struct:
            num = (it.get("num") or "").strip()
            title = (it.get("title") or "").strip()
            line = f"{num} {title}".strip()
            if line: out.append(line)
        return out
    return entry.get("toc") or []
