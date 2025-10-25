import logging
import os
import asyncio
import time
import re
import shutil
from pathlib import Path
from typing import List
import torch
import torch.nn.functional as F
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from huggingface_hub import login

# ----------- Config & Constants ----------- #
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", 0.5))
RERANK_PREVIEW = os.getenv("RERANK_PREVIEW", "0").lower() in ("1", "true", "yes")
PREVIEW_WINDOW = int(os.getenv("RERANK_PREVIEW_WINDOW", "550"))
PREVIEW_HEAD   = int(os.getenv("RERANK_PREVIEW_HEAD", "1200"))
PREVIEW_TAIL   = int(os.getenv("RERANK_PREVIEW_TAIL", "800"))
PREVIEW_MID    = int(os.getenv("RERANK_PREVIEW_MID", "800"))

_device = "cuda" if torch.cuda.is_available() else "cpu"
_cross_encoder = None

# ----------- Helper functions ----------- #
# imports
import threading
from pathlib import Path
import shutil

# lock toàn cục
_model_lock = threading.Lock()

def _hf_repo_cache_dir(repo_id: str) -> Path:
    home = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    safe = repo_id.replace("/", "--")
    return Path(home) / "hub" / f"models--{safe}"

def reset_reranker(clear_cache: bool = False):
    global _cross_encoder
    if clear_cache:
        # hub cache
        hub = _hf_hub_dir()
        safe = RERANK_MODEL.replace("/", "--")
        hub_model_dir = hub / f"models--{safe}"
        shutil.rmtree(hub_model_dir, ignore_errors=True)
        logging.info(f"[Reranker] Cleared HF hub cache: {hub_model_dir}")
        # modules cache
        _purge_jina_modules()
        logging.info(f"[Reranker] Cleared HF transformers_modules cache for Jina")
    _cross_encoder = None
    logging.info("[Reranker] Reset done (will reload on next use).")

def log_reranker_brief(ce):
    m, t = ce.model, ce.tokenizer
    repo = getattr(m.config, "_name_or_path", RERANK_MODEL)
    tok_repo = getattr(t, "name_or_path", None)
    logging.info(f"[Reranker] repo={repo} | tok_repo={tok_repo} | tok.vocab={getattr(t, 'vocab_size', None)} | mdl.vocab={getattr(m.config, 'vocab_size', None)} | device={next(m.parameters()).device}")


def get_reranker():
    global _cross_encoder
    if _cross_encoder is None:
        def _load_model():
            ce = CrossEncoder(
                RERANK_MODEL,
                device=_device,
                max_length=256,
                trust_remote_code=True,
                revision=os.getenv("RERANK_REVISION", "main"),   # 👈 pin revision
                local_files_only=False,                           # 👈 buộc check remote
                # bạn có thể thêm: force_download=True nếu muốn ép tải
            )
            if _device == "cuda":
                try: ce.model.half()
                except Exception: pass
            return ce

        logging.info(f"🚀 Loading reranker model: {RERANK_MODEL} on {_device}")
        ce = _load_model()

        # sanity check: tokenizer vs model
        tok_vs = getattr(ce.tokenizer, "vocab_size", None)
        mdl_vs = getattr(ce.model.config, "vocab_size", None)
        if tok_vs and mdl_vs and tok_vs != mdl_vs:
            logging.warning(f"[Reranker] Vocab mismatch tok={tok_vs} vs mdl={mdl_vs} → purge & reload")
            reset_reranker(clear_cache=True)
            ce = _load_model()

        log_reranker_brief(ce)  # 👈 log ngắn gọn

        _cross_encoder = ce
    return _cross_encoder

def log_reranker():
    ce = get_reranker()
    m, t = ce.model, ce.tokenizer
    repo = getattr(m.config, "_name_or_path", RERANK_MODEL)
    rev  = getattr(m.config, "_commit_hash", None) or getattr(m.config, "revision", None)
    tok_vs = getattr(t, "vocab_size", None)
    mdl_vs = getattr(m.config, "vocab_size", None)
    device = next(m.parameters()).device
    dtype  = getattr(m, "dtype", None)
    logging.info(f"[Reranker] repo={repo} rev={rev} tok.vocab={tok_vs} mdl.vocab={mdl_vs} dtype={dtype} device={device}")

def clean_doc(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"```([\s\S]*?)```", lambda m: "\n" + m.group(1) + "\n", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^>>>.*$", " ", text, flags=re.M)
    text = re.sub(r"^In\s*\[\d*\]:.*$", " ", text, flags=re.M)
    text = re.sub(r"^Out\s*\[\d*\]:.*$", " ", text, flags=re.M)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = re.sub(r"^[\*\-\+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.M)
    text = re.sub(r"</?[A-Za-z][A-Za-z0-9\-]*(\s[^<>]*)?>", " ", text)
    text = re.sub(r"[■◆●►◼▪¤•★☆※☞✓✔️❌✗➤→⇒⬇⬆⬅➡🔹🔸🔻🔺💡🔥🚀⭐🧠✅❗]", " ", text)
    text = re.sub(r"\{[\s\S]*?\}", " ", text)
    text = re.sub(r"---[\s\S]*?---", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _extract_keyword_spans(text: str, query: str, max_terms: int = 6):
    """
    Tạo danh sách (start, end) cho các từ khóa trong query (lọc stop-ish: độ dài >=3).
    Dùng regex không phân biệt hoa thường; trả về list có thể rỗng.
    """
    if not text or not query:
        return []
    # chọn term đơn giản: tách theo chữ/số, bỏ quá ngắn
    terms = [t.lower() for t in re.findall(r"[A-Za-zÀ-ỹ0-9_]+", query) if len(t) >= 3]
    terms = list(dict.fromkeys(terms))[:max_terms]  # de-dup, giới hạn
    spans = []
    low = text.lower()
    for t in terms:
        for m in re.finditer(re.escape(t), low):
            spans.append((m.start(), m.end()))
    return spans

def _pick_best_center(spans: list[tuple[int,int]], text_len: int) -> int | None:
    """
    Chọn “tâm” cửa sổ:
    - nếu có nhiều spans, chọn span trung tâm theo median vị trí (ổn định hơn “đầu tiên”).
    - nếu rỗng → None.
    """
    if not spans:
        return None
    centers = [(s+e)//2 for s, e in spans]
    centers.sort()
    return centers[len(centers)//2]

def _window_by_center(text: str, center: int, radius: int) -> str:
    a = max(center - radius, 0)
    b = min(center + radius, len(text))
    return text[a:b].strip()

def _fallback_head_mid_tail(text: str) -> str:
    """
    Khi không xác định được vị trí hit, lấy head + mid + tail để tránh bias “đầu nặng”.
    """
    n = len(text)
    if n <= (PREVIEW_HEAD + PREVIEW_MID + PREVIEW_TAIL + 200):
        return text  # ngắn → trả full
    head = text[:PREVIEW_HEAD]
    mid_start = max((n // 2) - (PREVIEW_MID // 2), 0)
    mid = text[mid_start: mid_start + PREVIEW_MID]
    tail = text[-PREVIEW_TAIL:]
    return (head + "\n...\n" + mid + "\n...\n" + tail).strip()

def build_preview_for_rerank(clean_text: str, query: str, meta: dict) -> str:
    """
    Trả về đoạn preview ~ tương đương <= vài ngàn ký tự cho CrossEncoder.
    Ưu tiên dùng spans từ metadata nếu có (bm25/keywords), nếu không thì rút từ query.
    """
    if not RERANK_PREVIEW:
        # Giữ hành vi cũ: đã clean ở clean_doc; không slice ở đây (để backward-compat)
        return clean_text

    # 1) thử lấy spans từ metadata (tuỳ pipeline BM25 của bạn)
    spans = []
    for key in ("bm25_spans", "keyword_spans", "match_positions"):
        if key in (meta or {}):
            # mong đợi dạng [(s,e), ...] hoặc [{"start":s,"end":e}, ...]
            raw = meta[key] or []
            for it in raw:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    s, e = int(it[0]), int(it[1])
                    if 0 <= s < e <= len(clean_text):
                        spans.append((s, e))
                elif isinstance(it, dict) and "start" in it and "end" in it:
                    s, e = int(it["start"]), int(it["end"])
                    if 0 <= s < e <= len(clean_text):
                        spans.append((s, e))
    # 2) nếu không có, tự tạo spans từ query
    if not spans:
        spans = _extract_keyword_spans(clean_text, query)

    center = _pick_best_center(spans, len(clean_text))
    if center is not None:
        return _window_by_center(clean_text, center, PREVIEW_WINDOW)

    # 3) fallback: head + mid + tail
    return _fallback_head_mid_tail(clean_text)


# ----------- Rerank Function ----------- #
async def rerank(query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
    if not docs:
        return []

    t0 = time.time()
    try:
        model = get_reranker()
    except Exception as e:
        logging.exception("[Reranker] load failed, falling back to pass-through")
        # trả về docs top_n không rerank để không vỡ pipeline
        return docs[:top_n]

    for d in docs:
        d.metadata["clean_text"] = clean_doc(d.page_content)

    pairs = []
    for d in docs:
        ct = d.metadata.get("clean_text") or d.page_content
        # NEW: build preview thông minh trước khi đưa vào CrossEncoder
        preview = build_preview_for_rerank(ct, query, d.metadata or {})
        pairs.append([query, preview])

    def _predict():
        with torch.inference_mode():
            scores_tensor = model.predict(pairs, convert_to_tensor=True)
            scores = torch.sigmoid(scores_tensor).detach().cpu().tolist()
        return scores

    if _device == "cuda":
        scores = _predict()
    else:
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, _predict)

    for d, s in zip(docs, scores):
        d.metadata["rerank_score"] = float(s)

    reranked = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
    top_docs = reranked[:top_n]
    filtered = [d for d in top_docs if d.metadata["rerank_score"] >= RERANK_THRESHOLD]

    dur = time.time() - t0
    logging.info(
        "🏅 [Reranker] scored=%d selected=%d/%d thr=%.2f max=%.3f min=%.3f ⏱️ %.3fs",
        len(docs), len(filtered), len(top_docs), RERANK_THRESHOLD,
        max(scores or [0]), min(scores or [0]), dur
    )

    return filtered or top_docs[:2]
