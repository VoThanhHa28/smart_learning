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

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", 0.5))
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
    text = text[:3000]
    text = re.sub(r"\s+", " ", text).strip()
    return text

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

    pairs = [[query, d.metadata.get("clean_text") or d.page_content] for d in docs]

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
