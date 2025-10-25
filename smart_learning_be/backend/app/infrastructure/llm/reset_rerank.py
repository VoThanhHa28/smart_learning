# app/services/reranker.py  (chỉ phần helper + reset)
import os, logging, shutil
from pathlib import Path

import logging
from app.services.reranker import reset_reranker, log_reranker
RERANK_MODEL = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

def _hf_home() -> Path:
    return Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface")

def _hf_hub_dir() -> Path:
    return _hf_home() / "hub"

def _hf_modules_dir() -> Path:
    return _hf_home() / "modules" / "transformers_modules"

def _purge_jina_modules():
    mod_dir = _hf_modules_dir()
    # xoá cả module Jina đã materialize
    if mod_dir.exists():
        for p in mod_dir.glob("jinaai/jina_hyphen_reranker_hyphen_v2*"):
            shutil.rmtree(p, ignore_errors=True)

def reset_reranker(clear_cache: bool = False) -> None:
    """Reset singleton + (tuỳ chọn) xoá cache model/tokenizer & modules của Jina."""
    print("🔄 Resetting reranker...")
    global _cross_encoder
    if clear_cache:
        safe = RERANK_MODEL.replace("/", "--")
        hub_model_dir = _hf_hub_dir() / f"models--{safe}"
        shutil.rmtree(hub_model_dir, ignore_errors=True)
        _purge_jina_modules()
        logging.info(f"[Reranker] Cleared cache: {hub_model_dir} & transformers_modules/Jina")
    _cross_encoder = None
    logging.info("[Reranker] Reset done (will reload on next use).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("Resetting reranker with cache clear...")
    reset_reranker(clear_cache=True)
    print("Reloading & logging reranker info...")
    log_reranker()