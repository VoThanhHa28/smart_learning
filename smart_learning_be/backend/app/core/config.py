# backend/app/core/config.py
import os

try:
    # nếu dùng python-dotenv, tự động nạp .env ở runtime
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def _to_bool(v: str | None, default=False) -> bool:
    if v is None: return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _to_int(v: str | None, default=0) -> int:
    try:
        return int(str(v).strip()) if v is not None else default
    except Exception:
        return default

def _to_float(v: str | None, default=0.0) -> float:
    try:
        return float(str(v).strip()) if v is not None else default
    except Exception:
        return default

class _CFG:
    # -------- GCP / Firebase ----------
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/app/firebase_key.json")
    FIREBASE_STORAGE_BUCKET        = os.getenv("FIREBASE_STORAGE_BUCKET", "")

    # -------- LangChain --------------
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "")

    # -------- Chunking ----------------
    CHUNK_SIZE    = _to_int(os.getenv("CHUNK_SIZE"), 1000)
    CHUNK_OVERLAP = _to_int(os.getenv("CHUNK_OVERLAP"), 120)

    # -------- LLM ---------------------
    LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "gemini")      # gemini | ollama | ...
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    LOCAL_MODEL    = os.getenv("LOCAL_MODEL", "")

    # -------- Embedding / Rerank ------
    USE_BM25      = _to_bool(os.getenv("USE_BM25"), True)
    USE_DOCLING   = _to_bool(os.getenv("USE_DOCLING"), True)
    USE_APPENDIX  = _to_bool(os.getenv("USE_APPENDIX"), False)
    USE_HYDE      = _to_bool(os.getenv("USE_HYDE"), False)
    USE_BILINGUAL = _to_bool(os.getenv("USE_BILINGUAL"), False)

    EMBED_MODEL   = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
    EMBED_BATCH   = _to_int(os.getenv("EMBED_BATCH"), 64)

    RERANKER_MODE   = os.getenv("RERANKER_MODE", "base")      # off | light | base
    RERANK_MODEL    = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    RERANK_THRESHOLD= _to_float(os.getenv("RERANK_THRESHOLD"), 0.5)

    FINAL_TOP_N   = _to_int(os.getenv("FINAL_TOP_N"), 8)
    MAX_INTENTS   = _to_int(os.getenv("MAX_INTENTS"), 2)

    # -------- HF / Transformers -------
    HF_TOKEN                  = os.getenv("HF_TOKEN", "")
    HF_HUB_ENABLE_HF_TRANSFER = _to_bool(os.getenv("HF_HUB_ENABLE_HF_TRANSFER"), True)
    TRANSFORMERS_SAFE_WEIGHTS = _to_bool(os.getenv("TRANSFORMERS_SAFE_WEIGHTS"), True)
    HF_HUB_CACHE              = os.getenv("HF_HUB_CACHE", "./.cache/huggingface")

    # -------- System / CUDA -----------
    PYTORCH_CUDA_ALLOC_CONF   = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

    # -------- Feature Flags -----------
    ENABLE_SEMANTIC_TAGGER = _to_bool(os.getenv("ENABLE_SEMANTIC_TAGGER"), True)
    OCR_DEVICE             = os.getenv("OCR_DEVICE", "gpu")
    NO_GRAPH_CACHE         = _to_bool(os.getenv("NO_GRAPH_CACHE"), True)
    DEBUG_TAGGER           = _to_bool(os.getenv("DEBUG_TAGGER"), False)

    # -------- Milvus ------------------
    MILVUS_HOST       = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT       = _to_int(os.getenv("MILVUS_PORT"), 19530)
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "smart_learning")

    # -------- Rerank Preview (debug) ---
    RERANK_PREVIEW        = _to_bool(os.getenv("RERANK_PREVIEW"), False)
    RERANK_PREVIEW_WINDOW = _to_int(os.getenv("RERANK_PREVIEW_WINDOW"), 550)
    RERANK_PREVIEW_HEAD   = _to_int(os.getenv("RERANK_PREVIEW_HEAD"), 1200)
    RERANK_PREVIEW_TAIL   = _to_int(os.getenv("RERANK_PREVIEW_TAIL"), 800)
    RERANK_PREVIEW_MID    = _to_int(os.getenv("RERANK_PREVIEW_MID"), 800)

    # -------- RAG fan-out -------------
    CONCURRENCY         = _to_int(os.getenv("RAG_SUBQ_CONCURRENCY"), 2)
    RRF_INTERQUERY_K    = _to_int(os.getenv("RAG_RRF_INTERQUERY_K"), 64)
    RRF_INTERQUERY_TOPN = _to_int(os.getenv("RAG_RRF_INTERQUERY_TOPN"), 50)
    SQ_MIN_PER_GROUP    = _to_int(os.getenv("RAG_QUOTA_MIN_PER_SUBQ"), 2)
    MAX_SUBQS_SQ        = _to_int(os.getenv("RAG_MAX_SUBQS_SQ"), 4)

    # -------- Misc --------------------
    PYTHONPATH = os.getenv("PYTHONPATH", "backend")

CFG = _CFG()
