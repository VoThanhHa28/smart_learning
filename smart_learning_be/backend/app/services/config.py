import os
RERANKER_MODE = os.getenv("RERANKER_MODE", "off")  # "off" | "light"
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "5"))

# Reranker nhẹ (nếu bật)
LIGHT_RERANKER_MODEL = os.getenv("LIGHT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LIGHT_RERANKER_TOPK = int(os.getenv("LIGHT_RERANKER_TOPK", "8"))
USE_BM25 = os.getenv("USE_BM25", "false").lower() == "true"
USE_DOCLING = os.getenv("USE_DOCLING", "true").lower() == "true"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")