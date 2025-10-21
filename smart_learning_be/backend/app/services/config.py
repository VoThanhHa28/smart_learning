import os
RERANKER_MODE = os.getenv("RERANKER_MODE", "off")  # "off" | "light"
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "5"))

# Reranker nhẹ (nếu bật)
LIGHT_RERANKER_MODEL = os.getenv("LIGHT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LIGHT_RERANKER_TOPK = int(os.getenv("LIGHT_RERANKER_TOPK", "8"))
USE_BM25 = os.getenv("USE_BM25", "false").lower() == "true"
USE_DOCLING = os.getenv("USE_DOCLING", "true").lower() == "true"
EMBED_MODEL = os.getenv("EMBED_MODEL")
USE_HYDE = os.getenv("USE_HYDE", "true").lower() == "true"
USE_BILINGUAL = os.getenv("USE_BILINGUAL", "false").lower() == "true"

# Multi-query / concurrency
# ---- RAG fan-out knobs ----
CONCURRENCY = int(os.getenv("RAG_SUBQ_CONCURRENCY", "2"))

# RRF giữa các sub-query (trong 1 nhóm) và giữa các nhóm
RRF_INTERQUERY_K = int(os.getenv("RAG_RRF_INTERQUERY_K", "64"))
RRF_INTERQUERY_TOPN = int(os.getenv("RAG_RRF_INTERQUERY_TOPN", "50"))

# Quota tối thiểu để đảm bảo phủ vế/ý
CMP_MIN_BOTH = int(os.getenv("RAG_QUOTA_MIN_BOTH", "3"))
CMP_MIN_PER_ENTITY = int(os.getenv("RAG_QUOTA_MIN_PER_ENTITY", "2"))
SQ_MIN_PER_GROUP = int(os.getenv("RAG_QUOTA_MIN_PER_SUBQ", "2"))

# Giới hạn số sub-queries để tránh “nổ quạt”
MAX_SUBQS_BOTH = int(os.getenv("RAG_MAX_SUBQS_BOTH", "6"))
MAX_SUBQS_ENTITY = int(os.getenv("RAG_MAX_SUBQS_ENTITY", "3"))
MAX_SUBQS_SQ = int(os.getenv("RAG_MAX_SUBQS_SQ", "4"))