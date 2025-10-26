# ============================================
# 🧠 Embedding Service (Expert Optimized)
# ============================================

import os
import asyncio
import threading
import warnings
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# ⚙️ CONFIG
# =========================
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", 32))  # batch tối ưu cho GPU 6–8GB

# ✅ Cache (singleton, an toàn cho cả sync/async)
_embeddings = None
_embeddings_lock_async = asyncio.Lock()
_embeddings_lock_sync = threading.Lock()

# ✅ Tắt warning tokenizer
warnings.filterwarnings("ignore", message="Token indices sequence length")


# ============================================
# 🧩 Custom Embedding Class
# ============================================
class CustomEmbeddings(HuggingFaceEmbeddings):
    """
    Expert wrapper:
    - Tự động thêm prefix E5: "passage:" / "query:"
    - Normalize thủ công (cosine)
    - Giữ batch lớn, tối ưu GPU
    """

    def embed_documents(self, texts):
        if not texts:
            return []
        texts = [f"passage: {t.strip()}" for t in texts]
        vecs = super().embed_documents(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        den = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (den + 1e-12)
        return vecs.tolist()

    def embed_query(self, text):
        text = f"query: {text.strip()}"
        vec = super().embed_query(text)
        vec = np.asarray(vec, dtype=np.float32)
        den = np.linalg.norm(vec, axis=-1, keepdims=True)
        vec = vec / (den + 1e-12)
        return vec.tolist()


# ============================================
# 🚀 Loader (Option B: Prefetch an toàn)
# ============================================
def _prefetch_to_cache_only_cpu():
    """Prefetch checkpoint về HF cache, KHÔNG move CUDA để tránh lỗi meta tensor."""
    try:
        _ = SentenceTransformer(EMBED_MODEL, device="cpu")
        del _
    except Exception as e:
        print(f"[Embedding] Prefetch warning (ignored): {e}")


def get_embedding_model():
    """
    Load multilingual embedding model optimized for COSINE search.
    Phương án B:
      - Prefetch bằng SentenceTransformer (CPU) để tải cache
      - Dùng HuggingFaceEmbeddings cho encode (ưu tiên GPU)
      - Fallback CPU nếu GPU move lỗi
    """
    print(f"🟩 Loading embedding model: {EMBED_MODEL}")

    # Prefetch chỉ CPU (tránh meta tensor khi move module sớm lên CUDA)
    _prefetch_to_cache_only_cpu()

    model_kwargs = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": False,  # E5 không cần True
    }
    encode_kwargs = {
        "batch_size": EMBED_BATCH,
        "normalize_embeddings": False,  # normalize thủ công trong CustomEmbeddings
        "truncate": True,
        "max_length": 512,
    }

    try:
        embedder = CustomEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    except NotImplementedError:
        # Fallback an toàn về CPU nếu CUDA move lỗi
        embedder = CustomEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu", "trust_remote_code": False},
            encode_kwargs=encode_kwargs,
        )

    print("✅ Embedding model initialized and normalized for COSINE search.")
    return embedder


# ============================================
# ⚡ Async Helpers
# ============================================
async def async_embed_docs(embeddings, texts, batch_size=128):
    """
    Async helper để batch-encode trong thread pool (không block event loop).
    """
    loop = asyncio.get_running_loop()

    def batch_iter(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    all_vectors = []
    for batch in batch_iter(texts, batch_size):
        vectors = await loop.run_in_executor(None, embeddings.embed_documents, batch)
        all_vectors.extend(vectors)
    return all_vectors


# ============================================
# 🧠 Cached Singleton (sync/async)
# ============================================
# === SAU (ĐÚNG) ===
def get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    with _embeddings_lock_sync:
        if _embeddings is None:
            _embeddings = get_embedding_model()
            print(f"✅ [Cache] Embedding model ready: {EMBED_MODEL}")
    return _embeddings

async def aget_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    async with _embeddings_lock_async:
        if _embeddings is None:
            _embeddings = get_embedding_model()
            print(f"✅ [Cache] Embedding model ready: {EMBED_MODEL}")
    return _embeddings

# ============================================
# 🧪 Warmup tiện dụng (gọi ở startup)
# ============================================
def warmup_embeddings() -> None:
    """
    Warmup đồng bộ để đảm bảo lần gọi đầu tiên sau khi server chạy là nhanh & ổn định.
    """
    try:
        _ = get_embeddings()
    except Exception as e:
        # Không làm chết server — chỉ log cảnh báo
        print(f"⚠️ [Embedding Warmup] Skipped/partial: {e}")
