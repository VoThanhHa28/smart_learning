# ============================================
# 🧠 Embedding Service (Expert Optimized)
# ============================================

import os
import asyncio
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# ⚙️ CONFIG
# =========================
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", 32))  # batch tối ưu cho GPU 6–8GB
_embeddings = None

# ✅ Tắt warning tokenizer
warnings.filterwarnings("ignore", message="Token indices sequence length")


# ============================================
# 🧩 Custom Embedding Class
# ============================================
class CustomEmbeddings(HuggingFaceEmbeddings):
    """
    Expert-level embedding wrapper:
    - Tự động thêm prefix "passage:" / "query:" (chuẩn Jina / E5)
    - Bảo đảm normalize vector để dùng với COSINE metric
    - Hỗ trợ batch encode lớn, tối ưu GPU
    """

    def embed_documents(self, texts):
        if not texts:
            return []
        # Thêm prefix chuẩn E5/Jina
        texts = [f"passage: {t.strip()}" for t in texts]
        vecs = super().embed_documents(texts)
        vecs = np.array(vecs, dtype=np.float32)
        # ✅ normalize thủ công để chắc chắn
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs.tolist()

    def embed_query(self, text):
        text = f"query: {text.strip()}"
        vec = super().embed_query(text)
        vec = np.array(vec, dtype=np.float32)
        vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec.tolist()


# ============================================
# 🚀 Loader
# ============================================
def get_embedding_model():
    """
    Load multilingual embedding model optimized for COSINE search.
    - Auto normalize
    - FP32 để tránh sai số cosine (E5 model cần chính xác)
    """
    print(f"🟩 Loading embedding model: {EMBED_MODEL}")

    # ✅ Load model và tokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

    model = AutoModel.from_pretrained(
        EMBED_MODEL,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,  # ❗ giữ FP32 để cosine chính xác
        trust_remote_code=True,
    )

    model_kwargs = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True,
    }

    encode_kwargs = {
        "batch_size": EMBED_BATCH,
        "normalize_embeddings": False,  # ❌ tắt ở đây để normalize thủ công
        "truncate": True,
        "max_length": 512,
    }

    embedder = CustomEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    print("✅ Embedding model initialized and normalized for COSINE search.")
    return embedder


# ============================================
# ⚡ Async Helpers
# ============================================
async def async_embed_docs(embeddings, texts, batch_size=128):
    """
    Async embedding helper để xử lý nhiều docs song song không block event loop.
    """
    loop = asyncio.get_event_loop()

    def batch_iter(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    all_vectors = []
    for batch in batch_iter(texts, batch_size):
        vectors = await loop.run_in_executor(None, embeddings.embed_documents, batch)
        all_vectors.extend(vectors)
    return all_vectors


# ============================================
# 🧠 Cached Singleton
# ============================================
def get_embeddings():
    """
    Cached singleton để load model 1 lần duy nhất trong suốt vòng đời app.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embedding_model()
        print(f"✅ [Cache] Embedding model ready: {EMBED_MODEL}")
    return _embeddings
