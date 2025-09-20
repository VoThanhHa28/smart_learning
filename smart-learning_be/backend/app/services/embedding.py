from langchain_huggingface import HuggingFaceEmbeddings
import os

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))

# Tạo 1 singleton để dùng chung
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"batch_size": EMBED_BATCH, "normalize_embeddings": True},
            multi_process=False,
        )
    return _embeddings
