import os
from functools import lru_cache

# Nếu chưa cài: pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Tự chọn GPU nếu có (có thể ép EMBED_DEVICE=cpu)
DEVICE = "cuda" if (torch.cuda.is_available() and os.getenv("EMBED_DEVICE", "auto") != "cpu") else "cpu"

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return model
