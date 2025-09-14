from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=1)
def get_model():
    # model nhẹ, chất lượng ổn
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs).tolist()

def embed_query(query: str):
    return embed_texts([query])[0]
