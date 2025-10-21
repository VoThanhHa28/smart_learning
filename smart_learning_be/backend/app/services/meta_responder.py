# app/services/meta_responder.py
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
import os

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "meta_intents"

model = SentenceTransformer(MODEL_NAME)

def search_meta(question: str, k: int = 3, threshold: float = 0.55):
    """Tìm câu trả lời meta gần nhất trong Milvus"""
    connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port="19530")
    collection = Collection(COLLECTION_NAME)
    collection.load()

    vec = model.encode([question], normalize_embeddings=True)
    vec = np.array(vec, dtype=np.float32).flatten()
    vec = vec.astype("float32")

    results = collection.search(
        data=[[x for x in vec]],  # đảm bảo list of float
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=k,
        output_fields=["question", "answer", "intent_type"]
    )

    if not results or len(results[0]) == 0:
        return None

    hit = results[0][0]
    score = hit.distance
    if score < threshold:
        return None

    ans = hit.entity.get("answer", "")
    intent = hit.entity.get("intent_type", "")
    print(f"🔍 Meta intent matched: {intent} (score={score:.3f})")
    return ans
