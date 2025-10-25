# app/services/meta_responder.py
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
import os

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "meta_intents"

model = SentenceTransformer(MODEL_NAME)

def search_meta(question: str, k: int = 3, threshold: float = 0.60):
    """Meta search dùng E5 chuẩn (query/passages), index COSINE"""
    connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port="19530")
    collection = Collection(COLLECTION_NAME)
    collection.load()

    q_raw = (question or "").strip().lower()
    q_for_e5 = f"query: {q_raw}"
    vec = model.encode([q_for_e5], normalize_embeddings=True)[0].astype("float32")

    results = collection.search(
        data=[[float(x) for x in vec]],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=k,
        output_fields=["question", "answer", "intent_type"]
    )

    if not results or len(results[0]) == 0:
        return None

    hit = results[0][0]
    distance = float(hit.distance)          # COSINE distance = 1 - cos_sim
    cos_sim = distance
    if cos_sim < threshold:
        print(f"DEBUG_meta_sim_E5: cos_sim={cos_sim:.3f} < threshold={threshold:.2f}")
        return None

    ans = hit.entity.get("answer", "")
    intent = hit.entity.get("intent_type", "")
    print(f"🔍 Meta intent matched: {intent} (cos_sim={cos_sim:.3f}) | q='{hit.entity.get('question','')}'")
    return ans

if __name__ == "__main__":
    m = SentenceTransformer(MODEL_NAME)
    q = m.encode(["query: chào bạn"], normalize_embeddings=True)[0]
    p = m.encode(["passage: chào"], normalize_embeddings=True)[0]
    import numpy as np
    cos = float(np.dot(q, p))
    print("LOCAL COS =", round(cos, 3))
