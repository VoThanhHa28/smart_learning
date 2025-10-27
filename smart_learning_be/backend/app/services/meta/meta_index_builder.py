# app/services/meta_index_builder.py
import os, json, numpy as np
from pathlib import Path
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "meta_intents"
DIM = 768  # e5-base = 768

def _ensure_collection():
    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="intent_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=1024),
        ]
        schema = CollectionSchema(fields, description="Meta intents Q/A")
        col = Collection(name=COLLECTION_NAME, schema=schema)

        # Index cho vector (HNSW / IVF_FLAT đều được)
        col.create_index(
            field_name="embedding",
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
            index_name="idx_embedding",
        )
        return col
    else:
        return Collection(COLLECTION_NAME)

def build_meta_index():
    connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port="19530")
    collection = _ensure_collection()
    model = SentenceTransformer(MODEL_NAME)

    # Tìm meta_qa_pairs.json ở backend/app/rag/prompts/
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "rag" / "prompts" / "meta_qa_pairs.json",        # app/services → app/rag/prompts
        here.parents[2] / "rag" / "prompts" / "meta_qa_pairs.json",        # app/services/meta → app/rag/prompts
        here.parents[3] / "app" / "rag" / "prompts" / "meta_qa_pairs.json" # chạy từ backend/... → backend/app/rag/prompts
    ]
    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        raise FileNotFoundError("meta_qa_pairs.json not found. Tried: " + " | ".join(str(p) for p in candidates))

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, answers, intents = [], [], []
    for item in data:
        for q in item["question"]:
            texts.append(q)
            answers.append(item["answer"])
            intents.append(item["intent_type"])

    texts_passage = [f"passage: {t}" for t in texts]
    vectors = model.encode(texts_passage, normalize_embeddings=True).astype("float32")

    # Milvus insert cần list-of-columns
    records = [vectors.tolist(), intents, texts, answers]
    collection.insert(records)
    collection.flush()

    # Load để sẵn sàng search
    collection.load()
    n = collection.num_entities
    print(f"### META COUNT = {n}")
    print(f"✅ Inserted {len(texts)} meta samples into Milvus ({COLLECTION_NAME})")

if __name__ == "__main__":
    build_meta_index()
