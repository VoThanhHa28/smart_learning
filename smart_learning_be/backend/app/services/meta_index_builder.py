# app/services/meta_index_builder.py
import os, json, numpy as np
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "meta_intents"

def build_meta_index():
    connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port="19530")
    collection = Collection(COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)

    with open("prompts/meta_qa_pairs.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, answers, intents = [], [], []

    for item in data:
        for q in item["question"]:
            texts.append(q)
            answers.append(item["answer"])
            intents.append(item["intent_type"])

    vectors = model.encode(texts, normalize_embeddings=True)
    records = [vectors.tolist(), intents, texts, answers]
    collection.insert(records)
    collection.flush()
    print(f"✅ Inserted {len(texts)} meta samples into Milvus ({COLLECTION_NAME})")

if __name__ == "__main__":
    build_meta_index()
