# app/services/reset_meta_collection.py
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import os

COLLECTION_NAME = "meta_intents"

def reset_meta_collection():
    connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port="19530")

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"✅ Dropped old collection: {COLLECTION_NAME}")

    # 🧩 Schema đơn giản hơn học liệu
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="intent_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2000),
    ]
    schema = CollectionSchema(fields=fields, description="Meta intent collection")

    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print("✅ Created new collection:", COLLECTION_NAME)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 100, "efConstruction": 200},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.flush()
    print("✅ Index created (HNSW)")

    return collection

if __name__ == "__main__":
    c = reset_meta_collection()
    print("📘 Meta-intents collection ready:", c.name)
