from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import utility
import os
COLLECTION_NAME = "smart_learning"

def reset_collection():
    # 1️⃣ Kết nối Milvus
    connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port="19530")


    # 2️⃣ Drop collection cũ (nếu có)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"✅ Dropped old collection: {COLLECTION_NAME}")
        connections.disconnect("default")
        connections.connect("default", host="localhost", port="19530")
    else:
        print(f"ℹ️ Collection {COLLECTION_NAME} not found, nothing to drop.")
    # 3️⃣ Định nghĩa schema mới (chuẩn cho RAG)
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # dim phải khớp model 
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="course_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # tăng lên max để tránh lỗi
    ]

    schema = CollectionSchema(fields=fields, description="Smart learning collection")

    # 4️⃣ Tạo collection mới
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"✅ Created new collection: {COLLECTION_NAME}")

    # 5️⃣ Tạo index cho trường embedding
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 100, "efConstruction": 200},
    }
    collection.create_index(
    field_name="embedding",
    index_params=index_params,
    index_name="embedding_hnsw",
    async_build=True  # ✅ build ở background
    )

    print("⚙️ Index building in background...")

    utility.wait_for_index_building_complete("smart_learning", "embedding_hnsw")
    print("✅ Index build completed!")
    collection.flush()
    print("✅ Index created (HNSW)")

    return collection

# 6️⃣ Thử load và in thông tin
if __name__ == "__main__":
    c = reset_collection()
    c.load()
    print("✅ Collection loaded successfully.")
    print("📋 Index params:", c.indexes[0].params)
    print("📘 Schema:")
    for f in c.schema.fields:
        print(" -", f.name, f.dtype)
