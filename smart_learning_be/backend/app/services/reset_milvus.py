from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

COLLECTION_NAME = "smart_learning"

def reset_collection():
    # Kết nối Milvus
    connections.connect("default", host="localhost", port="19530")

    # Drop cũ nếu có
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"✅ Dropped old collection: {COLLECTION_NAME}")
    else:
        print(f"ℹ️ Collection {COLLECTION_NAME} not found, nothing to drop.")

    # --- Tạo schema mới ---
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # dim phải khớp với model embed
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="course_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    ]

    schema = CollectionSchema(fields=fields, description="Smart learning collection")

    # Tạo collection
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"✅ Created new collection: {COLLECTION_NAME}")

    # --- Tạo index HNSW ---
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 200, "efConstruction": 200}
    }
    collection.create_index("embedding", index_params)
    print("✅ Index created (HNSW)")

if __name__ == "__main__":
    reset_collection()
    c = Collection("smart_learning")
    c.load()
    print(c.indexes[0].params) 
    print(c.schema)
