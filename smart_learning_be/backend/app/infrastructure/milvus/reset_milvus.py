# reset_collection.py
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "smart_learning")

def reset_collection():
    """
    Kết nối Milvus, xóa collection cũ (nếu có) và tạo collection mới
    với schema bao gồm user_id.
    """
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        try:
            print(f"ℹ️ Releasing collection {COLLECTION_NAME}...")
            Collection(COLLECTION_NAME).release()
        except Exception as e:
            print(f"   ⚠️ Could not release collection (might be ok): {e}")
        try:
            print(f"🗑️ Dropping old collection: {COLLECTION_NAME}...")
            utility.drop_collection(COLLECTION_NAME)
            print(f"✅ Dropped old collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"   ❌ Error dropping collection: {e}")
            return None # Dừng nếu không xóa được
    else:
        print(f"ℹ️ Collection {COLLECTION_NAME} not found, nothing to drop.")

    # Định nghĩa các trường cho collection mới
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Dim phải khớp model embedding
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="course_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64), # <-- THÊM user_id
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # Giới hạn lớn nhất
    ]

    schema = CollectionSchema(fields=fields, description="Smart learning collection with user ownership")

    # Tạo collection mới
    try:
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        print(f"✅ Created new collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"   ❌ Error creating collection: {e}")
        return None

    # Tạo index cho trường embedding để tìm kiếm nhanh
    index_params = {
        "index_type": "HNSW",      # Thuật toán index phổ biến
        "metric_type": "COSINE",   # Đo độ tương đồng cosine (phù hợp embedding)
        "params": {"M": 100, "efConstruction": 200}, # Tham số HNSW
    }
    try:
        collection.create_index(
            field_name="embedding",
            index_params=index_params,
            index_name="embedding_hnsw",
            # async_build=True # Chạy ngầm có thể gây lỗi nếu script kết thúc sớm
        )
        print("⏳ Creating index...")
        # Bỏ wait_for nếu không cần thiết ngay lập tức, nhưng flush là cần
        # utility.wait_for_index_building_complete(COLLECTION_NAME, "embedding_hnsw")
        # print("✅ Index build completed!")
        collection.flush() # Đảm bảo dữ liệu index được ghi
        print("✅ Index created (HNSW) and flushed.")
    except Exception as e:
         print(f"   ❌ Error creating index: {e}")
         # Vẫn trả về collection nhưng index có thể chưa sẵn sàng

    return collection

# Đoạn code này chỉ chạy khi bạn chạy trực tiếp file reset_collection.py
if __name__ == "__main__":
    print("--- Running Milvus Collection Reset ---")
    new_collection = reset_collection()
    if new_collection:
        try:
            new_collection.load() # Load collection vào bộ nhớ để sẵn sàng truy vấn
            print("✅ Collection loaded successfully into memory.")
            if new_collection.has_index:
                 print("📋 Index params:", new_collection.indexes[0].params)
            else:
                 print("⚠️ Collection created, but index might still be building or failed.")
            print("📘 Final Schema:")
            for f in new_collection.schema.fields:
                print(f"   - {f.name}: {f.dtype}")
        except Exception as e:
            print(f"   ❌ Error loading collection or getting info: {e}")
    else:
        print("--- Collection reset failed ---")