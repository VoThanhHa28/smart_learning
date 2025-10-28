# services/ingestion/vectorstore.py
import os
import json
from typing import List, Optional, Tuple, Dict, Any
import logging 
import traceback
import time
from pathlib import Path
import numpy as np 
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langsmith import traceable

# --- Milvus Imports ---
try:
    from pymilvus import connections, Collection, utility, DataType # DataType cần cho schema check
except ImportError:
    logging.error("Pymilvus library not found. Please install it: pip install pymilvus")
    raise

# --- Langchain Imports ---
from langchain_core.documents import Document
# Giả sử đường dẫn import
try:
    from ...infrastructure.milvus.test_milvus import DebugMilvus # Class wrapper của bạn
    from ...infrastructure.llm.embedding import get_embeddings # Hàm lấy embedding model
except ImportError:
    # Fallback paths (điều chỉnh nếu cần)
    from ....infrastructure.milvus.test_milvus import DebugMilvus  
    from ....infrastructure.llm.embedding import get_embeddings

# --- Langsmith Import ---
try:
    from langsmith import traceable
except ImportError:
    # Tạo decorator giả nếu không cài langsmith
    def traceable(func=None, *, name=None):
        if func: return func
        else: return lambda f: f
    logging.warning("langsmith library not found. Tracing disabled.")


# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# --- Cấu hình Đường dẫn & File ---
_THIS_DIR   = Path(__file__).resolve().parent                 
_SERVICE_DIR = _THIS_DIR.parent # Giả sử: backend/app/services
_APP_DIR = _SERVICE_DIR.parent # Giả sử: backend/app
_BACKEND_DIR = _APP_DIR.parent # Giả sử: backend
_DATA_DIR   = Path(os.getenv("DATA_DIR", _BACKEND_DIR / "data"))
try:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Data directory set to: {_DATA_DIR}")
except OSError as e:
     logging.error(f"Could not create data directory {_DATA_DIR}: {e}")
CORPUS_FILE = _DATA_DIR / "corpus.jsonl" 

# --- Cấu hình Milvus (Từ Environment Variables) ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "smart_learning")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768")) 
METRIC_TYPE = os.getenv("MILVUS_METRIC", "COSINE") 
MILVUS_TIMEOUT = int(os.getenv("MILVUS_TIMEOUT", "15")) 
MILVUS_SEARCH_TIMEOUT = float(os.getenv("MILVUS_SEARCH_TIMEOUT", "10.0")) 
MILVUS_CONSISTENCY = os.getenv("MILVUS_CONSISTENCY", "Bounded") # Mức độ nhất quán

# --- Globals (Cache) ---
_CONNECTED = False
_COLLECTION: Optional[Collection] = None
_vs = None 

# --- Hàm Kết nối & Khởi tạo ---
def _connect_once():
    """Kết nối tới Milvus một lần duy nhất, có retry."""
    global _CONNECTED
    if _CONNECTED:
        return True
    
    max_retries = 3
    retry_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempting Milvus connection (attempt {attempt+1}/{max_retries}) to {MILVUS_HOST}:{MILVUS_PORT}...")
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=MILVUS_TIMEOUT)
            # Kiểm tra kết nối thành công
            if utility.has_collection("dummy_check_collection_existence"): # Lệnh nhẹ để kiểm tra
                 pass # Tồn tại hay không không quan trọng, chỉ cần không lỗi
            logging.info(f"✅ Successfully connected to Milvus.")
            _CONNECTED = True
            return True
        except Exception as e:
            logging.error(f"❌ Milvus connection attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                logging.info(f"   Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("❌ Max connection retries reached. Could not connect to Milvus.")
                _CONNECTED = False
                return False # Trả về False nếu không kết nối được
                # Có thể raise lỗi ở đây nếu kết nối là bắt buộc
                # raise RuntimeError(f"Could not connect to Milvus after {max_retries} attempts.") from e

def _get_collection() -> Collection:
    """Lấy Milvus Collection instance, đảm bảo đã load và schema đúng."""
    global _COLLECTION
    if _COLLECTION is not None:
        # Kiểm tra nhanh xem kết nối còn không (có thể bị ngắt)
        try:
             if _COLLECTION.is_empty is not None: # Thử một thuộc tính nhanh
                  return _COLLECTION
             else: # Kết nối có vẻ mất
                  logging.warning("Milvus connection seems lost, attempting reconnect...")
                  _CONNECTED = False
                  _COLLECTION = None # Reset cache
                  connections.disconnect("default")
        except Exception: # Lỗi khi truy cập _COLLECTION -> kết nối mất
             logging.warning("Error accessing cached Milvus collection, attempting reconnect...")
             _CONNECTED = False
             _COLLECTION = None
             try: connections.disconnect("default") 
             except Exception: pass


    if not _connect_once(): # Thử kết nối lại nếu cần
        raise RuntimeError("Cannot proceed: Milvus connection failed.")

    try:
        if not utility.has_collection(COLLECTION_NAME):
            logging.error(f"CRITICAL: Milvus Collection '{COLLECTION_NAME}' not found!")
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' does not exist. Please run reset_collection.py first.")
        
        _COLLECTION = Collection(COLLECTION_NAME)
        
        # Kiểm tra Schema (quan trọng)
        expected_fields = {"id", "embedding", "page", "subject", "course_id", "user_id", "text"}
        actual_fields = {f.name for f in _COLLECTION.schema.fields}
        if not expected_fields.issubset(actual_fields):
            missing = expected_fields - actual_fields
            extra = actual_fields - expected_fields
            error_msg = f"Schema mismatch for collection '{COLLECTION_NAME}'!"
            if missing: error_msg += f" Missing fields: {missing}."
            if extra: error_msg += f" Unexpected fields: {extra}."
            logging.error(f"CRITICAL: {error_msg} Please run reset_collection.py.")
            raise RuntimeError(error_msg)
        
        # Kiểm tra dimension của embedding field
        embed_field = next((f for f in _COLLECTION.schema.fields if f.name == 'embedding'), None)
        if embed_field and embed_field.params.get('dim') != EMBED_DIM:
            error_msg = f"Schema mismatch: Embedding dimension is {embed_field.params.get('dim')}, expected {EMBED_DIM}."
            logging.error(f"CRITICAL: {error_msg} Please check EMBED_DIM env var and run reset_collection.py.")
            raise RuntimeError(error_msg)

        # Load collection vào bộ nhớ
        load_start = time.time()
        logging.info(f"Loading Milvus collection '{COLLECTION_NAME}' into memory...")
        _COLLECTION.load()
        logging.info(f"✅ Loaded Milvus collection '{COLLECTION_NAME}' (took {time.time()-load_start:.2f}s).")
        
        return _COLLECTION
            
    except Exception as e:
        logging.error(f"Failed to get or load Milvus collection '{COLLECTION_NAME}': {e}", exc_info=True)
        _COLLECTION = None # Reset cache nếu lỗi
        raise RuntimeError(f"Could not access or load Milvus collection: {e}") from e

def get_vectorstore():
    """Lấy instance của vector store (DebugMilvus), tạo nếu chưa có."""
    global _vs
    if _vs is not None:
        return _vs
    
    col = _get_collection() # Gọi hàm này để đảm bảo collection ok và đã load
    
    try:
        embeddings = get_embeddings() # Lấy hàm embedding
    except Exception as e:
        logging.error(f"Failed to get embeddings function: {e}", exc_info=True)
        raise RuntimeError("Could not initialize embeddings.") from e

    # Cấu hình index (chủ yếu để DebugMilvus biết)
    index_params = { "index_type": "HNSW", "metric_type": METRIC_TYPE, "params": {"M": 100, "efConstruction": 200} }
    # Cấu hình search params mặc định
    default_search_ef = int(os.getenv("MILVUS_SEARCH_EF", "128"))
    default_search_params = {"metric_type": METRIC_TYPE, "params": {"ef": default_search_ef}}

    try:
        _vs = DebugMilvus( 
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT, "timeout": MILVUS_TIMEOUT},
            index_params=index_params, # Cho DebugMilvus biết cấu hình index
            auto_id=True, 
            text_field="text",
            vector_field="embedding",
            consistency_level=MILVUS_CONSISTENCY, 
            search_params=default_search_params, # Cấu hình search mặc định
        )
        # Gán collection đã load vào instance (nếu DebugMilvus hỗ trợ)
        if hasattr(_vs, "_collection"):
             _vs._collection = col
        logging.info(f"✅ Milvus vector store instance (DebugMilvus) created for collection '{COLLECTION_NAME}'.")
        return _vs
    except Exception as e:
        logging.error(f"Failed to initialize DebugMilvus vector store: {e}", exc_info=True)
        raise RuntimeError("Could not initialize DebugMilvus vector store.") from e


def _ensure_partitions_loaded(col: Collection, subject: str | None):
    """Load partition cụ thể hoặc toàn bộ collection vào bộ nhớ nếu cần."""
    # Hàm này có thể không cần thiết nếu _get_collection đã load toàn bộ
    # Nhưng giữ lại để có thể tối ưu sau này nếu chỉ muốn load partition cụ thể
    try:
        target_partition_name = None
        if subject:
            target_partition_name = f"p_{subject.strip().lower()}"
            if not col.has_partition(target_partition_name):
                logging.debug(f"Partition '{target_partition_name}' not found, cannot ensure load.")
                return 

        # Kiểm tra trạng thái load (có thể tốn kém)
        # load_state = utility.load_state(col.name)
        # if load_state != "Loaded": # Chỉ load nếu chưa load
        
        # Tạm thời luôn gọi load (Milvus sẽ tự xử lý nếu đã load)
        load_start_time = time.time()
        if target_partition_name:
            col.load(partition_names=[target_partition_name])
            logging.debug(f"   [Milvus Load Ensure] Partition '{target_partition_name}' loaded/checked (took {time.time()-load_start_time:.2f}s).")
        else:
            # Load toàn bộ collection (đã làm trong _get_collection)
             logging.debug("   [Milvus Load Ensure] Entire collection load confirmed.")
             pass # Không cần load lại nếu _get_collection đã làm
            
    except Exception as e:
        logging.warning(f"[Milvus] Warning during _ensure_partitions_loaded: {e}")

# ====================== #
# 🧠 Hàm xử lý 1 batch   #
# ====================== #
def _normalize_batch(vectors):
    """Chuẩn hóa list các vector thành unit vectors."""
    try:
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized_arr = arr / norms
        return normalized_arr.tolist()
    except Exception as e:
        logging.error(f"Error normalizing vectors: {e}")
        return [] # Trả về list rỗng nếu lỗi

# @traceable(name="Process Milvus Batch") # Bỏ traceable ở đây vì chạy trong thread
def process_batch(vs, embeddings, batch: List[Document], embed: bool = True):
    """
    Xử lý một batch Document: embed (nếu cần) và insert vào Milvus.
    Chạy trong thread (gọi bằng asyncio.to_thread).
    """
    batch_start_time = time.time()
    num_docs = len(batch)
    logging.debug(f"Process Batch: Starting for {num_docs} documents.")
    
    try:
        # 1. Lọc doc rỗng (đã làm ở upsert_chunks, nhưng kiểm tra lại cho chắc)
        valid_docs = [d for d in batch if d.page_content and d.page_content.strip()]
        if not valid_docs: return 0
        
        # 2. Embedding (nếu embed=True)
        vectors = []
        if embed:
            embed_start = time.time()
            try:
                contents = [d.page_content for d in valid_docs]
                vectors = embeddings.embed_documents(contents)
                if not vectors or len(vectors) != len(valid_docs):
                     raise ValueError(f"Embedding result length mismatch: got {len(vectors)}, expected {len(valid_docs)}")
                vectors = _normalize_batch(vectors) # Chuẩn hóa L2
                if not vectors: raise ValueError("Normalization failed.")
                logging.debug(f"   Batch embedded {len(vectors)} docs in {time.time()-embed_start:.2f}s.")
            except Exception as e:
                logging.error(f"   Batch embedding failed: {e}", exc_info=True)
                return 0 # Lỗi nghiêm trọng, dừng batch này
        else:
            vectors = [[0.0] * EMBED_DIM for _ in valid_docs]

        # 3. Chuẩn bị data lists cho Milvus (theo đúng thứ tự schema)
        data_lists = {
            "embedding": [], "page": [], "subject": [], 
            "course_id": [], "user_id": [], "text": []
        }
        valid_indices = [] # Theo dõi index của docs hợp lệ
        
        for i, doc in enumerate(valid_docs):
            current_vector = vectors[i]
            if len(current_vector) != EMBED_DIM:
                 logging.warning(f"   Skipping doc {i}: Vector dim mismatch (got {len(current_vector)}, expected {EMBED_DIM}). Metadata: {doc.metadata}")
                 continue 
            
            meta = doc.metadata
            data_lists["embedding"].append(current_vector)
            data_lists["page"].append(int(meta.get("page", i + 1))) 
            data_lists["subject"].append(str(meta.get("subject", "unknown")))
            data_lists["course_id"].append(str(meta.get("course_id", "unknown")))
            data_lists["user_id"].append(str(meta.get("user_id", "unknown"))) # LẤY user_id
            data_lists["text"].append(str(doc.page_content))
            valid_indices.append(i) # Lưu index nếu doc hợp lệ

        num_valid_to_insert = len(data_lists["embedding"])
        if num_valid_to_insert == 0:
            logging.warning("Process Batch: No valid documents left after vector/metadata check.")
            return 0

        # 4. Lấy collection Milvus
        try:
             # Dùng _get_collection để đảm bảo collection tồn tại và đã load
             collection = _get_collection() 
        except Exception as e:
             logging.error(f"Process Batch: Cannot get Milvus collection: {e}")
             return 0 # Không thể insert nếu không có collection

        # 5. Partition (optional, dựa trên subject)
        first_subject = data_lists["subject"][0] if data_lists["subject"] else None
        target_partition_name = None
        if first_subject and first_subject != "unknown":
            target_partition_name = f"p_{first_subject.lower()}"
            try:
                if not collection.has_partition(target_partition_name):
                    logging.info(f"   Creating partition: {target_partition_name}")
                    collection.create_partition(target_partition_name)
            except Exception as e:
                logging.warning(f"   Could not create partition '{target_partition_name}' (might exist or concurrent creation): {e}")
                # Không đặt target_partition_name = None, cứ thử insert, Milvus sẽ báo lỗi nếu partition sai

        # 6. Insert vào Milvus
        insert_start = time.time()
        try:
            # Tạo list dữ liệu theo đúng thứ tự fields trong schema (quan trọng!)
            # Lấy thứ tự từ schema để đảm bảo
            schema_field_order = [f.name for f in collection.schema.fields if not f.is_primary]
            
            # Map dữ liệu từ data_lists vào đúng thứ tự
            data_to_insert_ordered = []
            for field_name in schema_field_order:
                if field_name in data_lists:
                    data_to_insert_ordered.append(data_lists[field_name])
                else:
                    # Trường hợp schema có field mà data_lists không có (lỗi)
                    logging.error(f"   Data preparation error: Missing data for schema field '{field_name}'")
                    raise ValueError(f"Missing data for field: {field_name}")

            logging.debug(f"   Inserting {num_valid_to_insert} records into partition '{target_partition_name or 'default'}'...")
            
            mutation_result = collection.insert(
                data=data_to_insert_ordered,
                partition_name=target_partition_name 
            )
            
            inserted_count = mutation_result.insert_count
            insert_duration = time.time() - insert_start
            
            if inserted_count != num_valid_to_insert:
                 logging.warning(f"   Milvus insert count mismatch: Expected {num_valid_to_insert}, Actual {inserted_count}. PKs: {mutation_result.primary_keys[:5]}...")
            else:
                 logging.debug(f"   Milvus insert successful ({inserted_count} docs) in {insert_duration:.2f}s.")

            # Không cần load partition ở đây, _get_collection đã load toàn bộ

            total_batch_duration = time.time() - batch_start_time
            logging.info(f"   ✅ Processed batch of {inserted_count} docs (Embed={embed}). Total time: {total_batch_duration:.2f}s")
            return inserted_count

        except Exception as e:
            logging.error(f"   ❌ Milvus insert failed: {e}", exc_info=True)
            logging.error(f"   Failed data details (first record): page={data_lists['page'][0]}, subject={data_lists['subject'][0]}, course={data_lists['course_id'][0]}, user={data_lists['user_id'][0]}, text_len={len(data_lists['text'][0])}")
            return 0 

    except Exception as outer_e:
        logging.error(f"❌ Unexpected error in process_batch for {num_docs} docs: {outer_e}", exc_info=True)
        return 0

# ========================= #
# 🚀 Hàm async upsert chính #
# ========================= #
@traceable(name="Upsert Chunks to Milvus")
async def upsert_chunks(docs: list[Document], batch_size: int = 128, embed: bool = True): # Giảm batch size
    """
    Upsert danh sách Document vào Milvus. Hàm này gọi process_batch trong thread.
    Metadata (user_id, course_id) phải có sẵn trong docs.
    """
    upsert_start_time = time.time()
    
    # 1. Lấy instances cần thiết (lỗi sớm nếu không có)
    try:
        vs = get_vectorstore() # Kích hoạt kết nối và load collection
        embeddings = get_embeddings()
    except Exception as setup_e:
        logging.error(f"Upsert Chunks: Failed setup - {setup_e}")
        return 0 # Không thể tiếp tục

    # 2. Làm sạch và chuẩn bị Docs
    clean_docs = []
    skipped_count = 0
    for i, d in enumerate(docs):
        if not isinstance(d, Document) or not d.page_content or not d.page_content.strip():
            skipped_count += 1
            continue
        meta = d.metadata.copy() if d.metadata else {}
        meta["subject"] = str(meta.get("subject", "unknown")).lower().strip()
        meta["course_id"] = str(meta.get("course_id", "unknown")).lower().strip()
        meta["user_id"] = str(meta.get("user_id", "unknown")).lower().strip() 
        try: # Xử lý page
            page_val = int(meta.get("page", -1))
            meta["page"] = page_val if page_val > 0 else (i + 1 - skipped_count)
        except (ValueError, TypeError): meta["page"] = i + 1 - skipped_count
        meta.pop("id", None); meta.pop("embedding", None) # Xóa ID và embedding cũ
        clean_docs.append(Document(page_content=d.page_content.strip(), metadata=meta))

    total_valid_docs = len(clean_docs)
    if skipped_count > 0: logging.warning(f"Upsert Chunks: Skipped {skipped_count} invalid/empty documents.")
    if total_valid_docs == 0:
        logging.warning("Upsert Chunks: No valid documents to upsert.")
        return 0
    logging.info(f"📤 Upsert Chunks: Starting for {total_valid_docs} documents (batch_size={batch_size})...")

    # 3. Chia batches và chạy song song
    batches = [clean_docs[i:i + batch_size] for i in range(0, total_valid_docs, batch_size)]
    logging.info(f"   Divided into {len(batches)} batches.")
    
    insert_tasks = [
        asyncio.to_thread(process_batch, vs, embeddings, current_batch, embed)
        for current_batch in batches
        ]
    
    try:
        batch_results = await asyncio.gather(*insert_tasks)
        total_inserted = sum(batch_results)
        num_failed_batches = len([r for r in batch_results if r == 0])
        logging.info(f"   📊 Batch processing finished. Successful inserts: {total_inserted}/{total_valid_docs}. Failed batches: {num_failed_batches}.")
    except Exception as e:
        logging.error(f"   ❌ Error during asyncio.gather for batch processing: {e}", exc_info=True)
        total_inserted = 0 

    # 4. Flush collection (quan trọng)
    flush_start_time = time.time()
    try:
        collection = _get_collection() # Lấy lại collection đã load
        if total_inserted > 0 and collection.has_index:
            logging.info("   ⏳ Flushing Milvus collection...")
            await asyncio.to_thread(collection.flush) 
            logging.info(f"   💾 Milvus collection flushed (took {time.time()-flush_start_time:.2f}s).")
        else: logging.info("   ⏭️ Skipping flush (no inserts or no index).")
    except Exception as e:
        logging.error(f"   ⚠️ Milvus flush failed: {e}", exc_info=True)

    # 5. Ghi corpus cache (chạy ngầm)
    if total_inserted > 0:
        async def append_corpus_task():
            await asyncio.sleep(1) # Đợi 1 chút để flush có thể bắt đầu
            await asyncio.to_thread(_append_to_corpus, clean_docs)
        asyncio.create_task(append_corpus_task(), name="append_corpus_cache")
    
    upsert_duration = time.time() - upsert_start_time
    logging.info(f"🏁 Upsert Chunks finished in {upsert_duration:.2f}s. Total indexed: {total_inserted}")
    return total_inserted

# ========= Helper Lấy Collection (Giữ nguyên) =========
# def _get_collection() -> Collection: ... (Code giữ nguyên)

# ========= Embedding helpers (Giữ nguyên) =========
def _ensure_vec(x) -> np.ndarray:
    """Đảm bảo vector là numpy array 1D đúng dimension và đã normalize."""
    try:
        arr = np.array(x, dtype=np.float32).reshape(-1) # Đảm bảo 1D
        if arr.shape[0] != EMBED_DIM:
            raise ValueError(f"Vector dimension mismatch: expected {EMBED_DIM}, got {arr.shape[0]}")
        
        # Normalize L2
        norm = np.linalg.norm(arr)
        if norm == 0: 
            # logging.warning("Zero vector encountered.") # Có thể log nếu cần
            return arr # Trả về vector 0 nếu norm là 0
        normalized_arr = arr / norm
        return normalized_arr.astype(np.float32) # Đảm bảo float32
    except Exception as e:
         logging.error(f"Error ensuring vector format: {e}")
         raise # Ném lại lỗi để hàm gọi xử lý

# ========= Public API (Tìm kiếm & Lấy khóa học) =========
def build_expr(where: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Xây dựng biểu thức lọc Milvus từ dict `where`.
    Hỗ trợ: subject, course_id, user_id. Bỏ qua giá trị rỗng/unknown.
    """
    if not where: return None
    parts = []
    subject = str(where.get("subject") or "").strip().lower()
    if subject and subject != "unknown": parts.append(f'subject == "{subject}"')
    course_id = str(where.get("course_id") or "").strip().lower()
    if course_id and course_id != "unknown": parts.append(f'course_id == "{course_id}"')
    user_id = str(where.get("user_id") or "").strip().lower()
    if user_id and user_id != "unknown": parts.append(f'user_id == "{user_id}"')
    expression = " and ".join(parts)
    logging.debug(f"Built Milvus expression: '{expression}' from: {where}")
    return expression if expression else None

# Hàm log context lỗi (Giữ nguyên)
# def _log_milvus_ctx(col: Collection, k: int, ef: int, expr: Optional[str]): ...

@traceable(name="Similarity Search Milvus")
async def similarity_search(
    query: str,
    k: int = 5, 
    where: Optional[dict] = None, # Chứa user_id, course_id từ RAG pipeline
    log: bool = True,
    embed_fn=None, 
) -> List[Tuple[Document, float]]:
    """
    Tìm kiếm vector tương đồng trong Milvus, có lọc theo metadata (`where`).
    Trả về List[(Document, score)], score là distance (0=giống hệt).
    """
    search_start_time = time.time()
    if not query or not query.strip():
        logging.warning("Similarity Search: Empty query received.")
        return []

    try:
        col = _get_collection() # Lấy collection đã load, schema validated
    except Exception as e:
        logging.error(f"Similarity Search: Cannot proceed, failed to get Milvus collection: {e}")
        return [] 

    # Không cần _ensure_partitions_loaded nếu _get_collection đã load toàn bộ

    # 1. Embed Query
    query_vector: Optional[np.ndarray] = None
    try:
        t_embed_start = time.time()
        raw_vector = embed_fn(query) if embed_fn else get_embeddings().embed_query(query)
        query_vector = _ensure_vec(raw_vector) # Chuẩn hóa và kiểm tra dim
        logging.debug(f"   Query embedded in {time.time()-t_embed_start:.3f}s")
    except Exception as e:
        logging.error(f"Similarity Search: Failed to embed query '{query[:50]}...': {e}", exc_info=True)
        return [] 

    # 2. Build Filter Expression
    expr = build_expr(where) # Sẽ bao gồm user_id, course_id nếu có trong `where`
    
    # 3. Search Parameters
    search_ef = int(os.getenv("MILVUS_SEARCH_EF", "128")) 
    search_params = {"metric_type": METRIC_TYPE, "params": {"ef": max(search_ef, k * 2)}} 
    output_fields = ["page", "subject", "course_id", "text", "user_id"] # Lấy đủ thông tin
    
    if log:
        logging.info(f"🔎 [Milvus Search] Starting...")
        logging.info(f"   Query: '{query[:80]}...'")
        logging.info(f"   Params: k={k}, ef={search_params['params']['ef']}, metric={METRIC_TYPE}")
        logging.info(f"   Filter (expr): {expr or 'None'}")

    # 4. Perform Search
    try:
        search_results = await asyncio.to_thread( # Chạy search trong thread
            col.search,
            data=[query_vector.tolist()], 
            anns_field="embedding",      
            param=search_params,         
            limit=k,                     
            expr=expr,                   
            output_fields=output_fields, 
            timeout=MILVUS_SEARCH_TIMEOUT, 
            consistency_level=MILVUS_CONSISTENCY, # Thêm consistency level
        )
        hits = search_results[0] if search_results else []
        search_duration = time.time() - search_start_time
        logging.info(f"   ✅ Milvus search completed in {search_duration:.3f}s. Hits found: {len(hits)}")

    except Exception as e:
        logging.error(f"❌ Milvus search failed after {time.time() - search_start_time:.3f}s: {e}", exc_info=False) 
        # _log_milvus_ctx(col, k, search_params['params']['ef'], expr) # Log context khi lỗi
        return [] # Trả về rỗng nếu lỗi

    # 5. Process Results
    output_docs: List[Tuple[Document, float]] = []
    for hit in hits:
        try:
            entity_data = hit.entity 
            score = hit.distance # Distance (0=exact match)
            metadata = {f: entity_data.get(f) for f in output_fields if f != "text"} # Lấy tất cả output fields trừ text
            metadata["score"] = score # Thêm score vào metadata
            
            doc = Document(page_content=entity_data.get("text", ""), metadata=metadata)
            output_docs.append((doc, score))
            
            if log and logging.getLogger().isEnabledFor(logging.DEBUG):
                 logging.debug(f"      Hit: score={score:.4f}, page={metadata.get('page')}, course={metadata.get('course_id')}, user={metadata.get('user_id')}, text='{doc.page_content[:100]}...'")
        except Exception as hit_e:
             logging.warning(f"   Error processing hit {hit.id}: {hit_e}")
             continue # Bỏ qua hit lỗi

    if log:
        logging.info(f"✅ Similarity search returning {len(output_docs)} processed documents.")

    return output_docs

# ========= Warmup (Giữ nguyên) =========
def warmup_vectorstore():
    logging.info("Warming up vector store...")
    try: _ = get_vectorstore() # Gọi hàm này để kích hoạt kết nối, load, schema check
    except Exception as e: logging.error(f"❌ Vector store warmup FAILED: {e}", exc_info=True)

# ========= Get All Docs (Giữ nguyên) =========
# async def get_all_docs(...): ...

# ---------- BM25 Corpus Helper (Giữ nguyên) ----------
def _append_to_corpus(docs: List[Document]):
    try:
        CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True) 
        with open(CORPUS_FILE, "a", encoding="utf-8") as f:
            for d in docs:
                record = {"text": d.page_content, "metadata": d.metadata}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.debug(f"Appended {len(docs)} records to corpus file: {CORPUS_FILE}")
    except Exception as e:
        logging.warning(f"Could not append to corpus file '{CORPUS_FILE}': {e}")


# === HÀM LẤY KHÓA HỌC CHO UI (Quan trọng) ===
@traceable(name="Get Courses By User")
async def get_courses_by_user(user_id: str) -> List[Dict[str, str]]:
    """
    Lấy danh sách khóa học (course_id, subject) duy nhất của user từ Milvus.
    """
    if not user_id or user_id == "unknown":
        logging.warning("get_courses_by_user: Called with invalid user_id.")
        return []

    normalized_user_id = user_id.lower().strip()
    if not normalized_user_id or normalized_user_id == "unknown":
         logging.warning(f"get_courses_by_user: Invalid user_id after normalization: '{user_id}'")
         return []
    
    logging.info(f"ℹ️ Querying Milvus for courses owned by user: '{user_id}'")
    query_start_time = time.time()
    
    try:
        col = _get_collection() # Lấy collection đã load
        # Không cần _ensure_partitions_loaded vì query metadata trên toàn collection

        expr = f'user_id == "{normalized_user_id}"' # Filter theo user_id
        output_fields = ["course_id", "subject"] # Chỉ lấy 2 trường này
        
        # Chạy query trong thread
        query_results = await asyncio.to_thread(
            col.query,
            expr=expr,
            output_fields=output_fields,
            limit=16384 # Giới hạn max của Milvus
        )
        
        query_duration = time.time() - query_start_time
        logging.info(f"   Milvus query completed in {query_duration:.3f}s. Found {len(query_results)} raw results for user '{user_id}'.")

        # De-duplicate kết quả
        seen_courses = {} # Dùng dict để lưu subject tương ứng
        for result_dict in query_results:
            course_id = result_dict.get("course_id")
            if course_id and course_id != "unknown" and course_id not in seen_courses:
                # Lưu subject lần đầu thấy course_id này
                seen_courses[course_id] = result_dict.get("subject", "N/A") 
                
        # Chuyển dict thành list theo format yêu cầu
        unique_courses = [{"course_id": cid, "subject": subj} for cid, subj in seen_courses.items()]
        
        total_duration = time.time() - query_start_time
        logging.info(f"✅ Found {len(unique_courses)} unique courses for user '{user_id}' (Total time: {total_duration:.3f}s)")
        # Sắp xếp
        unique_courses.sort(key=lambda x: x.get('subject', '').lower())
        return unique_courses

    except Exception as e:
        logging.error(f"❌ Error fetching courses for user '{user_id}': {e}", exc_info=True)
        return [] 
# =======================================