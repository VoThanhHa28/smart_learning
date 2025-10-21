import os
import json
from typing import List, Optional
# from langchain_milvus import Milvus
from .test_milvus import DebugMilvus  
from langchain_core.documents import Document
from .embedding import get_embeddings
import traceback
from pymilvus import connections, Collection
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
import re
import os, time
import logging
from typing import Optional, List, Tuple, Dict, Any
BASE_DIR = os.path.dirname(__file__)   # thư mục hiện tại (app/services)
CORPUS_FILE = os.getenv("CORPUS_FILE", os.path.join(BASE_DIR, "corpus.jsonl"))



# ========= Consts =========
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "smart_learning")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
METRIC_TYPE = os.getenv("MILVUS_METRIC", "COSINE")

# ========= Globals (cache) =========
_CONNECTED = False
_COLLECTION: Optional[Collection] = None

def _connect_once():
    global _CONNECTED
    if _CONNECTED:
        return
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=10)
    _CONNECTED = True
    logging.info(f"[milvus] connected {MILVUS_HOST}:{MILVUS_PORT}")

_vs = None

def get_vectorstore():
    global _vs
    if _vs is not None:
        return _vs
    embeddings = get_embeddings()

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 100, "efConstruction": 200},
    }

    _vs = DebugMilvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": os.getenv("MILVUS_HOST", "localhost"), "port": "19530"},
        index_params=index_params,
        auto_id=True,
        text_field="text",
        vector_field="embedding",
    )
    return _vs


# ====================== #
# 🧠 Hàm xử lý 1 batch   #
# ====================== #
import os, traceback, numpy as np

from pymilvus import Collection, connections, utility

def _normalize(vectors):
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / (norms + 1e-10)).tolist()


def process_batch(vs, embeddings, batch, embed=True):
    try:
        # ⚙️ Bỏ các chunk rỗng
        batch = [d for d in batch if d.page_content and d.page_content.strip()]
        if not batch:
            return 0

        # 🔹 Embed hoặc tạo dummy vector
        if embed:
            vectors = embeddings.embed_documents([d.page_content for d in batch])
            vectors = _normalize(vectors)   # ✅ tự normalize lại
            fixed_vectors = []
            for idx, v in enumerate(vectors):
                arr = np.array(v).flatten()  # đảm bảo 1D
                if arr.ndim != 1 or len(arr) != 768:  # dimension phải khớp schema
                    raise ValueError(f"❌ Vector dimension mismatch tại doc {idx}: {arr.shape}")
                fixed_vectors.append(arr.tolist())
            vectors = fixed_vectors
        else:
            vectors = [[0.0] * 768 for _ in batch]  # dummy vector đúng dim

        # 🔹 Chuẩn bị payload
        payload = []
        for i, d in enumerate(batch):
            meta = d.metadata
            page_val = meta.get("page", -1)
            if page_val in [None, -1]:
                page_val = i + 1

            payload.append([
                vectors[i],
                page_val,
                meta.get("subject", ""),
                meta.get("course_id", ""),
                d.page_content
            ])

        # 🔹 Kết nối Milvus (fallback nếu cần)
        collection = getattr(vs, "_collection", None)
        if collection is None:
            _connect_once()
            collection = Collection(vs.collection_name)
            print(f"⚙️ Fallback: connected directly to collection {vs.collection_name}")

        # 🔹 Chuẩn hóa dữ liệu chèn
        embeddings_list = [r[0] for r in payload]
        pages = [r[1] for r in payload]
        subjects = [r[2] for r in payload]
        courses = [r[3] for r in payload]
        texts = [r[4] for r in payload]

        # 🔹 Partition theo subject (optional)
        partition = f"p_{subjects[0]}" if subjects else None
        try:
            if partition and partition not in [p.name for p in collection.partitions]:
                collection.create_partition(partition)
        except Exception:
            pass  # partition có thể đã tồn tại

        # 🔹 Kiểm tra dimension trước insert
        for v in embeddings_list:
            if len(v) != 768:
                raise ValueError(f"❌ Embedding vector invalid length: {len(v)}")

        # 🔹 Insert vào Milvus
        collection.insert(
            [embeddings_list, pages, subjects, courses, texts],
            partition_name=partition if partition else None
        )

        return len(batch)

    except Exception as e:
        print(f"❌ Batch insert error: {e}\n{traceback.format_exc()}")
        print(f"📦 Inserted {len(batch)} docs into {vs.collection_name}")
        return 0




# ========================= #
# 🚀 Hàm async upsert chính #
# ========================= #
import asyncio
import os, json, traceback
from pymilvus import Collection, connections
from langchain_core.documents import Document

async def upsert_chunks(docs: list[Document], batch_size: int = 512, embed: bool = True):
    """
    ✅ Optimized Async Upsert for Milvus (Expert)
    - Tối ưu tốc độ và ổn định khi chạy song song với LLM pipeline
    - Async-safe, không block event loop
    - Flush 1 lần duy nhất, append corpus async
    """
    vs = get_vectorstore()
    embeddings = get_embeddings()

    # 1️⃣ Làm sạch metadata
    clean_docs = []
    for d in docs:
        if not d or not d.page_content.strip():
            continue

        meta = d.metadata.copy()
        meta["subject"] = meta.get("subject", "").lower().strip()
        meta["course_id"] = meta.get("course_id", "").lower().strip()

        # ✅ Fix auto page (fallback cho -1)
        try:
            page_val = int(meta.get("page", -1))
            if page_val == -1:
                page_val = len(clean_docs) + 1
            meta["page"] = page_val
        except Exception:
            meta["page"] = len(clean_docs) + 1

        meta.pop("id", None)
        meta.pop("embedding", None)
        clean_docs.append(Document(page_content=d.page_content, metadata=meta))

    total = len(clean_docs)
    if total == 0:
        print("⚠️ Không có tài liệu hợp lệ để index.")
        return 0

    print(f"📥 Upserting {total} docs (batch_size={batch_size}) ...")

    # 2️⃣ Chia batch & xử lý song song
    batches = [clean_docs[i:i+batch_size] for i in range(0, total, batch_size)]

    async def insert_batch(b):
        """Chạy process_batch trong thread pool (vì Milvus client là sync)."""
        return await asyncio.to_thread(process_batch, vs, embeddings, b, embed)

    results = await asyncio.gather(*[insert_batch(b) for b in batches])
    inserted = sum(results)

    print(f"✅ Indexed total {inserted}/{total} docs using {len(batches)} batches")

    # 3️⃣ ✅ Flush một lần sau toàn bộ insert
    try:
        collection = getattr(vs, "_collection", None)
        if collection is None:
            _connect_once()
            collection = Collection(vs.collection_name)

        # 🧩 Chỉ flush nếu collection đã có index
        if collection.has_index():
            await asyncio.to_thread(collection.flush)
            print("💾 Flushed collection successfully (once).")
        else:
            print("⚠️ Skip flush: collection has no index yet.")
    except Exception as e:
        print(f"⚠️ Flush skipped: {e}")


    # 4️⃣ ✅ Ghi corpus cache song song để không chặn event loop
    async def append_corpus_async():
        try:
            await asyncio.to_thread(_append_to_corpus, clean_docs)
            print(f"🧾 Saved {len(clean_docs)} docs vào corpus.jsonl")
        except Exception as e:
            print(f"⚠️ Không thể ghi corpus cache: {e}")

    asyncio.create_task(append_corpus_async())

    print("🏁 Upsert done.\n")
    return inserted

def _get_collection() -> Collection:
    global _COLLECTION
    _connect_once()
    if _COLLECTION is None:
        if not utility.has_collection(COLLECTION_NAME):
            raise RuntimeError(f"Collection not found: {COLLECTION_NAME}")
        _COLLECTION = Collection(COLLECTION_NAME)
        # Load 1 lần; Milvus sẽ quản lý cache bộ nhớ
        try:
            _COLLECTION.load()
        except Exception as e:
            logging.warning(f"[milvus] load() warn: {e}")
        logging.info(f"[milvus] ready collection={COLLECTION_NAME}")
    return _COLLECTION

# ========= Embedding helpers =========
def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n > 0:
        return (vec / n).astype(np.float32)
    return vec.astype(np.float32)

def _ensure_vec(x) -> np.ndarray:
    arr = np.array(x, dtype=np.float32).reshape(-1)
    if arr.shape[0] != EMBED_DIM:
        raise ValueError(f"❌ Embedding dim mismatch: got {arr.shape[0]}, expect {EMBED_DIM}")
    return _normalize(arr)

# ========= Public API =========
def build_expr(where: Optional[Dict[str, Any]]) -> Optional[str]:
    if not where:
        return None
    parts = []
    subj = (where.get("subject") or "").strip().lower()
    cid  = (where.get("course_id") or "").strip().lower()
    if subj:
        parts.append(f'subject == "{subj}"')
    if cid:
        parts.append(f'course_id == "{cid}"')
    # if topic:
    #     parts.append(f'lower(topic) == "{topic}"')  # chỉ nếu bạn có field "topic"
    return " and ".join(parts) if parts else None

def _log_milvus_ctx(col, k, ef, expr):
    try:
        ver = utility.get_server_version()
        n = col.num_entities
        prog = utility.loading_progress(col.name)
        idx_state = utility.index_building_progress(col.name)
    except Exception as e:
        ver, n, prog, idx_state = f"unreachable:{e}", "?", {}, {}

    logging.error(
        "[milvus][SEARCH_ERR] host=%s port=%s coll=%s ver=%s rows=%s k=%d ef=%d expr=%s idx=%s load=%s build=%s",
        os.getenv("MILVUS_HOST","localhost"), os.getenv("MILVUS_PORT","19530"),
        col.name, ver, n, k, ef, expr or "{}", [ix.index_name for ix in col.indexes],
        prog, idx_state
    )
    try:
        import psutil  # <- chỉ psutil, KHÔNG import os ở đây
        rss = psutil.Process(os.getpid()).memory_info().rss/1e6
        logging.error("[milvus][SEARCH_ERR] app_mem_rss=%.1fMB", rss)
    except Exception:
        pass


async def similarity_search(
    query: str,
    k: int = 6,
    where: Optional[dict] = None,
    threshold: float = 0.5,   # giữ tham số cũ (không hard filter)
    log: bool = True,
    embed_fn=None,            # nếu muốn truyền hàm embed sẵn có
) -> List[Tuple[Document, float]]:
    """
    Giữ nguyên hành vi trước: trả [(Document, score)].
    - Không connect/load mỗi lần nữa (cache).
    - Không thay đổi k/logic lọc; chỉ tối ưu chuẩn hoá vector & expr.
    """
    if not query or not query.strip():
        return []

    col = _get_collection()

    # Lấy embedding query từ embed_fn bên ngoài (giữ logic hiện tại)
    if embed_fn is None:
        # Reuse hàm có sẵn của bạn nếu muốn
        from .embedding import get_embeddings
        embeddings = get_embeddings()
        qv = embeddings.embed_query(query)
    else:
        qv = embed_fn(query)

    qv = _ensure_vec(qv)

    expr = build_expr(where)
    if log and expr:
        logging.info(f"[milvus] expr={expr}")

    # ef = max(128, k*2) → GIỮ logic/params như trước
    ef_value = max(32, k * 2)
    search_params = {"metric_type": METRIC_TYPE, "params": {"ef": ef_value}}

    try:
        t0 = time.time()
        res = col.search(
            data=[qv.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["page","subject","course_id","text"],
            timeout=8.0,               # <- đặt timeout rõ
        )
        logging.info("[milvus] search OK in %.3fs, hits=%d",time.time()-t0, len(res[0]) if res else 0)
    except Exception as e:
        _log_milvus_ctx(col, k, ef_value, expr)
        logging.exception("[milvus] search FAIL after %.3fs: %s",time.time()-t0, e)
        raise
    
    out: List[Tuple[Document, float]] = []
    hits = res[0] if res else []
    for hit in hits:
        meta = {
            "page": hit.entity.get("page"),
            "subject": hit.entity.get("subject"),
            "course_id": hit.entity.get("course_id"),
        }
        # Milvus với COSINE: `distance` là 1 - cosine_sim (tuỳ thiết lập),
        # bạn đã log 1-hit.distance trước đó → giữ nguyên cách tính nếu cần
        score = hit.distance
        doc = Document(page_content=hit.entity.get("text", ""), metadata=meta)
        out.append((doc, score))

    if log:
        logging.info(f"🔎 [milvus] q='{query[:80]}' k={k} ef={ef_value} → {len(out)} hits")

    return out

# ========= Warmup (gọi ở app start nếu muốn) =========
def warmup_vectorstore():
    try:
        _ = _get_collection()
        logging.info("[milvus] warmup done")
    except Exception as e:
        logging.warning(f"[milvus] warmup skipped: {e}")


async def get_all_docs(limit: int = None, batch_size: int = 5000):
    from app.services.milvus_client import get_milvus_client
    client = get_milvus_client()
    all_docs = []
    offset = 0

    while True:
        batch = await client.query(
            collection_name="smart_learning",
            filter=None,
            output_fields=["page_content", "metadata"],
            limit=batch_size,
            offset=offset
        )
        if not batch:
            break
        all_docs.extend([
            Document(page_content=r["page_content"], metadata=r["metadata"])
            for r in batch
        ])
        offset += batch_size
        if limit and len(all_docs) >= limit:
            break

    return all_docs[:limit] if limit else all_docs



# ---------- BM25 Corpus Helpers ----------
def _append_to_corpus(docs: List[Document]):
    with open(CORPUS_FILE, "a", encoding="utf-8") as f:
        for d in docs:
            rec = {"text": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

