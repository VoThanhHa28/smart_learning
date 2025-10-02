import os
import json
from typing import List, Optional
# from langchain_milvus import Milvus
from .test_milvus import DebugMilvus  
from langchain_core.documents import Document
from .embedding import get_embeddings
import traceback
from pymilvus import connections, Collection
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "smart_learning")
BASE_DIR = os.path.dirname(__file__)   # thư mục hiện tại (app/services)
CORPUS_FILE = os.getenv("CORPUS_FILE", os.path.join(BASE_DIR, "corpus.jsonl"))


_vs = None

def get_vectorstore():
    global _vs
    if _vs is not None:
        return _vs
    embeddings = get_embeddings()

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 200, "efConstruction": 200}
    }

    _vs = DebugMilvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": os.getenv("MILVUS_HOST", "milvus"), "port": "19530"},
        index_params=index_params,
        auto_id=True,
        text_field="text",
        vector_field="embedding",
    )
    return _vs

def upsert_chunks(docs: List[Document], batch_size: int = 500):
    vs = get_vectorstore()

    # 🔑 clean metadata
    safe_docs = []
    for d in docs:
        clean_meta = {k: v for k, v in d.metadata.items() if k != "id"}
        safe_docs.append(Document(page_content=d.page_content, metadata=clean_meta))

    for i in range(0, len(safe_docs), batch_size):
        batch = safe_docs[i:i+batch_size]
        vs.add_documents(batch)
        print(f"✅ Indexed {i+len(batch)}/{len(safe_docs)} docs")

    _append_to_corpus(safe_docs)
    return len(safe_docs)


def similarity_search(query: str, k: int = 6, where: Optional[dict] = None,
                      threshold: float = 0.75, log: bool = True):

    # 1. Kết nối
    connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port="19530")

    # 2. Lấy collection
    c = Collection("smart_learning")
    c.load()

    # 3. Encode query -> vector
    embeddings = get_embeddings()
    query_vec = embeddings.embed_query(query)

    # 4. Build expr filter
    expr = None
    if where:
        clauses = []
        if "subject" in where:
            clauses.append(f'subject == "{where["subject"]}"')
        if "course_id" in where:
            clauses.append(f'course_id == "{where["course_id"]}"')
        expr = " and ".join(clauses) if clauses else None

    # 5. Search trực tiếp
    ef_value = max(128, k * 2)
    search_params = {"metric_type": "COSINE", "params": {"ef": ef_value}}

    results = c.search(
        data=[query_vec],
        anns_field="embedding",
        param=search_params,
        limit=k,
        expr=expr,
        output_fields=["page", "subject", "course_id", "text"]
    )

    # 6. Parse kết quả
    docs = []
    for hit in results[0]:
        meta = {f: hit.entity.get(f) for f in ["page", "subject", "course_id"]}
        sim = 1 - hit.distance  # vì COSINE trong Milvus trả về distance
        docs.append((Document(page_content=hit.entity.get("text"), metadata=meta), sim))

    if threshold is not None:
        docs = [(d, s) for d, s in docs if s >= threshold]

    if log:
        print(f"\n🔎 [Direct Milvus] query='{query}', ef={ef_value}, k={k}, results={len(docs)}")
        for d, s in docs[:5]:
            print(f" - score={s:.3f} page={d.metadata.get('page')} {d.page_content[:100]}...")

    return docs




# ---------- BM25 Corpus Helpers ----------
def _append_to_corpus(docs: List[Document]):
    with open(CORPUS_FILE, "a", encoding="utf-8") as f:
        for d in docs:
            rec = {"text": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

