import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

def get_client():
    persist_dir = os.getenv("CHROMA_DIR", "./chroma_data")
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.Client(Settings(is_persistent=True, persist_directory=persist_dir))

def collection_name(doc_id: str) -> str:
    # để mỗi doc 1 collection riêng (đơn giản, dễ dọn dẹp)
    return f"doc_{doc_id}"

def upsert_chunks(doc_id: str, chunks: List[str], metadatas: List[Dict[str, Any]]):
    client = get_client()
    coll = client.get_or_create_collection(collection_name(doc_id))
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    coll.upsert(documents=chunks, metadatas=metadatas, ids=ids)

def query(doc_id: str, query_text: str, n_results: int = 5):
    client = get_client()
    coll = client.get_or_create_collection(collection_name(doc_id))
    res = coll.query(query_texts=[query_text], n_results=n_results)

    docs = res.get("documents")
    dists = res.get("distances")
    metas = res.get("metadatas")

    # Nếu bất kỳ cái nào None hoặc rỗng → trả [] luôn
    if not docs or not docs[0]:
        return []

    if not dists or not dists[0]:
        dists = [[0.0] * len(docs[0])]  # fallback safe

    if not metas or not metas[0]:
        metas = [[{}] * len(docs[0])]  # fallback safe

    return [
        {"document": doc, "score": dist, "metadata": md}
        for doc, dist, md in zip(docs[0], dists[0], metas[0])
    ]

