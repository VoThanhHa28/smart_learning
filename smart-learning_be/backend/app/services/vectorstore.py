import os
from typing import List, Optional
import pinecone
from langchain_pinecone import Pinecone
from langchain_core.documents import Document
from .embedding import get_embeddings
from pinecone import ServerlessSpec
from collections import defaultdict

INDEX_NAME = os.getenv("PINECONE_INDEX", "smart-learning")


_pc = None
_vs = None

def init_pinecone():
    global _pc
    if _pc is not None:
        return _pc
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("⚠️ PINECONE_API_KEY chưa được set trong .env")
    _pc = pinecone.Pinecone(api_key=api_key)
    # chỉ check tạo index 1 lần
    if INDEX_NAME not in [idx.name for idx in _pc.list_indexes()]:
        dim = len(get_embeddings().embed_query("test"))
        _pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return _pc

def get_vectorstore():
    global _vs
    if _vs is not None:
        return _vs
    pc = init_pinecone()
    embeddings = get_embeddings()
    _vs = Pinecone.from_existing_index(INDEX_NAME, embeddings)
    return _vs

def upsert_chunks(docs: List[Document]):
    vs = get_vectorstore()
    buckets = defaultdict(list)
    for d in docs:
        ns = (d.metadata.get("subject") or "general").lower()
        buckets[ns].append(d)
    for ns, group in buckets.items():
        vs.add_documents(group, namespace=ns)  # ✅ ghi theo namespace
    print(f"DEBUG added {sum(len(g) for g in buckets.values())} docs vào Pinecone index {INDEX_NAME}")
    return sum(len(g) for g in buckets.values())


def similarity_search(query: str, k: int = 6, where: Optional[dict] = None):
    vs = get_vectorstore()
    ns, filt = _split_ns(where)
    return vs.similarity_search_with_score(query, k=k, filter=filt, namespace=ns)


def _split_ns(where: dict | None):
    ns = None
    if where and "subject" in where:
        ns = str(where["subject"]).lower()
        where = {k: v for k, v in where.items() if k != "subject"}
    return ns, where

def get_all_docs(subject: Optional[str] = None) -> List[Document]:
    vs = get_vectorstore()
    if subject:
        return vs.similarity_search("dummy", k=10000, filter={"subject": subject})
    return vs.similarity_search("dummy", k=10000)
