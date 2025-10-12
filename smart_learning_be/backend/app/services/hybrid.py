import asyncio
from typing import List, Tuple
import hashlib
import time
from langchain_core.documents import Document
from .vectorstore import similarity_search
from .bm25 import get_bm25_retriever
from .config import USE_BM25


def _hash_doc(d: Document) -> str:
    key = (d.page_content or "") + "|" + str(d.metadata.get("page", ""))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def rrf_merge(dense: List[Document], sparse: List[Document],
              k: int = 60, top: int = 30) -> List[Document]:
    scores = {}

    def add_scores(docs: List[Document]):
        for rank, d in enumerate(docs):
            h = _hash_doc(d)
            scores[h] = scores.get(h, 0.0) + 1.0 / (k + rank + 1)

    add_scores(dense)
    add_scores(sparse)

    pool = {}
    for d in dense + sparse:
        h = _hash_doc(d)
        if h not in pool:
            pool[h] = d
        else:
            # ✅ Merge metadata thay vì ghi đè
            old_meta = pool[h].metadata or {}
            new_meta = d.metadata or {}
            merged_meta = {**old_meta, **{k: v for k, v in new_meta.items() if v not in [None, ""]}}
            pool[h].metadata = merged_meta

    merged = sorted(pool.items(), key=lambda kv: scores.get(kv[0], 0.0), reverse=True)
    return [kv[1] for kv in merged[:top]]



async def hybrid_retrieve(
    query: str,
    k_dense: int = 12,
    k_sparse: int = 12,
    top_after_rrf: int = 30,
    filters=None,
    return_raw: bool = False
) -> Tuple[List[Document], List[Document], List[Document]]:

    async def dense_search():
        # Gọi Milvus async search
        return await similarity_search(query, k=k_dense, where=filters, threshold=0.75, log=True)

    if not query or not query.strip():
        print("🔕 Empty query — skipping dense/sparse search")
        return [] 

    async def sparse_search():
        # BM25 là sync => nên chạy trong threadpool
        if not USE_BM25:
            return []
        bm25 = get_bm25_retriever(k=k_sparse, subject=filters.get("subject") if filters else None)
        if not bm25:
            return []
        return await asyncio.to_thread(bm25.get_relevant_documents, query)

    # ✅ Chạy song song cả 2
    dense_results, sparse_docs = await asyncio.gather(
        dense_search(),
        sparse_search()
    )

    dense_docs = [d for d, s in dense_results]
    print(f"⏱ Dense search ({k_dense}) → {len(dense_docs)} docs")
    print(f"⏱ BM25 search ({k_sparse}) → {len(sparse_docs)} docs")

    # Merge kết quả
    merged = rrf_merge(dense_docs, sparse_docs, k=60, top=top_after_rrf)
    print(f"👉 Sau RRF merge: {len(merged)} docs, lấy top {top_after_rrf}")

    return (merged, sparse_docs, dense_docs) if return_raw else merged
