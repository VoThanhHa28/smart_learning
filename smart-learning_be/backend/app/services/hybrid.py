# hybrid.py
from typing import List, Tuple
import hashlib

from langchain_core.documents import Document
from .vectorstore import get_vectorstore
from .bm25 import get_bm25_retriever

def _hash_doc(d: Document) -> str:
    # khử trùng lặp theo nội dung + page (ổn định)
    key = (d.page_content or "") + "|" + str(d.metadata.get("page", ""))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def rrf_merge(dense: List[Document], sparse: List[Document], k: int = 60, top: int = 30) -> List[Document]:
    """
    Reciprocal Rank Fusion: score = Σ(1 / (k + rank))
    """
    scores = {}
    def add_scores(docs: List[Document]):
        for rank, d in enumerate(docs):
            h = _hash_doc(d)
            scores[h] = scores.get(h, 0.0) + 1.0 / (k + rank + 1)

    add_scores(dense)
    add_scores(sparse)

    # giữ 1 bản duy nhất theo hash, lấy doc từ dense trước rồi sparse
    pool = {}
    for d in dense + sparse:
        h = _hash_doc(d)
        if h not in pool:
            pool[h] = d

    # sort theo RRF
    merged = sorted(pool.items(), key=lambda kv: scores.get(kv[0], 0.0), reverse=True)
    docs_sorted = [kv[1] for kv in merged[:top]]
    return docs_sorted

def hybrid_retrieve(query: str, k_dense: int = 12, k_sparse: int = 12,
                    top_after_rrf: int = 30, filters=None) -> List[Document]:
    # Dense retriever
    dense_retriever = get_vectorstore().as_retriever(
        search_kwargs={"k": k_dense, "filter": filters}
    )
    dense_docs = dense_retriever.get_relevant_documents(query)

    # Sparse (BM25)
    bm25 = get_bm25_retriever(k=k_sparse, subject=filters.get("subject") if filters else None)
    sparse_docs = bm25.get_relevant_documents(query) if bm25 else []


    # Áp dụng filter thủ công cho BM25
    if filters and "subject" in filters:
        sparse_docs = [d for d in sparse_docs if d.metadata.get("subject") == filters["subject"]]

    # RRF + dedup
    merged = rrf_merge(dense_docs, sparse_docs, k=60, top=top_after_rrf)
    return merged
