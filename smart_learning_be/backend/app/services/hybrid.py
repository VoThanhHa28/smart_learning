# hybrid.py
from typing import List, Tuple
import hashlib
import time
from .config import USE_BM25, EMBED_MODEL
from langchain_core.documents import Document
from .vectorstore import get_vectorstore, similarity_search
from .bm25 import get_bm25_retriever

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

    merged = sorted(pool.items(), key=lambda kv: scores.get(kv[0], 0.0), reverse=True)
    return [kv[1] for kv in merged[:top]]

def hybrid_retrieve(query: str, k_dense: int = 12, k_sparse: int = 12,
                    top_after_rrf: int = 30, filters=None,
                    return_raw: bool = False) -> List[Document] | Tuple[List[Document], List[Document], List[Document]]:
    # ---- Dense ----
    expr = None
    if filters:
        expr_clauses = []
        if "subject" in filters:
            expr_clauses.append(f'subject == "{filters["subject"]}"')
        if "course_id" in filters:
            expr_clauses.append(f'course_id == "{filters["course_id"]}"')
        expr = " and ".join(expr_clauses) if expr_clauses else None

    t0 = time.time()
    dense_results = similarity_search(query, k=k_dense, where=filters, threshold=0.75, log=True)
    dense_docs = [d for d, s in dense_results]
    t1 = time.time()
    print(f"⏱ Dense search ({k_dense}) → {len(dense_docs)} docs trong {t1 - t0:.2f}s")
    for d in dense_docs[:3]:
        print("DENSE DEBUG:", d.metadata, d.page_content[:80], "...")

    # ---- BM25 ----
    sparse_docs = []
    if USE_BM25:
        bm25 = get_bm25_retriever(k=k_sparse, subject=filters.get("subject") if filters else None)
        if bm25:
            t2 = time.time()
            sparse_docs = bm25.get_relevant_documents(query)
            t3 = time.time()
            print(f"⏱ BM25 search ({k_sparse}) → {len(sparse_docs)} docs trong {t3 - t2:.2f}s")
            for d in sparse_docs[:3]:
                print("BM25 DEBUG:", d.metadata, d.page_content[:80], "...")

    # ---- RRF merge ----
    merged = rrf_merge(dense_docs, sparse_docs, k=60, top=top_after_rrf)
    print(f"👉 Sau RRF merge: {len(merged)} docs, lấy top {top_after_rrf}")

    return (merged, sparse_docs, dense_docs) if return_raw else merged
