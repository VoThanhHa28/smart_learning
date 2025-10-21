# app/services/hybrid.py
import asyncio
from typing import List, Tuple, Literal, overload
import hashlib
import time
import logging

from langchain_core.documents import Document

from .vectorstore import similarity_search
from .bm25 import get_bm25_retriever
from .config import USE_BM25

def _hash_doc(d: Document) -> str:
    key = (d.page_content or "") + "|" + str(d.metadata.get("page", ""))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def rrf_merge(dense: List[Document], sparse: List[Document],
              k: int = 60, top: int = 30) -> List[Document]:
    # ⚠️ Giữ nguyên thuật toán/params — chỉ tối ưu nhỏ trong vòng lặp
    scores = {}

    def add_scores(docs: List[Document]):
        for rank, d in enumerate(docs):
            h = _hash_doc(d)
            # cộng dồn theo công thức RRF (không đổi)
            scores[h] = scores.get(h, 0.0) + 1.0 / (k + rank + 1)

    add_scores(dense)
    add_scores(sparse)

    pool = {}
    for d in dense + sparse:
        h = _hash_doc(d)
        if h not in pool:
            pool[h] = d
        else:
            # ✅ Merge metadata thay vì ghi đè (giữ nguyên hành vi cũ)
            old_meta = pool[h].metadata or {}
            new_meta = d.metadata or {}
            merged_meta = {**old_meta, **{mk: mv for mk, mv in new_meta.items() if mv not in [None, ""]}}
            pool[h].metadata = merged_meta

    # sort theo điểm RRF (không đổi)
    merged = sorted(pool.items(), key=lambda kv: scores.get(kv[0], 0.0), reverse=True)
    return [kv[1] for kv in merged[:top]]

@overload
async def hybrid_retrieve(
    query: str,
    k_dense: int = 12,
    k_sparse: int = 12,
    top_after_rrf: int = 30,
    filters=None,
    return_raw: Literal[True] = True,
) -> Tuple[List[Document], List[Document], List[Document]]: ...
@overload
async def hybrid_retrieve(
    query: str,
    k_dense: int = 12,
    k_sparse: int = 12,
    top_after_rrf: int = 30,
    filters=None,
    return_raw: Literal[False] = False,
) -> List[Document]: ...

async def hybrid_retrieve(  # implementation
    query: str,
    k_dense: int = 12,
    k_sparse: int = 12,
    top_after_rrf: int = 30,
    filters=None,
    return_raw: bool = False
):
    """
    Hybrid search song song (dense + BM25), giữ nguyên logic:
    - Không đổi k/top_after_rrf.
    - Không cắt/bỏ chunk.
    - Thêm tối ưu không block event loop và logging gọn.
    """
    if not query or not query.strip():
        logging.info("🔕 [hybrid] Empty query — skip dense/sparse")
        if return_raw:
            return ([], [], [])
        return []

    # copy filters để tránh mutate từ bên ngoài (giữ hành vi đã sửa)
    filters = dict(filters or {})
    subj = (filters.get("subject") or "").lower().strip()
    cid  = (filters.get("course_id") or "").lower().strip()

    t0 = time.time()

    async def dense_search():
        # similarity_search là async → không block, giữ nguyên tham số
        return await similarity_search(query, k=k_dense, where=filters, threshold=0.5, log=False)

    async def sparse_search():
        if not USE_BM25:
            return []
        # retriever đã cache theo (subject, course_id)
        bm25 = get_bm25_retriever(k=k_sparse, subject=subj or None, course_id=cid or None)
        if not bm25:
            return []
        # BM25.get_relevant_documents là sync → đưa vào thread để không block loop
        return await asyncio.to_thread(bm25.get_relevant_documents, query)

    # chạy song song 2 nguồn
    dense_results, sparse_docs = await asyncio.gather(dense_search(), sparse_search())

    # giữ nguyên: lấy docs từ dense (list[(doc, score)]) → list[doc]
    dense_docs: List[Document] = [d for d, _s in dense_results]

    # RRF giữ nguyên công thức/params
    merged = rrf_merge(dense_docs, sparse_docs, k=60, top=top_after_rrf)

    # ✅ Lưới an toàn: hậu kiểm theo filter (không đổi logic trước đó)
    def _enforce_filters(docs: List[Document]) -> List[Document]:
        if not subj and not cid:
            return docs
        out = []
        ls, lc = subj, cid
        for d in docs:
            m = d.metadata or {}
            if ls and (str(m.get("subject", "")).lower().strip() != ls):
                continue
            if lc and (str(m.get("course_id", "")).lower().strip() != lc):
                continue
            out.append(d)
        return out

    merged = _enforce_filters(merged)

    dur = (time.time() - t0) * 1000.0
    logging.info("🔎 [hybrid] dense=%d sparse=%d merged=%d top=%d ⏱ %.1fms",
                 len(dense_docs), len(sparse_docs), len(merged), top_after_rrf, dur)

    if return_raw:
        sparse_docs = _enforce_filters(sparse_docs)
        dense_docs  = _enforce_filters(dense_docs)
        return (merged, sparse_docs, dense_docs)
    return merged
