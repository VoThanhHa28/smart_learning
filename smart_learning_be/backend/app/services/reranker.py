import os
from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# model base (nhẹ, đa ngôn ngữ, phù hợp GPU/CPU)
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# cache model để không load lại nhiều lần
_cross_encoder = None

def get_reranker():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL, device="cuda")  # GPU 4050
    return _cross_encoder

def rerank(query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
    if not docs:
        return []
    model = get_reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)
    # attach scores vào docs
    for d, s in zip(docs, scores):
        d.metadata["rerank_score"] = float(s)
    # sort by score desc
    reranked = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
    return reranked[:top_n]
