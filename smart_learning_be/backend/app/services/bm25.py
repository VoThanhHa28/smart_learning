# bm25.py
import json, os
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from .config import USE_BM25

CORPUS_FILE = os.path.join(os.path.dirname(__file__), "corpus.jsonl")
# _bm25 = None  # cache
_bm25_cache: dict[str, BM25Retriever] = {}

def load_corpus() -> List[Document]:
    docs = []
    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    docs.append(Document(page_content=item["text"], metadata=item["metadata"]))
    return docs



def get_bm25_retriever(k: int = 12, subject: str | None = None) -> BM25Retriever | None:
    if not USE_BM25:
        return None
    key = (subject or "__ALL__").lower()
    if key not in _bm25_cache:
        docs = load_corpus() if subject is None else [d for d in load_corpus() if d.metadata.get("subject") == subject]
        if not docs:
            raise ValueError(f"❌ Không có tài liệu cho subject={subject}")
        _bm25_cache[key] = BM25Retriever.from_documents(docs)
    _bm25_cache[key].k = k
    return _bm25_cache[key]
