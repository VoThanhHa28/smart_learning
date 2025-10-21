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



def get_bm25_retriever(k: int = 12, subject: str | None = None, course_id: str | None = None) -> BM25Retriever | None:
    if not USE_BM25:
        return None
    key = f"{(subject or '__ALL__').lower()}::{(course_id or '__ALL__').lower()}"
    if key not in _bm25_cache:
        all_docs = load_corpus()
        if subject:
            all_docs = [d for d in all_docs if (d.metadata.get("subject","").lower()==subject.lower())]
        if course_id:
            all_docs = [d for d in all_docs if (d.metadata.get("course_id","").lower()==course_id.lower())]
        if not all_docs:
            # Có thể trả None thay vì raise nếu muốn “BM25 optional”
            return None
        _bm25_cache[key] = BM25Retriever.from_documents(all_docs)
    _bm25_cache[key].k = k
    return _bm25_cache[key]
