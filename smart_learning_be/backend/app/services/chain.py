import os
from fastapi.responses import Response
import time
import json
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from app.services.vectorstore import get_vectorstore
from app.services.llm import get_llm
from app.services.hybrid import hybrid_retrieve
from app.services.reranker import rerank as heavy_rerank
from app.services.rag_graph import graph, State
from app.services.prompt_utils import QA_PROMPT
from app.services.config import (
    RERANKER_MODE,
    FINAL_TOP_N,
    LIGHT_RERANKER_MODEL,
    LIGHT_RERANKER_TOPK,
)



# ================== #
# 🧩 Helper Functions #
# ================== #
def dump_docs(tag: str, docs: List[Document]):
    """🪶 In log gọn gàng các docs (debug retrieval pipeline)."""
    print(f"\n=== {tag} ({len(docs)}) ===")
    formatted = []
    for d in docs:
        meta = d.metadata
        formatted.append({
            "page": meta.get("page"),
            "headings": meta.get("headings"),
            "course_id": meta.get("course_id"),
            "score": meta.get("score"),
            "filename": os.path.basename(str(meta.get("origin", ""))),
            "text": d.page_content[:300].replace("\n", " "),
        })
    print(json.dumps(formatted, ensure_ascii=False, indent=2))

# ================== #
# 🚀 Main RAG Chain  #
# ================== #

async def retrieve_answer(
    question: str,
    subject: str = "General",
    course_id: str | None = None,
    k: int = 6,
    filters: dict | None = None,
):
    start = time.time()

    # ✅ Truyền đủ thông tin cho graph
    result = await graph.ainvoke({ #type: ignore
        "question": question,
        "subject": subject,
        "course_id": course_id,
    })

    print(f"⏱ Total LangGraph pipeline: {time.time() - start:.2f}s")
    return result["answer"], result["context"]
