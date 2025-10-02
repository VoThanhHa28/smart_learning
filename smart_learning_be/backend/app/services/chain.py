import os
import time
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate    # Ollama
from .vectorstore import get_vectorstore
from .llm import get_llm
from .hybrid import hybrid_retrieve
from app.services.reranker import rerank as heavy_rerank
import json

from app.services.config import (
    RERANKER_MODE,
    FINAL_TOP_N,
    LIGHT_RERANKER_MODEL,
    LIGHT_RERANKER_TOPK,
)
from app.services.llm import get_llm

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT = load_prompt("prompts/system_prompt.txt")
APPENDIX = load_prompt("prompts/appendix_examples.txt")

USE_APPENDIX = os.getenv("USE_APPENDIX", "false").lower() == "true"

if USE_APPENDIX:
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT + "\n\n" + APPENDIX)
else:
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)

# Biến thể ngắn gọn (A)
QA_PROMPT_COMPACT = PromptTemplate.from_template(
    QA_PROMPT.template.replace("2–5 câu", "tối đa 3 câu")
)

# Biến thể đầy đủ (B)
QA_PROMPT_EXTENDED = PromptTemplate.from_template(
    QA_PROMPT.template.replace("2–5 câu", "5–7 câu").replace(
        "Xuất kết quả theo Markdown",
        "Nếu người hỏi mở rộng, có thể thêm mục **Bối cảnh liên quan**.\nXuất kết quả theo Markdown"
    )
)

def dump_docs(tag, docs):
    print(f"\n=== {tag} ({len(docs)}) ===")
    print(json.dumps([
        {"page": d.metadata.get("page"), 
         "course_id": d.metadata.get("course_id"),
         "score": d.metadata.get("score"),
         "text": d.page_content[:300]} 
        for d in docs
    ], ensure_ascii=False, indent=2))

def build_context(docs: List[Document], limit_chars=5000):
    parts = []
    for d in docs:
        pg = d.metadata.get("page")
        sec = d.metadata.get("section")
        tag = f"[p:{pg or '-'}|{sec or '-'}]"
        parts.append(f"{tag} {d.page_content}")
    ctx = "\n\n".join(parts)
    return ctx[:limit_chars]

def retrieve_answer(
    question: str,
    k: int = 15,
    filters: dict | None = None,
    subject: str = "General",
    threshold: float = 0.35,   # 👈 threshold để lọc context kém
) -> Tuple[str, List[Document]]:
    t0 = time.time()

    # --- Step 1: Hybrid retrieve ---
    merged_docs, bm25_docs, dense_docs = hybrid_retrieve(
        question,
        k_dense=12,
        k_sparse=24,
        top_after_rrf=50,
        filters=filters,
        return_raw=True
    )
    print(f"\n🔎 [Hybrid] BM25={len(bm25_docs)}, Dense={len(dense_docs)}, "
          f"Merged(RRF)={len(merged_docs)}")
    
    if not merged_docs:
        return "(Không tìm thấy ngữ cảnh phù hợp. Hãy kiểm tra bộ lọc hoặc index.)", []

    # --- Step 2: Rerank ---
    if RERANKER_MODE == "off":
        reranked_docs = merged_docs[:FINAL_TOP_N]
    elif RERANKER_MODE == "base":
        reranked_docs = heavy_rerank(question, merged_docs, top_n=FINAL_TOP_N)
        print("\n✅ [Reranker] Chọn top-N sau rerank:")
        for d in reranked_docs:
            print(" -", d.page_content[:120], d.metadata)
    else:
        reranked_docs = merged_docs[:FINAL_TOP_N]

    # --- Step 2.5: Threshold filter ---
    # Nếu reranker trả score, bạn có thể lọc tại đây. Với similarity search, cần return score cùng docs.
    # Giả sử bạn mở rộng vectorstore để trả (doc, score), thì filter như sau:
    # reranked_docs = [d for d, s in reranked_with_score if s >= threshold]

    dump_docs("BM25", bm25_docs)
    dump_docs("Dense", dense_docs)
    dump_docs("Merged", merged_docs)
    dump_docs("Reranked", reranked_docs)

    if not reranked_docs:
        return "(Các tài liệu thu được đều dưới ngưỡng chất lượng, không thể trả lời chính xác.)", []

    # --- Step 3: Build context ---
    context = build_context(reranked_docs)
    print(f"\n📚 [Context] {len(reranked_docs)} docs, length={len(context)} chars")
    print(context[:1000], "...\n")

    # --- Step 4: Query LLM ---
    llm = get_llm()
    resp = llm.invoke(QA_PROMPT.format(question=question, context=context, subject=subject))
    t1 = time.time()
    print(f"⏱ Total pipeline: {t1 - t0:.2f}s")

    return resp.content.strip(), reranked_docs
