import logging
import os
from fastapi.responses import Response
import time
import json
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from ..rag.retrieval.vectorstore import get_vectorstore
from ..infrastructure.llm.llm import get_llm
from ..rag.retrieval.hybrid import hybrid_retrieve
from ..infrastructure.llm.reranker import rerank as heavy_rerank
from ..rag.rag_graph import get_graph
from ..rag.rag_state import State
from .prompts.prompt_utils import QA_PROMPT
from ..core.config import CFG

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
) -> tuple[str, List[Document]]:
    start = time.time()

    final = await get_graph().ainvoke({
        "question": question or "", #type: ignore
        "subject": subject or "General",
        "course_id": course_id,
        "filters": filters or {},
    })

    logging.info("[API] final_state_keys=%s", list(final.keys()))
    if "sub_answers" in final:
        sa = final.get("sub_answers") or {}
        logging.info("[API] sub_answers_count=%d | sample_keys=%s", len(sa), list(sa.keys())[:3])


    # In chẩn đoán có cấu trúc
    diag = final.get("diag") or {}
    if diag:
        try:
            import json as _json
            logging.info("[DIAG] pipeline=%s", _json.dumps(diag, ensure_ascii=False, indent=2))
        except Exception:
            logging.info("[DIAG] pipeline (non-jsonable) = %r", diag)

    answer: str = (final.get("answer") or "").strip()
    docs: List[Document] = final.get("context") or []

    # Nếu chưa có answer, thử per-subq
    if not answer:
        sub_answers = final.get("sub_answers")
        if isinstance(sub_answers, dict) and sub_answers:
            parts = []
            for q, a in sub_answers.items():
                if a:
                    parts.append(f"**{q}**\n{a}")
            answer = "\n\n".join(parts).strip()
            if answer:
                logging.warning("[FALLBACK] using merged sub_answers (reflect probably failed/empty).")

    # Fallback cuối cùng: ghép context
    if not answer:
        if docs:
            snippet = "\n\n".join((d.page_content or "")[:400] for d in docs[:2])
            answer = snippet or "Xin lỗi, hiện chưa tạo được câu trả lời."
            logging.error(
                "[FALLBACK] using context snippets (LLM failure). docs=%d answer_len=%d",
                len(docs), len(answer)
            )
        else:
            answer = "Xin lỗi, hiện chưa tạo được câu trả lời."
            logging.error("[FALLBACK] no docs and no answer — likely LLM + retrieve both failed")
    
    # log full answer (không cắt) để đối chiếu với Postman
    logging.info("[API] FULL_ANSWER_BEGIN\n%s\n[API] FULL_ANSWER_END", answer)

    # (Tuỳ chọn) nếu muốn gộp thêm nguồn per-subq vào docs trả về:
    per_docs = final.get("per_subq_docs") or {}
    if per_docs and isinstance(per_docs, dict):
        # gộp tất cả doc của các subq, dedup theo _sig; tránh phình quá nhiều
        seen = set()
        flat_docs = []
        for sq, lst in per_docs.items():
            for d in (lst or []):
                sig = (d.metadata or {}).get("_sig") or (d.page_content or "")[:120]
                if sig in seen:
                    continue
                seen.add(sig)
                flat_docs.append(d)
        # ưu tiên docs đã có sẵn trong 'docs' rồi mới nối thêm
        if flat_docs:
            docs = (docs or []) + [d for d in flat_docs if d not in docs]

    logging.info("⏱ Total LangGraph pipeline: %.2fs", time.time() - start)
    return answer, docs