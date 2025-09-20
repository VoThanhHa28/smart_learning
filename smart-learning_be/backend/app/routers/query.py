from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, List
from ..services.chain import retrieve_answer

router = APIRouter(tags=["query"])

class QueryRequest(BaseModel):
    question: str
    top_k: int = 6
    subject: str | None = None
    course_id: str | None = None

class ContextChunk(BaseModel):
    doc_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    text: str
    score: float | None = None
    source: str | None = None

class QueryResponse(BaseModel):
    answer: str
    contexts: List[ContextChunk]
    elapsed_ms: int

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    import time
    t0 = time.time()

    # ép subject (bắt buộc phải có để tránh query toàn bộ)
    if not req.subject:
        raise HTTPException(status_code=400, detail="Subject is required")

    if not req.course_id:
        raise HTTPException(status_code=400, detail="Course ID is required")
    # filter theo subject trong metadata
    filters = {"subject": req.subject.lower()}
    if req.course_id:
        filters["course_id"] = req.course_id
    answer, docs = retrieve_answer(req.question, k=req.top_k, filters=filters, subject=req.subject)

    ctxs: List[ContextChunk] = [
        ContextChunk(
            doc_id=d.metadata.get("source"),
            chunk_id=d.metadata.get("chunk_id"),
            page=d.metadata.get("page"),
            text=d.page_content[:1000],
            score=d.metadata.get("score"),
            source=d.metadata.get("source"),
        )
        for d in docs
    ]

    return QueryResponse(
        answer=answer,
        contexts=ctxs,
        elapsed_ms=int((time.time() - t0) * 1000),
    )

