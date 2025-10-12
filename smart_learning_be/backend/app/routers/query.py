from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, List, Optional
from ..services.chain import retrieve_answer
import os
import time

router = APIRouter(tags=["query"])


# ============================= #
# 📦 Models
# ============================= #
class QueryRequest(BaseModel):
    question: str
    top_k: int = 6
    subject: Optional[str] = None
    course_id: Optional[str] = None


class ContextChunk(BaseModel):
    page: Optional[int] = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[ContextChunk]
    elapsed_ms: int


# ============================= #
# 🚀 Query Endpoint
# ============================= #
@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    t0 = time.time()

    if not req.subject:
        raise HTTPException(status_code=400, detail="Subject is required")
    if not req.course_id:
        raise HTTPException(status_code=400, detail="Course ID is required")

    filters = {
        "subject": req.subject.lower().strip(),
        "course_id": req.course_id.lower().strip(),
    }

    # ✅ Gọi async RAG chain
    answer, docs = await retrieve_answer(
        question=req.question,
        subject=req.subject,              # ✅ truyền rõ ràng (đừng để trong filters)
        course_id=req.course_id,          # ✅ thêm dòng này
        k=req.top_k,
        filters=filters,
    )

    # ✅ Chuẩn hóa metadata cho client
    sources: List[ContextChunk] = []
    for d in docs:
        meta = d.metadata
        sources.append(
            ContextChunk(
                page=meta.get("page"),          
                text=d.page_content[:1000] + "...",  # chỉ preview 1000 ký tự
            )
        )

    elapsed = int((time.time() - t0) * 1000)

    # ✅ Giữ nguyên Markdown + newline (không JSON encode thêm lần nữa)
    return QueryResponse(
        answer=answer,
        sources=sources,
        elapsed_ms=elapsed,
    )
