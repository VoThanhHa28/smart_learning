from fastapi import APIRouter, HTTPException
from app.schemas import QueryRequest
from app.services.vectorstore import query as vs_query

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/")
def query_doc(req: QueryRequest):
    try:
        hits = vs_query(req.docId, req.question, n_results=req.top_k)
        if not hits:
            return {"answer": "Không tìm thấy nội dung phù hợp trong tài liệu.", "contexts": []}

        # TẠM THỜI: trả về top chunk như câu trả lời (mock minimal, nhưng search là real)
        # Bước sau mới gọi LLM (Gemini/Ollama) để tổng hợp + trích dẫn.
        best = hits[0]["document"]
        return {
            "answer": best,
            "contexts": hits  # để frontend hiển thị nguồn
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
