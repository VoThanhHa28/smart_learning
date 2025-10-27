import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI

# --- DỌN DẸP MÔI TRƯỜNG TRƯỚC KHI IMPORT MODEL ---
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
load_dotenv()
os.environ.pop("HF_TOKEN", None)
logging.info("Đã gỡ HF_TOKEN (nếu có) để tránh lỗi đăng nhập.")

# --- IMPORT SAU KHI DỌN DẸP ---
from app.api.routers import index, query
from .infrastructure.llm.llm import warmup_llm
from .infrastructure.llm.embedding import warmup_embeddings  # ✅
from .rag.retrieval.vectorstore import warmup_vectorstore
from .rag.rag_graph import warmup_rag
from .services.lms.lms_router import initialize_lms_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

app = FastAPI(title="Smart Learning API", version="0.1.0")

@app.on_event("startup")
async def _startup():
    logging.info("🚀 API đang khởi động...")

    # ⚡ Warmup theo thứ tự: vectorstore → LLM → embeddings → RAG graph → LMS tools
    warmup_vectorstore()      # connect + load collection
    warmup_llm()              # ping 1 câu cho LLM
    warmup_embeddings()       # ✅ materialize + cache embedder (ưu tiên GPU khi encode)
    await warmup_rag()        # preload reranker, etc.
    initialize_lms_retriever()

    logging.info("✅ API đã khởi động thành công.")

app.include_router(index.router)
app.include_router(query.router)

@app.get("/")
def root():
    return {"msg": "Smart Learning API (v2 - Hierarchical) running"}
