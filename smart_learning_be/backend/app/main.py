from fastapi import FastAPI
from app.routers import index, query
from dotenv import load_dotenv
import logging
from app.services.llm import warmup_llm
from app.services.vectorstore import warmup_vectorstore
from app.services.rag_graph import warmup_rag  # ✅ dùng hàm public

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

load_dotenv()
app = FastAPI(title="Smart Learning API", version="0.1.0")

@app.on_event("startup")
async def _startup():
    warmup_vectorstore()          # connect + load collection
    warmup_llm()                  # ping 1 câu
    await warmup_rag()            # ✅ preload E5 + reranker (GPU nếu có)

app.include_router(index.router)
app.include_router(query.router)

@app.get("/")
def root():
    return {"msg": "Smart Learning API running"}
