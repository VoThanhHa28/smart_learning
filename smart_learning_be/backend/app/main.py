from fastapi import FastAPI
from app.routers import index, query
from dotenv import load_dotenv
import os
from transformers import AutoModelForImageTextToText
from app.services.llm import warmup_llm
from app.services.vectorstore import warmup_vectorstore
from app.services.reranker import get_reranker  # chỉ gọi để load GPU model
from app.services.rag_graph import _get_mmre5   
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

load_dotenv()

app = FastAPI(title="Smart Learning API", version="0.1.0")

@app.on_event("startup")
async def _startup():
    warmup_vectorstore()  # connect + load collection
    warmup_llm()          # ping 1 câu
    try:
        get_reranker()    # load model vào GPU
    except Exception:
        pass
    try:
        _get_mmre5()      # load E5 cho MMR
    except Exception:
        pass



app.include_router(index.router)
app.include_router(query.router)

@app.get("/")
def root():
    return {"msg": "Smart Learning API running"}
