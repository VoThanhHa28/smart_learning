from fastapi import FastAPI
from app.routers import index, query
from dotenv import load_dotenv
import os
from transformers import AutoModelForImageTextToText
from app.routers import ocr_test
load_dotenv()

app = FastAPI(title="Smart Learning API", version="0.1.0")

app.include_router(ocr_test.router, prefix="/ocr", tags=["OCR"])
app.include_router(index.router)
app.include_router(query.router)

@app.get("/")
def root():
    return {"msg": "Smart Learning API running"}
