from fastapi import FastAPI
from app.routers import index, query, stats
from dotenv import load_dotenv
import os

load_dotenv()
print("DEBUG API KEY:", os.getenv("PINECONE_API_KEY"))
print("DEBUG ENV:", os.getenv("PINECONE_ENV"))

app = FastAPI(title="Smart Learning API", version="0.1.0")

app.include_router(index.router)
app.include_router(query.router)
app.include_router(stats.router)

@app.get("/")
def root():
    return {"msg": "Smart Learning API running"}
