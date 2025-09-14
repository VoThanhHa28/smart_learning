from fastapi import FastAPI
from app.routers import index, query

app = FastAPI(title="Smart Learning API", version="0.1.0")

app.include_router(index.router)
app.include_router(query.router)

@app.get("/")
def root():
    return {"msg": "Smart Learning API running"}
