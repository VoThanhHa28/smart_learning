from pydantic import BaseModel

class IndexRequest(BaseModel):
    docId: str
    storagePath: str

class QueryRequest(BaseModel):
    docId: str
    question: str
    top_k: int = 5
