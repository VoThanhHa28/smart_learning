from fastapi import APIRouter
from app.services.stats import count_docs_by_subject

router = APIRouter(prefix="/stats", tags=["stats"])

@router.get("")
def get_stats():
    return count_docs_by_subject()
