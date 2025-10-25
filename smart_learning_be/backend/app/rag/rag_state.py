# app/services/rag_state.py
from typing_extensions import TypedDict
from typing import Dict, List
from langchain_core.documents import Document

class State(TypedDict, total=False):
    question: str
    question_user: str
    question_norm: str
    subject: str
    course_id: str
    context: List[Document]
    analyze_meta: dict
    answer: str
    timing: dict
    sub_answers: Dict[str, str]
    filters: dict
    search_params: dict
    per_subq_docs: Dict[str, List[Document]]
    diag: dict

def debug_state(node_name, state):
    cid = state.get("course_id")
    print(f"🐛 [DEBUG] Node={node_name} | course_id={cid} | keys={list(state.keys())}")

# vài hằng số nhẹ nếu cần dùng ở nhiều node
PER_SUBQ_CONTEXT_N = 2
RERANK_THRESHOLD = 0.50
MIN_PER_SUBQ = 1
MAX_PER_SUBQ = 2
