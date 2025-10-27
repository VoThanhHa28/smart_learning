from typing_extensions import TypedDict
from typing import Dict, List, Any
from langchain_core.documents import Document
import time

from typing_extensions import TypedDict
from typing import Dict, List, Any
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
    timing: Dict[str, float]
    filters: dict
    search_params: dict
    diag: dict
    final_context_docs: List[Document]
    structured_context_for_prompt: List[Dict[str, Any]]
    per_subq_docs: Dict[str, List[Document]]
    sub_answers: Dict[str, str]
    ordered_blocks: List[Dict[str, Any]]

def debug_state(node_name, state):
    cid = state.get("course_id")
    print(f"🐛 [DEBUG] Node={node_name} | course_id={cid} | keys={list(state.keys())}")

# --- Hàm _update_timing (Giữ nguyên) ---
def _update_timing(state: State, node_name: str, start_time: float) -> State:
    """Cập nhật dictionary timing trong state."""
    duration = round((time.perf_counter() - start_time) * 1000) # milliseconds
    current_timing = state.get("timing", {})
    current_timing[node_name] = duration
    state["timing"] = current_timing
    print(f"⏱️  [Timing] Node '{node_name}' hoàn thành sau: {duration} ms")
    return state

# --- Hàm _dedup_by_sig (Giữ nguyên) ---
def _dedup_by_sig(docs: List[Document]) -> List[Document]:
    # ... (code _dedup_by_sig giữ nguyên) ...
    seen, out = set(), []
    for d in docs:
        if not isinstance(d, Document): continue
        sig = (d.metadata or {}).get("_sig") or (d.page_content or "")[:120]
        if sig in seen:
            continue
        seen.add(sig); out.append(d)
    return out

# --- Hàm build_context (Nếu chưa có, thêm vào đây) ---
def build_context(docs: List[Document]) -> str:
    """Xây dựng chuỗi context từ list Document."""
    context_parts = []
    sources_index = []
    # Thêm check isinstance để đảm bảo chỉ xử lý Document
    for doc in docs:
        if not isinstance(doc, Document): continue
        meta = doc.metadata or {}
        # Ưu tiên dùng D#-p# nếu có, nếu không chỉ dùng text
        source_tag = f"[{meta.get('doc_id', 'D?')}-p{meta.get('page', '?')}]" if meta.get('doc_id') and meta.get('page') else ""
        header = f"### {source_tag} | {meta.get('heading', '-')}"
        meta_info = f"(meta: type={meta.get('type', 'text')}; score={meta.get('rerank_score', ''):.3f})" if meta.get('rerank_score') else "(meta: type=text)"

        context_parts.append(f"{header}\n{meta_info}\n{doc.page_content or ''}")
        if source_tag:
            sources_index.append(source_tag)

    # Thêm phần liệt kê nguồn ở cuối
    if sources_index:
        context_parts.append("\n## SOURCES INDEX\n" + "\n".join(sources_index))

    return "\n\n".join(context_parts).strip()
# --- Kết thúc hàm build_context ---
