# app/services/rag_runtime.py
import hashlib, time, logging, torch, numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

_mmre5 = None
def _get_mmre5():
    global _mmre5
    if _mmre5 is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"[MMR] loading E5 on device={device}")
        _mmre5 = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    return _mmre5

def doc_sig(d: Document) -> str:
    meta = d.metadata or {}
    head = meta.get("heading") or meta.get("headings") or ""
    page = str(meta.get("page", ""))
    key = f"{(d.page_content or '')[:200]}|{head}|{page}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

def mark_doc(d: Document, *, source_group: str, source_subq: str, stage: str):
    d.metadata = dict(d.metadata or {})
    trace = d.metadata.get("_trace") or {}
    sources = trace.get("sources") or []
    sources.append({"stage": stage, "group": source_group, "subq": source_subq, "ts": time.time()})
    trace["sources"] = sources
    d.metadata["_trace"] = trace

def mmr_select(query: str, docs: list[Document], k: int = 24, lambda_mult: float = 0.65):
    if not docs:
        return []
    model = _get_mmre5()
    if torch.cuda.is_available():
        with torch.inference_mode():
            q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
            d_texts = [d.page_content or "" for d in docs]
            d_embs = model.encode(d_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    else:
        q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, batch_size=1, show_progress_bar=False)
        d_texts = [d.page_content or "" for d in docs]
        d_embs = model.encode(d_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, show_progress_bar=False)

    sims_q = (q_emb @ d_embs.T).ravel()
    selected = []
    rep = np.zeros_like(sims_q)
    while len(selected) < min(k, len(docs)):
        if selected:
            rep = np.max((d_embs[selected] @ d_embs.T), axis=0)
        scores = lambda_mult * sims_q - (1 - lambda_mult) * rep
        scores[selected] = -1e9
        pick = int(np.argmax(scores))
        selected.append(pick)
    return [docs[i] for i in selected]
