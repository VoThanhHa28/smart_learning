# -*- coding: utf-8 -*-
"""Reciprocal Rank Fusion (RRF) tiện ích độc lập."""
from __future__ import annotations
from typing import List, Sequence, Dict
from langchain_core.documents import Document

def _as_docs(seq) -> List[Document]:
    out = []
    for x in seq:
        if isinstance(x, tuple) and isinstance(x[0], Document):
            out.append(x[0])
        elif isinstance(x, Document):
            out.append(x)
    return out

def rrf_merge(ranked_lists: Sequence[Sequence], k: int = 60, top_n: int = 50) -> List[Document]:
    scores: Dict[str, float] = {}
    keeper: Dict[str, Document] = {}

    for lst in ranked_lists:
        docs = _as_docs(lst)
        for r, d in enumerate(docs, start=1):
            key = (d.page_content or "").strip()
            if not key:
                continue
            keeper.setdefault(key, d)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + r)

    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = [keeper[k] for k, _ in items[:top_n]]

    # dedup lần cuối
    seen, uniq = set(), []
    for d in out:
        sig = (d.page_content or "").strip()
        if sig and sig not in seen:
            seen.add(sig)
            uniq.append(d)
    return uniq
