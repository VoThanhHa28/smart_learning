from collections import defaultdict
from .vectorstore import get_vectorstore

def count_docs_by_subject():
    vs = get_vectorstore()
    col = vs._collection  # truy cập collection trong Chroma

    limit, offset = 5000, 0
    counts = defaultdict(int)
    total = 0

    while True:
        out = col.get(include=["metadatas"], limit=limit, offset=offset)
        metas = out.get("metadatas", [])
        if not metas:
            break
        for m in metas:
            subj = (m or {}).get("subject", "unknown").lower()
            counts[subj] += 1
            total += 1
        offset += len(metas)

    return {"total_docs": total, "by_subject": dict(counts)}
