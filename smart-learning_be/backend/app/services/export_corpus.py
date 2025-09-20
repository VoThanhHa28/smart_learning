# export_corpus.py
import json, os
from .vectorstore import get_vectorstore

OUT = os.path.join(os.path.dirname(__file__), "corpus.jsonl")

def main():
    vs = get_vectorstore()
    # ⚠️ Cách dưới dựa vào API Chroma wrapper phổ biến.
    # Nếu bản langchain của bạn khác, có thể dùng vs._collection.get(...) tương tự.
    try:
        raw = vs._collection.get(include=["documents", "metadatas"])
    except Exception:
        # Fallback nếu wrapper thay đổi
        raw = vs._client.get_collection(vs._collection.name).get(include=["documents","metadatas","ids"])
    ids = raw.get("ids", [])
    docs = raw.get("documents", [])
    metas = raw.get("metadatas", [])

    with open(OUT, "w", encoding="utf-8") as f:
        for i, txt in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            item = {"id": ids[i] if i < len(ids) else None, "text": txt, "metadata": meta}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Exported {len(docs)} docs → {OUT}")

if __name__ == "__main__":
    main()
