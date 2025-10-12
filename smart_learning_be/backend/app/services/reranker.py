import os
import asyncio
import time
from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from huggingface_hub import login
import torch
import torch.nn.functional as F

# ===========================
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


# ===========================
# ⚙️ Config
# ===========================
RERANK_MODEL = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base")
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", 0.3))  # ✅ hạ ngưỡng hợp lý cho Jina
_cross_encoder = None

import re

def clean_doc(text: str) -> str:
    """
    Làm sạch văn bản trước khi rerank.
    - Loại bỏ code, markdown, HTML, ký tự đặc biệt.
    - Giữ lại ngữ nghĩa chính của nội dung.
    - Không làm hỏng dấu câu, không hư tiếng Việt.
    """

    if not text:
        return ""

    # ✅ 1. Gộp khoảng trắng & normalize linebreaks
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)  # gộp nhiều dòng trống
    text = re.sub(r"[ \t]+", " ", text)

    # ✅ 2. Xóa code block (```...```) và inline code (`...`)
    text = re.sub(r"```[\s\S]*?```", " ", text)  # multi-line code
    text = re.sub(r"`[^`]+`", " ", text)         # inline code

    # ✅ 3. Xóa lệnh REPL (>>> hoặc In [x]:)
    text = re.sub(r"^>>>.*$", " ", text, flags=re.M)
    text = re.sub(r"^In\s*\[\d*\]:.*$", " ", text, flags=re.M)
    text = re.sub(r"^Out\s*\[\d*\]:.*$", " ", text, flags=re.M)

    # ✅ 4. Xóa markdown heading / bullet / numbering
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = re.sub(r"^[\*\-\+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.M)

    # ✅ 5. Xóa HTML / XML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # ✅ 6. Xóa ký tự rác & emoji không liên quan
    text = re.sub(r"[■◆●►◼▪¤•★☆※☞✓✔️❌✗➤→⇒⬇⬆⬅➡🔹🔸🔻🔺💡🔥🚀⭐🧠✅❗]", " ", text)

    # ✅ 7. Xóa JSON / YAML block (thường xuất hiện trong log)
    text = re.sub(r"\{[\s\S]*?\}", " ", text)  # JSON
    text = re.sub(r"---[\s\S]*?---", " ", text)  # YAML

    # ✅ 8. Giới hạn độ dài để tránh reranker quá tải
    text = text[:3000]

    # ✅ 9. Làm gọn lại khoảng trắng và dòng cuối
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ===========================
# 🚀 Lazy load CrossEncoder
# ===========================
def get_reranker():
    """Load model 1 lần duy nhất, GPU-optimized."""
    global _cross_encoder
    if _cross_encoder is None:
        print(f"🚀 Loading reranker model: {RERANK_MODEL}")
        _cross_encoder = CrossEncoder(
            RERANK_MODEL,
            device="cuda",
            max_length=512,
            trust_remote_code=True,   # ✅ cần cho Jina model
        )
    return _cross_encoder


# ===========================
# 🧠 Async Rerank Function
# ===========================
async def rerank(query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
    """
    Async reranker:
    - Gọi CrossEncoder trong thread pool để không block event loop
    - Áp dụng threshold lọc doc yếu
    - Log chi tiết thời gian & score range
    """
    if not docs:
        return []

    t0 = time.time()
    model = get_reranker()

    for d in docs:
        d.metadata["clean_text"] = clean_doc(d.page_content)

    # for d in docs[:3]:
    #     print(f"---DOC---\n{d.page_content[:1000]}")

    pairs = [[query, d.metadata.get("clean_text") or d.page_content] for d in docs]


    # ⚡ chạy trong thread pool để tránh block asyncio
    loop = asyncio.get_event_loop()
    # predict ra tensor thay vì list
    scores = model.predict(pairs, convert_to_tensor=True)
    scores = torch.sigmoid(scores).cpu().tolist()


    # 🧩 Gắn score vào metadata
    for d, s in zip(docs, scores):
        d.metadata["rerank_score"] = float(s)


    for i, d in enumerate(docs[:3]):
        print(f"\n[DOC {i}] Score={d.metadata['rerank_score']:.3f}")
        print(f"Clean text:\n{d.metadata['clean_text'][:1000]}\n")

    # 📊 Sort theo score giảm dần
    reranked = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
    top_docs = reranked[:top_n]

    # ✅ Lọc theo threshold hợp lý (Jina: điểm 0.4–0.9 là mạnh)
    filtered = [d for d in top_docs if d.metadata["rerank_score"] >= RERANK_THRESHOLD]

    t1 = time.time()
    dur = t1 - t0

    # 🪶 Logging chuẩn expert
    print(
        f"🏅 [Reranker] Scored={len(docs)}, "
        f"Selected={len(filtered)}/{len(top_docs)} "
        f"(thr={RERANK_THRESHOLD}) | "
        f"max={max(scores):.3f} min={min(scores):.3f} | ⏱️ {dur:.3f}s"
    )

    # ✅ Fallback nếu tất cả score thấp
    return filtered or top_docs[:2]
