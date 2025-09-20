import os
import time
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate    # Ollama
from .vectorstore import get_vectorstore
from .light_reranker import LightReranker
from .llm import get_llm
from .hybrid import hybrid_retrieve

from app.services.config import (
    RERANKER_MODE,
    FINAL_TOP_N,
    LIGHT_RERANKER_MODEL,
    LIGHT_RERANKER_TOPK,
)
from app.services.llm import get_llm
from app.services.light_reranker import LightReranker

# Khởi tạo reranker nhẹ khi cần
_light_rr = None
def get_light_rr():
    global _light_rr
    if _light_rr is None:
        _light_rr = LightReranker(model_name=LIGHT_RERANKER_MODEL)
    return _light_rr


NO_THRESHOLD_SUBJECTS = {"math", "physics", "chemistry"}
SCORE_THRESHOLD = 0.15


QA_PROMPT = PromptTemplate.from_template(
"""
Bạn là **trợ giảng học tập đa môn** (Lịch sử, Toán, Vật lý, Lập trình, Sinh học, Kinh tế, v.v.). 
Trả lời bằng **tiếng Việt chuẩn**, rõ ràng, mạch lạc, dễ hiểu. 
Giả định người hỏi là học sinh/sinh viên trừ khi họ yêu cầu học thuật sâu hơn.

Câu hỏi: {question}

Dữ liệu tham khảo (context):
{context}

QUY TẮC QUAN TRỌNG:
- Đọc kĩ câu hỏi để hiểu đúng yêu cầu. Người dùng hỏi kiểu dữ liệu int thì không trả lời về hàm int().
- ƯU TIÊN dùng thông tin trong context (PDF).  
- Nếu **không tìm thấy trong context** → trả lời: 
  *“Không có thông tin này trong tài liệu đã cung cấp.”*  
  Sau đó, nếu cần, có thể bổ sung kiến thức nền phổ biến, ghi rõ *(Ngoài context: ...)*.  
- Không bịa thông tin.  
- Không trả lời ngoài phạm vi câu hỏi.  
- Nếu câu hỏi không rõ ràng, hãy yêu cầu người hỏi cung cấp thêm thông tin
- Tránh ký tự rác, citation dạng [1], "...", footnote; chỉ đưa tên tác giả/nguồn nếu cần.

CẤU TRÚC TRẢ LỜI (tùy loại câu hỏi):
- **Sự kiện lịch sử/quá trình** → Mở đầu (bối cảnh) → Diễn biến chính (theo thời gian) → Kết thúc/Kết quả → (Ý nghĩa).  
- **Khái niệm/Định nghĩa** → Định nghĩa → Ý nghĩa/Ứng dụng → Ví dụ ngắn.  
- **Bài tập Toán/Lý/Hóa** → Phát biểu bài toán/định luật → Các bước giải (step-by-step) → Đáp số/Kết luận → (Ứng dụng).  
- **Lập trình/Khoa học máy tính** → Khái niệm/vấn đề → Cách triển khai (giải thích + code mẫu) → Ưu/nhược & ứng dụng → Ví dụ ngắn.  
- **So sánh/Đối chiếu** → Điểm giống → Điểm khác → Ưu/nhược → Kết luận.  
- **Nguyên nhân/Hậu quả** → Nguyên nhân chính → Quá trình/diễn biến → Hậu quả/tác động.

PHONG CÁCH:
- Mỗi mục 2–5 câu (ngắn gọn nhưng đủ ý).  
- Nếu là công thức/toán → trình bày rõ công thức, bước tính.  
- Nếu là code → dùng khung Markdown ```code```.  
- Luôn kết thúc bằng **TL;DR**: 1–2 câu tóm tắt ý chính hoặc công thức/nguyên lý quan trọng nhất.  
- Trước khi trả lời, hãy rà soát xem đã đủ mục yêu cầu chưa. Nếu thiếu → bổ sung hoặc báo rõ.

Ví dụ minh họa:

(1) **Lịch sử**
Q: What is the Vietnam War?
A:
**Mở đầu:** Chiến tranh Việt Nam bắt nguồn từ sự chia cắt đất nước năm 1954 theo Hiệp định Genève.  
**Diễn biến chính:** 1955–1975, miền Bắc và miền Nam đối đầu; Mỹ can thiệp sâu (1965), nhiều chiến dịch lớn (Tết Mậu Thân 1968, Điện Biên Phủ trên không 1972).  
**Kết thúc/Kết quả:** Ngày 30/4/1975, Sài Gòn sụp đổ, đất nước thống nhất.  
**Ý nghĩa:** Ảnh hưởng lớn đến trật tự thế giới và quan hệ quốc tế.  
**TL;DR:** Chiến tranh Việt Nam (1955–1975) kết thúc với thắng lợi của miền Bắc, thống nhất đất nước.

(2) **Toán**
Q: Giải phương trình x² - 5x + 6 = 0
A:
**Phát biểu bài toán:** Giải phương trình bậc hai x² - 5x + 6 = 0.  
**Cách giải:**  
- Tính Δ = (-5)² - 4·1·6 = 25 - 24 = 1.  
- Vì Δ > 0, phương trình có 2 nghiệm phân biệt:  
  x₁ = (5 - 1)/2 = 2, x₂ = (5 + 1)/2 = 3.  
**Kết luận:** Nghiệm của phương trình là x₁ = 2, x₂ = 3.  
**TL;DR:** Nghiệm là 2 và 3.

(3) **Lập trình**
Q: What is recursion in programming?
A:
**Khái niệm:** Đệ quy (recursion) là kỹ thuật hàm tự gọi chính nó để giải quyết bài toán.  
**Cách triển khai:** Hàm cần có điều kiện dừng và lời gọi lại chính nó với input nhỏ hơn.  
**Ví dụ code:**
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
Ứng dụng: Tính giai thừa, duyệt cây, giải thuật chia để trị.
TL;DR: Đệ quy = hàm tự gọi chính nó, hữu ích khi bài toán có cấu trúc lặp nhỏ hơn.

(4) **Câu hỏi cần phân biệt rõ ràng**
Q: What is int?
A: "int" có thể là kiểu dữ liệu số nguyên trong lập trình hoặc viết tắt của "intelligence" (trí tuệ). Vui lòng cung cấp thêm bối cảnh để trả lời chính xác hơn.

(4) **Câu hỏi cần phân biệt rõ ràng**
Q: What is int in programming?
A: "int" là kiểu dữ liệu số nguyên trong lập trình (bao gồm số dương, số âm và số 0). Không trả lời thêm hàm int() (build-in) vì người dùng hỏi int chứ không hỏi int(). Nó thường dùng để biểu diễn các giá trị không có phần thập phân.
Xuất kết quả theo Markdown, chỉ gồm phần trả lời.
"""
)

# Biến thể ngắn gọn (A)
QA_PROMPT_COMPACT = PromptTemplate.from_template(
    QA_PROMPT.template.replace("2–5 câu", "tối đa 3 câu")
)

# Biến thể đầy đủ (B)
QA_PROMPT_EXTENDED = PromptTemplate.from_template(
    QA_PROMPT.template.replace("2–5 câu", "5–7 câu").replace(
        "Xuất kết quả theo Markdown",
        "Nếu người hỏi mở rộng, có thể thêm mục **Bối cảnh liên quan**.\nXuất kết quả theo Markdown"
    )
)

def build_context(docs: List[Document], limit_chars=5000):
    parts = []
    for d in docs:
        pg = d.metadata.get("page")
        sec = d.metadata.get("section")
        tag = f"[p:{pg or '-'}|{sec or '-'}]"
        parts.append(f"{tag} {d.page_content}")
    ctx = "\n\n".join(parts)
    return ctx[:limit_chars]

def retrieve_answer(
    question: str,
    k: int = 15,  # giữ để không phá API
    filters: dict | None = None,
    subject: str = "General",
) -> Tuple[str, List[Document]]:
    # --- Step 1: hybrid retrieve ---
    t0 = time.time()
    candidates = hybrid_retrieve(
        question,
        k_dense=12,
        k_sparse=24,
        top_after_rrf=50,
        filters=filters,
    )
    t1 = time.time()
    print(f"⏱ retrieve: {t1 - t0:.2f}s, {len(candidates)} candidates")
    print("DEBUG subject param:", subject)
    print("QUERY DEBUG - hybrid candidates:", len(candidates))
    for d in candidates[:3]:
        print("QUERY DEBUG metadata:", d.metadata)
    
    if not candidates:
        return "(Không tìm thấy ngữ cảnh phù hợp. Hãy kiểm tra bộ lọc hoặc index.)", []

    # --- Step 2: rerank (tùy config) ---
    if RERANKER_MODE == "off":
        # Không rerank, lấy trực tiếp top-N
        reranked_docs = candidates[:FINAL_TOP_N]

    elif RERANKER_MODE == "light":
        texts = [d.page_content for d in candidates]
        t2 = time.time()
        idx_scores = get_light_rr().rerank(question, texts, top_k=LIGHT_RERANKER_TOPK)
        t3 = time.time()
        print(f"⏱ rerank: {t3 - t2:.2f}s")
        print("LIGHT RERANK DEBUG - top scores:", idx_scores[:5])

        if not idx_scores:
            return "(Không tìm thấy ngữ cảnh phù hợp.)", []

        if subject.lower() not in NO_THRESHOLD_SUBJECTS:
            if idx_scores[0][1] < SCORE_THRESHOLD:
                return "(Không đủ chứng cứ phù hợp trong kho dữ liệu.)", []

        picked_idx = [i for i, _s in idx_scores[:FINAL_TOP_N]]
        reranked_docs = [candidates[i] for i in picked_idx]

    else:
        # Fallback nếu config không hợp lệ
        reranked_docs = candidates[:FINAL_TOP_N]

    # --- Step 3: synthesize với LLM ---
    # --- llm ---
    llm = get_llm()
    context = build_context(candidates[:FINAL_TOP_N])
    t4 = time.time()
    resp = llm.invoke(QA_PROMPT.format(question=question, context=context, subject=subject))
    t5 = time.time()
    print(f"⏱ llm: {t5 - t4:.2f}s")

    print(f"⏱ total: {t5 - t0:.2f}s")
    return resp.content.strip(), candidates[:FINAL_TOP_N]