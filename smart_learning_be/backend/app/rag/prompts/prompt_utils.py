# app/services/prompt_utils.py
import os
from langchain_core.prompts import PromptTemplate
from typing import Union
from pathlib import Path

def load_prompt(path: str) -> str:
    base = Path(__file__).resolve().parent          # app/rag/prompts/
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    with p.open("r", encoding="utf-8") as f:
        return f.read()
# ===========================
# 🔁 Intent aliases (chuẩn hóa tên intent về file prompt)
# ===========================
INTENT_ALIASES = {
    "compare": "comparison",
    "problem": "exercise",
    "summarize": "summary",
    "summ": "summary",
    "def": "definition",
    "warn": "warning",
    "socratic": "socratic_question",
    "quiz": "quick_check",
    "simplify": "paraphrase",
    "router_fallback": "router_fallback",  # phòng khi router rơi vào fallback
    "multi_part": "multi_part",
}


PRIORITY = [
  "definition","comparison","multi_part","application","example","warning",
  "exercise","theory","summary","socratic_question","quick_check","paraphrase"
]

def normalize_and_order_intents(intents: list[str]) -> list[str]:
    # to lower + alias + unique (giữ thứ tự ưu tiên)
    seen = set()
    norm = []
    for raw in intents:
        i = (raw or "").lower().strip()
        i = INTENT_ALIASES.get(i, i) or "general"
        if i not in seen:
            seen.add(i)
            norm.append(i)
    # order by PRIORITY; những intent lạ đưa về cuối
    ordered = sorted(norm, key=lambda x: PRIORITY.index(x) if x in PRIORITY else 999)
    return ordered

def get_soft_hints(intents: list[str]) -> str:
    if not intents:
        return ""

    # alias + thứ tự ưu tiên (lấy tối đa 2 gợi ý đầu)
    ALIAS = {
        "compare": "comparison", "def": "definition", "summ": "summary",
        "simplify": "paraphrase", "quiz": "quick_check"
    }
    PRIORITY = [
        "comparison","definition","application","example","warning","summary",
        "exercise","theory","quick_check","paraphrase","multi_part",
        "socratic_question","toc","router_fallback","general"
    ]
    HINT = {
    "comparison": "So sánh theo 2–3 trục cốt lõi; nêu điểm giống/khác ngắn gọn. Ví dụ (nếu có) phải trích từ NGỮ CẢNH.",
    "definition": "Nêu định nghĩa. Nếu có hãy thêm các phần sau: 1-2 ví dụ đúng cùng loại, ứng dụng, Ngoài ra(nếu có) trích từ NGỮ CẢNH.",
    "application": "Chỉ mô tả các bước/qui trình nếu NGỮ CẢNH có nêu; nếu không, tóm nguyên tắc áp dụng ngắn gọn.",
    "example": "Chọn 1 ví dụ ngắn, đúng cùng loại với chủ đề, trích trực tiếp từ NGỮ CẢNH; không bịa thêm.",
    "warning": "Chỉ cảnh báo nếu NGỮ CẢNH có nêu bẫy/ngoại lệ; trình bày ngắn gọn, rõ điều kiện xảy ra.",
    "summary": "Kết thúc bằng TL;DR 1–2 câu, chỉ dùng ý từ NGỮ CẢNH.",
    "exercise": "Nếu NGỮ CẢNH cho phép, đề xuất 1–3 bài tập siêu ngắn bám NGỮ CẢNH; kèm gợi ý/đáp án tóm tắt khi có.",
    "theory": "Làm rõ nguyên lý/điều kiện áp dụng dựa trên NGỮ CẢNH; không suy diễn ngoài phạm vi.",
    "quick_check": "Soạn 3–5 câu hỏi kiểm tra ngắn, trả lời được từ NGỮ CẢNH; cuối mục nêu đáp án ngắn gọn.",
    "paraphrase": "Diễn đạt lại ý trong NGỮ CẢNH cho dễ hiểu, không thêm/giảm thông tin.",
    "multi_part": "Sắp xếp: định nghĩa/cơ sở → so sánh/ứng dụng → lưu ý. Dùng câu nối; không lặp ý.",
    "socratic_question": "Đặt 3–5 câu hỏi gợi mở từ cơ bản đến kết luận; mỗi câu phải trả lời được từ NGỮ CẢNH, không dẫn dắt ngoài phạm vi.",
    "toc": "Nếu NGỮ CẢNH là mục lục, chỉ định vị đúng mục liên quan; không suy diễn nội dung.",
    "router_fallback": "Nếu tín hiệu mơ hồ, ưu tiên nêu định nghĩa/ý chính bám NGỮ CẢNH, kèm TL;DR; tránh suy diễn.",
    "general": "Ưu tiên ý chính ngắn gọn, có câu nối; chỉ trích thông tin trong NGỮ CẢNH.",
    }


    # normalize + de-dup
    norm = []
    seen = set()
    for it in intents:
        k = (it or "").lower().strip()
        k = ALIAS.get(k, k)
        if k and k not in seen:
            seen.add(k)
            norm.append(k)

    # order by PRIORITY rồi lấy tối đa 2 hint
    norm.sort(key=lambda x: PRIORITY.index(x) if x in PRIORITY else 999)
    hints = [HINT[i] for i in norm if i in HINT][:2]
    return " ".join(hints)



def build_unified_block(subject: str, context: str, question: str, topic: str, doc_source: str, soft_hints: str="") -> str:
    return f"""
# 📘 NGỮ CẢNH
{context}

---

# 🎓 VAI TRÒ
Bạn là trợ giảng môn {subject}. Hãy dùng tiếng Việt để trả lời.

# 🎯 NHIỆM VỤ
Giải thích "{question}" trong phạm vi {doc_source}, bám sát NGỮ CẢNH, viết mạch lạc.

ĐẦU TIÊN VÀ QUAN TRỌNG:
1. XÁC ĐỊNH CÂU HỎI {question} có liên quan đến ngữ cảnh đươc cung cấp không? 
- Nếu có thì trả lời. 
- Nếu không hoặc mơ hồ, không chắc chắn thì trả lời "Tài liệu không đề cập gì về chủ đề {question}."
2. CÂU TRẢ LỜI PHẢI BẮT BUỘC PHẢI CÓ LIÊN QUAN TRỰC TIẾP 100% ĐẾN {question}.
3. QUAN TRỌNG: CÂU HỎI - NGỮ CẢNH - CÂU TRẢ LỜI LÀ BỘ 3 LIÊN QUAN, LIÊN KẾT VỚI NHAU .
-> NGỮ CẢNH LIÊN QUAN ĐẾN CÂU HỎI - VÀ CÂU TRẢ LỜI PHẢI LIÊN QUAN VỚI CÂU HỎI.

# 🧭 CÁCH LÀM (gợi ý mềm)
- Ưu tiên đoạn NGỮ CẢNH nói trực tiếp về "{question}".
- Tóm lược chính xác → nêu vai trò/bản chất → ví dụ ngắn (nếu có).
- Dùng văn phong mạch lạc, dễ hiểu, có câu nối giữa các ý, diễn đạt tự nhiên sao cho giống 1 giảng viên giỏi đang giảng dạy.
- Không dùng từ ngữ cứng nhắc, tránh liệt kê khô cứng.
{('- ' + soft_hints) if soft_hints else ''}

# 🚫 RÀNG BUỘC
- Chỉ dùng dữ kiện trong NGỮ CẢNH, không bịa, không dẫn ngoài.
- Không chào hỏi và nói các câu như "Chào bạn, dựa trên tài liệu, trong ngữ cảnh này,... các câu cứng nhắt" bởi vì nó không được hay như chatbot rag thực thụ.
- Nếu thiếu dữ kiện, trả đúng câu: "Tài liệu không đề cập gì về chủ đề này."
- Văn phong rõ ràng, liền mạch; tránh liệt kê khô cứng.


-> Quy tắc trích dẫn dẫn chứng: Khi dùng thông tin từ NGỮ CẢNH, gắn thẻ nguồn ngay sau câu theo tag [D#-p#] đúng như trong NGỮ CẢNH đc cung cấp. Không bịa số trang.
""".strip()

def build_toc_validation_block(toc_lines: list[str]) -> str:
    ctx = "\n".join(f"- {x}" for x in (toc_lines or [])[:200])
    return f"""
NHIỆM VỤ: Xác thực MỤC LỤC (TOC) từ NGỮ CẢNH và xuất đầu ra tối giản.

YÊU CẦU XUẤT DUY NHẤT MỘT TRONG HAI:
1) Nếu NGỮ CẢNH là mục lục hợp lệ mà 1 bài học có(đủ các phần,có ý nghĩa, không thiếu, rời rạc) → In lại danh sách, mỗi mục một dòng bắt đầu bằng "- ".
2) Nếu KHÔNG phải mục lục hợp lệ → In chính xác câu:
"Mục lục tôi nhận được đang lỗi, hãy hỏi lại sau"

RÀNG BUỘC:
- Không thêm tiêu đề, không TL;DR, không giải thích, không chú thích.
- Không in bất kỳ phần nào khác ngoài 1 trong 2 dạng trên.
- Không dùng code block, không thêm ký tự trang trí.

NGỮ CẢNH:
{ctx}
""".strip()


SYSTEM_PROMPT = load_prompt("system_prompt.txt")
APPENDIX = load_prompt("appendix_examples.txt")
USE_APPENDIX = os.getenv("USE_APPENDIX", "false").lower() == "true"

if USE_APPENDIX:
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT + "\n\n" + APPENDIX)
else:
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)

# MỚI (trả về string đã fill sẵn): 
CACHED_PROMPT = QA_PROMPT.format(subject="General").strip()  # <-- string