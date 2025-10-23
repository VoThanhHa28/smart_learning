# app/services/prompt_utils.py
import os
from langchain_core.prompts import PromptTemplate
from typing import Union

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
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
    "definition": "Nêu bản chất súc tích (1–2 câu). Nếu có, thêm 1 ví dụ đúng cùng loại, trích từ NGỮ CẢNH.",
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
Bạn là trợ giảng môn {subject}.

# 🎯 NHIỆM VỤ
Giải thích "{question}" trong phạm vi {doc_source}, bám sát NGỮ CẢNH, viết mạch lạc.

# 🧭 CÁCH LÀM (gợi ý mềm)
- Ưu tiên đoạn NGỮ CẢNH nói trực tiếp về "{question}".
- Tóm lược chính xác → nêu vai trò/bản chất → ví dụ ngắn (nếu có).
- Dùng văn phong mạch lạc, dễ hiểu, có câu nối giữa các ý, diễn đạt tự nhiên sao cho giống 1 giảng viên giỏi đang giảng dạy.
- Không dùng từ ngữ cứng nhắc, tránh liệt kê khô cứng.
{('- ' + soft_hints) if soft_hints else ''}

# 🚫 RÀNG BUỘC
- Chỉ dùng dữ kiện trong NGỮ CẢNH, không bịa, không dẫn ngoài.
- Nếu thiếu dữ kiện, trả đúng câu: "Tài liệu không đề cập gì về chủ đề này."
- Văn phong rõ ràng, liền mạch; tránh liệt kê khô cứng.

# 💬 ĐẦU RA (linh hoạt, không ép khung nếu không phù hợp)
• Định nghĩa/Ý chính (nếu có trong NGỮ CẢNH)
• Ứng dụng/Ý nghĩa (nếu có)
• Ví dụ ngắn (1–3) bám đúng phần chính
• Ngoài ra (nếu có trong NGỮ CẢNH): các chi tiết liên quan (vd. công cụ/hàm phụ trợ); kèm ví dụ đúng phần này
• Lưu ý/Cảnh báo (nếu có)
• TL;DR (1–2 câu)
-> Cuối cùng hãy viết đoạn giải thích cho từng đoạn trả lời của bạn, tại sao bạn trả lời như vậy dựa trên ngữ cảnh đã cho. 
từ định nghĩa, ví dụ, ứng dụng, lưu ý, tóm lại, bạn đã trả lời các phần nào thì hãy giải thích đủ lại tại sao bạn trả lời như vậy.
""".strip()

SYSTEM_PROMPT = load_prompt("prompts/system_prompt.txt")
APPENDIX = load_prompt("prompts/appendix_examples.txt")
USE_APPENDIX = os.getenv("USE_APPENDIX", "false").lower() == "true"

if USE_APPENDIX:
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT + "\n\n" + APPENDIX)
else:
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)

# MỚI (trả về string đã fill sẵn): 
CACHED_PROMPT = QA_PROMPT.format(subject="General").strip()  # <-- string