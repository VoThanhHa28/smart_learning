import os
from langchain_core.prompts import PromptTemplate
from typing import Union, List, Dict, Any # <-- Thêm Dict, Any
from pathlib import Path
import logging
from ..rag_utils import build_context
from langchain_core.documents import Document # <-- Thêm Document

# --- Hàm load_prompt (Giữ nguyên) ---

def load_prompt(path: str) -> str:
    base = Path(__file__).resolve().parent          # app/rag/prompts/
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    with p.open("r", encoding="utf-8") as f:
        return f.read()
    
# --- HÀM MỚI: BUILD PROMPT CÓ CẤU TRÚC ---
def build_structured_prompt_block(
    subject: str,
    structured_context: List[Dict[str, Any]],  # mỗi item: {"kind":"system"| "academic", ...}
    original_full_question: str,
    doc_source: str = "Tài liệu học tập",
) -> str:
    """
    Xây dựng prompt V4 theo thứ tự sub-question của user.
    - system block: giữ nguyên TEXT-TO-INSERT (meta/unclear/unsafe/toc/…)
    - academic block: cung cấp CONTEXT riêng cho từng sub
    """

    context_blocks: List[str] = []
    sub_questions_list: List[str] = []

    if not structured_context:
        logging.warning("[Prompt Builder] structured_context rỗng. Dùng prompt fallback.")
        return f"""
# 🎯 NHIỆM VỤ
Trả lời câu hỏi sau một cách tốt nhất có thể:
"{original_full_question}"

# 📝 BẮT ĐẦU TRẢ LỜI:
""".strip()

    # build từng block theo đúng thứ tự
    for i, item in enumerate(structured_context):
        kind = (item or {}).get("kind")  # 'system' | 'academic' | fallback
        subq = item.get("subq", f"Phần {i+1}")
        sub_questions_list.append(f"{i+1}. {subq}")

        if kind == "system":
            role = item.get("role")  # meta | unclear | unsafe | toc | ...
            sys_text = (item.get("text") or "").strip()
            context_blocks.append(f"""
## Câu hỏi {i+1}: {subq}  (SYSTEM: {role})
### TEXT-TO-INSERT (giữ nguyên, KHÔNG sửa)
{sys_text if sys_text else "—"}
""".strip())

        elif kind == "academic":
            docs = item.get("docs", [])
            context_str_for_subq = build_context(docs) if docs else "Không có ngữ cảnh cụ thể được tìm thấy cho phần này."
            context_blocks.append(f"""
## Câu hỏi {i+1}: {subq}
### CONTEXT (ONLY USE THIS)
{context_str_for_subq}
""".strip())

        else:
            # fallback an toàn để không rơi block
            context_blocks.append(f"""
## Câu hỏi {i+1}: {subq}  (SYSTEM: unknown)
### TEXT-TO-INSERT (giữ nguyên, KHÔNG sửa)
—
""".strip())

    full_context_section = "\n---\n".join(context_blocks)
    all_sub_questions = "\n".join(sub_questions_list)

    # ráp prompt cuối (vai trò, nhiệm vụ, quy tắc)
    return f"""
# 📘 NGỮ CẢNH CHI TIẾT (Tách theo Câu hỏi con)
{full_context_section}

---

# 🎓 VAI TRÒ
Bạn là trợ giảng môn {subject}. Đây là tên tài liệu {doc_source}. Hãy dùng tiếng Việt để trả lời.

# 🎯 NHIỆM VỤ TỔNG HỢP
Dựa **CHÍNH XÁC** vào **từng cặp** [Câu hỏi con] và [Ngữ cảnh/Insert tương ứng] ở trên, viết **MỘT câu trả lời DUY NHẤT** mạch lạc cho **tất cả** các câu hỏi sau, theo đúng thứ tự:
{all_sub_questions}

# 🧭 QUY TẮC BẮT BUỘC
1) Với khối **SYSTEM**, phần **TEXT-TO-INSERT** phải được giữ nguyên, không chế thêm.
2) Với khối **ACADEMIC**, CHỈ dùng **CONTEXT (ONLY USE THIS)** của đúng câu hỏi đó để trả lời.
3) Toàn bộ câu trả lời phải liền mạch, giọng văn thống nhất (không rời rạc), theo thứ tự Q1→Qn.
4) Gắn số trang là cái p của "[D#-p#]" ngay sau câu khi sử dụng dữ kiện trừ khối **SYSTEM** chỉ gắn cho **ACADEMIC**."
7) Tuyệt đối không dùng thông tin ngoài context, không bịa. Cho phép suy luận hợp lí để cho câu trả lời không quá cụt ngủn dựa trên context.

###Lưu ý: Câu trả lời cuối cùng phải mạch lạc, rõ ràng, có liên kết giữa các phần Q1->Qn(kể cả là TEXT-TO-INSERT hay ACADEMIC đều được nối lại thành 1 câu hoàn chỉnh mạch lạc, rõ ràng), không rời rạc.

# 📝 BẮT ĐẦU CÂU TRẢ LỜI TỔNG HỢP:
""".strip()


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
1. XÁC ĐỊNH CÂU HỎI {question} có liên quan đến ngữ cảnh được cung cấp không?
- Nếu có thì trả lời.
- Nếu không hoặc mơ hồ, không chắc chắn thì trả lời "Tài liệu không đề cập gì về chủ đề {question}."
2. CÂU TRẢ LỜI PHẢI BẮT BUỘC PHẢI CÓ LIÊN QUAN TRỰC TIẾP 100% ĐẾN {question}.
3. QUAN TRỌNG: CÂU HỎI - NGỮ CẢNH - CÂU TRẢ LỜI LÀ BỘ 3 LIÊN QUAN, LIÊN KẾT VỚI NHAU .
4. Các mục "[D#-p#]" tương ứng là "D# là Document ID (Mã tài liệu) - p# là Page Number (Số trang)" trong NGỮ CẢNH.
-> NGỮ CẢNH LIÊN QUAN ĐẾN CÂU HỎI - VÀ CÂU TRẢ LỜI PHẢI LIÊN QUAN VỚI CÂU HỎI.

# 🧭 CÁCH LÀM (gợi ý mềm)
- Ưu tiên đoạn NGỮ CẢNH nói trực tiếp về "{question}".
- Tóm lược chính xác → nêu vai trò/bản chất → ví dụ ngắn (nếu có).
- Dùng văn phong mạch lạc, dễ hiểu, có câu nối giữa các ý, diễn đạt tự nhiên sao cho giống 1 giảng viên giỏi đang giảng dạy.
- Không dùng từ ngữ cứng nhắc, tránh liệt kê khô cứng.
{('- ' + soft_hints) if soft_hints else ''}

# 🚫 RÀNG BUỘC
- Chỉ dùng dữ kiện trong NGỮ CẢNH, không bịa, không dẫn ngoài.
- Trích dẫn số trang là p của "[D#-p#]" nếu có trong đoạn NGỮ CẢNH bạn đang sử dụng để trả lời.
- Không chào hỏi và nói các câu như "Chào bạn, dựa trên tài liệu, trong ngữ cảnh này,... các câu cứng nhắt" bởi vì nó không được hay như chatbot rag thực thụ.
- CHỈ trả về câu "Tài liệu không đề cập gì về chủ đề này." KHI TOÀN BỘ NGỮ CẢNH HOÀN TOÀN KHÔNG CÓ THÔNG TIN LIÊN QUAN ĐẾN CÂU HỎI.
- Văn phong rõ ràng, liền mạch; tránh liệt kê khô cứng.

-> Quy tắc trích dẫn dẫn chứng: Khi dùng thông tin từ NGỮ CẢNH, gắn số trang ngay sau câu dựa vào tag [D#-p#] đúng như trong NGỮ CẢNH đc cung cấp. Không bịa số trang.
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
# --- CACHED_PROMPT (Giữ nguyên) ---
try:
    SYSTEM_PROMPT = load_prompt("system_prompt.txt")
    # ... (code tạo CACHED_PROMPT giữ nguyên) ...
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)
    CACHED_PROMPT = QA_PROMPT.format(subject="General").strip()
    logging.info("✅ Đã nạp system_prompt.txt")
except Exception as e:
     # ... (code fallback giữ nguyên) ...
     logging.error(f"Lỗi khi nạp system_prompt.txt: {e}. Sử dụng prompt mặc định.")
     SYSTEM_PROMPT = "# 🎓 VAI TRÒ\nBạn là trợ giảng AI..."
     QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)
     CACHED_PROMPT = QA_PROMPT.format(subject="General").strip()

