import os
from langchain_core.prompts import PromptTemplate
from typing import Union, List, Dict, Any # <-- Thêm Dict, Any
from pathlib import Path
import logging
from ..rag_utils import build_context
from langchain_core.documents import Document # <-- Thêm Document

# --- Hàm load_prompt (Giữ nguyên) ---
def load_prompt(path: str) -> str:
    base = Path(__file__).resolve().parent
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    try:
        with p.open("r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.warning(f"Không tìm thấy file prompt: {p}. Sẽ dùng prompt mặc định.")
        # Cung cấp một prompt hệ thống mặc định TỐI GIẢN và AN TOÀN
        return """# 🎓 VAI TRÒ
Bạn là trợ giảng AI môn học. Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.

# 🚫 RÀNG BUỘC
- Chỉ dùng thông tin trong NGỮ CẢNH.
- Nếu không có thông tin, trả lời: "Xin lỗi, ngữ cảnh không chứa thông tin để trả lời câu hỏi này."
- Trích dẫn nguồn [D#-p#] nếu có."""

# --- HÀM BUILD PROMPT CẤU TRÚC V5 (ĐÃ SỬA QUY TẮC 1) ---
def build_structured_prompt_block(
    subject: str,
    structured_context: List[Dict[str, Any]], # List [{'subq': str, 'docs': List[Document], 'kind': 'academic'|'system', 'role'?, 'text'?}]
    original_full_question: str,
    doc_source: str = "Tài liệu học tập",
) -> str:
    """Xây dựng khối prompt V5 với cấu trúc sub-question / context riêng."""

    context_blocks: List[str] = []
    sub_questions_list: List[str] = []
    has_system_block = False # Cờ để biết có block system hay không

    if not structured_context:
         logging.warning("[Prompt Builder V5] structured_context rỗng.")
         # Trả về prompt đơn giản yêu cầu trả lời câu hỏi gốc
         return f"""
# 🎯 NHIỆM VỤ
Trả lời câu hỏi sau một cách tốt nhất có thể: "{original_full_question}"
# 📝 BẮT ĐẦU TRẢ LỜI:""".strip()

    # Build từng block
    for i, item in enumerate(structured_context):
        kind = (item or {}).get("kind", "unknown")
        subq = item.get("subq", f"Phần {i+1}")
        sub_questions_list.append(f"{i+1}. {subq}")

        if kind == "system":
            has_system_block = True # Đánh dấu có system block
            role = item.get("role", "system info")
            sys_text = (item.get("text") or "").strip()
            # **Thay đổi:** Không còn gọi là TEXT-TO-INSERT nữa
            context_blocks.append(f"""
## Câu hỏi {i+1}: {subq} (Loại: Thông tin hệ thống - {role})
### Thông tin cần tích hợp:
{sys_text if sys_text else "(Không có thông tin)"}
""".strip())

        elif kind == "academic":
            docs = item.get("docs", [])
            context_str_for_subq = build_context(docs) if docs else "Không có ngữ cảnh cụ thể được tìm thấy cho phần này."
            context_blocks.append(f"""
## Câu hỏi {i+1}: {subq} (Loại: Học thuật)
### Ngữ cảnh liên quan (Chỉ dùng phần này):
{context_str_for_subq}
""".strip())
        # Bỏ qua kind 'unknown' hoặc các loại khác nếu có

    full_context_section = "\n\n---\n".join(context_blocks)
    all_sub_questions = "\n".join(sub_questions_list)

    # Tạo prompt cuối
    return f"""
# 📘 THÔNG TIN CHI TIẾT (Tách theo Câu hỏi con)
{full_context_section}

---

# 🎓 VAI TRÒ
Bạn là trợ giảng môn {subject}. Tên tài liệu tham khảo chính là {doc_source}. Hãy dùng tiếng Việt.

# 🎯 NHIỆM VỤ TỔNG HỢP
Dựa **CHÍNH XÁC** vào **từng phần** [Thông tin chi tiết] ở trên, hãy viết **MỘT câu trả lời DUY NHẤT** mạch lạc, tổng hợp và đáp ứng **tất cả** các câu hỏi sau, theo đúng thứ tự:
{all_sub_questions}

# 🧭 QUY TẮC BẮT BUỘC (TUÂN THỦ NGHIÊM NGẶT):
1.  **TÍCH HỢP TỰ NHIÊN (Cho khối Hệ thống):** Nếu có phần "Thông tin cần tích hợp", hãy **kết hợp nội dung đó một cách tự nhiên** vào phần mở đầu hoặc phần liên quan của câu trả lời tổng hợp. **KHÔNG** cần chép lại y hệt nếu nó làm câu trả lời thiếu tự nhiên khi đi kèm câu hỏi khác. Ví dụ, nếu thông tin là lời chào và có câu hỏi học thuật đi kèm, chỉ cần chào ngắn gọn rồi trả lời phần học thuật.
2.  **CHỈ DÙNG NGỮ CẢNH (Cho khối Học thuật):** Với các câu hỏi học thuật, chỉ sử dụng "Ngữ cảnh liên quan" của đúng câu hỏi đó. Tuyệt đối không suy diễn hay dùng kiến thức ngoài.
3.  **TỔNG HỢP MẠCH LẠC:** Câu trả lời cuối cùng phải là **một đoạn văn duy nhất**, liền mạch, có câu nối giữa các ý nếu cần. Tránh trả lời rời rạc từng câu. Giọng văn thống nhất.
4.  **TRÍCH DẪN NGUỒN ([D#-p#]):** Khi sử dụng thông tin từ "Ngữ cảnh liên quan" (khối Học thuật) có tag nguồn, BẮT BUỘC phải gắn tag đó vào cuối câu/đoạn tương ứng.
5.  **XỬ LÝ THIẾU THÔNG TIN:** Nếu "Ngữ cảnh liên quan" bị thiếu, hãy bỏ qua câu hỏi con đó trong phần tổng hợp hoặc ghi chú rất ngắn gọn (ví dụ: "...tuy nhiên ngữ cảnh không nói về Y."). **KHÔNG** dùng câu fallback chung nếu ít nhất một phần có thể trả lời.
6.  **VĂN PHONG:** Như trợ giảng: rõ ràng, thân thiện, không dùng câu dẫn dắt ("Dựa vào...", "Trong tài liệu..."). Không chào hỏi (trừ khi được yêu cầu tích hợp ở Quy tắc 1).

# 📝 BẮT ĐẦU CÂU TRẢ LỜI TỔNG HỢP:
""".strip()
# --- Kết thúc hàm ---

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
- Trích dẫn nguồn [D#-p#] nếu có trong đoạn NGỮ CẢNH bạn đang sử dụng để trả lời.
- Không chào hỏi và nói các câu như "Chào bạn, dựa trên tài liệu, trong ngữ cảnh này,... các câu cứng nhắt" bởi vì nó không được hay như chatbot rag thực thụ.
- CHỈ trả về câu "Tài liệu không đề cập gì về chủ đề này." KHI TOÀN BỘ NGỮ CẢNH HOÀN TOÀN KHÔNG CÓ THÔNG TIN LIÊN QUAN ĐẾN CÂU HỎI.
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

# --- CACHED_PROMPT (Giữ nguyên) ---
try:
    # HÃY ĐẢM BẢO BẠN ĐÃ ĐƠN GIẢN HÓA system_prompt.txt
    SYSTEM_PROMPT = load_prompt("system_prompt.txt")
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)
    CACHED_PROMPT = QA_PROMPT.format(subject="General").strip()
    logging.info("✅ Đã nạp system_prompt.txt")
except Exception as e:
     logging.error(f"Lỗi khi nạp system_prompt.txt: {e}. Sử dụng prompt mặc định.")
     SYSTEM_PROMPT = """# 🎓 VAI TRÒ
Bạn là trợ giảng AI môn học. Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.
# 🚫 RÀNG BUỘC
- Chỉ dùng thông tin trong NGỮ CẢNH.
- Nếu không có thông tin, trả lời: "Xin lỗi, ngữ cảnh không chứa thông tin để trả lời câu hỏi này."
- Trích dẫn nguồn [D#-p#] nếu có."""
     QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)
     CACHED_PROMPT = QA_PROMPT.format(subject="General").strip()

