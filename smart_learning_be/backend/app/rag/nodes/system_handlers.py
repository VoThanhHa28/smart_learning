import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Any

# 1. IMPORT HÀM SEARCH_META "THẬT"
from ...services.meta.meta_responder import search_meta 
from ...services.utils.toc_store import get_toc_by_key, get_toc_by_course, flatten_toc
from ..prompts.prompt_utils import build_toc_validation_block
from ...infrastructure.llm.llm_utils import sync_stream_generate

async def handle_system(sq: str, state: dict, topic: str) -> Tuple[str, str]:
    """
    Xử lý các intent hệ thống (meta, unsafe, unclear, toc).
    Trả về (sub_question, answer_string).
    """
    print(f"      [SystemHandler] Internal: Đang xử lý: sq='{sq[:50]}...', topic='{topic}'")
    
    if topic == "unsafe":
        return sq, f"Xin lỗi, mình không thể hỗ trợ với nội dung \"{sq}\"."

    if topic == "unclear":
        return sq, f"Mình chưa rõ \"{sq}\" nghĩa là gì; bạn có thể mô tả cụ thể hơn không?"

    if topic == "meta":
        try:
            print(f"      [SystemHandler] ➡️  Gọi 'Tool' [search_meta]...")
            loop = asyncio.get_event_loop()
            meta_ans = await loop.run_in_executor(None, search_meta, sq)
            
            if meta_ans:
                print(f"      [SystemHandler] ✅ 'Tool' [search_meta] thành công.")
                return sq, meta_ans
            else:
                print(f"      [SystemHandler] ⚠️  'Tool' [search_meta] không tìm thấy kết quả.")
                return sq, "Xin lỗi, mình chưa có câu trả lời meta cụ thể cho câu hỏi này."
                
        except Exception as e:
            print(f"      [SystemHandler] ❌ LỖI khi đang chạy search_meta: {e}")
            return sq, "Lỗi: Đã có sự cố khi mình tìm câu trả lời meta."

    if topic == "toc":
        try:
            print(f"      [SystemHandler] ➡️  Đang xử lý TOC...")
            course_id = str(state.get("course_id") or "").strip()
            doc_key = state.get("doc_key")
            entry = get_toc_by_key(doc_key) if doc_key else get_toc_by_course(course_id)
            toc_lines = flatten_toc(entry)
            prompt_text = build_toc_validation_block(toc_lines)
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                ans = await loop.run_in_executor(pool, sync_stream_generate, prompt_text)
            print(f"      [SystemHandler] ✅ Xử lý TOC thành công.")
            return sq, ans
        except Exception as e:
            print(f"      [SystemHandler] ❌ LỖI khi xử lý TOC: {e}")
            return sq, "Lỗi: Không thể tạo mục lục."

    return sq, "Lỗi: Không nhận dạng được intent hệ thống."

