import time, logging
from concurrent.futures import ThreadPoolExecutor
from google.api_core.exceptions import ServiceUnavailable
from grpc import RpcError
# Import get_llm
from ..llm.llm import get_llm # Đảm bảo đường dẫn đúng
import asyncio
from typing import AsyncGenerator

# --- Hàm sync_stream_generate (Giữ nguyên) ---
def sync_stream_generate(prompt_text: str) -> str:
    """
    Hàm cũ: Gọi LLM stream nhưng block và trả về chuỗi đầy đủ.
    """
    stage = "sync_stream_generate"
    # ... (Code sync_stream_generate giữ nguyên) ...
    max_attempts = 2
    delay = 0.8
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            llm = get_llm(streaming=True)
            out = ""
            # Dùng llm.stream (sync)
            for chunk in llm.stream(prompt_text):
                # Langchain Gemini trả về BaseMessageChunk, cần getattr
                token = getattr(chunk, "content", "") or ""
                out += token
            return out # Trả về chuỗi đầy đủ
        except (ServiceUnavailable, RpcError) as e:
            last_err = e
            logging.exception(f"[LLM:{stage}] attempt={attempt} ServiceUnavailable/RpcError: {e}")
            time.sleep(delay); delay *= 1.6
        except Exception as e:
            last_err = e
            logging.exception(f"[LLM:{stage}] attempt={attempt} unknown error: {e}")
            time.sleep(delay); delay *= 1.6
    logging.warning(f"[LLM:{stage}] Falling back to non-streaming invoke...")
    try:
        llm2 = get_llm(streaming=False)
        res = llm2.invoke(prompt_text)
        return getattr(res, "content", str(res)) or ""
    except Exception as e2:
        logging.exception(f"[LLM:{stage}] fallback invoke failed: {e2} | original={last_err}")
        return ""


# --- HÀM MỚI: ASYNC STREAM GENERATOR (ĐÃ SỬA LẠI) ---
async def async_stream_generate(prompt_text: str) -> AsyncGenerator[str, None]:
    """
    Hàm mới: Gọi LLM astream và yield các chunk/token.
    """
    stage = "async_stream_generate"
    max_attempts = 2
    delay = 0.8
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            llm = get_llm(streaming=True) # Lấy LLM streaming

            # --- SỬA LẠI Ở ĐÂY: DÙNG astream() ---
            logging.debug(f"[LLM:{stage}] Attempt {attempt}: Calling llm.astream()...")
            async for chunk in llm.astream(prompt_text):
            # --- KẾT THÚC SỬA ---

                # Trích xuất token (Langchain trả về BaseMessageChunk)
                token = getattr(chunk, "content", "") or "" # Lấy content attribute
                # Đôi khi chunk đầu tiên/cuối cùng không có content
                if token:
                    # print(token, end="", flush=True) # Bỏ comment nếu muốn debug token
                    yield token
            logging.debug(f"[LLM:{stage}] Attempt {attempt}: astream() completed.")
            return # Kết thúc generator thành công

        except AttributeError as ae:
             # Nếu lỗi AttributeError ('_GeminiAdapter' object has no attribute 'astream')
             last_err = ae
             logging.error(f"[LLM:{stage}] attempt={attempt} Lỗi AttributeError: {ae}. Phương thức astream() không tồn tại cho LLM này.")
             # Thử fallback ngay lập tức
             break
        except TypeError as te:
             # Bắt lỗi TypeError ('async for' requires...) nếu astream trả về generator thường
             last_err = te
             logging.error(f"[LLM:{stage}] attempt={attempt} Lỗi TypeError: {te}. Phương thức astream() không trả về async generator.")
             # Thử fallback ngay lập tức
             break
        except (ServiceUnavailable, RpcError) as e:
            # Xử lý lỗi mạng/dịch vụ
            last_err = e
            logging.warning(f"[LLM:{stage}] attempt={attempt} ServiceUnavailable/RpcError: {e}") # Đổi thành warning
            # Nếu là lần thử cuối, không cần sleep nữa
            if attempt < max_attempts:
                 await asyncio.sleep(delay)
                 delay *= 1.6
            else:
                 logging.error(f"[LLM:{stage}] Lỗi mạng/dịch vụ sau {max_attempts} lần thử.")
                 break # Thoát vòng lặp để fallback
        except Exception as e:
            # Xử lý lỗi khác
            last_err = e
            logging.exception(f"[LLM:{stage}] attempt={attempt} unknown error: {e}")
            if attempt < max_attempts:
                 await asyncio.sleep(delay)
                 delay *= 1.6
            else:
                 logging.error(f"[LLM:{stage}] Lỗi không xác định sau {max_attempts} lần thử.")
                 break # Thoát vòng lặp để fallback

    # --- Fallback (Giữ nguyên logic gọi ainvoke) ---
    logging.warning(f"[LLM:{stage}] Streaming failed. Falling back to ainvoke.")
    try:
        llm_fallback = get_llm(streaming=False)
        # Sử dụng ainvoke cho non-streaming async call
        res = await llm_fallback.ainvoke(prompt_text)
        fallback_content = getattr(res, "content", str(res)) or ""
        if fallback_content:
            logging.info(f"[LLM:{stage}] Fallback ainvoke successful (len={len(fallback_content)}).")
            yield fallback_content # Yield toàn bộ nội dung 1 lần
        else:
            yield f"[Lỗi: Không thể tạo câu trả lời sau {max_attempts} lần thử streaming và fallback]"
            logging.error(f"[LLM:{stage}] Fallback ainvoke cũng thất bại hoặc trả về rỗng.")
    except Exception as e2:
        logging.exception(f"[LLM:{stage}] fallback ainvoke failed: {e2} | original stream error={last_err}")
        yield f"[Lỗi nghiêm trọng khi tạo câu trả lời fallback: {e2}]"
