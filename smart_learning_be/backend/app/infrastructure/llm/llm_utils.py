import asyncio
import logging
from typing import AsyncGenerator
from .llm import get_llm  # đảm bảo đường dẫn đúng với llm.py

async def async_stream_generate(prompt_text: str) -> AsyncGenerator[str, None]:
    """
    Gọi LLM astream và yield từng token (LangChain streaming thật).
    Fallback về ainvoke nếu streaming không khả dụng.
    """
    stage = "async_stream_generate"
    max_attempts = 2
    delay = 0.8
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            llm = get_llm()  # Chỉ dùng LangChain

            logging.debug(f"[LLM:{stage}] Attempt {attempt}: Calling llm.astream()...")
            async for chunk in llm.astream(prompt_text):
                yield chunk
            logging.debug(f"[LLM:{stage}] Attempt {attempt}: astream() completed.")
            return  # Streaming thành công, thoát generator

        except (AttributeError, TypeError) as e:
            # astream không khả dụng hoặc trả về sai kiểu
            last_err = e
            logging.error(f"[LLM:{stage}] attempt={attempt} Lỗi streaming: {e}")
            break
        except Exception as e:
            last_err = e
            logging.exception(f"[LLM:{stage}] attempt={attempt} Lỗi không xác định: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(delay)
                delay *= 1.6
            else:
                logging.error(f"[LLM:{stage}] Lỗi không xác định sau {max_attempts} lần thử.")
                break

    # --- Fallback về ainvoke LangChain ---
    logging.warning(f"[LLM:{stage}] Streaming failed. Falling back to ainvoke...")
    try:
        llm_fallback = get_llm()
        res = await llm_fallback.ainvoke(prompt_text)
        content = getattr(res, "content", str(res)) or ""
        if content:
            yield content  # Yield toàn bộ nội dung 1 lần
        else:
            logging.error(f"[LLM:{stage}] Fallback ainvoke trả về rỗng.")
            yield f"[Lỗi: Không thể tạo câu trả lời sau {max_attempts} lần thử streaming và fallback]"
    except Exception as e2:
        logging.exception(f"[LLM:{stage}] fallback ainvoke failed: {e2} | original stream error={last_err}")
        yield f"[Lỗi nghiêm trọng khi tạo câu trả lời fallback: {e2}]"
