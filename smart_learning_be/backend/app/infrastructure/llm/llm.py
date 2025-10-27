import os, asyncio
from dataclasses import dataclass
# 1. Import AsyncIterable, Dict, Optional từ typing
from typing import Iterable, AsyncIterable, Dict, Optional # <-- Thêm Dict, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import logging # Thêm logging

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@dataclass
class _Msg:
    """Class đơn giản để chuẩn hóa output từ LLM."""
    content: str

class _GeminiAdapter:
    """
    Adapter class để bao bọc ChatGoogleGenerativeAI,
    cung cấp các phương thức invoke, ainvoke, stream, và astream nhất quán.
    """
    def __init__(self, streaming: bool = False):
        if not GOOGLE_API_KEY:
            raise ValueError("❌ GOOGLE_API_KEY not set")

        logging.info(f"Khởi tạo ChatGoogleGenerativeAI (Model: {GEMINI_MODEL}, Streaming: {streaming})...")
        try:
            # Khởi tạo LLM gốc từ Langchain
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                top_k=32,
                top_p=0.9,
                # Bỏ model_kwargs={"candidate_count": 1} nếu không chắc chắn model hỗ trợ
                # streaming=streaming, # Tham số này có thể không cần thiết khi gọi stream/astream
                # convert_system_message_to_human=True # Thêm nếu gặp lỗi về system message
            )
            logging.info("✅ ChatGoogleGenerativeAI đã khởi tạo.")
        except Exception as e:
             logging.exception(f"💥 Lỗi khi khởi tạo ChatGoogleGenerativeAI: {e}")
             raise e

    # --- Các phương thức đồng bộ (Sync) ---
    def invoke(self, prompt: str) -> _Msg:
        logging.debug(f"[LLM Adapter] Calling invoke...")
        res = self.llm.invoke(prompt)
        content = getattr(res, "content", str(res) or "")
        logging.debug(f"[LLM Adapter] invoke finished (len={len(content)}).")
        return _Msg(content=content)

    def stream(self, prompt: str) -> Iterable[_Msg]:
        logging.debug(f"[LLM Adapter] Calling stream...")
        try:
            for chunk in self.llm.stream(prompt):
                # Langchain trả về AIMessageChunk hoặc tương tự
                c = getattr(chunk, "content", None)
                if isinstance(c, str) and c: # Kiểm tra c là string và không rỗng
                    yield _Msg(content=c)
        except Exception as e:
            logging.exception(f"[LLM Adapter] Lỗi trong quá trình stream: {e}")
            raise e # Ném lại lỗi để bên ngoài xử lý
        finally:
            logging.debug(f"[LLM Adapter] stream finished.")

    # --- Các phương thức bất đồng bộ (Async) ---
    async def ainvoke(self, prompt: str) -> _Msg:
        logging.debug(f"[LLM Adapter] Calling ainvoke...")
        res = await self.llm.ainvoke(prompt)
        content = getattr(res, "content", str(res) or "")
        logging.debug(f"[LLM Adapter] ainvoke finished (len={len(content)}).")
        return _Msg(content=content)

    # --- 2. THÊM PHƯƠNG THỨC astream ---
    async def astream(self, prompt: str) -> AsyncIterable[_Msg]:
        """
        Ủy thác lệnh gọi astream đến self.llm và yield _Msg.
        """
        logging.debug(f"[LLM Adapter] Calling astream...")
        try:
            # Gọi self.llm.astream() gốc của Langchain
            async for chunk in self.llm.astream(prompt):
                # Langchain trả về AIMessageChunk hoặc tương tự
                c = getattr(chunk, "content", None)
                if isinstance(c, str) and c: # Kiểm tra c là string và không rỗng
                    yield _Msg(content=c)
        except AttributeError as ae:
             logging.error(f"[LLM Adapter] Lỗi: Đối tượng self.llm loại '{type(self.llm).__name__}' không có phương thức 'astream'. Kiểm tra phiên bản Langchain.")
             # Ném lại lỗi để bên ngoài biết
             raise ae from None
        except Exception as e:
            logging.exception(f"[LLM Adapter] Lỗi trong quá trình astream: {e}")
            raise e # Ném lại lỗi để bên ngoài xử lý (ví dụ: async_stream_generate)
        finally:
             logging.debug(f"[LLM Adapter] astream finished.")
    # --- KẾT THÚC THÊM astream ---

# --- Singleton Pattern (Giữ nguyên) ---
# Sử dụng Dict và Optional đã import
_llm_singletons: Dict[bool, Optional[_GeminiAdapter]] = {True: None, False: None}

def get_llm(streaming: bool = False) -> _GeminiAdapter:
    """Lấy instance LLM (singleton)."""
    # Key là boolean (streaming)
    key = streaming
    if _llm_singletons.get(key) is None:
        try:
             _llm_singletons[key] = _GeminiAdapter(streaming=streaming)
        except ValueError as ve: # Bắt lỗi API key thiếu
             logging.error(ve)
             raise ve from None # Dừng app nếu thiếu key
        except Exception as e: # Bắt lỗi khởi tạo LLM khác
             logging.exception(f"Không thể khởi tạo LLM (streaming={streaming}): {e}")
             # Có thể raise lỗi ở đây hoặc trả về None/dummy tùy chiến lược xử lý lỗi
             raise e from None
    # Thêm kiểm tra None để đảm bảo trả về đúng type hint
    adapter = _llm_singletons[key]
    if adapter is None:
        # Trường hợp này không nên xảy ra nếu raise lỗi ở trên
        raise RuntimeError(f"Không thể lấy LLM instance (streaming={streaming})")
    return adapter

# --- Warmup function (Giữ nguyên) ---
def warmup_llm():
    """Ping LLM để kiểm tra kết nối khi khởi động."""
    logging.info("Warming up LLM...")
    try:
        # Gọi invoke qua adapter
        a = get_llm(streaming=False).invoke("Warm up. Reply OK.")
        _ = a.content
        logging.info("✅ LLM warmup successful.")
    except Exception as e:
        # Chỉ warning, không dừng app
        logging.warning(f"⚠️ LLM warmup failed (có thể thử lại sau): {e}")

