import os
import asyncio
import logging
from typing import Any, AsyncGenerator
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY chưa được đặt trong .env")

@dataclass
class _Msg:
    content: str

class GeminiLLM:
    def __init__(self, model_name: str = GEMINI_MODEL):
        self.model_name = model_name
        self.model: Any = None

        try:
            self.model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                top_k=32,
                top_p=0.9,
                streaming=True,  # bật streaming thật
            )
            logging.info(f"✅ LangChain Model ({model_name}) initialized with streaming.")
        except Exception as e:
            raise RuntimeError(f"Không thể khởi tạo LLM LangChain: {e}") from e

    # -------------------- Sync invoke --------------------
    def invoke(self, prompt: str) -> _Msg:
        try:
            response = self.model.invoke(prompt)
            content = getattr(response, "content", str(response) or "")
        except Exception as e:
            logging.exception(f"[LLM Adapter {self.model_name}] Lỗi invoke: {e}")
            content = f"[Lỗi invoke: {e}]"
        return _Msg(content=content)

    # -------------------- Async invoke --------------------
    async def ainvoke(self, prompt: str) -> _Msg:
        try:
            response = await self.model.ainvoke(prompt)
            content = getattr(response, "content", str(response) or "")
        except Exception as e:
            logging.exception(f"[LLM Adapter {self.model_name}] Lỗi ainvoke: {e}")
            content = f"[Lỗi ainvoke: {e}]"
        return _Msg(content=content)

    # -------------------- Streaming --------------------
    async def astream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            async for chunk in self.model.astream(prompt):
                token = getattr(chunk, "content", "") or ""
                if token:
                    yield token
        except Exception as e:
            logging.exception(f"[LLM Adapter {self.model_name}] Lỗi astream: {e}")
            yield f"[Lỗi astream: {e}]"

# -------------------- Helpers --------------------
_llm_singleton: GeminiLLM | None = None

def get_llm() -> GeminiLLM:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = GeminiLLM(GEMINI_MODEL)
    return _llm_singleton

# -------------------- Warmup --------------------
def warmup_llm(prompt: str = "Hello") -> None:
    llm = get_llm()
    _ = llm.invoke(prompt)

async def warmup_llm_async(prompt: str = "Hello") -> None:
    llm = get_llm()
    _ = await llm.ainvoke(prompt)
