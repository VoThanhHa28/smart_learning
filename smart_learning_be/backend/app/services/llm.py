# app/services/llm.py
import os, asyncio
from dataclasses import dataclass
from typing import Iterable
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@dataclass
class _Msg:
    content: str

class _GeminiAdapter:
    def __init__(self, streaming: bool = False):
        if not GOOGLE_API_KEY:
            raise ValueError("❌ GOOGLE_API_KEY not set")
        # KHÔNG gắn StreamingStdOutCallbackHandler để tránh in đúp
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,  # type: ignore
            temperature=0.3,
            top_k=32,
            top_p=0.9,
            model_kwargs={"candidate_count": 1},  # type: ignore
            streaming=streaming,  # type: ignore
        )

    # sync
    def invoke(self, prompt: str) -> _Msg:
        res = self.llm.invoke(prompt)
        # LangChain trả AIMessage có .content
        return _Msg(content=getattr(res, "content", str(res) or ""))

    # async
    async def ainvoke(self, prompt: str) -> _Msg:
        res = await self.llm.ainvoke(prompt)
        return _Msg(content=getattr(res, "content", str(res) or ""))

    # streaming: yield các chunk có .content
    def stream(self, prompt: str) -> Iterable[_Msg]:
        for chunk in self.llm.stream(prompt):
            c = getattr(chunk, "content", None)
            if c:
                yield _Msg(content=c)
                
_llm_singletons = {True: None, False: None}

def get_llm(streaming: bool = False) -> _GeminiAdapter:
    if _llm_singletons[streaming] is None:
        _llm_singletons[streaming] = _GeminiAdapter(streaming=streaming)
    return _llm_singletons[streaming]

# tuỳ chọn
def warmup_llm():
    try:
        a = get_llm().invoke("Warm up. Reply OK.")
        _ = a.content
    except Exception:
        pass
