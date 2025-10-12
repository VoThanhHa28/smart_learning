import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_llm = None

def get_llm(streaming: bool = False):
    """
    Trả về Gemini LLM (hỗ trợ async).
    """
    global _llm
    if _llm is not None:
        return _llm

    if not GOOGLE_API_KEY:
        raise ValueError("❌ GOOGLE_API_KEY not set in .env")
    
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
    
    _llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        top_k=32,
        top_p=0.9,
        model_kwargs={"candidate_count": 1},
        streaming=streaming,       # ✅ dùng param này
        callbacks=callbacks        # ✅ để in token ra console
    )
    return _llm



# =======================
# 🧠 Local LLM cho tagging
# =======================
from langchain_ollama import ChatOllama

_local_llm = None

def get_local_llm(streaming: bool = False):
    """
    Dùng Ollama local cho semantic tagging:
    - Không cần API key
    - Tốc độ nhanh, JSON tốt
    - Giữ model preload (cache)
    """
    global _local_llm
    if _local_llm is not None:
        return _local_llm

    model = os.getenv("LOCAL_MODEL", "adrienbrault/nous-hermes2pro:Q4_K_S-json")  # hoặc qwen2.5:7b-instruct
    _local_llm = ChatOllama(
        model=model,
        temperature=0.2,
        top_p=0.9,
        num_thread=os.cpu_count(),
        num_ctx=4096,
        num_predict=512,
        format="json",      # ép output JSON
        streaming=streaming,
        keep_alive="2h"     # giữ model nóng 1 tiếng
    )
    return _local_llm
