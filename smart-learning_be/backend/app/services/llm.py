import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_llm = None

def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            top_k=32,
            top_p=0.9,
            model_kwargs={"candidate_count": 1}
        )
    elif LLM_PROVIDER == "ollama":
        _llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
    else:
        _llm = None

    return _llm
