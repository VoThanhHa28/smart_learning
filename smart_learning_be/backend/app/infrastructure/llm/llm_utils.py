# app/services/llm_utils.py
import time, logging
from concurrent.futures import ThreadPoolExecutor
from google.api_core.exceptions import ServiceUnavailable
from grpc import RpcError
from .llm import get_llm

def sync_stream_generate(prompt_text: str) -> str:
    stage = "stream_generate"
    max_attempts = 2
    delay = 0.8
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            llm = get_llm(streaming=True)
            out = ""
            for chunk in llm.stream(prompt_text):
                token = getattr(chunk, "content", "") or ""
                out += token
            logging.info(f"[LLM:{stage}] streaming ok | len={len(out)} chars")
            return out
        except (ServiceUnavailable, RpcError) as e:
            last_err = e
            logging.exception(f"[LLM:{stage}] attempt={attempt} ServiceUnavailable/RpcError: {e}")
            time.sleep(delay); delay *= 1.6
        except Exception as e:
            last_err = e
            logging.exception(f"[LLM:{stage}] attempt={attempt} unknown error: {e}")
            time.sleep(delay); delay *= 1.6

    logging.warning(f"[LLM:{stage}] Falling back to non-streaming invoke after failures. last_err={last_err}")
    try:
        llm2 = get_llm(streaming=False)
        res = llm2.invoke(prompt_text)
        return getattr(res, "content", str(res)) or ""
    except Exception as e2:
        logging.exception(f"[LLM:{stage}] fallback invoke failed: {e2} | original={last_err}")
        return ""
