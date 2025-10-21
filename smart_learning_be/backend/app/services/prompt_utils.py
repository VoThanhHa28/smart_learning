# app/services/prompt_utils.py
import os
from langchain_core.prompts import PromptTemplate
from typing import Union

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ===========================
# 🔁 Intent aliases (chuẩn hóa tên intent về file prompt)
# ===========================
INTENT_ALIASES = {
    "compare": "comparison",
    "problem": "exercise",
    "summarize": "summary",
    "summ": "summary",
    "def": "definition",
    "warn": "warning",
    "socratic": "socratic_question",
    "quiz": "quick_check",
    "simplify": "paraphrase",
    "router_fallback": "router_fallback",  # phòng khi router rơi vào fallback
    "multi_part": "multi_part",
}


PRIORITY = [
  "definition","comparison","multi_part","application","example","warning",
  "exercise","theory","summary","socratic_question","quick_check","paraphrase"
]

def normalize_and_order_intents(intents: list[str]) -> list[str]:
    # to lower + alias + unique (giữ thứ tự ưu tiên)
    seen = set()
    norm = []
    for raw in intents:
        i = (raw or "").lower().strip()
        i = INTENT_ALIASES.get(i, i) or "general"
        if i not in seen:
            seen.add(i)
            norm.append(i)
    # order by PRIORITY; những intent lạ đưa về cuối
    ordered = sorted(norm, key=lambda x: PRIORITY.index(x) if x in PRIORITY else 999)
    return ordered

def get_prompt_by_intent(intent: Union[str, list], doc_type: str = ""):
    # Ưu tiên TOC trước
    if doc_type and doc_type.lower() == "toc":
        toc_path = "prompts/intent_prompts/toc.txt"
        if os.path.exists(toc_path):
            return load_prompt(toc_path)

    # ⚠️ intent có thể là list (multi-intent) → lấy cái đầu tiên
    if isinstance(intent, list):
        intent = intent[0] if intent else "general"

    # 🔁 Chuẩn hóa intent theo aliases
    intent_norm = (intent or "").lower().strip()
    intent_norm = INTENT_ALIASES.get(intent_norm, intent_norm) or "general"

    intent_path = f"prompts/intent_prompts/{intent_norm}.txt"
    if os.path.exists(intent_path):
        return load_prompt(intent_path)

    return load_prompt("prompts/intent_prompts/general.txt")



SYSTEM_PROMPT = load_prompt("prompts/system_prompt.txt")
APPENDIX = load_prompt("prompts/appendix_examples.txt")
USE_APPENDIX = os.getenv("USE_APPENDIX", "false").lower() == "true"

if USE_APPENDIX:
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT + "\n\n" + APPENDIX)
else:
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)

# MỚI (trả về string đã fill sẵn): 
CACHED_PROMPT = QA_PROMPT.format(subject="General").strip()  # <-- string