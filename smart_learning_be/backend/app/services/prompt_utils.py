# app/services/prompt_utils.py
import os
from langchain_core.prompts import PromptTemplate

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT = load_prompt("prompts/system_prompt.txt")
APPENDIX = load_prompt("prompts/appendix_examples.txt")
USE_APPENDIX = os.getenv("USE_APPENDIX", "false").lower() == "true"

if USE_APPENDIX:
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT + "\n\n" + APPENDIX)
else:
    QA_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT)
CACHED_PROMPT = QA_PROMPT.partial(subject="General")