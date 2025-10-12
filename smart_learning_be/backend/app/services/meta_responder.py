# app/services/meta_responder.py
import json
import os
import random
import logging

META_PATH = os.path.join("prompts", "meta_responses.json")

try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        META_RESPONSES = json.load(f)
except Exception as e:
    logging.warning(f"[MetaResponder] ⚠️ Could not load {META_PATH}: {e}")
    META_RESPONSES = {}

def get_meta_response(meta_type: str) -> str:
    """
    Trả về phản hồi tương ứng với meta_type (greeting / identity / ...)
    Nếu không có trong file JSON → fallback về 'other'.
    """
    meta_type = meta_type.lower().strip()
    options = META_RESPONSES.get(meta_type) or META_RESPONSES.get("other", [])
    if not options:
        return "Xin chào! Mình là trợ giảng ảo, sẵn sàng hỗ trợ học tập."
    return random.choice(options)
