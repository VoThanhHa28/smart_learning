import yaml, os

ROUTING_FILE = "configs/intent_routing.yaml"
PROMPT_FILE = "configs/prompt_catalog.yaml"

def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

ROUTING_CONF = _load_yaml(ROUTING_FILE)
PROMPT_PATHS = _load_yaml(PROMPT_FILE)

def get_routing(intent: str) -> dict:
    intent = (intent or "").lower()
    return ROUTING_CONF.get(intent, ROUTING_CONF["_default"])

def get_prompt_path(intent: str) -> str:
    intent = (intent or "").lower()
    return PROMPT_PATHS.get(intent, PROMPT_PATHS.get("fallback"))
