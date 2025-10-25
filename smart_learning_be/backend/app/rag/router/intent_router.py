import yaml
from pathlib import Path

# thư mục hiện tại của file này: app/rag/router/
_THIS_DIR = Path(__file__).resolve().parent
# app/ (root của package app)
_APP_DIR = _THIS_DIR.parents[2]
# backend/ (root dự án backend)
_BACKEND_DIR = _THIS_DIR.parents[1] / ".."
_BACKEND_DIR = _BACKEND_DIR.resolve()

ROUTING_FILE = _THIS_DIR / "intent_routing.yaml"
PROMPT_FILE  = _BACKEND_DIR / "configs" / "prompt_catalog.yaml"

def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

ROUTING_CONF = _load_yaml(ROUTING_FILE)
PROMPT_PATHS = _load_yaml(PROMPT_FILE)

def get_routing(intent: str) -> dict:
    intent = (intent or "").lower()
    return ROUTING_CONF.get(intent, ROUTING_CONF["_default"])

def get_prompt_path(intent: str) -> str:
    intent = (intent or "").lower()
    return PROMPT_PATHS.get(intent, PROMPT_PATHS.get("fallback"))
