import requests
from pathlib import Path

OCR_URL = "http://ocr:9000/ocr"  # service name từ docker-compose

def call_ocr(file_path: str) -> str:
    """Gửi file sang OCR service, nhận text trả về"""
    with open(file_path, "rb") as f:
        resp = requests.post(OCR_URL, files={"file": (Path(file_path).name, f, "application/octet-stream")})
    resp.raise_for_status()
    return resp.json().get("text", "")
