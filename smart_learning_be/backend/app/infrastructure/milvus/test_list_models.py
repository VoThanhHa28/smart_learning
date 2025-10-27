import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
print("🔑 GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("📋 Listing available Gemini models...\n")

for m in genai.list_models():
    print(f"- {m.name}")
    # Nếu muốn xem chi tiết hỗ trợ gì:
    # print(vars(m))
