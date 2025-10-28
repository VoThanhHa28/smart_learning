"""
Test query API endpoint
"""
import requests
import json

def test_query():
    url = "http://localhost:8000/query"

    headers = {
        "Content-Type": "application/json",
        "X-User-ID": "eejzjccrylv10n94kgvunmecjb92"
    }

    payload = {
        "question": "Cây là gì?",
        "subject": "vatly",
        "course_id": "vatly_daa8be08",
        "top_k": 3
    }

    print("="*80)
    print("🔍 Testing Query API")
    print("="*80)
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    print("="*80)

    try:
        print("\n⏳ Sending request...")
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=30
        )

        print(f"✅ Response Status: {response.status_code}\n")

        if response.status_code != 200:
            print(f"❌ Error Response: {response.text}")
            return

        print("📝 Streaming Response:")
        print("="*80)

        # Read streaming response
        full_response = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end='', flush=True)
                full_response += chunk

        print("\n" + "="*80)

        # Parse metadata if present
        if "__METADATA__" in full_response:
            parts = full_response.split("__METADATA__")
            answer_text = parts[0].strip()
            metadata_part = parts[1].split("__END_METADATA__")[0]

            print("\n📊 Parsed Results:")
            print(f"Answer: {answer_text[:200]}...")

            try:
                metadata = json.loads(metadata_part.strip())
                print(f"\nSources found: {len(metadata.get('sources', []))}")
                print(f"Elapsed time: {metadata.get('elapsed_ms')}ms")
                print(f"User ID: {metadata.get('user_id')}")

                for i, src in enumerate(metadata.get('sources', [])[:3], 1):
                    print(f"\n  Source {i}:")
                    print(f"    Page: {src.get('page')}")
                    print(f"    Text: {src.get('text', '')[:100]}...")

            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse metadata: {e}")

        print("\n✅ Test completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error!")
        print("💡 Backend không chạy. Hãy chạy:")
        print("   cd smart-learning_be/backend")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_query()