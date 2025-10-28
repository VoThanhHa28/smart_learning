"""
Test với câu hỏi phù hợp với nội dung chunks
"""
import asyncio
from app.rag.retrieval.vectorstore import similarity_search

async def test():
    # ✅ Query phù hợp với nội dung chunks (về tư vấn nghề nghiệp)
    queries = [
        "Làm sao để hiểu rõ bản thân?",
        "Cách tìm mentor như thế nào?",
        "Bước đầu tiên để chọn nghề là gì?",
        "Cây là gì?"  # Query không liên quan
    ]

    where = {
        "user_id": "eejzjccrylv10n94kgvunmecjb92",
        "subject": "vatly",
        "course_id": "vatly_daa8be08"
    }

    for query in queries:
        print("="*80)
        print(f"🔍 Query: {query}")
        print("="*80)

        try:
            results = await similarity_search(
                query=query,
                k=3,
                where=where,
                log=False
            )

            print(f"✅ Found {len(results)} results\n")

            if results:
                for i, (doc, score) in enumerate(results, 1):
                    print(f"{i}. Score: {score:.4f}")
                    print(f"   Text: {doc.page_content[:80]}...")
                    print()
            else:
                print("❌ No results\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(test())