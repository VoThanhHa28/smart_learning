"""
Test hàm retrieve_answer đầy đủ với DEBUG logs
"""
import asyncio
import logging

# ✅ BẬT DEBUG LOGS
logging.basicConfig(
    level=logging.DEBUG,  # ← Changed to DEBUG
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)

from app.rag.chain import retrieve_answer

async def test():
    question = "Làm sao để hiểu rõ bản thân?"

    filters = {
        "user_id": "eejzjccrylv10n94kgvunmecjb92",
        "subject": "vatly",
        "course_id": "vatly_daa8be08"
    }

    print("="*80)
    print(f"🔍 Testing retrieve_answer")
    print("="*80)
    print(f"Question: {question}")
    print(f"Filters: {filters}")
    print("="*80)

    try:
        answer_gen, docs = await retrieve_answer(
            question=question,
            subject=filters["subject"],
            course_id=filters["course_id"],
            filters=filters
        )

        print(f"\n✅ Retrieved {len(docs)} documents\n")

        # Print sources
        if docs:
            print("📚 Source Documents:")
            for i, doc in enumerate(docs, 1):
                print(f"\n{i}. {doc.page_content[:100]}...")
                print(f"   Meta: {doc.metadata}")
        else:
            print("⚠️ NO DOCUMENTS RETURNED!")
            print("   This is why LLM says 'không đề cập'")

        # Get answer
        print("\n" + "="*80)
        print("📝 Generated Answer:")
        print("="*80)

        full_answer = ""
        async for token in answer_gen:
            print(token, end='', flush=True)
            full_answer += token
        
        print("\n" + "="*80)
        print(f"\n✅ Answer length: {len(full_answer)} chars")
        
        if "không đề cập" in full_answer.lower():
            print("⚠️ WARNING: LLM says 'không đề cập' despite having sources!")
            print("   This means the prompt or context is not working properly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())