"""
Kiểm tra data của course cụ thể
"""
from pymilvus import connections, Collection

connections.connect("default", host="localhost", port="19530")
collection = Collection("smart_learning")
collection.load()

user_id = "eejzjccrylv10n94kgvunmecjb92"
course_id = "vatly_daa8be08"

print("="*80)
print(f"🔍 Checking data for course: {course_id}")
print("="*80)

# Query với filter chính xác như API
expr = f'user_id == "{user_id}" and subject == "vatly" and course_id == "{course_id}"'
print(f"Filter expression: {expr}\n")

results = collection.query(
    expr=expr,
    output_fields=["user_id", "subject", "course_id", "text", "page"],
    limit=10
)

print(f"✅ Found {len(results)} chunks\n")

if results:
    for i, r in enumerate(results, 1):
        print(f"{i}. Page: {r.get('page')}")
        print(f"   User: {r.get('user_id')}")
        print(f"   Subject: {r.get('subject')}")
        print(f"   Course: {r.get('course_id')}")
        print(f"   Text: {r.get('text')[:100]}...")
        print()
else:
    print("❌ NO DATA FOUND!")
    print("\n💡 Debugging:")

    # Check without course_id filter
    expr2 = f'user_id == "{user_id}" and subject == "vatly"'
    results2 = collection.query(
        expr=expr2,
        output_fields=["course_id"],
        limit=100
    )

    if results2:
        courses = set(r.get('course_id') for r in results2)
        print(f"   Available course_ids for subject 'vatly': {courses}")
    else:
        print("   No data found for subject 'vatly' either")

connections.disconnect("default")