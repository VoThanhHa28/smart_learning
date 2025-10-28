from pymilvus import connections, Collection

connections.connect("default", host="localhost", port="19530")
collection = Collection("smart_learning")
collection.load()

print("="*80)
print("📊 DETAILED DATA CHECK")
print("="*80)

# Query tất cả 10 records
results = collection.query(
    expr="",
    output_fields=["user_id", "subject", "course_id", "page"],
    limit=10
)

print(f"\n✅ Found {len(results)} records:\n")

# Group by user_id
user_data = {}
for r in results:
    uid = r.get('user_id')
    if uid not in user_data:
        user_data[uid] = []
    user_data[uid].append(r)

# Display by user
for uid, records in user_data.items():
    print(f"👤 User ID: {uid}")
    print(f"   Total chunks: {len(records)}")
    
    # Group by course
    courses = {}
    for r in records:
        cid = r.get('course_id')
        if cid not in courses:
            courses[cid] = {'subject': r.get('subject'), 'chunks': []}
        courses[cid]['chunks'].append(r)
    
    print(f"   Courses:")
    for cid, info in courses.items():
        print(f"     • {cid}")
        print(f"       Subject: {info['subject']}")
        print(f"       Chunks: {len(info['chunks'])}")
    print()

print("="*80)
print("🔍 CHECKING SPECIFIC USER FROM FLUTTER")
print("="*80)

# User ID từ Flutter (Firebase UID)
flutter_user = "eejzjccrylv10n94kgvunmecjb92"
print(f"\n🔎 Looking for user: {flutter_user}")

flutter_results = collection.query(
    expr=f'user_id == "{flutter_user}"',
    output_fields=["subject", "course_id"],
    limit=100
)

if flutter_results:
    print(f"✅ Found {len(flutter_results)} chunks")
    courses_for_user = {}
    for r in flutter_results:
        cid = r.get('course_id')
        if cid not in courses_for_user:
            courses_for_user[cid] = {
                'subject': r.get('subject'),
                'count': 0
            }
        courses_for_user[cid]['count'] += 1
    
    print("\n📚 Courses for this user:")
    for cid, info in courses_for_user.items():
        print(f"  • course_id: {cid}")
        print(f"    subject: {info['subject']}")
        print(f"    chunks: {info['count']}")
else:
    print("❌ No data found for this user!")
    print("\n💡 Possible reasons:")
    print("  1. Upload chưa gửi đúng user_id")
    print("  2. Data thuộc về user_id khác")
    print("  3. Backend không lưu user_id khi insert")

connections.disconnect("default")