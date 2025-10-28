"""
Script kiểm tra dữ liệu trong Milvus
"""
from pymilvus import connections, Collection
import sys

def check_milvus_data(user_id=None):
    """Kiểm tra data trong Milvus"""

    print("="*80)
    print("🔍 MILVUS DATA CHECK")
    print("="*80)

    try:
        # Kết nối
        print("\n1️⃣ Connecting to Milvus...")
        connections.connect("default", host="localhost", port="19530")
        print("   ✅ Connected")

        # Load collection
        collection = Collection("smart_learning")
        collection.load()
        print(f"   ✅ Collection loaded: {collection.name}")

        # Đếm tổng số entities
        total = collection.num_entities
        print(f"\n2️⃣ Total entities in collection: {total}")

        if total == 0:
            print("\n   ⚠️ COLLECTION IS EMPTY!")
            print("   💡 Upload a PDF to add data")
            return

        # Liệt kê partitions
        print(f"\n3️⃣ Partitions:")
        for p in collection.partitions:
            print(f"   • {p.name}: {p.num_entities} entities")

        # Query tất cả users
        print(f"\n4️⃣ Users in database:")
        all_data = collection.query(
            expr="",
            output_fields=["user_id", "subject", "course_id"],
            limit=16384
        )

        # Group by user
        users = {}
        for r in all_data:
            uid = r.get('user_id', 'unknown')
            if uid not in users:
                users[uid] = {'courses': {}, 'total_chunks': 0}

            cid = r.get('course_id', 'unknown')
            if cid not in users[uid]['courses']:
                users[uid]['courses'][cid] = {
                    'subject': r.get('subject', 'unknown'),
                    'count': 0
                }
            users[uid]['courses'][cid]['count'] += 1
            users[uid]['total_chunks'] += 1

        print(f"   Found {len(users)} unique users:")
        for uid, data in users.items():
            print(f"\n   👤 User: {uid}")
            print(f"      Total chunks: {data['total_chunks']}")
            print(f"      Courses: {len(data['courses'])}")
            for cid, info in data['courses'].items():
                print(f"        • {cid} ({info['subject']}): {info['count']} chunks")

        # Kiểm tra user cụ thể
        if user_id:
            print(f"\n5️⃣ Checking specific user: {user_id}")
            user_data = collection.query(
                expr=f'user_id == "{user_id}"',
                output_fields=["subject", "course_id", "text"],
                limit=100
            )

            if user_data:
                print(f"   ✅ Found {len(user_data)} chunks for this user")

                # Group by course
                courses = {}
                for r in user_data:
                    cid = r.get('course_id')
                    if cid not in courses:
                        courses[cid] = {
                            'subject': r.get('subject'),
                            'chunks': []
                        }
                    courses[cid]['chunks'].append(r)

                print(f"\n   📚 Courses:")
                for cid, info in courses.items():
                    print(f"      • {cid}")
                    print(f"        Subject: {info['subject']}")
                    print(f"        Chunks: {len(info['chunks'])}")
                    # Sample text
                    sample_text = info['chunks'][0].get('text', '')
                    print(f"        Sample: {sample_text[:100]}...")
            else:
                print(f"   ❌ No data found for user: {user_id}")
                print(f"\n   💡 Tips:")
                print(f"      1. Check user_id is correct: {user_id}")
                print(f"      2. Upload a PDF file from Flutter app")
                print(f"      3. Check backend logs during upload")

        print("\n" + "="*80)
        print("✅ CHECK COMPLETE")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            connections.disconnect("default")
            print("\n👋 Disconnected")
        except:
            pass

if __name__ == "__main__":
    # Kiểm tra với user_id cụ thể
    user_id = "eejzjccrylv10n94kgvunmecjb92"  # ← User của bạn

    if len(sys.argv) > 1:
        user_id = sys.argv[1]

    check_milvus_data(user_id)