from pymilvus import connections, Collection, utility
import os
connections.connect("default", host=os.getenv("MILVUS_HOST","localhost"), port=os.getenv("MILVUS_PORT","19530"))
coll = Collection(os.getenv("MILVUS_COLLECTION","smart_learning"))
print("num_entities:", coll.num_entities)            # ❗ phải > 0
print("partitions:", [p.name for p in coll.partitions])
print("indexes:", [ix.index_name for ix in coll.indexes])
