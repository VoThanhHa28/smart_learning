from pymilvus import connections, Collection, list_collections

connections.connect("default", host="localhost", port="19530")
print("📦 localhost collections:", list_collections())

connections.disconnect("default")
connections.connect("default", host="milvus", port="19530")
print("📦 milvus (docker) collections:", list_collections())
