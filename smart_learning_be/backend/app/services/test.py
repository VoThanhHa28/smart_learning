from pymilvus import connections, Collection

# Kết nối trước
connections.connect("default", host="localhost", port="19530")
c = Collection("smart_learning")
res = c.search(
    data=[[0.1]*1024],
    anns_field="embedding",
    param={"ef": 128},
    limit=12,
    expr=None
)
print(res)
