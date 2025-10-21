from pymilvus import Collection, connections
connections.connect("default", host="localhost", port="19530")
c = Collection("smart_learning")
res = c.query(expr='course_id == "p1"', output_fields=["page", "course_id", "subject"])
print(len(res), res[:2])
