from .vectorstore import get_vectorstore
from .vectorstore import get_all_docs

docs = get_all_docs()
print("Docs in DB:", len(docs))

vs = get_vectorstore()
print("Docs in DB:", vs._collection.count())
