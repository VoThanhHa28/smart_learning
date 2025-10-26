from langchain_milvus import Milvus as _Milvus
import traceback

class DebugMilvus(_Milvus):
    def similarity_search(self, *args, **kwargs):
        import traceback
        print("[DEBUG Milvus] similarity_search called without score!")
        traceback.print_stack(limit=5)
        return super().similarity_search(*args, **kwargs)

    def similarity_search_with_score(self, *args, **kwargs):
        import traceback
        print("[DEBUG Milvus] similarity_search_with_score called!", kwargs)
        traceback.print_stack(limit=5)
        return super().similarity_search_with_score(*args, **kwargs)
