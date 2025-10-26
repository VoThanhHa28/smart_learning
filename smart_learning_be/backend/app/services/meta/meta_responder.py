import logging
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
import os

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "meta_intents"

# --- Sửa lỗi 1: Chuyển sang Lazy Loading ---
_model = None

def get_meta_model():
    """
    Hàm này sẽ nạp model (1 lần) khi được gọi.
    Và nó sẽ ra lệnh KHÔNG DÙNG TOKEN.
    """
    global _model
    if _model is None:
        logging.info(f"Đang nạp model 'meta_responder': {MODEL_NAME}")
        try:
            # --- Sửa lỗi 2: Thêm token=False để tắt xác thực (Fix 401) ---
            _model = SentenceTransformer(MODEL_NAME, token=False)
            logging.info("✅ Model 'meta_responder' đã nạp xong.")
        except Exception as e:
            logging.error(f"Lỗi khi nạp model 'meta_responder': {e}")
            # Báo lỗi để app dừng lại nếu không nạp được model
            raise e 
    return _model

def search_meta(question: str, k: int = 3, threshold: float = 0.60):
    """Meta search dùng E5 chuẩn (query/passages), index COSINE"""
    
    # --- Sửa lỗi 3: Gọi getter bên trong hàm ---
    try:
        model = get_meta_model()
    except Exception as e:
        logging.error(f"Không thể tải model cho search_meta: {e}")
        return "Lỗi: Bộ xử lý meta chưa sẵn sàng."

    try:
        connections.connect("default", host=os.getenv("MILVUS_HOST", "localhost"), port="19530")
        collection = Collection(COLLECTION_NAME)
        collection.load()
    except Exception as e:
        logging.error(f"Lỗi kết nối Milvus trong search_meta: {e}")
        return "Lỗi: Không thể kết nối đến cơ sở dữ liệu meta."

    q_raw = (question or "").strip().lower()
    q_for_e5 = f"query: {q_raw}"
    
    # Dùng biến `model` (đã được nạp)
    vec = model.encode([q_for_e5], normalize_embeddings=True)[0].astype("float32")

    results = collection.search(
        data=[[float(x) for x in vec]],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=k,
        output_fields=["question", "answer", "intent_type"]
    )

    if not results or len(results[0]) == 0:
        return None

    hit = results[0][0]
    distance = float(hit.distance)      # Milvus COSINE = 1 - cos_sim
    cos_sim = distance                  # (Bạn đã gán distance = cos_sim, ok)
    
    if cos_sim < threshold:
        logging.info(f"DEBUG_meta_sim_E5: cos_sim={cos_sim:.3f} < threshold={threshold:.2f}")
        return None

    ans = hit.entity.get("answer", "")
    intent = hit.entity.get("intent_type", "")
    logging.info(f"🔍 Meta intent matched: {intent} (cos_sim={cos_sim:.3f}) | q='{hit.entity.get('question','')}'")
    return ans

if __name__ == "__main__":
    # Sửa cả __main__ để dùng getter
    m = get_meta_model() # Dùng getter
    q = m.encode(["query: chào bạn"], normalize_embeddings=True)[0]
    p = m.encode(["passage: chào"], normalize_embeddings=True)[0]
    cos = float(np.dot(q, p))
    print("LOCAL COS =", round(cos, 3))