from typing import List, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class LightReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def rerank(self, query: str, docs: List[str], top_k: int = 10, batch_size: int = 32) -> List[Tuple[int, float]]:
        if not docs:
            return []
        scores: List[float] = []
        pairs = [[query, d] for d in docs]
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits.squeeze(-1)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            scores.extend(logits.float().cpu().tolist())
        order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        top_k = min(top_k, len(order))
        return [(j, float(scores[j])) for j in order[:top_k]]
