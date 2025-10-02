import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# Singleton
_embeddings = None

class HFCustomEmbeddings(Embeddings):
    def __init__(self, model_name: str, batch_size: int = 64):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔥 Embedding model {model_name} running on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
        ).to(self.device)
        self.batch_size = batch_size

    def _mean_pooling(self, outputs, attention_mask):
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

    def embed_query(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = self._mean_pooling(outputs, inputs["attention_mask"])
        return embedding[0].cpu().numpy()

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
            embeddings.extend(batch_embeddings.cpu().numpy())
        return embeddings

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HFCustomEmbeddings(EMBED_MODEL, batch_size=64)
    return _embeddings
