from typing import List, Dict
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import torch


class Retriever:
    def __init__(
        self,
        index_path: str = "index/faiss.index",
        model_name: str = "BAAI/bge-m3",
        top_k: int = 5,
    ):
        self.index_path = index_path
        self.top_k = top_k

        self.embeddings, self.metadata = joblib.load(index_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Retriever using {device=}")
        self.model = SentenceTransformer(model_name, device=device)

        print("embedding shape:", type(self.embeddings), getattr(self.embeddings, "shape", None))
        print("metadata count:", len(self.metadata))

    def retrieve(self, query: str) -> List[Dict]:
        query_vec = self.model.encode(query, convert_to_numpy=True)
        scores = self.embeddings @ query_vec / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
        )
        top_indices = np.argsort(scores)[::-1][:self.top_k]

        results = []
        for i in top_indices:
            meta = self.metadata[i]
            print("="*100)
            print(f"[DEBUG] retrieve meta: {meta}")
            results.append({
                "file_path": meta["file_path"],
                "symbol": meta.get("symbol", "unknown"),
                "content": meta.get("content", ""),
                "score": float(scores[i]),
            })
        return results
