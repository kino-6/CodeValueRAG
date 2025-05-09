from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib
import os
from pathlib import Path


class VectorIndexer:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.model = None
        self.embeddings = None
        self.metadata = []

    def add(self, embeddings: np.ndarray, metadata: list):
        self.embeddings = embeddings  # 必須
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.model.fit(embeddings)
        self.metadata = metadata

    def save(self):
        os.makedirs(self.index_path.parent, exist_ok=True)
        joblib.dump((self.embeddings, self.metadata), self.index_path)

    def load(self):
        self.embeddings, self.metadata = joblib.load(self.index_path)
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.model.fit(self.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        distances, indices = self.model.kneighbors(query_embedding, n_neighbors=top_k)
        return [(self.metadata[i], 1 - distances[0][i]) for i in indices[0]]
