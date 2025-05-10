from typing import List, Dict
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import torch
from rag.utils import setup_logger

logger = setup_logger(__name__)

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
        logger.info(f"Retriever using device: {device}")
        self.model = SentenceTransformer(model_name, device=device)

        logger.debug(f"embedding shape: {type(self.embeddings)}, {getattr(self.embeddings, 'shape', None)}")
        logger.debug(f"metadata count: {len(self.metadata)}")

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query to improve search results.
        """
        # Add code-specific context to the query
        query = query.lower()
        if "how" in query and "work" in query:
            # For "how does X work" queries, add implementation-related terms
            query += " implementation code example"
        elif "how" in query and "use" in query:
            # For "how to use X" queries, add usage-related terms
            query += " usage example code"
        elif "how" in query and "implement" in query:
            # For implementation queries, add implementation-related terms
            query += " implementation code"
        return query

    def retrieve(self, query: str) -> List[Dict]:
        # Preprocess the query
        processed_query = self._preprocess_query(query)
        logger.debug(f"Original query: {query}")
        logger.debug(f"Processed query: {processed_query}")

        # Get embeddings for the processed query
        query_vec = self.model.encode(processed_query, convert_to_numpy=True)
        
        # Calculate cosine similarity
        scores = self.embeddings @ query_vec / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
        )
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:self.top_k]

        # Prepare results with additional context
        results = []
        for i in top_indices:
            meta = self.metadata[i]
            logger.debug("="*100)
            logger.debug(f"retrieve meta: {meta}")
            
            # Add file type and component information
            file_path = meta["file_path"]
            file_type = "test" if "test" in file_path.lower() else "implementation"
            component = file_path.split("/")[-2] if len(file_path.split("/")) > 1 else "unknown"
            
            results.append({
                "file_path": file_path,
                "symbol": meta.get("symbol", "unknown"),
                "content": meta.get("content", ""),
                "score": float(scores[i]),
                "file_type": file_type,
                "component": component,
            })
        
        # Sort results by score and file type (implementation files first)
        results.sort(key=lambda x: (-x["score"], x["file_type"] == "test"))
        return results
