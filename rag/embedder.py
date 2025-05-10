import os
import hashlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import joblib
import torch
from rag.utils import setup_logger

logger = setup_logger(__name__)

class CodeEmbedder:
    """
    A class for embedding code/text using a pre-trained SentenceTransformer model,
    with GPU utilization and local caching.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: Path = Path("index/cache")):
        """
        Initialize the CodeEmbedder.

        Args:
            model_name (str): Hugging Face model name.
            cache_dir (Path): Directory to store cached embeddings.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _text_hash(self, texts: list[str]) -> str:
        """
        Generate a SHA256 hash for the list of texts to use as cache key.
        """
        joined = "\n".join(texts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts, using cache if available.

        Args:
            texts (list[str]): List of input strings.

        Returns:
            np.ndarray: Matrix of embeddings.
        """
        key = self._text_hash(texts)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            return joblib.load(cache_file)

        embeddings = []
        for text in tqdm(texts, desc="Embedding"):
            embedding = self.model.encode(text, convert_to_numpy=True)
            embeddings.append(embedding)

        result = np.vstack(embeddings)
        joblib.dump(result, cache_file)
        return result
