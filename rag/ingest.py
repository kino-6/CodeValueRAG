from pathlib import Path
from rag.data_loader import CodeDataLoader
from rag.embedder import CodeEmbedder
from rag.indexer import VectorIndexer
from rag.utils import setup_logger

logger = setup_logger(__name__)

class IngestPipeline:
    def __init__(
        self,
        repo_root: Path,
        model_name: str = "BAAI/bge-m3",
        index_path: Path = Path("index/faiss.index"),
    ):
        self.repo_root = repo_root
        self.embedder = CodeEmbedder(model_name)
        self.indexer = VectorIndexer(index_path)
        logger.debug(f"retriever metadata sample: {self.indexer.metadata[:3]}")

    def run(self):
        # Load code files (includes content and symbol)
        loader = CodeDataLoader(self.repo_root)
        documents = loader.load()

        if not documents:
            logger.warning("No code files found.")
            return

        # Extract content and full metadata
        texts = [doc["content"] for doc in documents]
        metadata = [
            {
                "file_path": doc["file_path"],
                "symbol": doc["symbol"],
                "content": doc["content"],
            }
            for doc in documents
        ]
        logger.debug(f"metadata sample: {metadata[:3]}")

        # Embed
        embeddings = self.embedder.embed(texts)

        # Save to index
        self.indexer.add(embeddings, metadata)
        logger.debug("="*100)
        logger.debug(f"indexer.metadata before save: {self.indexer.metadata[:3]}")
        self.indexer.save()
        logger.debug(f"indexer.metadata after save: {self.indexer.metadata[:3]}")
        logger.debug("="*100)
        logger.info(f"Index saved to {self.indexer.index_path}")
