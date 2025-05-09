from pathlib import Path
from rag.data_loader import CodeDataLoader
from rag.embedder import CodeEmbedder
from rag.indexer import VectorIndexer

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
        print(f"[DEBUG] retriever metadata sample: {self.indexer.metadata[:3]}")

    def run(self):
        # Load code files (includes content and symbol)
        loader = CodeDataLoader(self.repo_root)
        documents = loader.load()

        if not documents:
            print("No code files found.")
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
        # print(f"[DEBUG] metadata sample: {metadata[:3]}")

        # Embed
        embeddings = self.embedder.embed(texts)

        # Save to index
        self.indexer.add(embeddings, metadata)
        print("="*100)
        print(f"[DEBUG] indexer.metadata before save: {self.indexer.metadata[:3]}")
        self.indexer.save()
        print(f"[DEBUG] indexer.metadata after save: {self.indexer.metadata[:3]}")
        print("="*100)
        print(f"Index saved to {self.indexer.index_path}")
