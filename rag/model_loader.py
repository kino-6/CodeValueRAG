from transformers import AutoModel, AutoTokenizer
from rag.utils import setup_logger

logger = setup_logger(__name__)

class ModelLoader:
    def __init__(self, model_name: str = "BAAI/bge-code-v1.5"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self):
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        return self.model, self.tokenizer
