from transformers import AutoModel, AutoTokenizer

class ModelLoader:
    def __init__(self, model_name: str = "BAAI/bge-code-v1.5"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        return self.model, self.tokenizer
