from typing import List
from transformers import pipeline

class LLMResponder:
    """
    Generates natural language responses based on a query and contextual code using an LLM.
    """
    def __init__(self, model_name: str = "tiiuae/falcon-7b-instruct", device: int = -1):
        """
        Initialize the LLM pipeline.

        Args:
            model_name (str): Hugging Face model to use.
            device (int): Device to run on. Use -1 for CPU, >=0 for GPU index.
        """
        self.generator = pipeline("text-generation", model=model_name, device=device)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate a response based on the prompt.

        Args:
            prompt (str): Full prompt to send to the LLM.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: Generated text.
        """
        outputs = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()
