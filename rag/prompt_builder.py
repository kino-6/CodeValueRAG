from typing import List, Dict


class PromptBuilder:
    """
    Builds prompts for LLM based on user query and related code chunks.
    """

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or (
            "You are an expert embedded systems developer. "
            "Given a user's question and relevant code, provide a concise explanation or example in Python."
        )

    def build(self, query: str, code_chunks: List[Dict[str, str]]) -> str:
        """
        Constructs a prompt for the LLM from the query and code chunks.

        Args:
            query (str): The user's natural language query.
            code_chunks (List[Dict]): List of related code chunks with 'file_path', 'function_name', and 'content'.

        Returns:
            str: Prompt string suitable for LLM input.
        """
        prompt = [f"[System Instruction]\n{self.system_prompt}\n"]
        prompt.append(f"[User Question]\n{query}\n")

        for i, chunk in enumerate(code_chunks, 1):
            file = chunk.get("file_path", "unknown")
            name = chunk.get("function_name", "unknown")
            content = chunk.get("content", "")
            prompt.append(f"[Reference {i}] {file}#{name}\n```\n{content}\n```")

        prompt.append("[Answer]")
        return "\n\n".join(prompt)
