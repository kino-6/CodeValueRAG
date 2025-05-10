from typing import List, Dict
import os
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM


class PromptBuilder:
    """
    Builds prompts for LLM based on user query and related code chunks.
    """

    def __init__(self):
        self.system_prompt = """<s>[INST] You are a code generation assistant. Your task is to generate Python code based on the given query and context.

CRITICAL REQUIREMENTS:
1. ONLY generate valid, runnable Python code
2. NO explanations, NO markdown, NO comments except docstrings
3. ALL functions and methods MUST be complete and properly defined
4. ALL variables MUST be properly defined before use
5. ALL imports MUST be used in the code
6. ALL method calls MUST reference existing methods
7. ALL syntax MUST be valid Python
8. ALL type hints MUST be correct
9. ALL error handling MUST be implemented
10. ALL code MUST follow the reference implementation style
11. ALL variable names MUST be spelled correctly
12. ALL imports MUST be properly spelled
13. ALL method names MUST be properly spelled
14. ALL class names MUST be properly spelled
15. ALL constants MUST be properly defined
16. ALL code MUST be complete and not cut off
17. ALL code MUST be properly indented
18. ALL code MUST be properly formatted
19. ALL code MUST be properly structured
20. ALL code MUST be properly documented

CODE STRUCTURE:
1. Required imports (only used ones)
2. Class/function definitions with docstrings
3. Complete method implementations
4. Proper error handling
5. Return values with type hints

VERIFICATION:
- Check all variable names are defined and spelled correctly
- Check all method calls exist and are spelled correctly
- Check all imports are used and spelled correctly
- Check syntax is valid Python
- Check error handling is complete
- Check all constants are properly defined
- Check all class names are properly spelled
- Check all method names are properly spelled
- Check all code is complete and not cut off
- Check all code is properly indented
- Check all code is properly formatted
- Check all code is properly structured
- Check all code is properly documented

DO NOT include any text before or after the code. ONLY output the code itself. [/INST]"""

    def _extract_relevant_code(self, results: List[Dict]) -> List[Dict]:
        """
        Extract and format relevant code chunks from search results.
        """
        relevant_code = []
        for result in results:
            if result.get("content"):
                # Add minimal context
                context = f"# {result['file_path']} - {result.get('symbol', 'Unknown')}\n"
                relevant_code.append({
                    "context": context,
                    "code": result["content"]
                })
        return relevant_code

    def _format_code_example(self, example: Dict) -> str:
        """
        Format a code example with its context.
        """
        return f"{example['context']}\n{example['code']}\n"

    def build(self, query: str, results: List[Dict]) -> str:
        """
        Build a prompt for the LLM.
        """
        # Format code examples with clear structure
        code_examples = []
        for result in results:
            code_examples.append(
                f"Reference Implementation:\n"
                f"File: {result['file_path']}\n"
                f"Symbol: {result['symbol']}\n"
                f"Code:\n{result['content']}\n"
                f"---\n"
            )

        # Construct the prompt with clear instructions
        prompt = f"{self.system_prompt}\n\n<s>[INST] Query: {query}\n\n"
        prompt += "Reference code examples to follow:\n"
        prompt += "\n".join(code_examples)
        prompt += "\nGenerate a complete, working implementation that follows the reference code style and structure. "
        prompt += "Ensure all methods are properly defined, all variables are properly initialized, and all names are spelled correctly. "
        prompt += "Output ONLY the code, no explanations. [/INST]"
        
        return prompt

    def load_model(self, model_name: str):
        """
        Load the model with memory optimizations.
        """
        # Set environment variables for better memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load model with memory optimizations
        try:
            print("Loading model with 4-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory={0: "12GiB"},  # Increased from 10GiB to 12GiB
                offload_folder="offload",
                offload_state_dict=True,
            )
        except Exception as e:
            print(f"Error loading model with 4-bit quantization: {e}")
            print("Trying with 8-bit quantization...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={0: "12GiB"},  # Increased from 10GiB to 12GiB
                    offload_folder="offload",
                    offload_state_dict=True,
                )
            except Exception as e:
                print(f"Error loading model with 8-bit quantization: {e}")
                print("Trying with CPU offloading...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={0: "12GiB", "cpu": "16GiB"},
                    offload_folder="offload",
                    offload_state_dict=True,
                )
