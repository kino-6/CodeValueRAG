from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
from datetime import datetime
from tqdm import tqdm
import threading
import sys
import gc
import os

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
        self.model_name = model_name
        self.device = device
        self.is_generating = False
        self.current_tokens = 0
        self.start_time = None
        
        # Set environment variables for better memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Check GPU availability and optimize memory
        if torch.cuda.is_available():
            self.device = 0  # Use first GPU
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            # Set memory optimization
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of available VRAM
        else:
            self.device = -1  # Use CPU
            print("No GPU available, using CPU")

        # Initialize tokenizer with proper settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=2048,
            padding_side="left",
            truncation_side="left"
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure 4-bit quantization with more conservative settings
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_quant_storage=torch.float16
        )

        # Load model with memory optimizations
        try:
            print("Loading model with 4-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory={0: "6GiB", "cpu": "16GiB"},  # Further reduced GPU memory
                offload_folder="offload",
                offload_state_dict=True,
                torch_dtype=torch.float16,  # Explicitly set dtype
            )
            print(f"Model loaded successfully with 4-bit quantization")
            print(f"Memory usage after loading model: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
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
                # Print memory usage after loading the model
                print(f"Memory usage after loading model: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
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
                # Print memory usage after loading the model
                print(f"Memory usage after loading model: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Initialize generator with optimized settings
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,  # Increased from 512 to 1024
            do_sample=True,
            temperature=0.1,  # Increased from 0.05 to 0.1
            top_p=0.95,
            top_k=40,        # Increased from 20 to 40
            repetition_penalty=1.2,  # Increased from 1.1 to 1.2
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
            batch_size=1,
            use_cache=True,
            return_full_text=False,
            clean_up_tokenization_spaces=True,
            bad_words_ids=[
                [self.tokenizer.encode("Here is")[0]],
                [self.tokenizer.encode("Here's")[0]],
                [self.tokenizer.encode("The code")[0]],
                [self.tokenizer.encode("This code")[0]],
                [self.tokenizer.encode("Explanation")[0]],
                [self.tokenizer.encode("Note")[0]],
                [self.tokenizer.encode("Important")[0]],
                [self.tokenizer.encode("Output")[0]],
                [self.tokenizer.encode("Result")[0]],
                [self.tokenizer.encode("Generated")[0]],
                [self.tokenizer.encode("Example")[0]],
                [self.tokenizer.encode("Usage")[0]],
                [self.tokenizer.encode("Note that")[0]],
            ],
        )

    def _truncate_prompt(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Truncate the prompt to fit within token limits while preserving important context.
        """
        # Tokenize the prompt
        tokens = self.tokenizer.encode(prompt, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _post_process_response(self, response: str) -> str:
        """
        Post-process the generated response to improve quality.
        """
        # 説明文を削除
        if any(marker in response for marker in ["Here is", "Explanation", "Output", "Result", "Generated"]):
            # コードブロックを探す
            code_blocks = response.split("```")
            if len(code_blocks) > 1:
                # 最後のコードブロックを使用
                response = code_blocks[-1].strip()
            else:
                # コードブロックがない場合は最初の行を削除
                response = "\n".join(response.split("\n")[1:]).strip()
        
        # コードブロックのマークを削除
        response = response.replace("```python", "").replace("```", "").strip()
        
        # 空行を整理
        lines = [line for line in response.split("\n") if line.strip()]
        response = "\n".join(lines)
        
        # 構文エラーの修正
        response = response.replace("od.(ODrive)", "od = ODrive()")
        response = response.replace("od.-(ODrive.OD_HANDLES)", "od.reset()")
        response = response.replace("ser <<", "ser.write")
        response = response.replace("ser &lt;&lt;", "ser.write")
        response = response.replace("ser._write", "ser.write")
        response = response.replace("ser_os_pos.__get__", "get_encoder_position")
        response = response.replace("ser_os.write()", "write_encoder_position()")
        
        return response

    def _monitor_progress(self, max_tokens: int):
        """
        Monitor generation progress and update progress bar.
        """
        with tqdm(total=max_tokens, desc="Generating code", unit="tokens") as pbar:
            while self.is_generating:
                # Estimate progress based on time elapsed and average generation speed
                if self.start_time:
                    elapsed_time = time.time() - self.start_time
                    # Estimate tokens based on average speed (tokens per second)
                    estimated_tokens = int(elapsed_time * 2)  # Assuming 2 tokens per second
                    if estimated_tokens > self.current_tokens:
                        new_tokens = estimated_tokens - self.current_tokens
                        self.current_tokens = estimated_tokens
                        pbar.update(new_tokens)
                        # Update progress bar description with estimated time remaining
                        if estimated_tokens > 0:
                            tokens_per_second = estimated_tokens / elapsed_time
                            remaining_tokens = max_tokens - estimated_tokens
                            if tokens_per_second > 0:
                                remaining_time = remaining_tokens / tokens_per_second
                                pbar.set_postfix({
                                    'tokens/s': f'{tokens_per_second:.1f}',
                                    'ETA': f'{remaining_time:.0f}s',
                                    'VRAM': f'{torch.cuda.memory_allocated() / 1024**2:.0f}MB'
                                })
                time.sleep(0.1)

    def generate(self, prompt: str, max_tokens: int = 384, temperature: float = 0.7) -> str:
        """
        Generate a response based on the prompt.

        Args:
            prompt (str): Full prompt to send to the LLM.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: Generated text.
        """
        # Truncate prompt if necessary
        truncated_prompt = self._truncate_prompt(prompt)
        
        try:
            # Reset progress tracking
            self.current_tokens = 0
            self.is_generating = True
            self.start_time = time.time()
            
            # Start progress monitoring in a separate thread
            progress_thread = threading.Thread(target=self._monitor_progress, args=(max_tokens,))
            progress_thread.start()
            
            # Generate response with optimized settings
            outputs = self.generator(
                truncated_prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                return_full_text=False,
                num_return_sequences=1,
                use_cache=True,
            )
            
            # Stop progress monitoring
            self.is_generating = False
            progress_thread.join()
            
            # End timing
            end_time = time.time()
            end_datetime = datetime.now()
            
            # Calculate timing metrics
            generation_time = end_time - self.start_time
            tokens_per_second = max_tokens / generation_time if generation_time > 0 else 0
            
            # Format timing information
            timing_info = (
                f"\n=== Generation Timing ===\n"
                f"Start: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"End: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Total time: {generation_time:.2f} seconds\n"
                f"Tokens per second: {tokens_per_second:.2f}\n"
                f"Device: {'GPU' if self.device >= 0 else 'CPU'}\n"
                f"VRAM Usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB\n"
                f"VRAM Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f}MB\n"
                f"========================\n"
            )
            
            response = outputs[0]["generated_text"].strip()
            if not response:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question or providing more specific details."
            
            # Post-process the response
            processed_response = self._post_process_response(response)
            
            return processed_response + timing_info
            
        except Exception as e:
            self.is_generating = False  # Ensure progress monitoring stops on error
            return f"Error generating response: {str(e)}"
