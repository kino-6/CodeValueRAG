import requests
import json
import logging
from typing import Optional, Dict, Any, Generator
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LMStudioClient:
    """
    Client for interacting with LM Studio's local API endpoint.
    """

    def __init__(self, base_url: str = "http://192.168.10.141:1234", model: str = "logipt-codellama-13b-instruct-hf-proofwriter-i1"):
        """
        Initialize the LM Studio client.
        
        Args:
            base_url: The base URL of the LM Studio API endpoint
            model: The name of the model to use for generation
        """
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        
        # リトライ設定
        retry_strategy = Retry(
            total=3,  # 最大3回リトライ
            backoff_factor=1,  # リトライ間隔を1秒から開始
            status_forcelist=[500, 502, 503, 504]  # これらのステータスコードでリトライ
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # タイムアウト設定
        self.session.timeout = (5, 30)  # (connect timeout, read timeout)
        
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def test_connection(self) -> bool:
        """Test the connection to LM Studio server."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def _format_prompt(self, prompt: str, is_code: bool = False) -> str:
        """
        Format the prompt for better model response.
        
        Args:
            prompt: The input prompt
            is_code: Whether this is a code generation request
            
        Returns:
            Formatted prompt string
        """
        if is_code:
            return f"""<s>[INST] You are a code generation assistant. Please generate code based on the following requirements:

{prompt}

Please provide ONLY the code implementation, no explanations or additional text. [/INST]"""
        else:
            return f"""<s>[INST] {prompt} [/INST]"""

    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """
        Stream the response from LM Studio (SSE or JSON lines).
        Accumulates chunks into meaningful units and shows progress.
        """
        buffer = ""
        total_chunks = 0
        last_progress = 0
        
        logger.info("Starting to receive stream...")
        
        for line in response.iter_lines(decode_unicode=True):
            if line:
                # SSE形式なら "data: ..." を処理
                if line.startswith("data: "):
                    line = line[len("data: "):]
                if line.strip() == "[DONE]":
                    # バッファに残っている内容があれば出力
                    if buffer:
                        logger.info(f"\nFinal chunk received. Total chunks: {total_chunks}")
                        yield buffer
                    break
                try:
                    data = json.loads(line)
                    if 'choices' in data and len(data['choices']) > 0:
                        text = data['choices'][0].get('text', '')
                        if text:
                            buffer += text
                            total_chunks += 1
                            
                            # 進捗表示（10チャンクごと）
                            if total_chunks % 10 == 0:
                                logger.info(f"Received {total_chunks} chunks...")
                            
                            # 意味のある単位で区切って出力
                            if text.endswith('\n') or text.endswith('.') or text.endswith(';'):
                                logger.info(f"\nCombined output ({len(buffer)} chars):")
                                logger.info("-" * 40)
                                logger.info(buffer)
                                logger.info("-" * 40)
                                yield buffer
                                buffer = ""
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error: {e}")
                    continue
        
        # 最後のバッファがあれば出力
        if buffer:
            logger.info(f"\nFinal output ({len(buffer)} chars):")
            logger.info("-" * 40)
            logger.info(buffer)
            logger.info("-" * 40)
            yield buffer

    def _make_request(self, endpoint: str, payload: Dict[str, Any], timeout: tuple = (5, 30), stream: bool = False) -> Optional[Dict]:
        """
        Make a request to the LM Studio API with retry logic.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            timeout: Request timeout tuple (connect, read)
            stream: Whether to stream the response
            
        Returns:
            Response JSON or None if request failed
        """
        try:
            # モデル名を追加
            payload['model'] = self.model
            
            response = self.session.post(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                json=payload,
                timeout=timeout,
                stream=stream
            )
            
            if response.status_code == 200:
                if stream:
                    return response
                return response.json()
            else:
                logger.error(f"Request failed with status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except requests.Timeout:
            logger.error("Request timed out")
            return None
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def generate_response(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response text or None if generation failed
        """
        formatted_prompt = self._format_prompt(prompt)
        payload = {
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["</s>", "[/INST]"],
            "stream": True
        }
        
        logger.info("Sending request to LM Studio...")
        response = self._make_request("v1/completions", payload, timeout=(5, 60), stream=True)
        
        if response:
            full_text = ""
            try:
                for chunk in self._stream_response(response):
                    full_text += chunk
                return full_text.strip()
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                return None
        return None

    def generate_code(self, prompt: str) -> Optional[str]:
        """
        Generate code with specific parameters optimized for code generation.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Generated code or None if generation failed
        """
        formatted_prompt = self._format_prompt(prompt, is_code=True)
        
        # コード生成用のパラメータを設定
        payload = {
            "prompt": formatted_prompt,
            "max_tokens": 2000,  # コード生成にはより多くのトークンが必要
            "temperature": 0.2,  # より決定論的な出力
            "top_p": 0.95,
            "stop": ["</s>", "[/INST]"],
            "stream": True,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        logger.info("\n=== Sending Code Generation Prompt to LM Studio ===")
        logger.info("Prompt:")
        logger.info("=" * 50)
        logger.info(formatted_prompt)
        logger.info("=" * 50)
        
        logger.info("Sending request to LM Studio...")
        response = self._make_request("v1/completions", payload, timeout=(5, 90), stream=True)
        
        if response:
            full_text = ""
            try:
                for chunk in self._stream_response(response):
                    full_text += chunk
                return full_text.strip()
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                return None
        return None 