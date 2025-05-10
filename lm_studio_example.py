from rag.lm_studio_client import LMStudioClient
from rag.retriever import Retriever
import json
from pathlib import Path
from typing import List, Dict, Optional

def format_response(response: str) -> str:
    """
    Format the response to make it more readable.
    
    Args:
        response: The raw response text
        
    Returns:
        Formatted response string
    """
    # コードブロックを保持しながら、テキストを整形
    lines = response.split('\n')
    formatted_lines = []
    in_code_block = False
    
    for line in lines:
        if line.startswith('```'):
            in_code_block = not in_code_block
            formatted_lines.append(line)
        elif in_code_block:
            formatted_lines.append(line)
        else:
            # テキスト行の整形
            line = line.strip()
            if line:
                formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def count_lines(text: str) -> int:
    """
    Count the number of lines in a text.
    
    Args:
        text: The text to count lines in
        
    Returns:
        Number of lines
    """
    return len(text.splitlines())

def build_enhanced_prompt(query: str, results: List[Dict]) -> str:
    """
    Build an enhanced prompt for code generation with better context and requirements.
    
    Args:
        query: The original query
        results: Retrieved code results
        
    Returns:
        Enhanced prompt string
    """
    # 参照コードの抽出
    reference_code = results[0]['content'] if results else 'No reference code available'
    
    # 依存関係の抽出
    imports = []
    for line in reference_code.split('\n'):
        if line.startswith('import ') or line.startswith('from '):
            imports.append(line.strip())
    
    # プロンプトの構築
    prompt = f"""Based on the following reference code, implement a complete and production-ready class:

Reference Code:
{reference_code}

Required Dependencies:
{chr(10).join(imports)}

Implementation Requirements:
1. Complete type hints for all methods and variables
2. Comprehensive error handling with custom exceptions
3. Detailed docstrings following Google style
4. Input validation for all parameters
5. Proper initialization of all instance variables
6. Follow the same style and structure as the reference code
7. Include all necessary methods for the class functionality
8. Add unit tests for the implementation

Please provide the implementation in the following format:

1. Main class implementation
2. Custom exceptions
3. Unit tests

Please provide ONLY the code implementation, no explanations or additional text."""

    return prompt

def main():
    client = LMStudioClient()

    # まず接続テストを実行
    if client.test_connection():
        print("Connection test successful!")
        
        # RAGコンポーネントの初期化（top_k=2を設定）
        retriever = Retriever(index_path="index/faiss.index", top_k=2)
        
        # コード生成のクエリ
        query = "Implement a PID controller class similar to the one in ODrive, including error handling and type hints"
        
        print(f"\nRetrieving relevant code for: {query}")
        results = retriever.retrieve(query)
        
        # 検索結果の表示
        print("\n=== Retrieved Code ===")
        for res in results:
            print(f"\nScore: {res['score']:.4f}")
            print(f"File: {res['file_path']}")
            print(f"Symbol: {res['symbol']}")
            print("Code:")
            print(res['content'])
            print("-" * 80)
        
        # プロンプトの構築
        prompt = build_enhanced_prompt(query, results)

        print("\n=== Generated Prompt ===")
        print(prompt)
        print("=======================\n")
        
        # コード生成の実行
        print("Generating code based on retrieved context...")
        response = client.generate_code(prompt)
        
        if response:
            print("\n=== Generated Code ===")
            formatted_response = format_response(response)
            print(formatted_response)
            print("=====================\n")
            
            # レスポンスの統計情報を表示
            print("Response Statistics:")
            print("- Total lines:", count_lines(response))
            print("- Contains code block:", '```' in response)
            print("- Response length:", len(response), "characters")
            
            # 追加の分析
            if '```' in response:
                print("\nCode Analysis:")
                print("- Contains docstrings:", '"""' in response or "'''" in response)
                print("- Contains type hints:", ': ' in response and '->' in response)
                print("- Contains error handling:", 'try' in response or 'except' in response)
                print("- Contains unit tests:", 'test_' in response or 'Test' in response)
        else:
            print("Code generation failed!")
    else:
        print("Connection test failed!")

if __name__ == "__main__":
    main()
