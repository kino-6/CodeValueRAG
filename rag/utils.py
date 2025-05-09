# rag/utils.py

import ast
from typing import List, Tuple

def extract_functions_with_source(source_code: str) -> List[Tuple[str, str]]:
    """
    Extracts function names and their corresponding source code blocks from a Python script.
    
    Args:
        source_code (str): The entire Python source code.

    Returns:
        List[Tuple[str, str]]: List of (function_name, source_code) tuples.
    """
    tree = ast.parse(source_code)
    functions = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = max([getattr(child, "end_lineno", node.lineno) for child in ast.walk(node)])
            lines = source_code.splitlines()[start_line:end_line]
            function_code = "\n".join(lines)
            functions.append((node.name, function_code))

    return functions
