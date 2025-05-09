# rag/display.py

from typing import List, Dict
from difflib import unified_diff
import ast
from pathlib import Path

def show_full_code(results):
    for i, item in enumerate(results, start=1):
        print(f"[DEBUG] display item: {item}")  # ←ここ
        file_path = Path(item["file_path"])
        func_name = item.get("symbol")
        content = extract_function_source(file_path, func_name) if func_name else ""
        print(f"\n[{i}] {file_path} #{func_name} #")
        print("-" * 80)
        print(content if content else "[No content found]")
        print("-" * 80)


def extract_function_source(file_path: Path, func_name: str) -> str:
    print("="*100)
    print(f"[DEBUG] extract_function_source: {file_path=}, {func_name=}")
    print(f"[DEBUG] file exists: {file_path.exists()}")

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        print(f"[DEBUG] AST nodes: {[node.name for node in ast.iter_child_nodes(tree) if hasattr(node, 'name')]}")

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):  # ←クラスを探す
                for child in ast.iter_child_nodes(node):
                    if hasattr(child, 'name') and child.name == func_name:  # ←クラス内のメソッドを探す
                        print(f"[DEBUG] found function: {child.name}")
                        start = child.lineno - 1
                        end = max([n.lineno for n in ast.walk(child) if hasattr(n, 'lineno')])
                        return "\n".join(source.splitlines()[start:end])
    except Exception as e:
        print(f"[Error] Failed to extract from {file_path}: {e}")
    return ""


def show_diff(query: str, results: List[Dict], top_k: int = 5):
    query_lines = query.strip().splitlines()
    for i, result in enumerate(results[:top_k]):
        result_lines = result.get("content", "").strip().splitlines()
        diff = unified_diff(query_lines, result_lines, lineterm="")
        print(f"\n[{i+1}] {result['file_path']} #{result.get('function_name', '')}")
        print("-" * 80)
        print("\n".join(diff) or "No diff (identical or empty)")
        print("-" * 80)
