# rag/display.py

from typing import List, Dict
from difflib import unified_diff
import ast
from pathlib import Path
from rag.utils import setup_logger

logger = setup_logger(__name__)

def show_full_code(results):
    for i, item in enumerate(results, start=1):
        logger.debug(f"display item: {item}")
        file_path = Path(item["file_path"])
        func_name = item.get("symbol")
        content = extract_function_source(file_path, func_name) if func_name else ""
        logger.info(f"\n[{i}] {file_path} #{func_name} #")
        logger.info("-" * 80)
        logger.info(content if content else "[No content found]")
        logger.info("-" * 80)


def extract_function_source(file_path: Path, func_name: str) -> str:
    logger.debug("="*100)
    logger.debug(f"extract_function_source: {file_path=}, {func_name=}")
    logger.debug(f"file exists: {file_path.exists()}")

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        logger.debug(f"AST nodes: {[node.name for node in ast.iter_child_nodes(tree) if hasattr(node, 'name')]}")

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):  # ←クラスを探す
                for child in ast.iter_child_nodes(node):
                    if hasattr(child, 'name') and child.name == func_name:  # ←クラス内のメソッドを探す
                        logger.debug(f"found function: {child.name}")
                        start = child.lineno - 1
                        end = max([n.lineno for n in ast.walk(child) if hasattr(n, 'lineno')])
                        return "\n".join(source.splitlines()[start:end])
    except Exception as e:
        logger.error(f"Failed to extract from {file_path}: {e}")
    return ""


def show_diff(query: str, results: List[Dict], top_k: int = 5):
    query_lines = query.strip().splitlines()
    for i, result in enumerate(results[:top_k]):
        result_lines = result.get("content", "").strip().splitlines()
        diff = unified_diff(query_lines, result_lines, lineterm="")
        logger.info(f"\n[{i+1}] {result['file_path']} #{result.get('function_name', '')}")
        logger.info("-" * 80)
        logger.info("\n".join(diff) or "No diff (identical or empty)")
        logger.info("-" * 80)
