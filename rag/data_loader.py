from pathlib import Path
import ast
from typing import List, Dict
from rag.utils import setup_logger

logger = setup_logger(__name__)

class CodeDataLoader:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def load(self) -> List[Dict]:
        documents = []
        for file_path in self.root_dir.rglob("*.py"):
            try:
                source = file_path.read_text(encoding="utf-8")
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start_line = node.lineno - 1
                        end_line = self._get_end_line(node)
                        code_lines = source.splitlines()[start_line:end_line]
                        code_text = "\n".join(code_lines)
                        symbol = node.name
                        documents.append({
                            "file_path": str(file_path),
                            "symbol": symbol,
                            "content": code_text,
                        })
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
        logger.debug(f"Documents sample: {documents[:3]}")
        return documents

    def _get_end_line(self, node: ast.AST) -> int:
        last_lineno = getattr(node, "lineno", 0)
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                last_lineno = max(last_lineno, child.lineno)
        return last_lineno
