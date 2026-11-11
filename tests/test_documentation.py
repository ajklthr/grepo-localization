"""Guard the source-level documentation contract."""

import ast
import unittest
from pathlib import Path


class DocumentationContractsTest(unittest.TestCase):
    def test_every_module_function_and_class_has_a_docstring(self):
        missing = []
        for path in Path("src/grepo_localization").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if ast.get_docstring(tree) is None:
                missing.append(f"{path}: module")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if ast.get_docstring(node) is None:
                        missing.append(f"{path}:{node.lineno}: {node.name}")

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
