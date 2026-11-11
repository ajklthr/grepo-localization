"""Static quality checks for the public exploration notebook."""

import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "01_explore_grepo.ipynb"


class ExplorationNotebookTest(unittest.TestCase):
    """Keep notebook-only code documented and independent of private APIs."""

    @classmethod
    def code_cells(cls) -> list[tuple[int, str]]:
        """Return cell numbers and source for every code cell in the notebook."""
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        return [
            (index, "".join(cell.get("source", [])))
            for index, cell in enumerate(notebook["cells"])
            if cell["cell_type"] == "code"
        ]

    def test_code_cells_parse_and_avoid_private_corpus_methods(self):
        """Notebook orchestration must compile and use the public corpus API."""
        for index, source in self.code_cells():
            with self.subTest(cell=index):
                ast.parse(source)
                self.assertNotIn("corpus._", source)

    def test_notebook_functions_have_documented_typed_interfaces(self):
        """Display helpers retained in the notebook must state their interface."""
        for index, source in self.code_cells():
            tree = ast.parse(source)
            functions = [
                node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for function in functions:
                with self.subTest(cell=index, function=function.name):
                    self.assertIsNotNone(ast.get_docstring(function))
                    self.assertIsNotNone(function.returns)
                    self.assertTrue(
                        all(argument.annotation is not None for argument in function.args.args)
                    )


if __name__ == "__main__":
    unittest.main()
