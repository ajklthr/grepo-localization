"""Tests for reusable transformations used by the exploration notebook."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data

from grepo_localization.exploration import (
    candidate_table,
    choose_nodes,
    count_types,
    export_issue_graphs,
    first_nonempty_batch,
    graphsage_logits,
    issue_graph_view,
    select_issue_row,
)


class ExplorationHelpersTest(unittest.TestCase):
    """Verify notebook helpers without network access or notebook state."""

    def test_type_counts_and_issue_selection(self):
        """Pure summary helpers preserve IDs and validate explicit selection."""
        self.assertEqual(count_types(torch.tensor([[4, 3], [4, 1]])), {1: 1, 3: 1, 4: 2})
        issues = pd.DataFrame([
            {"repo": "demo", "issue_id": 1, "gold": [], "seeds": [2]},
            {"repo": "demo", "issue_id": 2, "gold": [3], "seeds": [2]},
        ])
        self.assertEqual(select_issue_row(issues)["issue_id"], 2)
        self.assertEqual(select_issue_row(issues, "demo", 1)["issue_id"], 1)
        with self.assertRaises(ValueError):
            select_issue_row(issues, repo="demo")

    def test_candidate_ranking_and_graph_view(self):
        """Candidate and display selection are deterministic and label-aware."""
        data = Data(
            original_node_id=torch.tensor([10, 20, 30]),
            candidate_mask=torch.tensor([True, False, True]),
        )
        records = {
            10: SimpleNamespace(node_type=4, path="a.py", name="a"),
            30: SimpleNamespace(node_type=3, path="b.py", name="B"),
        }
        issue = {"seeds": [30], "gold": [10]}
        candidates = candidate_table(issue, data, records)
        selected, name = choose_nodes(candidates, 2, None)
        self.assertEqual(selected, [30, 10])
        self.assertEqual(name, "Anchor-first baseline")
        with self.assertRaises(ValueError):
            choose_nodes(candidates, -1, None)

        graph = nx.MultiDiGraph([(10, 20), (20, 30), (30, 40)])
        view, groups = issue_graph_view(issue, graph, max_nodes=3)
        self.assertEqual(set(view), {10, 20, 30})
        self.assertEqual(groups["gold"], {10})
        self.assertEqual(groups["seeds"], {30})

    def test_graphsage_smoke_test_and_export(self):
        """Batch validation and atomic export expose inspectable results."""
        empty = Data(x=torch.empty((0, 2)), edge_index=torch.empty((2, 0), dtype=torch.long))
        graph = Data(
            x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor([1.0, 0.0]),
            candidate_mask=torch.tensor([True, True]),
            repo="demo",
            issue_id=7,
        )
        self.assertIs(first_nonempty_batch([empty, graph]), graph)
        self.assertEqual(graphsage_logits(graph, torch.device("cpu")).shape, (2,))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "export"
            manifest = export_issue_graphs(
                {"train": [graph]}, pd.DataFrame([{"issue_id": 7}]), root
            )
            self.assertEqual(manifest.loc[0, "split"], "train")
            self.assertTrue((root / manifest.loc[0, "file"]).is_file())
            self.assertTrue((root / "splits.csv").is_file())
            self.assertTrue((root / "coverage.csv").is_file())


if __name__ == "__main__":
    unittest.main()
