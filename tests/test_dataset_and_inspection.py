"""Contract tests for dataset adapters and coverage diagnostics."""

import unittest

import pandas as pd

from grepo_localization.dataset import GrepoIssueDataset, chronological_split
from grepo_localization.graphs import attach_candidate_labels, make_issue_graph
from grepo_localization.inspection import coverage_record, to_networkx_issue_graph
from tests.test_graphs import repository_graph


class DatasetAndInspectionContractsTest(unittest.TestCase):
    def test_chronological_split_and_lazy_dataset(self):
        issues = pd.DataFrame(
            [
                {"repo": "a", "time": time, "gold": [time], "value": time}
                for time in range(1, 6)
            ] + [{"repo": "b", "time": 1, "gold": [1], "value": 9}]
        )
        skipped = []

        split = chronological_split(issues, on_skip=skipped.append)
        dataset = GrepoIssueDataset(split, lambda row: row["value"] * 2)

        self.assertEqual(split["split"].tolist(), ["train"] * 4 + ["test"])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset[-1], 10)

    def test_networkx_conversion_and_coverage_record(self):
        issue = {
            "repo": "demo", "issue_id": 11, "time": 4,
            "seeds": [0], "gold": [1, 2],
        }
        graph = repository_graph()
        data = attach_candidate_labels(make_issue_graph(graph, issue, hops=1), issue["gold"])

        converted = to_networkx_issue_graph(data)
        metrics = coverage_record(issue, graph, data)

        self.assertEqual(set(converted), {0, 1})
        self.assertEqual(metrics["recovered_gold"], 1)
        self.assertEqual(metrics["gold_unavailable_at_issue_time"], 1)
        self.assertEqual(metrics["gold_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
