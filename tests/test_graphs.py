"""Contract tests for temporal graph transformations."""

import unittest

import torch
from torch_geometric.data import Data

from grepo_localization.graphs import (
    active_edges,
    active_nodes,
    attach_candidate_labels,
    expand_seeds,
    live_ids,
    make_issue_graph,
)


def repository_graph():
    """Return a small temporal graph with alive, future, and expired nodes."""
    return Data(
        type=torch.tensor([0, 3, 4, 2]),
        starttimestamp=torch.tensor([0, 0, 5, 0]),
        endtimestamp=torch.tensor([10, 10, 10, 3]),
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 3]]),
        edge_attr=torch.tensor([7, 8, 9]),
    )


class GraphContractsTest(unittest.TestCase):
    def test_temporal_selection_and_expansion(self):
        graph = repository_graph()
        alive = active_nodes(graph, 4)
        edges, edge_types = active_edges(graph, alive)

        self.assertEqual(alive.tolist(), [True, True, False, False])
        self.assertEqual(live_ids([-1, 0, 2, 9], alive).tolist(), [0])
        self.assertEqual(edges.tolist(), [[0], [1]])
        self.assertEqual(edge_types.tolist(), [7])
        self.assertEqual(expand_seeds(torch.tensor([0]), edges, 1).tolist(), [0, 1])
        with self.assertRaises(ValueError):
            expand_seeds(torch.tensor([0]), edges, -1)

    def test_issue_graph_and_labels_preserve_global_ids(self):
        issue = {"repo": "demo", "issue_id": 11, "time": 4, "seeds": [0]}

        data = make_issue_graph(repository_graph(), issue, hops=1)
        attach_candidate_labels(data, [1])

        self.assertEqual(data.original_node_id.tolist(), [0, 1])
        self.assertEqual(data.candidate_mask.tolist(), [False, True])
        self.assertEqual(data.x.shape, (2, 6))
        self.assertEqual(data.y.tolist(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
