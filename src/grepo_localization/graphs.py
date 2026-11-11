"""Turn a repository graph into a GNN example for one bug report.

A GREPO repository graph contains code from many points in time. A model should
only see code that existed when the bug was reported. This module selects that
historical state, starts from issue-specific anchor nodes, expands a local
neighborhood, builds node features, and optionally attaches labels from the
later fix.

The small transformation functions can be studied and tested independently
before using :func:`make_issue_graph` or :func:`prepare_issue` end to end.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from .corpus import IssueRef


def active_nodes(graph: Data, issue_time: int) -> torch.Tensor:
    """Mark the repository nodes that existed when an issue was reported.

    Each repository node has a start timestamp and an end timestamp. A node is
    present from its start time up to, but not including, its end time.

    Args:
        graph: Temporal repository graph with ``starttimestamp`` and
            ``endtimestamp`` tensors, one value per node.
        issue_time: Timestamp stored for the GREPO issue.

    Returns:
        A Boolean tensor with one value per repository node. ``True`` means the
        node may be used in this issue's historical graph.
    """
    start = graph.starttimestamp.flatten()
    end = graph.endtimestamp.flatten()
    return (start <= int(issue_time)) & (int(issue_time) < end)


def live_ids(ids: Sequence[int], alive: torch.Tensor) -> torch.Tensor:
    """Keep node IDs that are valid and present at the issue time.

    Args:
        ids: Repository node IDs to check, such as an issue's anchor nodes.
        alive: Boolean mask returned by :func:`active_nodes`.

    Returns:
        A ``long`` tensor containing the in-range, active IDs. Their original
        order and duplicates are preserved. Invalid or inactive IDs are
        omitted.
    """
    ids = torch.tensor(ids, dtype=torch.long)
    in_bounds = ids[(ids >= 0) & (ids < alive.numel())]
    return in_bounds[alive[in_bounds]]


def active_edges(graph: Data, alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove relationships connected to code that did not yet exist.

    Args:
        graph: Repository graph with ``edge_index`` and edge-type ``edge_attr``.
        alive: Boolean node mask returned by :func:`active_nodes`.

    Returns:
        ``(edge_index, edge_types)`` containing only edges whose source and
        target are both active. The release's edge order is preserved.
    """
    source, target = graph.edge_index.long()
    keep = alive[source] & alive[target]
    return graph.edge_index[:, keep].long(), graph.edge_attr.flatten()[keep].long()


def expand_seeds(seeds: torch.Tensor, edges: torch.Tensor, hops: int) -> torch.Tensor:
    """Find repository nodes near an issue's starting nodes.

    One *hop* crosses one graph edge. Expansion follows edges in either
    direction so a seed can find both incoming and outgoing neighbors. The
    original edge directions are restored when the final subgraph is built.

    Args:
        seeds: Repository node IDs selected as starting points for the issue.
        edges: Active repository edges using repository node IDs.
        hops: Neighborhood radius. ``0`` returns only the seeds; ``1`` also
            includes immediate neighbors.

    Returns:
        Sorted, unique repository node IDs in the expanded neighborhood.

    Raises:
        ValueError: If ``hops`` is negative.
    """
    if hops < 0:
        raise ValueError("hops must be non-negative")
    selected = seeds
    for _ in range(hops if seeds.numel() else 0):
        left, right = edges
        neighbors = torch.cat([left[torch.isin(right, selected)],
                               right[torch.isin(left, selected)]])
        selected = torch.cat([selected, neighbors]).unique(sorted=True)
    return selected


@lru_cache(maxsize=1)
def embedding_row_index(index_path: Path) -> dict[int, int]:
    """Map repository node IDs to rows in a content-embedding tensor.

    Args:
        index_path: GREPO JSON index stored beside a repository's safetensors
            embedding file.

    Returns:
        A dictionary from stable repository node ID to embedding row number.
        The parsed mapping is cached for repeated issue construction.
    """
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return {int(node): row for row, node in enumerate(index["node_ids"])}


def selected_embeddings(
    selected: torch.Tensor,
    embedding_files: Mapping[str, Path],
) -> torch.Tensor:
    """Load optional source-content vectors for selected graph nodes.

    Args:
        selected: Repository node IDs in the order used by the issue graph.
        embedding_files: Local paths named ``"embedding_index"`` and
            ``"embeddings"``.

    Returns:
        A floating-point matrix with one row per selected node. Any NaN values
        in the released vectors are replaced with zero.

    Raises:
        KeyError: If a selected node has no row in the embedding index.
    """
    node_to_row = embedding_row_index(embedding_files["embedding_index"])
    rows = [node_to_row[int(node)] for node in selected]
    with safe_open(embedding_files["embeddings"], framework="pt", device="cpu") as file:
        vectors = file.get_slice("embeddings")
        return torch.nan_to_num(vectors[rows].float())


def build_node_features(
    node_type: torch.Tensor,
    selected: torch.Tensor,
    seeds: torch.Tensor,
    embedding_files: Mapping[str, Path] | None,
) -> torch.Tensor:
    """Create the input feature matrix consumed by a GNN.

    Args:
        node_type: GREPO type ID for every selected node. This release uses IDs
            from 0 through 4.
        selected: Stable repository IDs in issue-graph row order.
        seeds: Stable IDs of the issue's active starting nodes.
        embedding_files: Optional content-embedding paths. Pass ``None`` to use
            only structural metadata.

    Returns:
        A matrix with one row per selected node. Columns 0–4 are a one-hot node
        type, column 5 indicates whether the node is a seed, and optional
        content-vector columns follow. Fix labels are never included.
    """
    seed_feature = torch.isin(selected, seeds).float().unsqueeze(1)
    features = [one_hot(node_type, num_classes=5).float(), seed_feature]
    if embedding_files is not None:
        features.append(selected_embeddings(selected, embedding_files))
    return torch.cat(features, dim=1)


def make_issue_graph(
    graph: Data,
    issue: Mapping[str, Any],
    hops: int,
    embedding_files: Mapping[str, Path] | None = None,
) -> Data:
    """Build the unlabeled PyTorch Geometric graph for one bug report.

    Args:
        graph: Full temporal graph for the issue's repository.
        issue: Mapping containing ``repo``, GREPO ``issue_id``, issue ``time``,
            and repository node IDs in ``seeds``.
        hops: Number of edges to traverse outward from the active seeds.
        embedding_files: Optional paths for content embeddings.

    Returns:
        An issue-scoped :class:`~torch_geometric.data.Data`. ``edge_index`` uses
        compact local row numbers, while ``original_node_id`` maps those rows
        back to stable repository IDs. ``candidate_mask`` is true for class and
        function nodes (GREPO types 3 and 4), which are the nodes a localizer
        should score.

    Note:
        This function does not read or attach the future fix labels. Keeping
        retrieval separate from labels prevents the answer key from changing
        the model input.
    """
    alive = active_nodes(graph, issue["time"])
    seeds = live_ids(issue["seeds"], alive)
    edges, edge_types = active_edges(graph, alive)
    selected = expand_seeds(seeds, edges, hops)
    local_edges, local_types = subgraph(
        selected, edges, edge_attr=edge_types, relabel_nodes=True, num_nodes=alive.numel()
    )
    node_type = graph.type.flatten()[selected].long()
    x = build_node_features(node_type, selected, seeds, embedding_files)
    candidate_mask = torch.isin(node_type, torch.tensor([3, 4]))
    return Data(x=x, edge_index=local_edges, edge_attr=local_types,
                candidate_mask=candidate_mask, original_node_id=selected,
                issue_id=int(issue["issue_id"]), repo=issue["repo"])


def attach_candidate_labels(data: Data, gold_ids: Iterable[int]) -> Data:
    """Label candidate nodes that were changed by the historical fix.

    Args:
        data: Unlabeled issue graph from :func:`make_issue_graph`.
        gold_ids: Stable repository IDs of nodes changed by the fix.

    Returns:
        The same graph object with a new floating-point ``y`` tensor. A row is
        ``1`` only if its original node ID is both gold and a candidate; every
        other row is ``0``.

    Note:
        The graph is mutated in place. Gold nodes missing from the retrieved
        graph are not inserted; coverage diagnostics report those misses.
    """
    gold = torch.tensor(sorted(gold_ids), dtype=torch.long)
    data.y = (torch.isin(data.original_node_id, gold) & data.candidate_mask).float()
    return data


def prepare_issue(
    issue: Mapping[str, Any],
    corpus,
    hops: int,
    files: Mapping[str, Mapping[str, Path]],
    use_embeddings: bool,
) -> Data:
    """Load, build, and label one issue for training or evaluation.

    Args:
        issue: Mapping containing ``repo``, ``issue_id``, ``time``, and
            ``seeds``.
        corpus: Object providing ``repository_graph`` and ``gold`` methods,
            normally :class:`grepo_localization.corpus.GrepoCorpus`.
        hops: Neighborhood radius around the issue's seeds.
        files: Local artifact paths keyed by repository name.
        use_embeddings: Whether to append released content embeddings to the
            basic type and seed features.

    Returns:
        A labeled issue graph suitable for a PyTorch Geometric dataset.
    """
    repo = issue["repo"]
    graph = corpus.repository_graph(repo)
    embedding_files = files[repo] if use_embeddings else None
    data = make_issue_graph(graph, issue, hops, embedding_files)
    gold = corpus.gold(IssueRef(repo, int(issue["issue_id"]))) or ()
    return attach_candidate_labels(data, gold)


def graph_for_issue(corpus, ref: IssueRef, *, hops: int = 1) -> Data:
    """Convenience function for building an unlabeled graph from an issue ID.

    Args:
        corpus: Configured :class:`grepo_localization.corpus.GrepoCorpus`.
        ref: Repository name and GREPO issue-row identifier.
        hops: Neighborhood radius around the issue's extractor anchors.

    Returns:
        An issue graph using node-type and seed features, without content
        embeddings or fix labels. This form is useful for interactive browsing
        and inference.
    """
    issue = corpus.issue(ref)
    inputs = {
        "repo": ref.repo, "issue_id": ref.issue_id,
        "time": issue.graph_time, "seeds": issue.extractor_anchor_ids,
    }
    return make_issue_graph(corpus.repository_graph(ref.repo), inputs, hops)
