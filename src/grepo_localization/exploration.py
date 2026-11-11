"""Reusable helpers for the GREPO exploration notebook.

The notebook should explain and orchestrate an experiment, not hide data
transformations inside stateful cells. This module keeps selection, summary,
ranking, visualization preparation, smoke testing, and export behavior in
ordinary functions that can be imported and tested independently.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


def count_types(values: torch.Tensor) -> dict[int, int]:
    """Count how many times each integer type ID occurs in a tensor.

    Args:
        values: Tensor containing node or edge type IDs. Its shape does not
            matter because all values are counted.

    Returns:
        A dictionary from each observed type ID to its occurrence count.
    """
    ids, counts = torch.unique(values, return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist()))


def load_repository_tables(
    repos: Sequence[str],
    corpus: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load repository summaries and issue rows for notebook exploration.

    This is the I/O boundary for the notebook's initial tables. It uses only
    the public corpus API; formatting and display remain the caller's job.

    Args:
        repos: GREPO repository names in the desired display order.
        corpus: Object implementing ``repository_graph``, ``issue_refs``,
            ``issue``, and ``gold``, normally a ``GrepoCorpus``.

    Returns:
        ``(graph_stats, issues, examples)``. ``graph_stats`` has one row per
        repository. ``issues`` has one row per GREPO issue and contains the
        inputs needed by graph construction. ``examples`` contains at most
        three labeled issues with search text per repository.

    Raises:
        ValueError: If ``repos`` is empty or no issue rows are available.
    """
    if not repos:
        raise ValueError("Choose at least one repository to describe.")

    graph_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for repo in repos:
        graph = corpus.repository_graph(repo)
        refs = corpus.issue_refs(repo)
        issues_with_queries = 0
        for ref in refs:
            issue = corpus.issue(ref)
            gold = sorted(corpus.gold(ref) or ())
            seeds = list(issue.extractor_anchor_ids)
            queries = issue.search_queries
            issues_with_queries += bool(queries)
            issue_rows.append({
                "repo": repo,
                "issue_id": int(ref.issue_id),
                "time": issue.graph_time,
                "gold": gold,
                "seeds": seeds,
                "has_query": bool(queries),
            })
            if gold and queries:
                samples.append({
                    "repo": repo,
                    "issue_id": int(ref.issue_id),
                    "time": issue.graph_time,
                    "extractor_seeds": len(seeds),
                    "gold_nodes": gold,
                    "query": queries[0],
                })
        graph_rows.append({
            "repo": repo,
            "nodes": graph.type.numel(),
            "edges": graph.num_edges,
            "node_types": count_types(graph.type),
            "edge_types": count_types(graph.edge_attr),
            "issue_records": len(refs),
            "issues_with_queries": issues_with_queries,
        })

    if not issue_rows:
        raise ValueError("The selected repositories contain no GREPO issue rows.")
    issues = pd.DataFrame(issue_rows).sort_values(
        ["repo", "time", "issue_id"]
    ).reset_index(drop=True)
    examples = pd.DataFrame(samples)
    if not examples.empty:
        examples = examples.groupby("repo", sort=False).head(3).reset_index(drop=True)
    return pd.DataFrame(graph_rows), issues, examples


def select_issue_row(
    issues: pd.DataFrame,
    repo: str | None = None,
    issue_id: int | None = None,
) -> dict[str, Any]:
    """Select one issue row for the notebook's detailed walkthrough.

    Args:
        issues: Table produced by :func:`load_repository_tables`.
        repo: Optional repository name. Supply it together with ``issue_id``.
        issue_id: Optional GREPO issue-row identifier.

    Returns:
        A copy of the selected row as a dictionary. When no explicit identity
        is supplied, the first labeled issue with at least one anchor is used.

    Raises:
        ValueError: If only one identity field is supplied, the requested row
            is absent, or no default eligible row exists.
    """
    if (repo is None) != (issue_id is None):
        raise ValueError("Set both the repository and issue ID, or leave both blank.")
    if repo is not None:
        matches = issues[(issues["repo"] == repo) & (issues["issue_id"] == issue_id)]
        if matches.empty:
            raise ValueError(f"Issue {issue_id} was not found in {repo}.")
        return matches.iloc[0].to_dict()

    eligible = issues[issues["gold"].map(bool) & issues["seeds"].map(bool)]
    if eligible.empty:
        raise ValueError("Choose a labeled issue with at least one anchor.")
    return eligible.iloc[0].to_dict()


def issue_graph_view(
    issue: Mapping[str, Any],
    graph: nx.MultiDiGraph,
    max_nodes: int,
) -> tuple[nx.Graph, dict[str, set[int]]]:
    """Choose nodes and color groups for a readable issue-graph drawing.

    Gold and anchor nodes receive display priority so a truncated diagnostic
    plot still explains retrieval successes and failures. This label-aware
    selection is for post-hoc visualization only; it must not be model input.

    Args:
        issue: Issue row containing repository node IDs in ``seeds`` and
            ``gold``.
        graph: Full NetworkX issue graph using repository node IDs.
        max_nodes: Maximum nodes to include in the drawing. Must be positive.

    Returns:
        ``(view, groups)`` where ``view`` is a simple undirected graph for
        drawing and ``groups`` contains the visible IDs grouped as ``both``,
        ``gold``, ``seeds``, and ``nearby``.

    Raises:
        ValueError: If ``max_nodes`` is not positive.
    """
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive.")
    node_ids = set(graph)
    seeds = set(issue["seeds"]).intersection(node_ids)
    gold = set(issue["gold"]).intersection(node_ids)
    priority = (
        sorted(seeds & gold)
        + sorted(gold - seeds)
        + sorted(seeds - gold)
        + sorted(node_ids - seeds - gold)
    )
    shown = set(priority[:max_nodes])
    groups = {
        "both": seeds & gold & shown,
        "gold": (gold - seeds) & shown,
        "seeds": (seeds - gold) & shown,
        "nearby": shown - seeds - gold,
    }
    return nx.Graph(graph.subgraph(priority[:max_nodes])), groups


def candidate_table(
    issue: Mapping[str, Any],
    data: Data,
    source_nodes: Mapping[int, Any],
) -> pd.DataFrame:
    """Describe every class or function candidate in an issue graph.

    Args:
        issue: Issue row containing anchor IDs in ``seeds``.
        data: Issue graph containing ``original_node_id`` and
            ``candidate_mask`` tensors.
        source_nodes: Node records keyed by stable repository node ID.

    Returns:
        A table with node ID, human-readable type, path, name, and whether the
        node came directly from an anchor or from graph expansion.

    Raises:
        KeyError: If a candidate is absent from ``source_nodes`` or has an
            unsupported GREPO node type.
    """
    candidate_ids = data.original_node_id[data.candidate_mask].tolist()
    seeds = set(issue["seeds"])
    rows = []
    for node_id in candidate_ids:
        node = source_nodes[node_id]
        rows.append({
            "node_id": node_id,
            "type": {3: "class", 4: "function"}[node.node_type],
            "path": node.path,
            "name": node.name,
            "found_at": "Starting anchor" if node_id in seeds else "Graph expansion",
        })
    return pd.DataFrame(
        rows, columns=["node_id", "type", "path", "name", "found_at"]
    )


def choose_nodes(
    candidates: pd.DataFrame,
    top_k: int,
    ranked_node_ids: Iterable[int] | None,
) -> tuple[list[int], str]:
    """Choose candidates for inspection from a baseline or supplied ranking.

    Args:
        candidates: Table from :func:`candidate_table` containing ``node_id``
            and ``found_at`` columns.
        top_k: Maximum number of nodes to return. Must be non-negative.
        ranked_node_ids: Optional model ranking expressed as stable repository
            IDs. ``None`` uses anchors first and node ID as a tie-breaker.

    Returns:
        ``(node_ids, ranking_name)`` with at most ``top_k`` unique candidates.

    Raises:
        ValueError: If ``top_k`` is negative or a supplied ID is not a
            candidate in this issue graph.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")
    if ranked_node_ids is None:
        order = candidates.assign(
            is_anchor=candidates["found_at"].eq("Starting anchor")
        ).sort_values(["is_anchor", "node_id"], ascending=[False, True])
        return order["node_id"].head(top_k).tolist(), "Anchor-first baseline"

    ranked = list(dict.fromkeys(ranked_node_ids))
    unknown = set(ranked) - set(candidates["node_id"])
    if unknown:
        raise ValueError(f"Ranked IDs must be candidates. Unknown IDs: {sorted(unknown)}")
    return ranked[:top_k], "Supplied model ranking"


def first_nonempty_batch(loader: Iterable[Data]) -> Data:
    """Return the first PyG batch containing at least one node.

    Args:
        loader: Iterable of PyTorch Geometric graphs or batches.

    Returns:
        The first item whose ``num_nodes`` is greater than zero.

    Raises:
        ValueError: If the iterable contains no nonempty graph.
    """
    batch = next((item for item in loader if item.num_nodes), None)
    if batch is None:
        raise ValueError("No nonempty issue graph is available for the forward pass.")
    return batch


def graphsage_logits(batch: Data, device: torch.device) -> torch.Tensor:
    """Run a one-layer GraphSAGE shape and device smoke test.

    This function does not train a useful localizer. It verifies that a
    prepared batch can pass through a GraphSAGE layer on the selected device.

    Args:
        batch: PyG graph or batch with ``x`` and ``edge_index``.
        device: CPU, CUDA, or MPS device used for the check.

    Returns:
        One detached CPU logit per node. The input batch is not moved or
        modified.
    """
    batch.validate(raise_on_error=True)
    moved = batch.clone().to(device)
    model = SAGEConv(moved.num_node_features, 1).to(device).eval()
    with torch.no_grad():
        return model(moved.x, moved.edge_index).flatten().cpu()


def export_issue_graphs(
    datasets: Mapping[str, Iterable[Data]],
    coverage: pd.DataFrame,
    export_root: Path,
) -> pd.DataFrame:
    """Export prepared issue tensors plus split and coverage tables.

    Each issue is written to a temporary sibling file and atomically moved to
    its final path. Saved values are plain tensors and primitive metadata, so
    callers can reload them with ``torch.load(..., weights_only=True)``.

    Args:
        datasets: Mapping from split name, such as ``"train"``, to issue graphs.
        coverage: Issue-level coverage table to save as ``coverage.csv``.
        export_root: Destination directory for graphs and CSV manifests.

    Returns:
        The split-manifest DataFrame also saved as ``splits.csv``.
    """
    export_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for split, dataset in datasets.items():
        for data in dataset:
            path = export_root / split / data.repo / f"{data.issue_id}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".pt.part")
            torch.save(data.to_dict(), temporary)
            temporary.replace(path)
            manifest_rows.append({
                "repo": data.repo,
                "issue_id": data.issue_id,
                "split": split,
                "file": str(path.relative_to(export_root)),
            })
    manifest = pd.DataFrame(
        manifest_rows, columns=["repo", "issue_id", "split", "file"]
    )
    manifest.to_csv(export_root / "splits.csv", index=False)
    coverage.to_csv(export_root / "coverage.csv", index=False)
    return manifest
