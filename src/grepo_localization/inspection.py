"""Inspect issue graphs and measure what candidate retrieval missed.

A GNN can rank only the candidates it receives. These helpers separate that
retrieval limit from model quality by measuring how many nodes from the actual
fix appear in an issue graph before ranking begins.
"""

from collections.abc import Callable, Mapping
from typing import Any

import networkx as nx
import pandas as pd
from torch_geometric.data import Data

from .graphs import active_nodes, live_ids


def to_networkx_issue_graph(data: Data) -> nx.MultiDiGraph:
    """Convert an issue graph into NetworkX form for inspection or plotting.

    Args:
        data: PyTorch Geometric issue graph produced by ``make_issue_graph``.

    Returns:
        A directed NetworkX multigraph. Its node keys are stable repository IDs
        rather than PyG's local row numbers. Candidate flags, edge types,
        parallel edges, repository name, and issue ID are preserved.
    """
    graph = nx.MultiDiGraph(repo=data.repo, issue_id=data.issue_id)
    node_ids = data.original_node_id.tolist()
    for node_id, is_candidate in zip(node_ids, data.candidate_mask.tolist()):
        graph.add_node(node_id, candidate=is_candidate)
    for (source, target), edge_type in zip(data.edge_index.t().tolist(), data.edge_attr.tolist()):
        graph.add_edge(node_ids[source], node_ids[target], edge_type=edge_type)
    return graph


def coverage_record(issue: Mapping[str, Any], graph: Data, data: Data) -> dict[str, Any]:
    """Measure whether candidate retrieval found the nodes changed by a fix.

    Args:
        issue: Mapping containing ``repo``, ``issue_id``, ``time``, ``seeds``,
            and the complete changed-node list in ``gold``.
        graph: Full temporal repository graph used to determine which gold
            nodes existed at the issue time.
        data: Labeled issue graph containing ``original_node_id``,
            ``candidate_mask``, and ``y``.

    Returns:
        A dictionary of counts and ratios for one issue. It distinguishes gold
        nodes that were unavailable at the issue time from nodes that existed
        but retrieval omitted. Coverage ratios are ``None`` when the issue has
        no gold labels. The inputs are not modified.
    """
    found = set(data.original_node_id[data.y.bool()].tolist())
    gold = set(issue["gold"])
    alive = active_nodes(graph, issue["time"])
    live_gold = set(live_ids(sorted(gold), alive).tolist())
    return dict(
        repo=issue["repo"], issue_id=issue["issue_id"], time=issue["time"],
        gold_size=len(gold), graph_nodes=data.num_nodes, graph_edges=data.num_edges,
        candidates=int(data.candidate_mask.sum()), recovered_gold=len(found),
        missing_gold=sorted(gold - found),
        seed_gold_coverage=len(live_gold.intersection(issue["seeds"])) / len(gold) if gold else None,
        gold_unavailable_at_issue_time=len(gold - live_gold),
        live_gold_missing_from_candidates=len(live_gold - found),
        gold_coverage=len(found) / len(gold) if gold else None,
        any_gold_reachable=bool(found), all_gold_reachable=bool(gold) and found == gold,
    )


def measure_issue_coverage(
    issue: Mapping[str, Any],
    corpus,
    build_issue: Callable[[Mapping[str, Any]], Data],
) -> dict[str, Any]:
    """Build one issue graph and calculate its candidate-coverage metrics.

    Args:
        issue: Issue mapping accepted by :func:`coverage_record`.
        corpus: Object whose ``repository_graph(repo)`` method loads the full
            temporal graph.
        build_issue: Function that turns the issue mapping into a labeled PyG
            graph, such as a configured ``prepare_issue``.

    Returns:
        The issue-level metric dictionary from :func:`coverage_record`.
    """
    graph = corpus.repository_graph(issue["repo"])
    data = build_issue(issue)
    return coverage_record(issue, graph, data)


def summarize_coverage(labeled: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate-retrieval quality for each repository.

    Args:
        labeled: DataFrame with one row per issue and the columns returned by
            :func:`coverage_record`.

    Returns:
        A DataFrame indexed by repository with issue counts, median candidate
        and gold sizes, mean coverage, unavailable or missed gold counts, and
        rates for finding any or all changed nodes. Numeric values are rounded
        to three decimal places.
    """
    return labeled.groupby("repo").agg(
        labeled_issues=("issue_id", "size"),
        median_gold_size=("gold_size", "median"),
        median_candidates=("candidates", "median"),
        mean_seed_coverage=("seed_gold_coverage", "mean"),
        mean_expanded_coverage=("gold_coverage", "mean"),
        unavailable_gold=("gold_unavailable_at_issue_time", "sum"),
        reachable_time_gold_missed=("live_gold_missing_from_candidates", "sum"),
        any_gold_rate=("any_gold_reachable", "mean"),
        complete_coverage_rate=("all_gold_reachable", "mean"),
    ).round(3)
