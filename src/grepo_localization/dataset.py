"""Prepare labeled GREPO issues for chronological GNN experiments.

The split helper keeps later issues out of training, while
:class:`GrepoIssueDataset` builds each PyTorch Geometric example only when it is
requested. Graph construction remains an injected function so experiments and
tests can choose their own preprocessing.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd
import torch


def chronological_split(
    issues: pd.DataFrame,
    *,
    on_skip: Callable[[str], None] | None = print,
) -> pd.DataFrame:
    """Assign earlier labeled issues to training and later ones to testing.

    The split is calculated separately for each repository so a model is never
    tested on an issue earlier than that repository's own training examples.
    Rows without changed-node labels are omitted. The cutoff uses the first 80%
    of distinct issue times for training and the remaining times for testing.

    Args:
        issues: DataFrame with at least ``repo``, ``gold``, and ``time``.
            ``gold`` should contain the changed node IDs for each issue.
        on_skip: Optional callback for explaining why a repository was skipped.
            Pass ``None`` to suppress these messages.

    Returns:
        A new DataFrame containing eligible input rows plus a ``split`` column
        whose value is ``"train"`` or ``"test"``.

    Raises:
        ValueError: If no repository has labeled issues at two or more distinct
            times, so a chronological split cannot be formed.
    """
    split_rows = []
    for repo, rows in issues[issues["gold"].map(bool)].groupby("repo", sort=False):
        times = sorted(rows["time"].unique())
        if len(times) < 2:
            if on_skip is not None:
                on_skip(f"Skipping split for {repo}: fewer than two distinct labeled issue times.")
            continue
        cutoff = times[min(len(times) - 1, max(1, int(0.8 * len(times))))]
        for row in rows.to_dict("records"):
            split_rows.append({**row, "split": "train" if row["time"] < cutoff else "test"})

    if not split_rows:
        raise ValueError("No temporal split could be created for this repository selection.")
    return pd.DataFrame(split_rows)


class GrepoIssueDataset(torch.utils.data.Dataset):
    """PyTorch dataset that creates an issue graph only when it is requested.

    Keeping graph construction lazy avoids storing every expanded graph in
    memory and makes preprocessing easy to replace during an experiment.
    """

    def __init__(self, rows: pd.DataFrame, build_issue: Callable[[dict], Any]):
        """Create a lazy dataset from issue metadata.

        Args:
            rows: One DataFrame row per issue, in the desired dataset order.
            build_issue: Function that accepts one row as a dictionary and
                returns its model-ready graph or another training example.
        """
        self.rows = rows.to_dict("records")
        self.build_issue = build_issue

    def __len__(self) -> int:
        """Return the number of issues in the dataset."""
        return len(self.rows)

    def __getitem__(self, index: int):
        """Build one model example from the issue row at ``index``.

        Negative indices follow normal Python sequence behavior. An invalid
        index raises ``IndexError`` before the builder is called.
        """
        return self.build_issue(self.rows[index])
