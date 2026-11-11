# Learning GNNs with GREPO

This guide is for someone who is learning graph neural networks (GNNs) and has not used GREPO before. By the end, you should understand what one GREPO training example represents, how it becomes a PyTorch Geometric graph, and which parts belong to retrieval, learning, and evaluation.

## 1. The problem in plain language

A developer reports a bug. Before reading the future fix, we want to identify the classes and functions most likely to require a change.

GREPO treats a source repository as a graph:

- **Nodes** are code entities such as directories, files, classes, and functions.
- **Edges** are relationships between those entities.
- **Node features** describe each entity.
- **Labels** say which class or function nodes the historical fix changed.

This is an issue-conditioned **node-ranking** problem. It resembles binary node classification because each candidate gets a 0 or 1 label, but the useful output is normally a ranked list for each issue.

One repository graph can produce many examples. Each issue selects a different time, starting set, neighborhood, and answer key.

```text
bug report
    │
    ├── search-derived starting nodes (anchors/seeds)
    │           │
    │           ▼
    └── issue time ──► historical repository graph
                            │
                            ▼
                    expand local neighborhood
                            │
                            ▼
                     issue-scoped PyG graph
                            │
                            ▼
                   score class/function nodes
                            │
                            ▼
                 compare with the future fix
```

## 2. GREPO vocabulary translated into GNN vocabulary

| GREPO term | Meaning in this project | GNN interpretation |
|---|---|---|
| Repository graph | A graph containing historical code entities and relationships | The large source graph |
| Issue | A bug report paired with a graph time and starting nodes | One graph-learning example |
| Graph time | The repository time associated with the report | A temporal cutoff that prevents using future nodes |
| Anchor or seed | A code node identified as a useful starting point | The center set for neighborhood retrieval and an input feature |
| Hop | One edge traversal in either direction during retrieval | Neighborhood radius |
| Candidate | A selected class or function that may be ranked | A node included in the supervised objective |
| Gold node | A class or function changed by the historical fix | A positive label used for training or evaluation |
| Coverage | The fraction of gold nodes present in the candidates | The upper bound imposed on the ranker by retrieval |

Two stages must stay conceptually separate:

1. **Candidate retrieval** decides which nodes enter the issue graph and which class/function nodes may be scored.
2. **GNN ranking** assigns scores to those candidates.

A ranker cannot recover a changed function that retrieval omitted. Measure candidate coverage before judging the model.

## 3. What is inside one PyTorch Geometric example?

`make_issue_graph` returns a `torch_geometric.data.Data` object. `prepare_issue` adds labels to it.

| Field | Shape | Meaning |
|---|---:|---|
| `x` | `[num_nodes, num_features]` | Node features |
| `edge_index` | `[2, num_edges]` | Directed edges using local row indices |
| `edge_attr` | `[num_edges]` | GREPO relationship type for each edge |
| `original_node_id` | `[num_nodes]` | Stable repository node ID for every local node |
| `candidate_mask` | `[num_nodes]` | `True` for class and function nodes eligible for scoring |
| `y` | `[num_nodes]` | `1` for candidates changed by the fix; otherwise `0` |
| `repo` | scalar metadata | Repository name |
| `issue_id` | scalar metadata | GREPO issue-row identifier |

### Local IDs versus original IDs

PyTorch Geometric expects nodes in each example to use dense local indices `0..num_nodes-1`. GREPO's repository node IDs are global and may be much larger. Therefore:

- `edge_index` contains **local indices**;
- `original_node_id` maps each local row back to its **repository ID**.

Use original IDs when looking up source code or comparing predictions with gold nodes. Use local indices when indexing `x` or passing messages along `edge_index`.

### Default node features

Without content embeddings, every node has six features:

```text
[five-way node-type one-hot encoding | is_seed]
```

The seed indicator tells the GNN which nodes made this graph relevant to the current issue. Optional GREPO content embeddings are appended after these six columns.

The gold label is never part of `x`.

## 4. How an issue graph is built

For each issue, the graph code performs these transformations in order:

1. `active_nodes` keeps nodes whose lifetime contains the issue time. Lifetimes use `[start, end)`: the start is included and the end is excluded.
2. `live_ids` discards anchors that are absent at that time.
3. `active_edges` keeps edges whose two endpoints are alive.
4. `expand_seeds` walks the requested number of hops in either direction.
5. PyTorch Geometric extracts the induced subgraph and relabels its nodes locally.
6. `build_node_features` creates type, seed, and optional content features.
7. Class and function nodes become candidates.
8. `attach_candidate_labels` assigns labels from the historical fix.

The graph keeps directed edges after retrieval. Only the neighborhood search treats edges as traversable in either direction.

This implementation uses released extractor anchors and simple hop expansion. It is a useful learning baseline, but it is not a complete reproduction of every retrieval component described by GREPO.

## 5. Build a first real example

Install the notebook dependencies and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook]'
```

Then run the following from the repository root. PyTorch graph files use pickle-backed serialization, so enable trust only after confirming that you accept the official release artifacts.

```python
from pathlib import Path

from grepo_localization.corpus import (
    API,
    REVISION,
    GrepoCorpus,
    download_repositories,
    release_manifest,
    repo_paths,
)
from grepo_localization.graphs import prepare_issue

repo = "csvkit"
cache = Path("grepo_workspace")

manifest = release_manifest(API, REVISION)
required = ("graph", "anchors", "queries")
selected = {
    repo: {kind: repo_paths(repo)[kind] for kind in required}
}
local_files = download_repositories(
    selected, manifest, cache, API, REVISION
)

corpus = GrepoCorpus(
    manifest,
    cache,
    API,
    REVISION,
    trust_graph_files=True,
)

ref = corpus.issue_refs(repo)[0]
metadata = corpus.issue(ref)
example = {
    "repo": repo,
    "issue_id": ref.issue_id,
    "time": metadata.graph_time,
    "seeds": list(metadata.extractor_anchor_ids),
}

data = prepare_issue(
    example,
    corpus=corpus,
    hops=1,
    files=local_files,
    use_embeddings=False,
)

print(data)
print("nodes:", data.num_nodes)
print("edges:", data.num_edges)
print("features:", tuple(data.x.shape))
print("candidates:", int(data.candidate_mask.sum()))
print("positive candidates:", int(data.y.sum()))
```

The first reference is convenient for understanding the API, but it may not be labeled. For supervised learning, select references for which `corpus.gold(ref)` is not `None`.

## 6. Read the tensors before training

Inspect a few nodes to connect tensor rows back to source entities:

```python
candidate_ids = data.original_node_id[data.candidate_mask].tolist()
records = corpus.nodes(repo, candidate_ids[:10])

for node_id in candidate_ids[:10]:
    record = records[node_id]
    print(node_id, record.path, record.name)
```

Check the core invariants:

```python
assert data.x.shape[0] == data.num_nodes
assert data.original_node_id.shape == (data.num_nodes,)
assert data.candidate_mask.shape == (data.num_nodes,)
assert data.y.shape == (data.num_nodes,)
assert not bool((data.y.bool() & ~data.candidate_mask).any())
```

The final assertion matters: non-candidate nodes may pass messages, but they are not positive prediction targets.

## 7. A minimal GNN forward pass

The following model produces one logit per node. Mask the result so the loss and ranking operate only on candidates.

```python
import torch
from torch_geometric.nn import SAGEConv


class NodeRanker(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        hidden = self.conv1(data.x, data.edge_index).relu()
        hidden = self.conv2(hidden, data.edge_index).relu()
        return self.head(hidden).squeeze(-1)


model = NodeRanker(data.num_node_features)
logits = model(data)

candidate_logits = logits[data.candidate_mask]
candidate_labels = data.y[data.candidate_mask]
loss = torch.nn.functional.binary_cross_entropy_with_logits(
    candidate_logits, candidate_labels
)
loss.backward()

ranked_local_ids = torch.where(data.candidate_mask)[0][
    candidate_logits.argsort(descending=True)
]
ranked_original_ids = data.original_node_id[ranked_local_ids]
print(ranked_original_ids[:10])
```

This is deliberately a learning example, not a strong training recipe. Positive nodes are sparse, so a real experiment should address class imbalance and evaluate rankings per issue rather than relying on accuracy.

Also note that this baseline does not use `edge_attr`. GraphSAGE aggregates neighboring states without distinguishing relationship types. A relation-aware model can incorporate GREPO's edge types.

## 8. Prepare train and test data safely

Use a chronological split, not a random issue split. `chronological_split` assigns earlier labeled issues in each repository to training and later issues to testing.

```python
from functools import partial

import pandas as pd

from grepo_localization.dataset import GrepoIssueDataset, chronological_split

rows = []
for ref in corpus.issue_refs(repo):
    issue = corpus.issue(ref)
    rows.append({
        "repo": repo,
        "issue_id": ref.issue_id,
        "time": issue.graph_time,
        "seeds": list(issue.extractor_anchor_ids),
        "gold": sorted(corpus.gold(ref) or ()),
    })

issues = pd.DataFrame(rows)
split = chronological_split(issues)

builder = partial(
    prepare_issue,
    corpus=corpus,
    hops=1,
    files=local_files,
    use_embeddings=False,
)
train_dataset = GrepoIssueDataset(
    split[split["split"] == "train"], builder
)
test_dataset = GrepoIssueDataset(
    split[split["split"] == "test"], builder
)
```

The dataset is lazy: indexing it constructs the issue graph at that moment. This keeps preprocessing composable and makes the builder easy to replace in tests.

## 9. Measure retrieval before model quality

Suppose a fix changed three functions:

- one is already an anchor;
- a second appears after one-hop expansion;
- the third never enters the issue graph.

Seed coverage is `1/3`. Expanded candidate coverage is `2/3`. No ranker can achieve recall above `2/3` on that issue with this candidate set.

Use the inspection helpers to diagnose this ceiling:

```python
from grepo_localization.inspection import (
    measure_issue_coverage,
    summarize_coverage,
)

coverage_rows = [
    measure_issue_coverage(row, corpus, builder)
    for row in split.to_dict("records")
]
summary = summarize_coverage(pd.DataFrame(coverage_rows))
print(summary)
```

Useful measures include:

- `gold_coverage`: share of all changed nodes present among candidates;
- `any_gold_reachable`: whether at least one changed node is a candidate;
- `all_gold_reachable`: whether every changed node is a candidate;
- `gold_unavailable_at_issue_time`: changed nodes absent from the temporal graph;
- `live_gold_missing_from_candidates`: changed nodes that existed but retrieval missed.

After checking coverage, evaluate ranking with metrics such as Hit@K, Recall@K, mean reciprocal rank, and the share of issues whose complete change set occurs in the top K. State clearly whether issues with zero reachable positives remain in the denominator.

## 10. Avoid the common leakage mistakes

The historical fix is the answer key. It must not influence graph construction or input features.

- Do not add gold nodes to the graph merely because retrieval missed them.
- Do not use post-fix source code as a node feature for a pre-fix issue.
- Do not split issues randomly across time when later examples can reveal repository evolution.
- Do not tune hop count on the test set.
- Do not report ranking quality only on issues where retrieval found a positive unless that filtered protocol is explicit.
- Do not confuse GREPO issue-row IDs with GitHub issue numbers. Linked report information is separate metadata.

`prepare_issue` keeps retrieval and labeling separate internally: it first builds the issue graph, then attaches labels. `coverage_record` preserves missed gold IDs for diagnosis without inserting them into the model input.

## 11. Package map

| Module | Responsibility | Start here when... |
|---|---|---|
| `corpus.py` | Manifest access, verified caching, issue metadata, temporal repository graphs, source records, and labels | You need data or want to inspect a prediction |
| `graphs.py` | Temporal filtering, seed expansion, feature construction, subgraph creation, and label attachment | You want to understand one model input |
| `dataset.py` | Chronological splitting and lazy PyTorch datasets | You are preparing training and test examples |
| `inspection.py` | NetworkX conversion and candidate-coverage diagnostics | You are debugging retrieval or visualizing a graph |
| `exploration.py` | Reusable selection, summary, visualization-preparation, smoke-test, and export helpers | You are extending or testing the exploration notebook |

Each transformation has a small contract in its docstring. Functions such as `active_nodes`, `expand_seeds`, `build_node_features`, `attach_candidate_labels`, and `coverage_record` are intentionally usable with small synthetic inputs, which makes them good places to experiment while learning.

## 12. Suggested learning path

1. Run the web browser and inspect one issue, its anchors, candidates, graph, and historical source.
2. Run `notebooks/01_explore_grepo.ipynb` from top to bottom with `csvkit`.
3. Print one `Data` object and verify every tensor against the table in this guide.
4. Change `hops` from 0 to 1 to 2 and compare graph size and candidate coverage.
5. Train the minimal GraphSAGE ranker and compare it with a feature-only MLP.
6. Remove or shuffle edges to test whether message passing actually helps.
7. Add content embeddings, then compare quality, memory use, and training time.
8. Try a relation-aware model that uses `edge_attr`.

The key experimental habit is to change one stage at a time. Keep candidate generation fixed when comparing rankers, and keep the ranker fixed when comparing retrieval strategies.

## Further reading

- The project notebook: [`notebooks/01_explore_grepo.ipynb`](../notebooks/01_explore_grepo.ipynb)
- Google Colab setup: [`docs/colab/README.md`](colab/README.md)
- Local VS Code setup: [`docs/vscode/README.md`](vscode/README.md)
- The interactive browser instructions: [project README](../README.md)
- [GREPO paper](https://arxiv.org/abs/2602.13921)
- [Official GREPO implementation](https://github.com/qingpingmo/GREPO)
- [PyTorch Geometric introduction](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)
