# Using `GrepoCorpus`

`GrepoCorpus` is the main interface for reading the GREPO dataset. It gives you access to:

- repositories included in the release;
- historical repository graphs;
- issue times, anchor nodes, and search queries;
- classes, functions, paths, and historical source text;
- the nodes changed by each issue's eventual fix.

It downloads files only when they are needed and keeps verified copies in a local cache.

## Install the package

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The following examples assume that commands run from the project root.

## Create a corpus

First retrieve the release manifest, then use it to create the corpus:

```python
from pathlib import Path

from grepo_localization.corpus import API, REVISION, GrepoCorpus, release_manifest

manifest = release_manifest(API, REVISION)

corpus = GrepoCorpus(
    manifest=manifest,
    root=Path("grepo_workspace"),
    api=API,
    revision=REVISION,
)
```

The **manifest** is an inventory of files in the remote GREPO release. It includes each file's path, size, and checksum. Creating the corpus does not download repository data.

`root` is the local cache directory. Downloaded files are stored under:

```text
grepo_workspace/raw/
```

You can delete this cache and download the files again. Do not place manually edited data in it because failed checksum verification will cause the file to be replaced.

## Discover available repositories

```python
repositories = corpus.repositories()

print(f"Found {len(repositories)} repositories")
print(repositories[:10])
```

`repositories()` returns only repositories with the basic artifacts needed by the browser: a repository graph, issue anchors, and search queries.

Use the release name returned here when calling other corpus methods. A name such as `"csvkit"` is a GREPO dataset name, not a local Git path.

## Check and download one repository

```python
repo = "csvkit"

ready, download_bytes = corpus.repository_status(repo)
print("Already cached:", ready)
print("Download size in MB:", round(download_bytes / 1_000_000, 2))

if not ready:
    corpus.download_repository(repo)
```

`download_repository` retrieves three basic artifacts:

- the temporal repository graph;
- issue anchor metadata;
- generated search queries.

Every downloaded file is checked against the size and SHA-256 checksum in the release manifest. A valid cached file is reused.

You do not have to call `download_repository` before other methods. Corpus methods download their required files on first use. Calling it explicitly is useful when you want download progress to finish before interactive analysis.

## List issues

```python
refs = corpus.issue_refs(repo)

print("Number of GREPO issues:", len(refs))
print("First reference:", refs[0])
```

Each value is an `IssueRef` containing:

- `repo`: the GREPO repository name;
- `issue_id`: the row identifier used inside the GREPO release.

The GREPO `issue_id` is not necessarily the public GitHub issue number. Linked GitHub report information is available separately.

You can also construct a reference directly:

```python
from grepo_localization.corpus import IssueRef

ref = IssueRef(repo="csvkit", issue_id=107)
```

## Read issue metadata

```python
issue = corpus.issue(ref)

print("Graph time:", issue.graph_time)
print("Anchor node IDs:", issue.extractor_anchor_ids)
print("Search queries:")
for query in issue.search_queries:
    print("-", query)
```

An issue record contains the information needed to construct an issue-specific graph:

| Field | Meaning |
|---|---|
| `ref` | Repository and GREPO issue-row identifier |
| `graph_time` | Historical time used to decide which code nodes existed |
| `extractor_anchor_ids` | Repository node IDs used as graph starting points |
| `search_queries` | Text queries derived from the report |
| `reports` | Linked report records, loaded only when requested |
| `fix_pr` | Fix pull-request row, loaded only when requested |

An **anchor** is a code node that GREPO selected as a useful place to begin searching for the fix location.

### Include the linked report

```python
issue_with_report = corpus.issue(ref, include_report=True)

for report in issue_with_report.reports:
    print(report.get("title"))
    print(report.get("body"))

if issue_with_report.fix_pr:
    print("Fix PR title:", issue_with_report.fix_pr.get("title"))
```

Linked reports and pull-request data live in a larger table, so `include_report=False` is the faster default.

## Load a repository graph safely

GREPO repository graphs are stored as PyTorch `.pt` files. PyTorch loads these files with pickle-backed deserialization, which should be used only for files from a source you trust.

After reviewing and accepting the official release as the source, create the corpus with graph loading enabled:

```python
corpus = GrepoCorpus(
    manifest=manifest,
    root=Path("grepo_workspace"),
    api=API,
    revision=REVISION,
    trust_graph_files=True,
)

graph = corpus.repository_graph(repo)

print("Historical nodes:", graph.num_nodes)
print("Edges:", graph.num_edges)
print("Node type tensor:", graph.type.shape)
print("Start timestamps:", graph.starttimestamp.shape)
print("End timestamps:", graph.endtimestamp.shape)
```

The repository graph contains historical nodes from multiple points in time. Do not treat every node as visible to every issue. Graph-building functions filter nodes using the issue's `graph_time`.

If `trust_graph_files` remains false, `repository_graph` raises `RuntimeError` before loading the `.pt` file.

## Build a graph for one issue

The convenience function below creates an unlabeled issue graph using the issue time and anchor nodes stored in the corpus:

```python
from grepo_localization.graphs import graph_for_issue

data = graph_for_issue(corpus, ref, hops=1)

print(data)
print("Selected nodes:", data.num_nodes)
print("Selected edges:", data.num_edges)
print("Candidates:", int(data.candidate_mask.sum()))
```

`hops=1` includes the active anchors and their immediate neighbors. Increasing the hop count retrieves a larger neighborhood and may add candidates, but also increases memory and computation.

The returned PyTorch Geometric graph contains:

- `x`: node type and anchor-indicator features;
- `edge_index`: edges using compact local node positions;
- `edge_attr`: GREPO relationship types;
- `original_node_id`: stable repository IDs for the selected nodes;
- `candidate_mask`: classes and functions eligible for localization.

`graph_for_issue` does not attach labels or content embeddings.

## Look up names and source code

Use `original_node_id`, rather than local positions in `edge_index`, when asking the corpus for node records:

```python
candidate_ids = data.original_node_id[data.candidate_mask].tolist()
records = corpus.nodes(repo, candidate_ids[:10])

for node_id in candidate_ids[:10]:
    node = records[node_id]
    print("ID:", node.node_id)
    print("Type:", node.node_type)
    print("Path:", node.path)
    print("Name:", node.name)
    print("Source available:", node.text is not None)
    print()
```

`corpus.nodes` accepts any iterable of repository node IDs and returns a dictionary keyed by those IDs. Duplicate inputs are collapsed. Unknown IDs raise `KeyError` rather than silently disappearing.

The first source lookup downloads the repository's larger node table. Later lookups reuse it from the cache and memory.

## Read the changed-node labels

```python
gold_ids = corpus.gold(ref)

if gold_ids is None:
    print("This issue has no changed-node labels")
else:
    print("Nodes changed by the fix:", sorted(gold_ids))
```

The changed nodes are commonly called **gold nodes** because they are the historical answer key for training and evaluation.

Gold labels must not influence which nodes enter an issue graph. If retrieval misses a gold node, keep it missing and report the coverage failure. Adding it to the graph would leak the answer into the model input.

To inspect the source of both retrieved candidates and missed gold nodes:

```python
candidate_ids = set(
    data.original_node_id[data.candidate_mask].tolist()
)
gold_ids = set(corpus.gold(ref) or ())

nodes_to_inspect = candidate_ids | gold_ids
records = corpus.nodes(repo, nodes_to_inspect)

for node_id in sorted(gold_ids):
    node = records[node_id]
    status = "candidate" if node_id in candidate_ids else "missed by retrieval"
    print(node_id, status, node.path, node.name)
```

This union is appropriate for human inspection only. It must not be used as the model's candidate set.

## Build labeled examples

For training or evaluation, use `prepare_issue`. It builds the graph first and attaches the historical labels afterward:

```python
from grepo_localization.graphs import prepare_issue

metadata = corpus.issue(ref)
example = {
    "repo": ref.repo,
    "issue_id": ref.issue_id,
    "time": metadata.graph_time,
    "seeds": list(metadata.extractor_anchor_ids),
}

data = prepare_issue(
    example,
    corpus=corpus,
    hops=1,
    files={},
    use_embeddings=False,
)

print("Candidate labels:", data.y[data.candidate_mask])
print("Positive candidates:", int(data.y.sum()))
```

With `use_embeddings=False`, `files` is not accessed and may be an empty dictionary. See [Learning GNNs with GREPO](../learning-gnns-with-grepo.md) for content embeddings, datasets, GraphSAGE, chronological splitting, and coverage evaluation.

## Choose a different cache directory

Pass any writable `Path` as `root`:

```python
cache = Path("/path/to/shared/grepo-cache")
corpus = GrepoCorpus(manifest, cache, API, REVISION)
```

The web application also supports the `GREPO_CACHE_DIR` environment variable. The Python class itself does not read that variable; your code must pass the selected path as `root`.

## Use the corpus without network access in tests

`GrepoCorpus` and the lower-level download functions accept an optional `opener`. Production code normally leaves it unchanged. Tests can provide a function that returns in-memory responses:

```python
corpus = GrepoCorpus(
    manifest=fake_manifest,
    root=temporary_directory,
    api="https://example.invalid/dataset",
    revision="test",
    opener=fake_opener,
)
```

This allows cache, JSON, and record behavior to be tested without contacting ModelScope. Pure conversion functions such as `node_record` and `linked_reports` can be tested without creating a corpus.

## Common failures

| Error | Likely cause | What to do |
|---|---|---|
| `KeyError` for a release path | The selected repository lacks that artifact in this release | Use `corpus.repositories()` or inspect the manifest |
| `ValueError` after a download | File size or checksum differs from the manifest | Retry; persistent failures may indicate a changed or incomplete release |
| `RuntimeError` when loading a graph | `trust_graph_files` is false | Review the graph source, then explicitly enable trusted loading |
| `KeyError` from `corpus.nodes` | A requested repository node ID does not exist | Check that the ID belongs to the same repository |
| `FileNotFoundError` for Arrow shards | The requested large table is absent from the release | Confirm the repository and release revision |
| Network or timeout error | ModelScope could not be reached | Retry later; already verified cache files remain usable |

## API summary

| Method | Downloads data? | Result |
|---|---|---|
| `repositories()` | No | Names with required basic artifacts |
| `repository_status(repo)` | No | Quick cache status and download size |
| `download_repository(repo)` | Yes, if not cached | Basic repository artifacts |
| `issue_refs(repo)` | Yes, if not cached | Available GREPO issue references |
| `issue(ref)` | Yes, if not cached | Time, anchors, and search queries |
| `issue(ref, include_report=True)` | Yes, including a large table | Issue data plus linked report and fix PR |
| `repository_graph(repo)` | Yes, if not cached | Trusted temporal PyG graph |
| `nodes(repo, node_ids)` | Yes, including a large table | Paths, names, types, and source text |
| `gold(ref)` | Yes, if not cached | Changed repository node IDs or `None` |

## Complete small example

```python
from pathlib import Path

from grepo_localization.corpus import API, REVISION, GrepoCorpus, release_manifest
from grepo_localization.graphs import graph_for_issue

manifest = release_manifest(API, REVISION)
corpus = GrepoCorpus(
    manifest,
    Path("grepo_workspace"),
    API,
    REVISION,
    trust_graph_files=True,
)

repo = "csvkit"
ref = corpus.issue_refs(repo)[0]
issue = corpus.issue(ref)
data = graph_for_issue(corpus, ref, hops=1)

candidate_ids = data.original_node_id[data.candidate_mask].tolist()
records = corpus.nodes(repo, candidate_ids[:5])

print("GREPO issue:", ref.issue_id)
print("Search query:", issue.search_queries[0] if issue.search_queries else None)
print("Gold nodes:", sorted(corpus.gold(ref) or ()))

for node_id, record in records.items():
    print(node_id, record.path, record.name)
```

For a known labeled walkthrough, the web application demonstrates `csvkit` GREPO issue `107`. The full notebook in [`notebooks/01_explore_grepo.ipynb`](../../notebooks/01_explore_grepo.ipynb) expands this workflow into candidate coverage and a GNN forward pass.
