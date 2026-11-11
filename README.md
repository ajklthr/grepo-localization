# GREPO Code Localization

Explore which classes and functions a bug report leads us to, inspect missed changes, and prepare the same issue graphs for PyTorch Geometric.

## New to GREPO or GNNs?

Start with **[Learning GNNs with GREPO](docs/learning-gnns-with-grepo.md)**. It explains the task without assuming prior GREPO knowledge, translates the dataset into graph-learning terms, walks through every PyTorch Geometric tensor, and shows how to build a first issue graph without accidentally leaking the answer into the model.

When you are ready to access real data, use the **[GrepoCorpus README](docs/corpus/README.md)** for repository discovery, verified downloads, issue metadata, source lookup, graphs, and labels.

The short version is: GREPO represents a repository as a graph of code entities and their relationships. For each historical bug report, this project takes a time-correct neighborhood of that graph and asks a GNN to score which class or function nodes were changed by the eventual fix.

## Start the issue browser

From the repository root, using Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[web]'
streamlit run app/streamlit_app.py
```

Select a repository and click **Download repository**. Pick a GREPO issue ID, then click a graph node or a row in the node table to read its historical source. **Changed by the fix** includes labeled nodes outside the candidate set. The report and source tables are fetched on demand. Downloads are checked against GREPO's release manifest and cached in `grepo_workspace/`. Set `GREPO_CACHE_DIR` to reuse a different cache. The browser reads PyG graph files from the authors' release, which uses pickle serialization. Run it in an environment where you trust those artifacts.

Try `csvkit` GREPO issue **107**, which links to GitHub issue **#553** and has all three changed nodes in its candidates. Then try issue **116**, which links to GitHub issue **#185**. Its candidate set includes two of three changed nodes. Open node **12809** under **Changed by the fix** to inspect the missed code.

## Run the notebook

For a hosted runtime, the recommended path is to **[open the notebook directly from GitHub in Google Colab](docs/colab/README.md)**. No repository clone is required for the normal walkthrough.

```bash
python -m pip install -e '.[notebook]'
jupyter lab notebooks/01_explore_grepo.ipynb
```

For Colab, open the notebook from the published GitHub repository and uncomment the first install cell. Replace `OWNER/REPO` with that repository's path. Keep Colab's PyTorch installation so a GPU runtime can use CUDA. The default walkthrough runs on CPU and downloads only the selected repositories. Set `TRUST_OFFICIAL_PYG_FILES = True` in the setup cell after reviewing the graph file provenance.

## Project layout

| Path | Purpose |
|---|---|
| `src/grepo_localization/corpus.py` | GREPO release listing, verified downloads, issue and source lookups, historical graphs, and gold labels |
| `src/grepo_localization/graphs.py` | Build PyG issue graphs from time and anchors, with separate label attachment |
| `src/grepo_localization/inspection.py` | Coverage diagnostics and NetworkX conversion |
| `src/grepo_localization/dataset.py` | Chronological split and PyTorch dataset yielding PyG graphs |
| `src/grepo_localization/exploration.py` | Tested table, selection, visualization-preparation, smoke-test, and export helpers used by the notebook |
| `app/streamlit_app.py` | Repository picker, issue graph, node table, and source viewer |
| `notebooks/01_explore_grepo.ipynb` | Dataset walkthrough, coverage analysis, PyG batching, and export |

The browser reports candidate coverage. The notebook includes a GraphSAGE forward-pass check and an anchor-first example ranking.

## Source contracts and tests

Every function and class in `src/grepo_localization/` has a concise docstring describing its inputs, output, side effects, and expected failures. Network access, file verification, record decoding, graph transformations, label attachment, and coverage calculations are separated so pure logic can be tested without downloading the GREPO release.

Run the contract tests with only the standard-library test runner:

```bash
python -m unittest discover -s tests -v
```

[GREPO paper](https://arxiv.org/abs/2602.13921) · [Official implementation](https://github.com/qingpingmo/GREPO) · [Dataset release](https://modelscope.cn/datasets/qingpingmomo/Grepo)
