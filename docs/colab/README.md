# Set up the GREPO notebook in Google Colab

This guide runs [`01_explore_grepo.ipynb`](../../notebooks/01_explore_grepo.ipynb) in Google Colab without installing Python locally.

The notebook downloads a small part of the GREPO release, constructs issue-specific PyTorch Geometric graphs, measures candidate coverage, creates chronological train and test datasets, and runs a GraphSAGE forward-pass check.

## Before you begin

You need:

- a Google account;
- this project published in a GitHub repository that Colab can read;
- permission to download files from the GREPO ModelScope release.

Start with the default `csvkit` repository and keep content embeddings disabled. The basic `csvkit` graph, anchors, and queries total approximately 4.62 MB. Inspecting reports and historical source downloads additional tables.

## Recommended setup: open directly from GitHub

Opening the notebook directly from GitHub is the simplest setup. Colab loads the `.ipynb` file without requiring you to clone the repository or manage a working directory.

The notebook still needs to install this project's Python package. Its first cell installs the matching GitHub repository before the GREPO imports run.

## 1. Open the notebook directly from GitHub

Replace `OWNER`, `REPO`, and `BRANCH` in this address:

```text
https://colab.research.google.com/github/OWNER/REPO/blob/BRANCH/notebooks/01_explore_grepo.ipynb
```

For example, if the repository uses the `main` branch:

```text
https://colab.research.google.com/github/OWNER/REPO/blob/main/notebooks/01_explore_grepo.ipynb
```

You can also open [Google Colab](https://colab.research.google.com/), select the **GitHub** tab, paste the repository URL, and choose `notebooks/01_explore_grepo.ipynb`.

Before editing, save your own copy using **File → Save a copy in Drive**. A notebook opened directly from GitHub does not automatically write changes back to that repository.

## 2. Choose a runtime

A standard CPU runtime is enough for the complete walkthrough. Graph construction and dataset preparation remain on CPU.

To use CUDA for the GraphSAGE check:

1. Open **Runtime → Change runtime type**.
2. Select an available GPU hardware accelerator.
3. Save the runtime settings.

The notebook automatically uses CUDA when `torch.cuda.is_available()` is true and otherwise uses CPU. It intentionally does not select Apple MPS, which is not part of a hosted Colab runtime.

Colab GPU availability and runtime limits vary. Using a GPU is optional for this notebook; choose a standard runtime if you do not need one. See Google's [Colab FAQ](https://research.google.com/colaboratory/faq.html) for current runtime behavior.

## 3. Install this project

The notebook's first code cell contains a commented Colab installation command. Replace `OWNER/REPO` with the GitHub repository and remove the leading `#`:

```python
%pip install "grepo-localization[notebook] @ git+https://github.com/OWNER/REPO.git"
```

To install a branch other than the repository default:

```python
%pip install "grepo-localization[notebook] @ git+https://github.com/OWNER/REPO.git@BRANCH"
```

Run the installation cell before any cell that imports `grepo_localization`.

If you update the package in GitHub during the same Colab session, reinstall it and then use **Runtime → Restart session** so Python does not keep older imported modules in memory.

### Optional alternative: clone the repository

You do not need to clone the repository for the normal walkthrough. Clone it only when you want to inspect or edit source files alongside the notebook:

```python
!git clone https://github.com/OWNER/REPO.git
%cd REPO
%pip install -e '.[notebook]'
```

Use either the recommended direct GitHub installation or the clone workflow; do not run both installation paths in the same session.

For a private repository, use Colab's supported GitHub authorization rather than placing a personal access token in a notebook cell. Tokens can leak through saved cell contents, outputs, or shared notebook history.

## 4. Review the notebook settings

The settings cell begins with values similar to these:

```python
REPOS = ["csvkit"]
ROOT = Path("grepo_workspace")
REVISION = "master"
HOPS = 1
USE_CONTENT_EMBEDDINGS = False
TRUST_OFFICIAL_PYG_FILES = True
```

For a first run:

- keep `REPOS = ["csvkit"]`;
- keep `HOPS = 1`;
- keep `USE_CONTENT_EMBEDDINGS = False`;
- review the graph-file warning before enabling trusted loading.

### Why graph-file trust is explicit

GREPO repository graphs are released as PyTorch `.pt` files. Loading these files uses pickle-backed deserialization. Only set:

```python
TRUST_OFFICIAL_PYG_FILES = True
```

when you accept the official GREPO release as a trusted source. Leave it false for files from an unknown or modified source.

## 5. Optionally persist the GREPO cache in Google Drive

Colab's local virtual machine is temporary. If the runtime is deleted, files under `/content` disappear and must be downloaded again.

To preserve verified GREPO files between sessions, insert this cell before the settings cell:

```python
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
COLAB_CACHE = Path("/content/drive/MyDrive/grepo-localization-cache")
COLAB_CACHE.mkdir(parents=True, exist_ok=True)
```

Then change the settings line to:

```python
ROOT = COLAB_CACHE
```

Mounting Drive gives the notebook access to files in your Drive, so review notebook code before granting access. Drive-backed operations can also be slower than the runtime's local disk. For a one-time run, keep `ROOT = Path("grepo_workspace")` and use the local runtime cache.

Google documents Drive mounting and external-data access in its [Colab I/O notebook](https://colab.research.google.com/notebooks/io.ipynb).

## 6. Run the notebook in order

Use **Runtime → Run all** or execute Sections 1–6 from top to bottom.

The expected stages are:

1. Install dependencies and select CUDA or CPU.
2. Read the GREPO release manifest.
3. Download and verify the selected repository artifacts.
4. Inspect repository and issue metadata.
5. Build and visualize an issue-specific graph.
6. Measure candidate coverage.
7. Create chronological train and test datasets.
8. Run one GraphSAGE forward pass.

Section 7 of the notebook exports prepared graphs only when `EXPORT = True`.

With the default settings, the final forward-pass message should resemble:

```text
GraphSAGE forward pass on cpu: 2 graphs, 97 nodes, 80 scored candidates
```

The device may say `cuda`, and exact batch sizes can change if repository data or settings change.

## 7. Save the notebook and generated files

Colab saves notebook text, code, and outputs to the notebook copy in Google Drive. It does not preserve the runtime's installed packages or temporary filesystem after the virtual machine is deleted.

After the run:

- use **File → Save** to retain cell outputs in the Drive copy;
- use **File → Download → Download .ipynb** for a local notebook file;
- copy exported datasets to Drive before ending the runtime if `ROOT` uses local storage.

If you intend to commit the executed notebook, review its outputs first. Issue text and historical source excerpts are included in the saved output.

## Optional dataset export

To write prepared graph files, change:

```python
EXPORT = True
```

The notebook writes:

```text
ROOT/prepared/hops1_structural/
├── train/
├── test/
├── splits.csv
└── coverage.csv
```

If `ROOT` points to Google Drive, these exports persist across Colab sessions. If it points to local runtime storage, copy the directory to Drive or download it before disconnecting.

Do not commit the downloaded `grepo_workspace/raw` release files to GitHub. The notebook and package can reproduce them from the verified release manifest.

## Troubleshooting

### `ModuleNotFoundError: grepo_localization`

Run the `%pip install` cell before the imports. Confirm that `OWNER/REPO` was replaced and that Colab can read the GitHub repository. Restart the session after reinstalling an updated package.

### GitHub installation fails

Open the repository URL in a private browser window to confirm it is publicly readable. For a private repository, authorize GitHub access without embedding credentials in notebook code.

### ModelScope download fails

Retry the manifest or download cell. Verified cache files are reused, so a retry does not redownload valid artifacts. If the failure continues, confirm that the Colab runtime can reach `modelscope.cn`.

### `RuntimeError` when loading the repository graph

Graph loading is still disabled. Review the security note and set `TRUST_OFFICIAL_PYG_FILES = True` only if you trust the graph artifacts.

### GPU is unavailable

The notebook works on CPU. Colab does not guarantee a particular GPU or GPU availability. Leave the runtime on CPU and rerun the settings cell.

### CUDA runs out of memory

Start with one repository, `HOPS = 1`, and `USE_CONTENT_EMBEDDINGS = False`. For the smoke test, you can force CPU by replacing the device-selection line with:

```python
device = torch.device("cpu")
```

### Google Drive reports an I/O or quota error

Switch `ROOT` back to local runtime storage for the current run, then copy only final exports to Drive. Google Drive applies operation and bandwidth limits, and mounted Drive access can be slower than local runtime storage.

## What to read next

- [Using `GrepoCorpus`](../corpus/README.md)
- [Learning GNNs with GREPO](../learning-gnns-with-grepo.md)
- [Main project README](../../README.md)
- [Official Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
