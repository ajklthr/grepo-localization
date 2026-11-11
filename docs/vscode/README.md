# Set up GREPO locally in Visual Studio Code

This guide prepares the GREPO project for local notebook exploration, package development, testing, and the Streamlit issue browser.

## What you need

Install these tools before cloning the project:

- [Git](https://git-scm.com/downloads)
- [Python 3.10 or later](https://www.python.org/downloads/)
- [Visual Studio Code](https://code.visualstudio.com/Download)
- VS Code's [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- VS Code's [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

The Python extension discovers workspace virtual environments and supplies language features. The Jupyter extension runs `.ipynb` files inside VS Code. Microsoft's [Python environment guide](https://code.visualstudio.com/docs/python/environments) and [Jupyter notebook guide](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) describe the current VS Code interface.

## 1. Clone and open the repository

In a terminal:

```bash
git clone https://github.com/ajklthr/grepo-localization.git
cd grepo-localization
code .
```

If the `code` command is unavailable, open VS Code and use **File → Open Folder** to select the cloned `grepo-localization` directory.

Open the repository root rather than only the `notebooks` directory. This lets VS Code discover the package, tests, documentation, and workspace virtual environment together.

When VS Code asks whether you trust the folder, review its contents before accepting. Restricted Mode hides rich notebook output and prevents notebook execution.

## 2. Create the project virtual environment

The `.venv` directory is local to your machine and excluded from Git.

### macOS or Linux

Run these commands in the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[notebook,web]'
```

### Windows PowerShell

Run these commands in the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[notebook,web]"
```

The editable installation means changes under `src/grepo_localization` are available without reinstalling the project. The extras install:

- `notebook`: JupyterLab and plotting support;
- `web`: Streamlit and Plotly for the issue browser.

Using the environment's Python executable directly avoids activation differences between shells.

### Create the environment through VS Code instead

You can create the same workspace environment from the Command Palette:

1. Open the Command Palette with `Cmd+Shift+P` on macOS or `Ctrl+Shift+P` on Windows and Linux.
2. Run **Python: Create Environment**.
3. Choose **Venv**.
4. Select Python 3.10 or later.
5. Select `pyproject.toml` when VS Code asks which dependencies to install.

If this installs only the base package, run the platform-specific `pip install -e` command above to add the `notebook` and `web` extras.

## 3. Select the `.venv` interpreter

VS Code normally discovers a `.venv` in the workspace automatically. Confirm the selection before running code:

1. Open the Command Palette.
2. Run **Python: Select Interpreter**.
3. Choose the interpreter whose path ends with:
   - `.venv/bin/python` on macOS or Linux;
   - `.venv\Scripts\python.exe` on Windows.

The selected interpreter controls Python files, terminals, debugging, IntelliSense, and test discovery.

If `.venv` is missing from the list, run **Python Environments: Refresh All Environment Managers** or reload the VS Code window.

## 4. Open the exploration notebook

Open:

```text
notebooks/01_explore_grepo.ipynb
```

The repository includes saved output from a verified `csvkit` run. You can read that output without executing the notebook.

To run it locally:

1. Select the kernel name in the upper-right corner of the notebook editor.
2. Choose **Python Environments**.
3. Select the repository's `.venv`.
4. Review the settings cell.
5. Select **Run All** or execute Sections 1–6 in order.

The notebook imports the editable package installed from this checkout. Its commented Colab installation command should remain commented during local use.

### Recommended first-run settings

```python
REPOS = ["csvkit"]
ROOT = Path("grepo_workspace")
HOPS = 1
USE_CONTENT_EMBEDDINGS = False
TRUST_OFFICIAL_PYG_FILES = True
```

Start with `csvkit`, one-hop graph expansion, and structural features. Content embeddings add a much larger download.

### Review graph-file trust

GREPO repository graphs are distributed as PyTorch `.pt` files. Loading them uses pickle-backed deserialization. Set `TRUST_OFFICIAL_PYG_FILES = True` only after confirming that the files come from the official GREPO release.

Leave it false for unknown or modified graph files.

## 5. Understand local downloads

The notebook downloads only the selected repository artifacts. By default it stores them in a directory named:

```text
grepo_workspace/
```

Depending on the notebook kernel's working directory, this folder may appear at the repository root or below `notebooks`. Both locations are excluded by `.gitignore`.

Downloaded files are checked against the release manifest's size and SHA-256 digest. Valid files are reused on later runs.

To share one cache between the notebook and web application, choose an absolute directory in the notebook:

```python
ROOT = Path("/absolute/path/to/grepo-cache")
```

Then set the same directory before starting Streamlit.

### macOS or Linux

```bash
export GREPO_CACHE_DIR="/absolute/path/to/grepo-cache"
```

### Windows PowerShell

```powershell
$env:GREPO_CACHE_DIR = "C:\path\to\grepo-cache"
```

Do not commit GREPO downloads, prepared graph exports, virtual environments, editor settings, or credentials. The project's `.gitignore` excludes its standard local locations.

## 6. Run the Streamlit issue browser

From the repository root:

### macOS or Linux

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py
```

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. Open it in a browser, select a repository, and download its basic artifacts. Selecting a graph node or node-table row displays its historical source.

Stop the server with `Ctrl+C` in the terminal.

## 7. Run the tests

The tests do not download the GREPO release.

### macOS or Linux

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite checks corpus downloading and verification, graph construction, datasets, coverage diagnostics, notebook helper behavior, documentation, and notebook code quality.

### Run tests from the VS Code interface

1. Open the **Testing** view in the Activity Bar.
2. Select **Configure Python Tests** if prompted.
3. Choose `unittest`.
4. Choose the `tests` directory.
5. Run all tests or select an individual test.

Confirm that `.venv` remains the selected interpreter if VS Code cannot import the package during test discovery.

## 8. Work on the package

The source modules have distinct responsibilities:

| Module | Responsibility |
|---|---|
| `corpus.py` | Release inventory, verified downloads, issue metadata, source records, and labels |
| `graphs.py` | Temporal filtering, neighborhood expansion, node features, graph construction, and labels |
| `dataset.py` | Chronological splitting and lazy PyTorch datasets |
| `inspection.py` | NetworkX conversion and candidate-coverage diagnostics |
| `exploration.py` | Tested helpers used by the exploration notebook |

When changing a reusable transformation, edit the corresponding source module and add or update a test. Keep notebook cells focused on orchestration and display.

## 9. Save notebook output intentionally

Executing the notebook changes `notebooks/01_explore_grepo.ipynb` because cell output and execution counts are stored inside the file.

Before committing it:

1. Review the saved issue text and source excerpts.
2. Check for warnings containing machine-specific absolute paths.
3. Decide whether the output is useful to other readers.
4. Run the notebook quality tests.

To discard only uncommitted notebook output changes and return to the checked-in version:

```bash
git restore notebooks/01_explore_grepo.ipynb
```

This command discards all uncommitted changes to that notebook, including code and Markdown edits, so review `git diff` first.

## Troubleshooting

### VS Code uses the wrong Python

Run **Python: Select Interpreter** and choose `.venv`. Then open a new VS Code terminal and verify:

```bash
python -c "import sys; print(sys.executable)"
```

The printed path should point into the repository's `.venv`.

### The notebook cannot find a kernel

Confirm that the Jupyter extension is enabled and the notebook kernel is `.venv`. Reinstall the notebook extra with the platform-specific command from Section 2 if necessary.

### `ModuleNotFoundError: grepo_localization`

The project is not installed in the selected environment. Run:

```bash
python -m pip install -e '.[notebook,web]'
```

On Windows PowerShell, use double quotes around `.[notebook,web]`.

### Graph loading raises `RuntimeError`

Review the graph-file security warning, then explicitly enable trusted loading only for official artifacts.

### The notebook kernel terminates during GraphSAGE

Use CPU by setting:

```python
device = torch.device("cpu")
```

The notebook automatically selects CUDA when available and otherwise uses CPU. It does not automatically select Apple MPS because support varies across PyTorch and PyTorch Geometric versions.

### A download is interrupted

Run the cell again. Partial downloads use temporary files and do not replace verified cache entries.

## What to read next

- [Using `GrepoCorpus`](../corpus/README.md)
- [Learning GNNs with GREPO](../learning-gnns-with-grepo.md)
- [Google Colab setup](../colab/README.md)
- [Main project README](../../README.md)
- [Official VS Code Python environments documentation](https://code.visualstudio.com/docs/python/environments)
- [Official VS Code Jupyter documentation](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
