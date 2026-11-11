"""Download and inspect the files that make up the GREPO dataset.

GREPO is published as many files in a ModelScope dataset release. This module
lists those files, downloads only the ones needed for a selected repository,
checks downloaded bytes for corruption, and exposes issue, graph, label, and
source-code records through :class:`GrepoCorpus`.

Most users should begin with :func:`release_manifest`, create a
:class:`GrepoCorpus`, and then use its public methods. The smaller functions are
kept separate so downloading, verification, and record conversion can be tested
without loading a complete repository.
"""

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import torch
from torch_geometric.data import Data

API = "https://modelscope.cn/api/v1/datasets/qingpingmomo/Grepo/repo"
REVISION = "master"


def _manifest_page(api: str, revision: str, page: int, opener: Callable[..., Any]) -> dict:
    """Fetch one page from ModelScope's paginated file listing.

    A GREPO release contains more files than ModelScope returns in one HTTP
    response. ModelScope therefore divides the listing into numbered pages,
    with up to 1,000 file or directory entries on each page.

    Args:
        api: Base URL of the ModelScope dataset API.
        revision: Dataset version to list, usually ``"master"``.
        page: Page number in the remote listing. ModelScope starts at page 1.
        opener: Function used to open the URL. It normally is
            :func:`urllib.request.urlopen`; tests can provide an in-memory
            replacement.

    Returns:
        The ``Data`` object from ModelScope's JSON response. It contains the
        entries for this page and the total number of entries in the release.

    Raises:
        Any network error from ``opener`` or decoding error from ``json.load``.
    """
    params = urlencode(dict(Revision=revision, Recursive="true",
                            PageNumber=page, PageSize=1000))
    with opener(f"{api}/tree?{params}", timeout=60) as response:
        return json.load(response)["Data"]


def release_manifest(api: str, revision: str, *, opener: Callable[..., Any] = urlopen) -> dict[str, dict]:
    """List every downloadable file in one GREPO release.

    The *manifest* is an inventory of remote files. Each value includes the
    file's byte size and SHA-256 checksum, which later downloads use to detect
    incomplete or corrupted data. Directory entries are not included.

    Args:
        api: Base URL of the ModelScope dataset API.
        revision: Dataset version to inspect, usually ``"master"``.
        opener: Optional HTTP-opening function. Supplying one is mainly useful
            in tests that must not access the network.

    Returns:
        A dictionary whose keys are paths inside the release and whose values
        are ModelScope metadata dictionaries for those files.

    Raises:
        Any network, JSON-decoding, or missing-field error encountered while
        reading the remote listing.
    """
    files = {}
    page = 1
    while True:
        payload = _manifest_page(api, revision, page, opener)
        entries = payload["Files"]
        for entry in entries:
            if entry["Type"] == "blob":
                files[entry["Path"]] = entry
        if not entries or page * 1000 >= payload["TotalCount"]:
            return files
        page += 1


anchor_prefix = "03_graph_feature_construction/rewriter"
embedding_prefix = "04_node_content_embeddings/get_content_embedding"


def repo_paths(repo: str) -> dict[str, str]:
    """Build the expected release paths for a repository's GREPO artifacts.

    Args:
        repo: Repository name used by GREPO, for example ``"csvkit"``.

    Returns:
        A mapping from artifact name—such as ``"graph"`` or ``"anchors"``—to
        its path inside the ModelScope release. The paths are only constructed;
        this function does not access the network or confirm they exist.
    """
    return {
        "graph": f"01_graph_construction/pyggraph/{repo}.timed.pt",
        "anchors": f"{anchor_prefix}/{repo}/anchor_node.json",
        "queries": f"{anchor_prefix}/{repo}/rewriter_output_post.json",
        "embeddings": f"{embedding_prefix}/{repo}/{repo}_embeddings.safetensors",
        "embedding_index": f"{embedding_prefix}/{repo}/{repo}_embedding_index.json",
    }


def file_matches_manifest(path: Path, item: Mapping[str, Any]) -> bool:
    """Check whether a local file is a complete copy of a release file.

    Args:
        path: Local file to inspect.
        item: Manifest metadata containing the expected ``Size`` in bytes and
            hexadecimal ``Sha256`` digest.

    Returns:
        ``True`` only when the local file exists and both its size and checksum
        match the release metadata. Missing or mismatched files return
        ``False``.
    """
    if not path.is_file() or path.stat().st_size != item["Size"]:
        return False
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() == item["Sha256"]


def download_file(
    remote_path: str,
    item: Mapping[str, Any],
    root: Path,
    api: str,
    revision: str,
    *,
    opener: Callable[..., Any] = urlopen,
) -> Path:
    """Download one GREPO file, verify it, and return its local path.

    An already verified cache entry is reused. A new download first goes to a
    temporary file beside the final target. Only after its size and SHA-256
    match the manifest is it moved into place, so an interrupted download does
    not leave a corrupt cache entry.

    Args:
        remote_path: File path inside the ModelScope release.
        item: Manifest metadata for that path.
        root: Local cache directory. Release files are stored below
            ``root / "raw"``.
        api: Base URL of the ModelScope dataset API.
        revision: Dataset version to download.
        opener: Optional HTTP-opening function, primarily for tests.

    Returns:
        Path to the verified local file.

    Raises:
        ValueError: If the downloaded bytes do not match the manifest.
        OSError: If the network or local filesystem cannot complete the work.
    """
    target = root / "raw" / remote_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_matches_manifest(target, item):
        return target

    partial = None
    params = urlencode(dict(Revision=revision, FilePath=remote_path), safe="/")
    try:
        with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".part",
                                delete=False) as file:
            partial = Path(file.name)
            with opener(f"{api}?{params}", timeout=120) as response:
                shutil.copyfileobj(response, file, length=1024 * 1024)
        if not file_matches_manifest(partial, item):
            raise ValueError(f"Downloaded bytes differ from ModelScope manifest: {remote_path}")
        partial.replace(target)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return target


def download_repositories(
    selected_paths: Mapping[str, Mapping[str, str]],
    manifest: Mapping[str, Mapping[str, Any]],
    root: Path,
    api: str,
    revision: str,
    *,
    opener: Callable[..., Any] = urlopen,
    progress: Callable[[str], None] | None = print,
) -> dict[str, dict[str, Path]]:
    """Cache a chosen set of artifacts for one or more repositories.

    Args:
        selected_paths: Nested mapping of repository name, artifact name, and
            release path. :func:`repo_paths` produces the inner paths.
        manifest: Release inventory returned by :func:`release_manifest`.
        root: Local cache directory.
        api: Base URL of the ModelScope dataset API.
        revision: Dataset version to download.
        opener: Optional HTTP-opening function, primarily for tests.
        progress: Function called with a short description before each file.
            Pass ``None`` for silent operation.

    Returns:
        A nested mapping with the same repository and artifact names, but with
        local :class:`~pathlib.Path` objects as values.

    Raises:
        KeyError: If a selected path is absent from the manifest.
        ValueError: If any downloaded file fails integrity verification.
    """
    local_files = {}
    for repo, paths in selected_paths.items():
        local_files[repo] = {}
        for kind, path in paths.items():
            if progress is not None:
                progress(f"{repo} · {kind}")
            local_files[repo][kind] = download_file(
                path, manifest[path], root, api, revision, opener=opener
            )
    return local_files


def read_release_table(
    prefix: str,
    columns: Iterable[str],
    manifest: Mapping[str, Mapping[str, Any]],
    root: Path,
    api: str,
    revision: str,
    *,
    opener: Callable[..., Any] = urlopen,
):
    """Load part of a large GREPO table stored across Arrow files.

    GREPO splits large tables into numbered *shards*. This function finds every
    shard below a release path, downloads it if needed, selects only the named
    columns, and combines the batches in file-name order.

    Args:
        prefix: Release directory containing ``data-*.arrow`` shard files.
        columns: Column names to retain from each shard.
        manifest: Release inventory returned by :func:`release_manifest`.
        root: Local cache directory.
        api: Base URL of the ModelScope dataset API.
        revision: Dataset version to read.
        opener: Optional HTTP-opening function, primarily for tests.

    Returns:
        A :class:`pyarrow.Table` containing the selected columns.

    Raises:
        FileNotFoundError: If the release contains no shards below ``prefix``.
        KeyError: If a requested column is absent.
        ValueError: If a downloaded shard fails verification.
    """
    import pyarrow as pa

    shards = sorted(
        name for name in manifest
        if name.startswith(f"{prefix}/data-") and name.endswith(".arrow")
    )
    if not shards:
        raise FileNotFoundError(f"No GREPO Arrow files found under {prefix}")
    batches = []
    for name in shards:
        path = download_file(name, manifest[name], root, api, revision, opener=opener)
        with pa.memory_map(str(path)) as source:
            batches.extend(batch.select(columns) for batch in pa.ipc.open_stream(source))
    return pa.Table.from_batches(batches)


@dataclass(frozen=True)
class IssueRef:
    """Identify one GREPO issue row by repository name and row number.

    ``issue_id`` is GREPO's row identifier. It is not necessarily the public
    GitHub issue number found in the linked report metadata.
    """

    repo: str
    issue_id: int


@dataclass(frozen=True)
class IssueRecord:
    """Inputs needed to build and explain a graph for one bug report.

    ``graph_time`` selects the historical repository state. The extractor
    anchors are code-node IDs used as graph starting points. Search queries are
    text derived from the report. ``reports`` and ``fix_pr`` are populated only
    when the caller explicitly requests the larger report table.
    """

    ref: IssueRef
    graph_time: int
    extractor_anchor_ids: tuple[int, ...]
    search_queries: tuple[str, ...]
    reports: tuple[dict, ...] = ()
    fix_pr: dict | None = None


@dataclass(frozen=True)
class NodeRecord:
    """Human-readable information about one node in a repository graph.

    ``node_id`` is the stable GREPO repository ID, while ``node_type`` tells
    whether the node represents a directory, file, class, function, or another
    released category. ``text`` contains historical source when GREPO provides
    it and is otherwise ``None``.
    """

    node_id: int
    node_type: int
    path: str
    name: str | None
    text: str | None


@lru_cache(maxsize=1)
def read_repository_graph(path: Path) -> Data:
    """Load a repository's temporal graph from a trusted PyTorch file.

    Args:
        path: Local ``.pt`` graph file from the GREPO release.

    Returns:
        A PyTorch Geometric :class:`~torch_geometric.data.Data` object. Repeated
        calls for the same path reuse the in-memory graph.

    Raises:
        ValueError: If node types and node-lifetime tensors have different
            lengths, which indicates an unusable graph.

    Warning:
        PyTorch graph files use pickle-backed deserialization. Only call this
        function for artifacts from a source you trust.
    """
    graph = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if graph.starttimestamp.numel() != graph.type.numel() or graph.endtimestamp.numel() != graph.type.numel():
        raise ValueError(f"Misaligned node tensors for {path}")
    return graph


def linked_reports(fix_pr: Mapping[str, Any]) -> tuple[dict, ...]:
    """Extract bug-report records stored inside a fix pull-request row.

    Args:
        fix_pr: Row from GREPO's pull-request table. Its ``issues_info`` field
            contains JSON describing linked reports.

    Returns:
        A tuple of report dictionaries. Empty or missing report data produces
        an empty tuple; a single JSON object produces a one-item tuple.

    Raises:
        json.JSONDecodeError: If ``issues_info`` is not valid JSON.
    """
    raw = fix_pr.get("issues_info")
    if not raw:
        return ()
    linked = json.loads(raw)
    return (linked,) if isinstance(linked, dict) else tuple(linked or ())


def node_record(row: Mapping[str, Any]) -> NodeRecord:
    """Convert one row from GREPO's node table into a :class:`NodeRecord`.

    Args:
        row: Mapping with ``id``, ``type``, ``path``, ``name``, and ``attr``.
            The ``attr`` column is JSON and may contain a ``code`` value.

    Returns:
        A record whose ``text`` is the stored code, or ``None`` when the row
        contains no code.
    """
    text = json.loads(row["attr"] or "{}").get("code") or None
    return NodeRecord(row["id"], row["type"], row["path"], row["name"], text)


class GrepoCorpus:
    """High-level, cached access to GREPO data for selected repositories.

    Creating the object does not download anything. Public methods fetch and
    cache the files they need on first use. This lets an application inspect a
    small number of repositories without downloading the full GREPO release.
    """

    def __init__(
        self,
        manifest: Mapping[str, Mapping[str, Any]],
        root: Path,
        api: str,
        revision: str,
        *,
        trust_graph_files: bool = False,
        opener: Callable[..., Any] = urlopen,
    ):
        """Configure where and how this corpus obtains GREPO files.

        Args:
            manifest: Release inventory returned by :func:`release_manifest`.
            root: Directory used for verified local caching.
            api: Base URL of the ModelScope dataset API.
            revision: Dataset version to read.
            trust_graph_files: Whether trusted PyTorch graph deserialization is
                allowed. It is disabled by default because ``.pt`` files use
                pickle-backed loading.
            opener: Optional HTTP-opening function, primarily for tests.
        """
        self.manifest = manifest
        self.root = root
        self.api = api
        self.revision = revision
        self.trust_graph_files = trust_graph_files
        self.opener = opener

    @lru_cache(maxsize=128)
    def _file(self, repo, kind):
        """Find an artifact in the cache, downloading and verifying it if needed."""
        path = repo_paths(repo)[kind]
        return download_file(
            path, self.manifest[path], self.root, self.api, self.revision,
            opener=self.opener,
        )

    @lru_cache(maxsize=16)
    def _metadata(self, repo, kind):
        """Load and cache a repository's JSON artifact, such as its anchors."""
        return json.loads(self._file(repo, kind).read_text(encoding="utf-8"))

    @lru_cache(maxsize=2)
    def _table(self, prefix, columns):
        """Load and cache selected columns from a large sharded GREPO table."""
        return read_release_table(
            prefix, columns, self.manifest, self.root, self.api, self.revision,
            opener=self.opener,
        )

    def issue_refs(self, repo: str) -> tuple[IssueRef, ...]:
        """List the GREPO issues available for a repository.

        Args:
            repo: GREPO repository name, for example ``"csvkit"``.

        Returns:
            Issue references in the order stored by the release.
        """
        return tuple(IssueRef(repo, int(key)) for key in self._metadata(repo, "anchors"))

    def issue(self, ref: IssueRef, *, include_report: bool = False) -> IssueRecord:
        """Load the time, starting nodes, and search text for one issue.

        Args:
            ref: Repository and GREPO issue-row identifier.
            include_report: Also load the linked bug report and fix pull request.
                This is slower because it reads a large Arrow table.

        Returns:
            An :class:`IssueRecord` used to explain or build the issue graph.

        Raises:
            KeyError: If the repository has no matching GREPO issue row.
        """
        key = str(ref.issue_id)
        anchor = self._metadata(ref.repo, "anchors")[key]
        queries = self._metadata(ref.repo, "queries").get(key, {}).get("query") or ()
        reports, fix_pr = (), None
        if include_report:
            table = self._table(f"02_pr_issue_data/pulldata_re_full_1002/{ref.repo}",
                                ("number", "title", "body", "issues_info"))
            fix_pr = table.slice(ref.issue_id, 1).to_pylist()[0]
            reports = linked_reports(fix_pr)
        return IssueRecord(
            ref, anchor["issue_time"], tuple(sorted(set(anchor.get("extractor_anchor_nodes") or ()))),
            tuple(queries), reports, fix_pr,
        )

    def repository_graph(self, repo: str) -> Data:
        """Load a repository graph containing code entities and their lifetimes.

        Args:
            repo: GREPO repository name.

        Returns:
            The repository's temporal PyTorch Geometric graph.

        Raises:
            RuntimeError: If trusted graph-file loading was not enabled when
                the corpus was created.
        """
        if not self.trust_graph_files:
            raise RuntimeError("After reviewing Section 1, enable TRUST_OFFICIAL_PYG_FILES and rerun setup.")
        return read_repository_graph(self._file(repo, "graph"))

    def nodes(self, repo: str, node_ids: Iterable[int]) -> dict[int, NodeRecord]:
        """Look up names, paths, and historical source for graph nodes.

        Args:
            repo: GREPO repository name.
            node_ids: Stable repository node IDs, usually taken from a graph's
                ``original_node_id`` tensor.

        Returns:
            A dictionary from each unique requested ID to its :class:`NodeRecord`.
            An empty request returns an empty dictionary without loading the
            node table.

        Raises:
            KeyError: If any requested ID is absent from the node table.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        wanted = set(node_ids)
        if not wanted:
            return {}
        table = self._table(f"01_graph_construction/savedata/repos/{repo}",
                            ("id", "type", "path", "name", "attr"))
        ids = pa.array(sorted(wanted), type=pa.int64())
        rows = table.filter(pc.is_in(table["id"], value_set=ids)).to_pylist()
        records = {row["id"]: node_record(row) for row in rows}
        if missing := wanted - records.keys():
            raise KeyError(f"Unknown node IDs in {repo}: {sorted(missing)}")
        return records

    def gold(self, ref: IssueRef) -> frozenset[int] | None:
        """Return the code nodes changed by the issue's historical fix.

        These IDs are the supervised answer key. They should be used for
        training or evaluation, never to decide which nodes enter an issue
        graph. ``None`` means the release has no changed-node label for this
        issue.
        """
        anchor = self._metadata(ref.repo, "anchors")[str(ref.issue_id)]
        return frozenset(anchor.get("patch_related_node_ids") or ()) or None

    def repositories(self) -> tuple[str, ...]:
        """List repositories with a graph, anchors, and search-query metadata."""
        required = ("graph", "anchors", "queries")
        repos = {
            path.split("/")[-2] for path in self.manifest
            if path.startswith(f"{anchor_prefix}/") and path.endswith("/anchor_node.json")
        }
        return tuple(sorted(repo for repo in repos
                            if all(repo_paths(repo)[kind] in self.manifest for kind in required)))

    def repository_status(self, repo: str) -> tuple[bool, int]:
        """Report whether a repository's basic files appear locally cached.

        Args:
            repo: GREPO repository name.

        Returns:
            ``(ready, bytes)`` where ``bytes`` is the combined download size.
            ``ready`` performs a quick existence and size check; files receive
            full SHA-256 verification when opened or downloaded.
        """
        required = ("graph", "anchors", "queries")
        paths = [repo_paths(repo)[kind] for kind in required]
        size = sum(self.manifest[path]["Size"] for path in paths)
        ready = all((self.root / "raw" / path).is_file()
                    and (self.root / "raw" / path).stat().st_size == self.manifest[path]["Size"]
                    for path in paths)
        return ready, size

    def download_repository(self, repo: str) -> None:
        """Download and verify the graph, anchors, and queries for a repository."""
        for kind in ("graph", "anchors", "queries"):
            self._file(repo, kind)
