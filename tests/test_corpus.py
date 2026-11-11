"""Contract tests for release access and record decoding."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from grepo_localization.corpus import (
    download_file,
    file_matches_manifest,
    linked_reports,
    node_record,
    release_manifest,
    repo_paths,
)


class RecordingOpener:
    """Return fixed bytes while recording requested URLs."""

    def __init__(self, body):
        self.body = body
        self.calls = []

    def __call__(self, url, *, timeout):
        self.calls.append((url, timeout))
        return io.BytesIO(self.body)


class CorpusContractsTest(unittest.TestCase):
    def test_release_manifest_keeps_only_blobs(self):
        payload = {
            "Data": {
                "Files": [
                    {"Type": "blob", "Path": "a.txt", "Size": 1},
                    {"Type": "tree", "Path": "folder"},
                ],
                "TotalCount": 2,
            }
        }
        opener = RecordingOpener(json.dumps(payload).encode())

        manifest = release_manifest("https://example.test/repo", "rev", opener=opener)

        self.assertEqual(list(manifest), ["a.txt"])
        self.assertEqual(len(opener.calls), 1)

    def test_download_file_verifies_and_reuses_cached_bytes(self):
        body = b"verified content"
        item = {"Size": len(body), "Sha256": hashlib.sha256(body).hexdigest()}
        opener = RecordingOpener(body)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = download_file(
                "nested/file.bin", item, root, "https://example.test/file", "rev",
                opener=opener,
            )
            cached = download_file(
                "nested/file.bin", item, root, "https://example.test/file", "rev",
                opener=opener,
            )

            self.assertEqual(path, cached)
            self.assertEqual(path.read_bytes(), body)
            self.assertTrue(file_matches_manifest(path, item))
            self.assertEqual(len(opener.calls), 1)

    def test_record_decoders_are_pure(self):
        report = {"number": 1, "issues_info": json.dumps({"title": "Bug"})}
        row = {"id": 7, "type": 4, "path": "a.py", "name": "run", "attr": '{"code":"pass"}'}

        self.assertEqual(linked_reports(report), ({"title": "Bug"},))
        self.assertEqual(node_record(row).text, "pass")
        self.assertTrue(repo_paths("demo")["graph"].endswith("demo.timed.pt"))


if __name__ == "__main__":
    unittest.main()
