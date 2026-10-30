"""Offline safety checks for the new campaign cache restorer."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("restore_large_v2", ROOT / "scripts/restore_large_campaign_resources_v2.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RestoreSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.content = {name: ("small fixture " + name).encode() for name in MODULE.FILES}
        self.rows = [{"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "source_url": f"https://huggingface.co/{MODULE.MODEL}/resolve/{MODULE.REVISION}/{name}"}
                     for name, data in sorted(self.content.items())]
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def populate(self):
        self.cache.mkdir()
        for name, data in self.content.items():
            (self.cache / name).write_bytes(data)

    def sender(self, request, timeout):
        self.calls.append(request.full_url)
        return io.BytesIO(self.content[request.full_url.rsplit("/", 1)[1]])

    def test_verify_complete_cache_never_requests_network(self):
        self.populate()
        before = {p.name: p.read_bytes() for p in self.cache.iterdir()}
        result = MODULE.restore(self.rows, self.cache, opener=self.sender)
        self.assertEqual(result["status"], "all_seven_resources_verified")
        self.assertEqual(result["public_download_attempts"], 0)
        self.assertFalse(self.calls)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.cache.iterdir()})

    def test_default_missing_cache_stays_absent_and_never_requests_network(self):
        result = MODULE.restore(self.rows, self.cache, opener=self.sender)
        self.assertEqual(result["status"], "blocked_missing_resources")
        self.assertEqual(len(result["files"]), 7)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.calls)

    def test_same_size_existing_corruption_blocks_all_downloads_and_is_preserved(self):
        self.populate()
        bad = self.cache / self.rows[0]["name"]
        corrupted = b"x" * bad.stat().st_size
        bad.write_bytes(corrupted)
        (self.cache / self.rows[1]["name"]).unlink()
        result = MODULE.restore(self.rows, self.cache, allow_download=True, opener=self.sender)
        self.assertEqual(result["status"], "blocked_existing_mismatch")
        self.assertFalse(self.calls)
        self.assertEqual(bad.read_bytes(), corrupted)
        self.assertFalse((self.cache / self.rows[1]["name"]).exists())

    def test_download_missing_publishes_only_verified_bytes(self):
        result = MODULE.restore(self.rows, self.cache, allow_download=True, opener=self.sender)
        self.assertEqual(result["status"], "all_seven_resources_verified")
        self.assertEqual(result["public_download_attempts"], 7)
        self.assertEqual(len(self.calls), 7)
        self.assertEqual(self.content, {p.name: p.read_bytes() for p in self.cache.iterdir()})

    def test_bad_download_is_not_published_and_temporary_file_is_removed(self):
        def wrong(request, timeout):
            return io.BytesIO(b"x" * self.rows[0]["size"])
        result = MODULE.restore(self.rows, self.cache, allow_download=True, opener=wrong)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_concurrently_created_destination_is_never_replaced(self):
        row = self.rows[0]
        def racing_sender(request, timeout):
            (self.cache / row["name"]).write_bytes(b"concurrent data")
            return io.BytesIO(self.content[row["name"]])
        result = MODULE.restore(self.rows, self.cache, allow_download=True, opener=racing_sender)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual((self.cache / row["name"]).read_bytes(), b"concurrent data")
        self.assertEqual(len(list(self.cache.iterdir())), 1)

    def test_symlink_is_refused_without_touching_target(self):
        self.populate()
        path = self.cache / self.rows[0]["name"]
        path.unlink()
        external = self.root / "external"
        external.write_bytes(b"untouched")
        path.symlink_to(external)
        result = MODULE.restore(self.rows, self.cache, allow_download=True, opener=self.sender)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(external.read_bytes(), b"untouched")
        self.assertFalse(self.calls)

    def test_changed_frozen_receipt_is_refused(self):
        path = self.root / "receipt.json"
        path.write_text("{}\n")
        with self.assertRaises(MODULE.RestoreError):
            MODULE.read_frozen_receipt(path)

    def test_unrelated_or_insecure_redirect_is_refused(self):
        for url in ("https://example.com/model", "http://huggingface.co/model",
                    "https://huggingface.co.evil.example/model", "https://user:secret@huggingface.co/model"):
            with self.subTest(url=url), self.assertRaises(MODULE.RestoreError):
                MODULE.validate_public_url(url)


if __name__ == "__main__":
    unittest.main()

