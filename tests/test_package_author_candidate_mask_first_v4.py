import fcntl
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('v4_package', Path(__file__).resolve().parents[1] / 'scripts/package_author_candidate_mask_first_v4.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'checkout'; self.root.mkdir()
        self.lease = self.base / 'canonical.lock'; self.lease.write_bytes(b'')
        raw = self.root / m.RAW; raw.mkdir(parents=True)
        post = self.root / m.POST; (post / 'root-process').mkdir(parents=True)
        (self.root / 'results/project-nn.lock').write_bytes(b'')
        (raw / 'initial-denominator.json').write_text('{"requested_chunks":2}')
        (raw / 'result.json').write_text('{"status":"unknown_terminal"}')
        (raw / 'empty-directory').mkdir()
        (raw / 'binary-partial.bin').write_bytes(b'\xff\x00partial')
        (post / 'source-readback.json').write_text('{}')
        for name, value in [('process.json', '{"exit_code":3}'), ('stdout.log', ''), ('stderr.log', 'unknown')]:
            (post / 'root-process' / name).write_text(value)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_bytes_unknown_and_empty_directories(self):
        before = m.storage().snapshot(self.root, 'results')
        result = m.package(self.root, self.lease)
        self.assertFalse(result['scientific_acceptance_inferred'])
        destination = self.base / 'restored'
        m.storage().restore(self.root / m.OUT / 'index.json', destination)
        self.assertEqual(before, m.storage().snapshot(destination, 'results'))
        self.assertEqual(before, m.storage().snapshot(self.root, 'results'))

    def test_missing_nested_result_preserves_incomplete(self):
        (self.root / m.RAW / 'result.json').unlink()
        result = m.package(self.root, self.lease)
        self.assertFalse(result['nested_campaign_result_present'])
        self.assertEqual(result['scientific_auditor_calls'], 0)

    def test_canonical_active_lease_refused(self):
        with self.lease.open('r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): m.package(self.root, self.lease)
        self.assertFalse((self.root / m.OUT).exists())

    def test_no_overwrite_and_unknown_root_exit_refused(self):
        p = self.root / m.POST / 'root-process/process.json'; p.write_text('{"exit_code":null}')
        with self.assertRaises(ValueError): m.package(self.root, self.lease)
        p.write_text('{"exit_code":0}'); m.package(self.root, self.lease)
        with self.assertRaises(FileExistsError): m.package(self.root, self.lease)

    def test_multiple_parts_and_corruption_refused(self):
        (self.root / m.RAW / 'synthetic.bin').write_bytes(os.urandom(450000))
        result = m.package(self.root, self.lease)
        self.assertGreater(result['preservation']['parts'], 1)
        part = self.root / m.OUT / 'full.zip.part000'
        data = part.read_bytes(); part.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with self.assertRaises(ValueError): m.storage().restore(self.root / m.OUT / 'index.json', self.base / 'bad')

    def test_foreign_tree_and_link_refused(self):
        (self.root / 'results/unrelated').mkdir()
        with self.assertRaises(ValueError): m.package(self.root, self.lease)
        (self.root / 'results/unrelated').rmdir()
        (self.root / m.RAW / 'link').symlink_to(self.lease)
        with self.assertRaises(ValueError): m.package(self.root, self.lease)


if __name__ == '__main__': unittest.main()
