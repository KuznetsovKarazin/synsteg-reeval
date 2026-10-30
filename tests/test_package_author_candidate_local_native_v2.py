"""Synthetic archive tests only; no native/tokenizer calls or actual output reads."""
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('candidate_v2_packager', ROOT / 'scripts/package_author_candidate_local_native_v2.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


class PackagingV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.primary = self.root / p.BASE
        self.primary.mkdir(parents=True)
        for name, data in {'result.json': b'{"status":"unknown_terminal"}\n',
                           'chunk0-attempt0/native/stderr.txt': b'process failure\n',
                           'chunk0-attempt0/native/resident.json': b'{}',
                           'chunk0-attempt0/sealed/journal.jsonl': b'{"partial":',
                           'chunk0-attempt0/sealed/terminal.json.writing': b'partial',
                           'chunk0-attempt0/sealed/writer.lock': b'',
                           'chunk0-attempt0/observed-response.raw': b'\xff\x00\x80',
                           'reservations/chunk0-attempt0.json': b'{"outcome_known":false}'}.items():
            target = self.primary / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        (self.primary / 'chunk1-reserved-empty/native').mkdir(parents=True)
        self.output = self.root / 'artifacts'

    def package(self, **kwargs): return p.package(self.root, self.output, **kwargs)

    def test_all_unknown_partial_binary_empty_files_and_dirs_survive(self):
        before = p.snapshot(self.root); result = self.package()
        self.assertEqual(result['primary_files'], 8)
        index = p.read_json(self.output / 'index.json')
        self.assertFalse(index['scientific_acceptance_inferred'])
        self.assertEqual(index['campaign_files']['chunk0-attempt0/observed-response.raw']['bytes'], 3)
        self.assertIn(p.BASE + '/chunk1-reserved-empty/native', index['directories'])
        restored = self.root / 'restored'; p.restore(self.output / 'index.json', restored)
        self.assertEqual(p.snapshot(restored), before)
        self.assertEqual(p.snapshot(self.root), before)

    def test_archives_are_deterministic_across_mtime_and_output_paths(self):
        self.package(); first = (self.output / 'full.zip').read_bytes()
        for path in self.primary.rglob('*'): os.utime(path, (5, 5))
        p.package(self.root, self.root / 'artifacts-second')
        self.assertEqual(first, (self.root / 'artifacts-second/full.zip').read_bytes())
        self.assertEqual((self.output / 'index.json').read_bytes(), (self.root / 'artifacts-second/index.json').read_bytes())

    def test_parts_respect_raw_192kib_bound_and_reconstruct(self):
        raw = b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(15000))
        (self.primary / 'uncompressible.raw').write_bytes(raw)
        result = self.package(); self.assertGreater(result['parts'], 2)
        self.assertLessEqual(result['largest_part'], 192 * 1024)
        index, archive = p.read_archive(self.output / 'index.json')
        self.assertEqual(archive, (self.output / 'full.zip').read_bytes())
        listing = p.read_json(self.output / 'upload-files.json')
        self.assertFalse(any(row['path'].endswith('/full.zip') for row in listing['files']))
        self.assertEqual(len(index['parts']), result['parts'])

    def test_missing_result_requires_explicit_incomplete_preservation(self):
        (self.primary / 'result.json').unlink()
        with self.assertRaises(ValueError): self.package()
        self.assertFalse(self.output.exists())
        self.package(allow_incomplete=True)
        index = p.read_json(self.output / 'index.json')
        self.assertFalse(index['terminal_result_present'])
        self.assertTrue(index['allow_incomplete_requested'])

    def test_active_cooperative_nn_lease_prevents_packaging(self):
        with (self.root / 'results/project-nn.lock').open('w') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): self.package()
        self.assertFalse(self.output.exists())

    def test_file_or_directory_symlink_is_not_followed(self):
        for target_name in ('result.json', 'chunk0-attempt0/native'):
            with self.subTest(target=target_name):
                link = self.primary / 'link'; link.symlink_to(self.primary / target_name)
                with self.assertRaises(ValueError): self.package()
                link.unlink()
        self.assertFalse(self.output.exists())

    def test_fifo_snapshot_refuses_without_blocking(self):
        os.mkfifo(self.primary / 'pipe')
        checked = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/package_author_candidate_local_native_v2.py'),
            'package', '--root', str(self.root), '--output', str(self.output)], capture_output=True, timeout=2)
        self.assertNotEqual(checked.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_unreadable_subdirectory_cannot_be_silently_omitted(self):
        original = os.scandir
        def denied(path):
            if Path(path) == self.primary / 'chunk0-attempt0/native':
                raise PermissionError('mock unreadable primary subtree')
            return original(path)
        with patch.object(p.os, 'scandir', side_effect=denied):
            with self.assertRaises(PermissionError): self.package()
        self.assertFalse(self.output.exists())

    def test_primary_mutation_during_zip_creation_is_rejected(self):
        original = zipfile.ZipFile.writestr; modified = False
        def mutate(archive, *args, **kwargs):
            nonlocal modified
            result = original(archive, *args, **kwargs)
            if not modified:
                modified = True; (self.primary / 'late-extra.raw').write_bytes(b'changed')
            return result
        with patch.object(zipfile.ZipFile, 'writestr', mutate):
            with self.assertRaisesRegex(ValueError, 'changed'): self.package()
        self.assertFalse((self.output / 'index.json').exists())

    def test_existing_package_cannot_be_replaced(self):
        self.package(); original = (self.output / 'index.json').read_bytes()
        with self.assertRaises(FileExistsError): self.package()
        self.assertEqual(original, (self.output / 'index.json').read_bytes())

    def test_output_alias_inside_primary_is_rejected(self):
        (self.root / 'extra').mkdir()
        path = self.root / 'extra' / '..' / p.BASE / 'artifact'
        with self.assertRaises(ValueError): p.package(self.root, path)
        self.assertFalse((self.primary / 'artifact').exists())

    def test_tampered_or_truncated_part_prevents_restore(self):
        self.package(); part = self.output / 'full.zip.part000'; before = part.read_bytes()
        for raw in (before[:-1], bytes([before[0] ^ 1]) + before[1:]):
            with self.subTest(length=len(raw)):
                part.write_bytes(raw)
                with self.assertRaises(ValueError): p.restore(self.output / 'index.json', self.root / 'restore-bad')
        self.assertFalse((self.root / 'restore-bad').exists())

    def test_extra_zip_member_rejected_even_with_updated_archive_hashes(self):
        self.package(); archive = io.BytesIO((self.output / 'full.zip').read_bytes())
        with zipfile.ZipFile(archive, 'a') as z: z.writestr(p.BASE + '/extra', b'x')
        raw = archive.getvalue(); self.assertLess(len(raw), p.PART_BYTES)
        (self.output / 'full.zip.part000').write_bytes(raw)
        index = p.read_json(self.output / 'index.json')
        index['parts'][0].update(bytes=len(raw), sha256=p.sha(raw))
        index['archive'].update(bytes=len(raw), sha256=p.sha(raw))
        (self.output / 'index.json').write_bytes(p.json_bytes(index))
        with self.assertRaisesRegex(ValueError, 'member set'): p.restore(self.output / 'index.json', self.root / 'bad')

    def test_inventory_path_traversal_is_rejected(self):
        self.package(); index = p.read_json(self.output / 'index.json')
        index['directories'].append('../escape')
        (self.output / 'index.json').write_bytes(p.json_bytes(index))
        with self.assertRaises(ValueError): p.restore(self.output / 'index.json', self.root / 'bad')

    def test_restore_refuses_existing_files_or_symlink_destination(self):
        self.package(); destination = self.root / 'existing'; destination.mkdir()
        (destination / 'keep').write_bytes(b'preserved')
        with self.assertRaises(FileExistsError): p.restore(self.output / 'index.json', destination)
        link = self.root / 'linked'; link.symlink_to(destination)
        with self.assertRaises(ValueError): p.restore(self.output / 'index.json', link)
        self.assertEqual((destination / 'keep').read_bytes(), b'preserved')

    def test_duplicate_manifest_key_rejected(self):
        self.package(); path = self.output / 'index.json'; raw = path.read_bytes()
        path.write_bytes(b'{"schema":"other",' + raw[1:])
        with self.assertRaisesRegex(ValueError, 'duplicate'): p.read_archive(path)


if __name__ == '__main__': unittest.main(verbosity=2)
