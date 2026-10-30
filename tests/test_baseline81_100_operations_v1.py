"""Operational boundary tests; no model, tokenizer, codec or scientific audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PACKAGE = module('new_baseline_packager', ROOT/'scripts/package_baseline81_100_v1.py')
CLOSE = module('new_baseline_closer', ROOT/'scripts/close_baseline81_100_v1.py')
PREPARE = module('new_baseline_preparer', ROOT/'scripts/prepare_baseline81_100_v1.py')


class OperationalBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='baseline81-100-synthetic-')
        self.root = Path(self.temporary.name)
        (self.root/'scripts').mkdir()
        for relative in (PACKAGE.REASSEMBLER, PACKAGE.RESTORER):
            shutil.copyfile(ROOT/relative, self.root/relative)
        self.reassemble = module('fixture_reassembler', self.root/PACKAGE.REASSEMBLER)
        PACKAGE.configure(self.root, 41)
        (self.root/PACKAGE.OUT).mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self):
        # Hash-counter bytes are deterministic and incompressible enough to cross
        # the obsolete single-blob1MiB limit without a network/model resource.
        big = b''.join(hashlib.sha256(i.to_bytes(4, 'big')).digest()
                       for i in range(40000))
        members = {'synthetic/large.bin': big, 'synthetic/receipt.json': b'{"fixture":true}\n'}
        archive_path = self.root/PACKAGE.OUT/'full.zip'
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as stream:
            for path, raw in sorted(members.items()):
                stream.writestr(path, raw)
        raw = archive_path.read_bytes()
        self.assertGreater(len(raw), 1024*1024)
        manifest = {'schema': 'native-continuation-artifact-bundle-v1',
                    'artifact_id': 'synthetic-boundary-only',
                    'archive': {'path': str(PACKAGE.OUT/'full.zip'),
                                'bytes': len(raw), 'sha256': PACKAGE.sha(raw)},
                    'files': {p: {'bytes': len(v), 'sha256': PACKAGE.sha(v)}
                              for p, v in members.items()}}
        PACKAGE.write_json_new(self.root/PACKAGE.OUT/'manifest.json', manifest)
        transport = PACKAGE.chunk_archive(raw, manifest)
        return manifest, transport, members

    def reassemble_to(self, destination):
        return self.reassemble.reassemble(self.root, self.root/PACKAGE.OUT/'transport-manifest.json', destination)

    def test_large_archive_production_chunk_restore(self):
        manifest, transport, members = self.fixture()
        self.assertGreater(transport['chunk_count'], 5)
        self.assertTrue(all(0 < p['bytes'] <= 196608 for p in transport['chunks']))
        result = PACKAGE.verify_chunk_restore(manifest, members)
        self.assertTrue(result['reassembled_from_parts'])
        self.assertEqual(result['verified_members'], 2)
        self.assertEqual(result['written_members'], 2)
        self.assertEqual(result['archive_sha256'], manifest['archive']['sha256'])
        # Demonstrate no dependency on the original ZIP at the reassembly input.
        (self.root/PACKAGE.OUT/'full.zip').unlink()
        destination = self.root/'from-parts-alone'
        restored = self.reassemble_to(destination)
        self.assertTrue(restored['archive_written'])
        self.assertEqual(PACKAGE.sha((destination/PACKAGE.OUT/'full.zip').read_bytes()),
                         manifest['archive']['sha256'])

    def test_missing_part_rejected_before_archive_write(self):
        _, transport, _ = self.fixture()
        (self.root/transport['chunks'][2]['path']).unlink()
        destination = self.root/'missing'
        with self.assertRaises(FileNotFoundError):
            self.reassemble_to(destination)
        self.assertFalse((destination/PACKAGE.OUT/'full.zip').exists())

    def test_corrupted_part_rejected_before_archive_write(self):
        _, transport, _ = self.fixture()
        part = self.root/transport['chunks'][0]['path']
        raw = bytearray(part.read_bytes()); raw[100] ^= 1; part.write_bytes(raw)
        destination = self.root/'corrupt'
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.reassemble_to(destination)
        self.assertFalse((destination/PACKAGE.OUT/'full.zip').exists())

    def test_reordered_and_duplicated_parts_rejected(self):
        _, transport, _ = self.fixture()
        path = self.root/PACKAGE.OUT/'transport-manifest.json'
        original = path.read_bytes()
        for mode in ('swap', 'duplicate'):
            with self.subTest(mode=mode):
                value = json.loads(original)
                if mode == 'swap':
                    value['chunks'][0], value['chunks'][1] = value['chunks'][1], value['chunks'][0]
                else:
                    value['chunks'][1] = value['chunks'][0].copy()
                path.write_text(json.dumps(value))
                destination = self.root/mode
                with self.assertRaisesRegex(ValueError, 'Duplicate chunk|discontinuous'):
                    self.reassemble_to(destination)
                self.assertFalse((destination/PACKAGE.OUT/'full.zip').exists())

    def test_changed_manifest_and_conflicting_output_rejected(self):
        _, _, _ = self.fixture()
        manifest_path = self.root/PACKAGE.OUT/'manifest.json'
        original = manifest_path.read_bytes()
        manifest_path.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.reassemble_to(self.root/'bad-manifest')
        manifest_path.write_bytes(original)
        destination = self.root/'conflict'
        target = destination/PACKAGE.OUT/'full.zip'
        target.parent.mkdir(parents=True); target.write_bytes(b'preserve-existing')
        with self.assertRaisesRegex(ValueError, 'refuse overwrite'):
            self.reassemble_to(destination)
        self.assertEqual(target.read_bytes(), b'preserve-existing')

    def test_end_of_tinyllama_has_no_next_tinyllama_record(self):
        CLOSE.configure(50)
        batch, orders, count, gate = CLOSE.next_stage()
        self.assertIsNone(batch); self.assertEqual(orders, []); self.assertIsNone(count)
        self.assertIn('separate', gate)
        self.assertEqual((CLOSE.BEFORE, CLOSE.AFTER, CLOSE.RUN_COUNT, CLOSE.PRIMARY_COUNT),
                         (98, 100, 106, 327))
        CLOSE.configure(41)
        self.assertEqual(CLOSE.next_stage()[:3], ('batch42', [83, 84], 82))

    def test_old_out_of_scope_and_noninteger_batches_rejected(self):
        for value in (40, 51, True, 41.0, '41'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): PACKAGE.configure(self.root, value)
                with self.assertRaises(ValueError): CLOSE.configure(value)
                with self.assertRaises(ValueError): PREPARE.configure(self.root, value)

    def test_optimized_preparation_fails_before_resource_access(self):
        result = subprocess.run([sys.executable, '-O', str(ROOT/'scripts/prepare_baseline81_100_v1.py'),
                                 '--root', str(self.root/'absent'), '--batch-number', '41'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Optimized Python disables operational assertions', result.stderr)
        self.assertFalse((self.root/'absent').exists())

    def test_packager_does_not_replay_existing_audit(self):
        (self.root/PACKAGE.OUT).rmdir()
        audit = self.root/PACKAGE.AUDIT
        audit.parent.mkdir(parents=True); audit.write_text('{"existing":true}\n')
        with self.assertRaisesRegex(ValueError, 'do not repeat'):
            PACKAGE.main()
        self.assertEqual(audit.read_text(), '{"existing":true}\n')
        self.assertFalse((self.root/PACKAGE.OUT).exists())


if __name__ == '__main__':
    unittest.main()
