"""Synthetic archive transport tests; no scientific codec or model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/package_first80_transport_results_v1.py'
spec = importlib.util.spec_from_file_location('artifact', SOURCE)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'root'
        self.root.mkdir()
        helper = self.root / P.HELPER
        helper.parent.mkdir()
        helper.write_bytes(SOURCE.read_bytes())
        m = self.root / P.MILESTONE
        m.mkdir(parents=True)
        raw = random.Random(25).randbytes(450000)
        (m / 'synthetic.bin').write_bytes(raw)
        sources = {'synthetic.bin': raw}
        binding = self.root / P.BINDING
        binding.mkdir(parents=True)
        (m / 'input_binding').mkdir()
        for name in P.BINDING_NAMES:
            data = ('synthetic-binding-' + name).encode()
            (binding / name).write_bytes(data)
            (m / 'input_binding' / name).write_bytes(data)
            sources['input_binding/' + name] = data
        manifest = P.canonical({name: P.sha(data) for name, data in sources.items()}) + b'\n'
        (m / 'output_manifest.json').write_bytes(manifest)
        a = self.root / P.AUDITS
        a.mkdir(parents=True)
        (a / 'tinyllama-first80.json').write_bytes(P.canonical({'synthetic_fixture': True, 'status': 'passed', 'output_manifest_sha256': P.sha(manifest), 'cells': 15300, 'input_records': 300}))
        self.dest = Path(self.temp.name) / 'fresh'

    def tearDown(self):
        self.temp.cleanup()

    def save(self, manifest):
        (self.root / P.ART / 'manifest.json').write_bytes(P.canonical(manifest))

    def test_exact_roundtrip_bounded_multiple_chunks(self):
        m = P.package(self.root)
        self.assertEqual(m['file_count'], 9)
        self.assertEqual(m['chunk_count'], 3)
        self.assertTrue(all(x['bytes'] <= 196608 for x in m['chunks']))
        restored = P.restore(self.dest, self.root)
        self.assertEqual(restored['verified_members'], 9)
        for path in m['files']:
            self.assertEqual((self.root / path).read_bytes(), (self.dest / path).read_bytes())

    def test_source_corruption_and_omission_refused(self):
        p = self.root / P.MILESTONE / 'synthetic.bin'
        p.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'source hash'):
            P.package(self.root)
        p.unlink()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            P.package(self.root)

    def test_existing_destinations_no_overwrite(self):
        P.package(self.root)
        with self.assertRaisesRegex(ValueError, 'no overwrite'):
            P.package(self.root)
        self.dest.mkdir()
        with self.assertRaisesRegex(ValueError, 'no overwrite'):
            P.restore(self.dest, self.root)

    def test_corrupt_part_fails_before_restore_creation(self):
        m = P.package(self.root)
        path = self.root / m['chunks'][0]['path']
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        with self.assertRaisesRegex(ValueError, 'part identity'):
            P.restore(self.dest, self.root)
        self.assertFalse(self.dest.exists())

    def test_hash_and_chunk_mutations_refused(self):
        original = P.package(self.root)
        for field, value in [('offset', False), ('offset', 17), ('bytes', P.LIMIT + 1), ('path', '../outside')]:
            m = copy.deepcopy(original)
            m['chunks'][0][field] = value
            self.save(m)
            with self.assertRaises(ValueError):
                P.restore(self.dest, self.root)
        m = copy.deepcopy(original)
        m['archive']['sha256'] = '0' * 64
        self.save(m)
        with self.assertRaisesRegex(ValueError, 'archive identity'):
            P.restore(self.dest, self.root)

    def test_manifest_strict_types_duplicate_and_nonfinite(self):
        original = P.package(self.root)
        for field, value in [('chunk_count', True), ('encoder_calls', False), ('decoder_calls', 0.0), ('unexpected', True)]:
            m = copy.deepcopy(original)
            m[field] = value
            self.save(m)
            with self.assertRaises(ValueError):
                P.restore(self.dest, self.root)
        for prefix in ('"encoder_calls":0,', '"unsupported":NaN,'):
            (self.root / P.ART / 'manifest.json').write_bytes(b'{' + prefix.encode() + P.canonical(original)[1:])
            with self.assertRaises(ValueError):
                P.restore(self.dest, self.root)

    def test_source_and_part_symlinks_refused(self):
        source = self.root / P.MILESTONE / 'synthetic.bin'
        other = self.root / 'outside.bin'
        other.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'symlink|outside root'):
            P.package(self.root)
        source.unlink()
        source.write_bytes(other.read_bytes())
        m = P.package(self.root)
        part = self.root / m['chunks'][0]['path']
        other.write_bytes(part.read_bytes())
        part.unlink()
        part.symlink_to(other)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            P.restore(self.dest, self.root)

    def test_binding_bytes_must_equal_audited_snapshot(self):
        (self.root / P.BINDING / 'records.jsonl.gz').write_bytes(b'changed binding')
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            P.package(self.root)
        self.assertFalse((self.root / P.ART).exists())

    def test_binding_omitted_or_extra_member_refused(self):
        path = self.root / P.BINDING / 'records.jsonl.gz'
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'binding exact inventory'):
            P.package(self.root)
        path.write_bytes(original)
        (self.root / P.BINDING / 'unexpected.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'binding exact inventory'):
            P.package(self.root)

    def test_extra_part_and_helper_drift_refused(self):
        P.package(self.root)
        path = self.root / P.ART / 'parts' / 'extra.bin'
        path.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            P.restore(self.dest, self.root)
        path.unlink()
        (self.root / P.HELPER).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'helper identity'):
            P.restore(self.dest, self.root)


if __name__ == '__main__':
    unittest.main()
