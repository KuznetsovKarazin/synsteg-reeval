"""Offline v2 builder tests; all future Qwen outcomes are synthetic fixtures."""
import importlib.util
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
V1 = Path(os.environ.get('MIXED_BUILDER_V1_CHECKOUT', str(HERE)))
def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
b = module(HERE/'scripts/build_mixed_development_snapshot_v2.py', 'builder_v2')
legacy = module(V1/'tests/test_mixed_development_snapshot_builder_v1.py', 'legacy_builder_regressions')
m = legacy.m
legacy.b = b


class LegacyRegressionTests(legacy.BuilderCases):
    """Retain all20 v1 behavioral tests against the additive implementation."""


class SuccessorBuilderTests(unittest.TestCase):
    setUp = legacy.BuilderCases.setUp
    qwen = legacy.BuilderCases.qwen
    reject = legacy.BuilderCases.reject

    def add_successor(self, order, state='completed'):
        old, identity = self.f.shard('qwen05', order, state)
        new = b.QWEN_SEQUENTIAL_ROOT+f'/order{order:03d}'
        (self.f.root/new).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(self.f.root/old, self.f.root/new)
        return new, identity

    def request(self):
        value = b.draft_request(m, self.f.root, 'c'*40)
        value['remote_census'].update(readback_verified=True, branches_and_prs_checked=True)
        value['qwen_preflight']['commit'] = 'd'*40
        return value

    def build(self, value):
        raw = b.encoded(value); output = Path(self.tmp.name)/'v2-bundle'
        b.build(m, self.f.root, raw, m.sha(raw), output)
        return m.parse((output/'full300.json').read_bytes())

    def test_first_plus_source2_with_shared_full100_contract(self):
        self.qwen(); self.add_successor(2)
        result = self.build(self.request())
        self.assertEqual(result['model_counts']['qwen05'], {'completed': 2, 'not_attempted': 98})
        self.assertEqual(result['full_denominator'], 300)
        self.assertEqual(result['new_shard_evidence'][0]['runtime_sha256'], result['new_shard_evidence'][1]['runtime_sha256'])

    def test_sparse_evidence_keeps_every_unattempted_identity(self):
        self.qwen(); self.add_successor(2); self.add_successor(100)
        result = self.build(self.request())
        self.assertEqual(result['model_counts']['qwen05'], {'completed': 3, 'not_attempted': 97})
        rows = [r for r in result['records'] if r['model_alias'] == 'qwen05']
        self.assertEqual({r['source_order'] for r in rows if r['status'] == 'not_attempted'}, set(range(3, 100)))

    def test_all100_synthetic_qwen_preserves_full300(self):
        self.qwen()
        for order in range(2, 101):
            self.add_successor(order)
        result = self.build(self.request())
        self.assertEqual(result['model_counts']['qwen05'], {'completed': 100})
        self.assertEqual(result['counts'], {'completed': 189, 'reserved_outcome_unknown': 1, 'not_attempted': 110})
        self.assertEqual(len(result['records']), 300)

    def test_successor_requires_preserved_first_in_same_census(self):
        _, first, _ = self.qwen(); self.add_successor(2)
        shutil.rmtree(self.f.root/first)
        self.reject(self.request())

    def test_successor_without_receipt_refused(self):
        self.qwen(); prefix, _ = self.add_successor(2); value = self.request()
        (self.f.root/prefix/'git-preservation-receipt.json').unlink()
        self.reject(value)

    def test_new_root_is_in_exact_remote_census(self):
        self.qwen(); self.add_successor(2); value = self.request()
        value['remote_census']['files_sha256'] = {k: v for k, v in value['remote_census']['files_sha256'].items()
                                                 if not k.startswith(b.QWEN_SEQUENTIAL_ROOT+'/')}
        self.reject(value)

    def test_orphan_file_in_new_root_refused(self):
        self.qwen(); self.add_successor(2)
        self.f.write(b.QWEN_SEQUENTIAL_ROOT+'/orphan.json', {'fixture': True})
        self.reject(self.request())

    def test_new_root_boundaries_no_duplicate_first_or101(self):
        for order in [1, 101, 0]:
            with self.assertRaises(ValueError): b.shard_identity(m, b.QWEN_SEQUENTIAL_ROOT+f'/order{order:03d}')
        self.assertEqual(b.shard_identity(m,b.QWEN_SEQUENTIAL_ROOT+'/order100'), ('qwen05',100))

    def test_new_root_dangling_symlink_refused(self):
        self.qwen(); link = self.f.root/b.QWEN_SEQUENTIAL_ROOT
        link.symlink_to(self.f.root/'absent', target_is_directory=True)
        with self.assertRaises(ValueError): b.census(m, self.f.root)

    def test_new_root_supervisor_lock_excluded_explicitly(self):
        self.qwen(); self.add_successor(2); value = self.request()
        self.f.write(b.QWEN_SEQUENTIAL_ROOT+'/supervisor.lock', {'fixture': True})
        payloads, manifest, provenance = b.stage(m, self.f.root, value)
        self.assertNotIn(b.QWEN_SEQUENTIAL_ROOT+'/supervisor.lock', payloads)
        self.assertIn(b.QWEN_SEQUENTIAL_ROOT+'/supervisor.lock', provenance['local_operational_files_excluded'])
        self.assertEqual(manifest['reservation_scope']['roots'], [b.TINY_ROOT,b.QWEN_ROOT,b.QWEN_SEQUENTIAL_ROOT])

    def test_preserved_unknown_successor_is_retained(self):
        self.qwen(); prefix, identity = self.add_successor(2, 'reservation_only')
        self.f.write(prefix+'/process.json', {'identity': identity, 'source_order': 2, 'timeout_seconds': 1200,
            'automatic_retries': 0, 'status': 'stopped_requires_review_no_retry', 'timed_out': False})
        self.f.refresh(prefix)
        result = self.build(self.request())
        self.assertEqual(result['model_counts']['qwen05'], {'completed': 1, 'reserved_outcome_unknown': 1, 'not_attempted': 98})

    def test_preserved_failed_successor_is_retained(self):
        self.qwen(); self.add_successor(2, 'failed')
        result = self.build(self.request())
        self.assertEqual(result['model_counts']['qwen05'], {'completed': 1, 'failed': 1, 'not_attempted': 98})

    def test_late_new_successor_reservation_during_stage_refused(self):
        self.qwen(); self.add_successor(2); value = self.request(); original = b.bound_read
        def altered(merger, root, path, expected=None):
            result = original(merger, root, path, expected)
            if path == b.QWEN_FILES[-1]:
                self.f.write(b.QWEN_SEQUENTIAL_ROOT+'/order003/reservation.json', {'late': True})
            return result
        with patch.object(b, 'bound_read', side_effect=altered):
            self.reject(value)


if __name__ == '__main__':
    unittest.main()
