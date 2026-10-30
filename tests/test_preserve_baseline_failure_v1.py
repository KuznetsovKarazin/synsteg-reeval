#!/usr/bin/env python3
"""Synthetic storage/availability checks only; no active experiment input."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('failure_preservation', SOURCE / 'scripts/preserve_baseline_failure_v1.py')
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
RUN = helper.RUN
BATCH = helper.BATCHES / 'batch41'
OUT = Path('results/baseline-failure-preservation-v1/batch41/artifacts')
PROOF = OUT.parent / 'quiescence.json'


def dump(root, relative, value):
    path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def hashes(root):
    return {str(p.relative_to(root)): helper.sha(p.read_bytes())
            for p in root.rglob('*') if p.is_file() and not p.is_symlink()}


def refresh_proof(root):
    obj = {'schema': helper.PROOF_SCHEMA, 'batch_id': 'batch41',
           'root_observed_supervisor_terminated': True,
           'root_observed_child_group_terminated': True,
           'exclusive_project_nn_lease_held_by_root': True,
           'observation_basis': 'Synthetic fixture only; no live process was started'}
    for name, field in [('process.json', 'process_json_sha256'),
                        ('artifact-manifest.json', 'artifact_manifest_sha256')]:
        path = root / BATCH / name
        obj[field] = helper.sha(path.read_bytes()) if path.exists() else None
    dump(root, PROOF, obj)


def fixture(root, completed=80, unknown=True, random_partial=False):
    for relative in [*helper.SOURCE_FILES, helper.MANIFEST, helper.PACKAGER,
                     Path('scripts/reassemble_baseline_primary_chunks_v1.py'),
                     Path('scripts/restore_native_continuation_artifact_v1.py')]:
        target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / relative, target)
    for relative in [helper.BATCHES / 'resume-supervisor.lock',
                     RUN / 'local_ledger/campaign.lock', RUN / 'run_guard/campaign.lock']:
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'0')
    selected = [{'record_id': f'synthetic-{i:03d}', 'split': 'development'} for i in range(1, 101)]
    run = {'provider': 'local', 'model': 'synthetic-model-only',
           'selected_records_count': 100, 'selected_records': selected}
    dump(root, RUN / 'run.json', run)
    entries = {}
    for i, meta in enumerate(selected[:completed + int(unknown)], 1):
        ident = helper.canonical(['local', run['model'], meta['record_id']])
        result = {'cover_text': f'Synthetic retained text {i}.'} if i <= completed else None
        status = 'completed' if i <= completed else 'reserved_outcome_unknown'
        entry = {'identity': ident, 'metadata': {'record': meta, 'provider': 'local', 'model': run['model']},
                 'automatic_retries': 0, 'request_fingerprint': 'f' * 64, 'status': status, 'result': result}
        entries[ident] = entry
        dump(root, RUN / 'attempts' / (ident + '.json'),
             {'identity': ident, 'record': meta, 'provider': 'local', 'model': run['model'],
              'automatic_retries': 0, 'request_fingerprint': 'f' * 64, 'status': status, 'result': result})
    dump(root, RUN / 'local_ledger/ledger.json',
         {'schema': 'synsteg-budget-ledger-recovery-v1', 'attempts': entries})
    dump(root, RUN / 'summary.json', {'status': 'running', 'completed': completed})
    dump(root, BATCH / 'process.json',
         {'status': 'stopped_requires_review_no_retry', 'exit_code': -15, 'timed_out': True})
    (root / BATCH / 'stdout.log').write_bytes(b'Synthetic standard output\n')
    (root / BATCH / 'stderr.log').write_bytes(b'Synthetic standard error\n')
    if random_partial:
        (root / RUN / 'partial.bin').write_bytes(random.Random(20261026).randbytes(1200000))
    members = {str(p.relative_to(root)): helper.sha(p.read_bytes())
               for p in (root / BATCH).rglob('*') if p.is_file()}
    dump(root, BATCH / 'artifact-manifest.json', {'files_sha256': members})
    refresh_proof(root)


class FailurePreservationTests(unittest.TestCase):
    def test_full_binary_partial_roundtrip_and_unknown_stays_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root, random_partial=True); before = hashes(root)
            status = helper.preserve(root, 41, PROOF)
            self.assertEqual(status['availability_counts'], {'available': 80, 'not_attempted': 19, 'unknown': 1})
            self.assertFalse(status['scientific_completed_count_asserted'])
            self.assertFalse(status['next_batch_allowed'])
            self.assertFalse(status['successful_closure_allowed'])
            self.assertEqual(status['model_calls'], 0)
            self.assertGreater(status['archive']['bytes'], 1024 * 1024)
            self.assertTrue(status['fresh_restore']['reassembled_from_parts'])
            self.assertEqual({name: helper.sha((root/name).read_bytes()) for name in before}, before)
            rows = json.loads((root / OUT / 'availability.json').read_text())['records']
            self.assertEqual(rows[80]['availability'], 'unknown')
            self.assertEqual(rows[80]['ledger_status'], 'reserved_outcome_unknown')
            with self.assertRaisesRegex(ValueError, 'output exists'):
                helper.preserve(root, 41, PROOF)

    def test_one_new_available_then_unknown_is_not_successful_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root, completed=81)
            files, _ = helper.snapshot(root, 41, PROOF)
            inv = helper.availability(files, 41)
            self.assertEqual(inv['counts'], {'available': 81, 'not_attempted': 18, 'unknown': 1})
            self.assertEqual([r['availability'] for r in inv['records'][80:82]], ['available', 'unknown'])

    def test_malformed_or_orphan_evidence_never_proves_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            files, _ = helper.snapshot(root, 41, PROOF)
            broken = dict(files); broken[str(RUN / 'local_ledger/ledger.json')] = b'{broken'
            inv = helper.availability(broken, 41)
            self.assertEqual(inv['counts'], {'unknown': 100})
            orphan = dict(files); orphan[str(RUN / 'attempts/orphan.tmp')] = b'partial'
            inv = helper.availability(orphan, 41)
            self.assertEqual(inv['counts'], {'available': 80, 'unknown': 20})
            bad_plan = dict(files); bad_plan[str(RUN / 'run.json')] = b'null'
            inv = helper.availability(bad_plan, 41)
            self.assertEqual(inv['unclassified_unknown_records'], 100)
            self.assertEqual(inv['records'], [])

    def test_missing_supervisor_files_preserved_only_with_external_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            (root / BATCH / 'process.json').unlink()
            (root / BATCH / 'artifact-manifest.json').unlink(); refresh_proof(root)
            status = helper.preserve(root, 41, PROOF)
            self.assertIsNone(status['supervisor_status_observed'])
            self.assertFalse(status['external_termination_independently_verified'])
            self.assertEqual(status['availability_counts']['unknown'], 1)

    def test_success_unattested_or_changed_receipt_rejected_before_output(self):
        for mode in ['success', 'no_termination', 'wrong_digest']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); fixture(root)
                if mode == 'success':
                    dump(root, BATCH / 'process.json', {'status': helper.SUCCESS}); refresh_proof(root)
                else:
                    proof = json.loads((root / PROOF).read_text())
                    if mode == 'no_termination': proof['root_observed_child_group_terminated'] = False
                    else: proof['process_json_sha256'] = '0' * 64
                    dump(root, PROOF, proof)
                with self.assertRaises(ValueError): helper.preserve(root, 41, PROOF)
                self.assertFalse((root / OUT).exists())

    def test_live_lock_and_symlink_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            with (root / helper.BATCHES / 'resume-supervisor.lock').open('rb') as held:
                fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, 'writer may be active'):
                    helper.preserve(root, 41, PROOF)
            self.assertFalse((root / OUT).exists())
            (root / RUN / 'unexpected-link').symlink_to(root / RUN / 'run.json')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                helper.preserve(root, 41, PROOF)
            self.assertFalse((root / OUT).exists())


if __name__ == '__main__':
    unittest.main()
