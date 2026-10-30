"""Independent adversarial fixtures; all subprocesses are tiny Python stubs."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = Path(os.environ['RECOVERY83_REVIEW_HELPER'])


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


a = module('independent_audit', ROOT / 'scripts/audit_baseline_single_record_recovery_v1.py')
h = module('reviewed_helper', HELPER_PATH)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class IndependentRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'resource'
        self.prior = Path(self.tmp.name) / 'prior'
        self.plan = Path(self.tmp.name) / 'plan'
        self.shard = self.root / 'runs' / h.ID / 'order083'
        self.row = {'record_id': 'independent-row83', 'source_group_id': 'independent-group83',
                    'split': 'development', 'topic': 'literature', 'genre': 'encyclopedic',
                    'target_words': 500, 'prompt': 'Only independently fixed source facts.'}
        self.meta = {k: self.row[k] for k in self.row if k != 'prompt'}
        self.meta['prompt_sha256'] = hashlib.sha256(self.row['prompt'].encode()).hexdigest()
        self.ident = a.canonical(['local', a.MODEL, self.row['record_id']])
        self.prompt = self.plan / 'prompts/order083.jsonl'
        write(self.prompt, self.row)
        self.job = {'source_order': 83, 'record_id': self.row['record_id'], 'identity': self.ident,
                    'prompt_path': 'prompts/order083.jsonl', 'prompt_file_sha256': h.sha(self.prompt)}
        self.protocol = self.plan / 'protocol.json'
        write(self.protocol, {'jobs': [self.job]})
        self.original_run = {'schema': 'synsteg-budget-generation-run-v1', 'provider': 'local',
            'model': a.MODEL, 'revision': a.REVISION, 'dtype': 'float32',
            'settings': {'seed': 20261024, 'max_output_tokens': 1024, 'temperature': 0.8,
                         'top_k': 50, 'top_p': 0.95, 'cpu_threads': 6},
            'pricing_sha256': None, 'prompts_sha256': 'original100',
            'selected_records': [dict(self.meta, record_id='prior' + str(i)) for i in range(100)],
            'selected_records_count': 100, 'runner_sha256': 'r' * 64, 'ledger_code_sha256': 'l' * 64,
            'transport_code_sha256': None}
        self.original_run['selected_records'][82] = self.meta
        self.runtime = {'synthetic_runtime': 'metadata only'}
        write(self.prior / a.RUN / 'run.json', self.original_run)
        write(self.prior / a.RUN / 'runtime.json', self.runtime)
        write(self.prior / a.RUN / 'local_ledger/ledger.json', {'attempts': {'old82': {'status': 'reserved_outcome_unknown'}}})
        run = copy.deepcopy(self.original_run)
        run.update(prompts_sha256=h.sha(self.prompt), selected_records=[self.meta], selected_records_count=1)
        self.fingerprint = a.canonical({'record': self.meta, 'provider': 'local', 'model': a.MODEL,
            'revision': a.REVISION, 'dtype': 'float32', 'settings': run['settings'], 'pricing_sha256': None,
            'prompts_sha256': h.sha(self.prompt), 'runtime_sha256': a.canonical(self.runtime),
            'implementation': {k: run[k] for k in ('runner_sha256', 'ledger_code_sha256', 'transport_code_sha256')}})
        self.result = {'cover_text': 'Synthetic retained text.', 'usage': {'input_tokens': 30, 'output_tokens': 20},
            'returned_model': a.MODEL, 'revision': a.REVISION, 'dtype': 'float32',
            'seed': int(a.canonical([20261024, a.MODEL, self.row['record_id']])[:8], 16),
            'stop_reason': 'end_turn', 'hit_output_limit': False}
        self.entry = {'identity': self.ident, 'status': 'completed', 'request_fingerprint': self.fingerprint,
            'automatic_retries': 0, 'reserved_nanodollars': 0, 'accounted_nanodollars': 0,
            'metadata': {'record': self.meta, 'provider': 'local', 'model': a.MODEL}, 'result': self.result}
        self.attempt = {'schema': 'synsteg-budget-generation-attempt-v1', 'identity': self.ident,
            'provider': 'local', 'model': a.MODEL, 'record': self.meta, 'status': 'completed',
            'request_fingerprint': self.fingerprint, 'automatic_retries': 0, 'result': self.result}
        self.summary = {'status': 'selected_records_completed', 'exit_code': 0, 'selected_records_count': 1,
            'completed': 1, 'resumed_completed': 0, 'requests_attempted_this_invocation': 1,
            'max_new_records_this_invocation': 1, 'automatic_retries': 0,
            'attempts': [{'identity': self.ident, 'record_id': self.row['record_id'], 'status': 'completed'}],
            'ledger': {'attempt_count': 1, 'unknown_or_failed_attempts': 0}}
        write(self.shard / 'run/run.json', run)
        write(self.shard / 'run/runtime.json', self.runtime)
        self.reset_outcome()

    def reset_outcome(self):
        write(self.shard / 'run/local_ledger/ledger.json', {'schema': 'synsteg-budget-ledger-recovery-v1',
              'budget_nanodollars': 20_000_000_000, 'attempts': {self.ident: self.entry}})
        write(self.shard / 'run/attempts' / (self.ident + '.json'), self.attempt)
        write(self.shard / 'run/summary.json', self.summary)

    def test_matching_shard_and_seed(self):
        result = a.audit(self.prior, self.protocol, self.shard, 83)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(h.shard_status(self.shard, self.job, self.original_run, self.runtime),
                         {'status': 'completed', 'details': []})

    def test_result_mutations_rejected_by_both_implementations(self):
        originals = copy.deepcopy(self.result)
        for field, bad in [('returned_model', 'wrong'), ('revision', 'wrong'), ('dtype', 'bfloat16'),
                           ('seed', 0), ('cover_text', ''), ('stop_reason', 'max_tokens'), ('hit_output_limit', True)]:
            with self.subTest(field=field):
                self.result.clear(); self.result.update(originals); self.result[field] = bad
                self.reset_outcome()
                self.assertEqual(a.audit(self.prior, self.protocol, self.shard, 83)['status'], 'failed')
                with self.assertRaises(ValueError):
                    h.shard_status(self.shard, self.job, self.original_run, self.runtime)

    def test_metadata_mutations_and_token_boundary(self):
        for tokens in (0, -1, 1025, True, 20.0):
            with self.subTest(tokens=tokens):
                self.result['usage']['output_tokens'] = tokens; self.reset_outcome()
                self.assertEqual(a.audit(self.prior, self.protocol, self.shard, 83)['status'], 'failed')
        self.result['usage']['output_tokens'] = 1024
        self.result.update(stop_reason='max_tokens', hit_output_limit=True); self.reset_outcome()
        self.assertEqual(a.audit(self.prior, self.protocol, self.shard, 83)['status'], 'passed')
        self.entry['request_fingerprint'] = 'wrong'; self.reset_outcome()
        self.assertIn('both request fingerprints', a.audit(self.prior, self.protocol, self.shard, 83)['errors'])

    def test_summary_and_duplicate_ledger_rejected(self):
        for key, bad in [('exit_code', 3), ('resumed_completed', 1), ('automatic_retries', 1),
                         ('selected_records_count', 100), ('max_new_records_this_invocation', 2)]:
            with self.subTest(key=key):
                original = self.summary[key]; self.summary[key] = bad; self.reset_outcome()
                self.assertEqual(a.audit(self.prior, self.protocol, self.shard, 83)['status'], 'failed')
                self.summary[key] = original
        self.reset_outcome()
        p = self.shard / 'run/local_ledger/ledger.json'
        ledger = a.load(p); ledger['attempts']['duplicate'] = self.entry; write(p, ledger)
        self.assertIn('one ledger identity', a.audit(self.prior, self.protocol, self.shard, 83)['errors'])

    def test_namespace_and_lock_substitution_refused_before_launch(self):
        with patch.object(h.subprocess, 'Popen', side_effect=AssertionError('must never launch')):
            for output, lock in [(self.root / 'alternate', self.root / 'results/project-nn.lock'),
                                 (self.root / 'runs' / h.ID, self.root / 'alternate.lock')]:
                with self.subTest(output=str(output), lock=str(lock)), self.assertRaisesRegex(ValueError, 'frozen output namespace'):
                    h.execute(self.root, self.prior, self.protocol, Path('absent'), output, 83, lock)

    def test_whole_preservation_inventory_blocks_raw_mutation(self):
        protocol = {'execution_enabled': True, 'jobs': [self.job, dict(self.job, source_order=84)], 'prior_run_files_sha256': {}}
        receipt = self.plan / 'receipt84.json'
        write(receipt, {'readback_verified': True, 'source_order': 84, 'protocol_sha256': h.sha(self.protocol),
                        'helper_sha256': h.sha(HELPER_PATH), 'commit': 'f' * 40})
        write(self.shard / 'process.json', {'status': 'completed_pending_remote_preservation'})
        write(self.shard / 'git-preservation-receipt.json', {'readback_verified': True, 'commit': 'e' * 40,
            'process_sha256': h.sha(self.shard / 'process.json'), 'files_sha256': h.file_inventory(self.shard)})
        with (self.shard / 'stdout.log').open('w') as f: f.write('post-save added bytes')
        with patch.object(h, 'verify_protocol', return_value=protocol), patch.object(h, 'shard_status', return_value={'status':'completed','details':[]}), patch.object(h.subprocess, 'Popen', side_effect=AssertionError('must never launch')):
            with self.assertRaisesRegex(ValueError, 'Previous shard not cleanly completed and Git preserved'):
                h.execute(self.root, self.prior, self.protocol, receipt, self.root / 'runs' / h.ID, 84, self.root / 'results/project-nn.lock')

    def test_exited_parent_descendant_killed_before_reap(self):
        marker = Path(self.tmp.name) / 'descendant.pid'
        child_code = 'import os,signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(' + repr(str(marker)) + ').write_text(str(os.getpid()));time.sleep(20)'
        parent_code = 'import subprocess,sys;subprocess.Popen([sys.executable,"-c",' + repr(child_code) + '])'
        proc = subprocess.Popen([sys.executable, '-c', parent_code], start_new_session=True, stdout=subprocess.PIPE)
        try:
            limit = time.monotonic() + 3
            while (not marker.exists() or not h.terminal_observed(proc)) and time.monotonic() < limit:
                time.sleep(0.02)
            self.assertTrue(marker.exists()); self.assertTrue(h.terminal_observed(proc))
            self.assertIsNone(proc.returncode)
            self.assertEqual(h.stop_and_reap(proc), 0)
            # /proc may expose another PID namespace. The descendant inherits
            # this pipe; EOF proves no live process retains that stdout FD.
            ready, _, _ = select.select([proc.stdout], [], [], 2)
            self.assertEqual(ready, [proc.stdout])
            self.assertEqual(os.read(proc.stdout.fileno(), 1), b'')
        finally:
            if proc.returncode is None:
                h.stop_and_reap(proc)
            proc.stdout.close()


if __name__ == '__main__':
    unittest.main()
