"""Synthetic operations only: no torch, tokenizer, model or paid API."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/baseline_single_record_recovery_v1.py'
spec = importlib.util.spec_from_file_location('recovery', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'root'
        self.prior = self.base / 'prior'
        self.out = self.root / 'runs' / m.ID
        self.plan = self.base / 'plan'
        self.rows = [{'record_id': f'row{n:03d}', 'source_group_id': f'group{n:03d}',
                      'split': 'development', 'topic': 'history', 'genre': 'encyclopedic',
                      'target_words': 500, 'prompt': f'Exact prompt {n}'} for n in range(1, 101)]
        p = self.root / m.PROMPTS
        p.parent.mkdir(parents=True)
        p.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))
        self.first = {'argv': ['scripts/run_budget_generation.py', '--provider', 'local',
                              '--model', m.MODEL, '--revision', m.REVISION, '--prompts', m.PROMPTS,
                              '--output', m.RUN, '--max-records', '100', '--max-new-records', '2',
                              '--max-output-tokens', '1024', '--dtype', 'float32', '--cpu-threads', '6',
                              '--seed', '20261024', '--split', 'development', '--execute'],
                      'python_relative': 'venv/bin/python', 'environment': {},
                      'frozen_files_sha256': {'scripts/run_budget_generation.py': 'a' * 64}}
        m.write_new(self.root / m.FIRST, self.first)
        self.patches = [patch.object(m, 'PROMPTS_SHA', m.sha(p)),
                        patch.object(m, 'FIRST_SHA', m.sha(self.root / m.FIRST))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        entries = {}
        for order, row in enumerate(self.rows[:82], 1):
            ident = m.canonical(['local', m.MODEL, row['record_id']])
            state = 'completed' if order <= 81 else 'reserved_outcome_unknown'
            result = {'cover_text': f'cover{order}'} if order <= 81 else None
            entries[ident] = {'status': state, 'metadata': {'record': row}, 'automatic_retries': 0, 'result': result}
            m.write_new(self.prior / m.RUN / 'attempts' / (ident + '.json'),
                        {'status': state, 'record': row, 'automatic_retries': 0, 'result': result})
        m.write_new(self.prior / m.RUN / 'local_ledger/ledger.json', {'attempts': entries})
        m.write_new(self.prior / m.RUN / 'run.json', {'model': m.MODEL, 'revision': m.REVISION, 'selected_records_count': 100})
        m.write_new(self.prior / m.RUN / 'runtime.json', {'synthetic': True})
        self.protocol = m.prepare(self.root, self.prior, self.plan)
        self.protocol_path = self.plan / 'protocol.proposed.json'

    def test_fixed18_and_unknown_full300(self):
        self.assertEqual([j['source_order'] for j in self.protocol['jobs']], list(range(83, 101)))
        projection = m.inspect(self.root, self.prior, self.protocol_path, self.out)
        self.assertEqual(projection['counts'], {'completed': 81, 'not_attempted': 218, 'reserved_outcome_unknown': 1})
        self.assertEqual(len(projection['records']), 300)
        self.assertEqual([r for r in projection['records'] if r['status'] == 'reserved_outcome_unknown'][0]['source_order'], 82)

    def test_exact_row_and_cli_parity(self):
        for j in self.protocol['jobs']:
            self.assertEqual(json.loads((self.plan / j['prompt_path']).read_text()), self.rows[j['source_order'] - 1])
        command = m.command_for(self.root, self.first, self.plan / 'prompts/order083.jsonl', self.out)
        args = command[1:]
        for flag in ['--model', '--revision', '--seed', '--dtype', '--cpu-threads', '--max-output-tokens', '--split']:
            self.assertEqual(args[args.index(flag) + 1], self.first['argv'][self.first['argv'].index(flag) + 1])
        self.assertEqual(args[args.index('--max-records') + 1], '1')
        self.assertEqual(args[args.index('--max-new-records') + 1], '1')

    def test_unknown82_cannot_be_replaced(self):
        ident = m.canonical(['local', m.MODEL, self.rows[81]['record_id']])
        path = self.prior / m.RUN / 'attempts' / (ident + '.json')
        value = m.load(path)
        value['status'] = 'completed'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Historical evidence changed'):
            m.verify_protocol(self.root, self.prior, self.protocol_path)

    def test_no82_or_order_skip(self):
        with self.assertRaisesRegex(ValueError, 'Only never-attempted'):
            m.execute(self.root, self.prior, self.protocol_path, self.base / 'none', self.out, 82, self.root / 'results/project-nn.lock')
        self.activate(84)
        with self.assertRaises(FileNotFoundError):
            m.execute(self.root, self.prior, self.protocol_path, self.receipt, self.out, 84, self.root / 'results/project-nn.lock')

    def test_protocol_disabled_and_prompt_mutation(self):
        self.write_receipt(83)
        with self.assertRaisesRegex(ValueError, 'Source-only'):
            m.execute(self.root, self.prior, self.protocol_path, self.receipt, self.out, 83, self.root / 'results/project-nn.lock')
        (self.plan / 'prompts/order083.jsonl').write_text(json.dumps(self.rows[83]) + '\n')
        with self.assertRaisesRegex(ValueError, 'Shard differs'):
            m.verify_protocol(self.root, self.prior, self.protocol_path)

    def write_receipt(self, order):
        self.receipt = self.base / f'receipt{order}.json'
        value = {'readback_verified': True, 'source_order': order, 'protocol_sha256': m.sha(self.protocol_path),
                 'helper_sha256': m.sha(SCRIPT), 'commit': 'b' * 40}
        self.receipt.write_text(json.dumps(value))

    def activate(self, order):
        p = m.load(self.protocol_path)
        p['execution_enabled'] = True
        self.protocol_path = self.plan / 'protocol.enabled.json'
        self.protocol_path.write_text(json.dumps(p))
        self.write_receipt(order)

    def test_reservation_spawn_failure_consumed_and_retained(self):
        self.activate(83)
        before = m.file_inventory(self.prior)
        with patch.object(m, 'verify_environment', return_value=self.first), patch.object(m.subprocess, 'Popen', side_effect=OSError('synthetic')):
            result = m.execute(self.root, self.prior, self.protocol_path, self.receipt, self.out, 83, self.root / 'results/project-nn.lock')
        self.assertEqual(result['status'], 'stopped_requires_review_no_retry')
        self.assertEqual(result['observed_outcome']['status'], 'reserved_outcome_unknown')
        self.assertTrue((self.out / 'order083/artifact-manifest.json').exists())
        self.assertEqual(before, m.file_inventory(self.prior))
        with self.assertRaisesRegex(ValueError, 'already consumed'):
            m.execute(self.root, self.prior, self.protocol_path, self.receipt, self.out, 83, self.root / 'results/project-nn.lock')
        projection = m.inspect(self.root, self.prior, self.protocol_path, self.out)
        self.assertEqual(projection['counts'], {'completed': 81, 'not_attempted': 217, 'reserved_outcome_unknown': 2})

    def test_historical_output_and_nn_collision_refused(self):
        with self.assertRaisesRegex(ValueError, 'historical evidence'):
            m.execute(self.root, self.prior, self.protocol_path, self.base / 'none', self.prior / 'new', 83, self.root / 'results/project-nn.lock')
        self.activate(83)
        with m.exclusive(self.root / 'results/project-nn.lock'):
            with self.assertRaises(BlockingIOError):
                m.execute(self.root, self.prior, self.protocol_path, self.receipt, self.out, 83, self.root / 'results/project-nn.lock')
        self.assertFalse((self.out / 'order083').exists())

    def test_status_detects_conflicting_attempt(self):
        job = self.protocol['jobs'][0]
        directory = self.out / 'order083'
        m.write_new(directory / 'run/local_ledger/ledger.json', {'attempts': {'wrong': {}}})
        m.write_new(directory / 'run/attempts' / (job['identity'] + '.json'), {})
        with self.assertRaisesRegex(ValueError, 'Unexpected shard'):
            m.shard_status(directory, job)

    def test_process_group_signal_before_reap(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        self.assertFalse(m.terminal_observed(child))
        code = m.stop_and_reap(child)
        self.assertLess(code, 0)
        with self.assertRaises(ProcessLookupError):
            os.killpg(child.pid, 0)

    def test_no_overwrite_preparation_and_write(self):
        with self.assertRaisesRegex(ValueError, 'destination already exists'):
            m.prepare(self.root, self.prior, self.plan)
        with self.assertRaises(FileExistsError):
            m.write_new(self.protocol_path, {})


if __name__ == '__main__':
    unittest.main()
