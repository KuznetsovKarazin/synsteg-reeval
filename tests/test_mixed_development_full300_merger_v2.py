"""Offline contract tests. All NEW model outcomes here are synthetic fixtures.

Only the unchanged base300 compatibility test uses actual historical outcomes.
Set MERGER_TEST_INPUTS to a checkout containing the five pinned historical files.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/merge_large_development_full300_v2.py'
spec = importlib.util.spec_from_file_location('mixed_merger_v2', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
SOURCE = Path(os.environ.get('MERGER_TEST_INPUTS', str(SCRIPT.parents[1])))


class Fixture:
    def __init__(self, parent):
        self.root = Path(parent) / 'evidence'
        for path in m.BASE_FILES.values():
            dst = self.root / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / path, dst)
        self.sources = [json.loads(s) for s in (self.root / m.BASE_FILES['source']).read_text().splitlines()]
        self.contracts = dict.fromkeys(m.MODELS)
        self.shards = []

    def write(self, path, value):
        dst = self.root / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')

    def read(self, path):
        return json.loads((self.root / path).read_text())

    def contract(self, alias):
        base = 'fixtures/contracts/' + alias
        model, revision = m.MODELS[alias]
        runtime = base + '/runtime.json'
        config = base + '/config.json'
        context = base + '/context.json'
        contract = base + '/contract.json'
        self.write(runtime, {'python': 'synthetic-fixture-only', 'packages': {'fixture': '0'}})
        self.write(config, {'max_position_embeddings': 2048})
        self.write(context, {
            'schema': m.ID + '-context-report', 'model_id': model, 'revision': revision,
            'source_sha256': m.BASE_PINS['source'],
            'model_config_sha256': m.sha((self.root / config).read_bytes()),
            'model_context_limit': 2048,
            'records': [dict(source_order=n, record_id=r['record_id'], prompt_sha256=r['prompt_sha256'], input_tokens=200)
                        for n, r in enumerate(self.sources, 1)]})
        self.write(contract, {
            'schema': m.ID + '-model-contract', 'model_alias': alias, 'model_id': model,
            'revision': revision, 'dtype': 'float32', 'settings': m.SETTINGS,
            'runtime_path': runtime, 'runtime_sha256': m.sha((self.root / runtime).read_bytes()),
            'context_report_path': context, 'model_config_path': config,
            'execution_stratum': 'synthetic-fixture-only-not-an-experiment', 'timeout_seconds_including_load': 1200})
        self.contracts[alias] = contract
        return self.read(contract)

    def shard(self, alias='qwen05', order=1, state='completed'):
        contract = self.read(self.contracts[alias]) if self.contracts[alias] else self.contract(alias)
        model, revision = m.MODELS[alias]
        src = self.sources[order-1]
        identity = m.canonical_sha(['local', model, src['record_id']])
        prefix = f'runs/synthetic-fixture-only/{alias}/order{order:03d}'
        self.shards.append(dict(model_alias=alias, source_order=order, path=prefix))
        self.write(prefix + '/reservation.json', {
            'identity': identity, 'source_order': order, 'record_id': src['record_id'],
            'automatic_retries': 0, 'status': 'reserved_outcome_unknown'})
        if state != 'reservation_only':
            record = {key: src[key] for key in ['genre', 'prompt_sha256', 'record_id', 'source_group_id', 'split', 'target_words', 'topic']}
            result = {'cover_text': 'Synthetic fixture; never a generated experimental cover.',
                      'dtype': 'float32', 'returned_model': model, 'revision': revision,
                      'seed': int(m.canonical_sha([20261024, model, src['record_id']])[:8], 16),
                      'usage': {'input_tokens': 200, 'output_tokens': 10},
                      'stop_reason': 'end_turn', 'hit_output_limit': False} if state == 'completed' else None
            attempt = dict(identity=identity, model=model, provider='local', automatic_retries=0,
                           record=record, status=state, result=result)
            self.write(prefix + '/run/attempts/' + identity + '.json', attempt)
            self.write(prefix + '/run/local_ledger/ledger.json', {'attempts': {identity: {
                'status': state, 'result': result, 'metadata': {'record': record}}}})
            self.write(prefix + '/run/run.json', {
                'schema': 'synsteg-budget-generation-run-v1', 'model': model, 'revision': revision,
                'dtype': 'float32', 'provider': 'local', 'settings': m.SETTINGS,
                'runner_sha256': m.RUNNER_SHA256, 'ledger_code_sha256': m.LEDGER_SHA256,
                'split': 'development', 'automatic_retries': 0,
                'selected_records_count': 1, 'selected_records': [record]})
            shutil.copyfile(self.root / contract['runtime_path'], self.root / prefix / 'run/runtime.json')
            self.write(prefix + '/process.json', {
                'identity': identity, 'source_order': order,
                'timeout_seconds': 1200, 'automatic_retries': 0,
                'status': 'completed_pending_remote_preservation' if state == 'completed' else 'stopped_requires_review_no_retry',
                'exit_code': 0 if state == 'completed' else 1, 'timed_out': False})
            for path in m.COMPLETED_CORE - {'artifact-manifest.json'}:
                if not (self.root / prefix / path).exists():
                    self.write(prefix + '/' + path, {'fixture': 'synthetic-only'})
        self.refresh(prefix)
        return prefix, identity

    def refresh(self, prefix):
        root = self.root / prefix
        raw = {str(p.relative_to(root)): m.sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()
               and p.name not in ['artifact-manifest.json', 'git-preservation-receipt.json']}
        if 'process.json' in raw or (root / 'artifact-manifest.json').exists():
            self.write(prefix + '/artifact-manifest.json', {'files_sha256': raw})
            raw['artifact-manifest.json'] = m.sha((root / 'artifact-manifest.json').read_bytes())
        receipt = {'readback_verified': True, 'commit': 'a' * 40, 'files_sha256': raw}
        if 'process.json' in raw:
            receipt['process_sha256'] = raw['process.json']
        self.write(prefix + '/git-preservation-receipt.json', receipt)

    def manifest(self):
        pins = m.inventory(self.root)
        return {'schema': m.ID + '-input', 'campaign': m.CAMPAIGN, 'base_ref': m.BASE_REF,
                'snapshot_ref': m.BASE_REF, 'files_sha256': pins, 'model_contracts': self.contracts,
                'new_shards': self.shards, 'reservation_scope': {
                    'snapshot_ref': m.BASE_REF, 'index_complete_for_supplied_roots': True,
                    'roots': ['runs/synthetic-fixture-only'] if self.shards else [],
                    'files': sorted(p for p in pins if p.startswith('runs/'))}}

    def merge(self):
        return m.merge(self.root, self.manifest())


class MergerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(self.tmp.name)

    def reject(self):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.f.merge()

    def test_actual_current_base_compatibility(self):
        result = self.f.merge()
        base = self.f.read(m.BASE_FILES['base'])
        self.assertEqual(result['records'], base['records'])
        self.assertEqual(result['counts'], {'completed': 89, 'reserved_outcome_unknown': 1, 'not_attempted': 210})
        self.assertEqual(result['model_denominators'], dict.fromkeys(m.MODELS, 100))
        self.assertFalse(result['inherited_base_raw_reaudited'])
        self.assertEqual(result['scheduling_overlays'][0]['source_order'], 91)

    def test_synthetic_three_model_completion_updates_full_denominator(self):
        for alias, order in [('tinyllama11', 92), ('qwen05', 1), ('falcon3_1b', 1)]:
            self.f.shard(alias, order)
        result = self.f.merge()
        self.assertEqual(result['counts']['completed'], 92)
        self.assertEqual(result['counts']['not_attempted'], 207)
        self.assertEqual(result['model_counts']['qwen05'], {'completed': 1, 'not_attempted': 99})
        self.assertEqual(result['model_counts']['falcon3_1b'], {'completed': 1, 'not_attempted': 99})

    def test_unknown_reservation_is_retained(self):
        self.f.shard(state='reservation_only')
        result = self.f.merge()
        self.assertEqual(result['counts']['reserved_outcome_unknown'], 2)
        self.assertEqual(result['new_shard_evidence'][0]['evidence_gap_reason'], 'attempt_absent')

    def test_completed_without_terminal_is_unknown(self):
        p, _ = self.f.shard()
        (self.f.root / p / 'process.json').unlink(); self.f.refresh(p)
        result = self.f.merge()
        self.assertEqual(result['new_shard_evidence'][0]['status'], 'reserved_outcome_unknown')
        self.assertEqual(result['new_shard_evidence'][0]['evidence_gap_reason'], 'terminal_process_absent')

    def test_failed_is_not_completed_or_not_attempted(self):
        self.f.shard(state='failed')
        self.assertEqual(self.f.merge()['counts']['failed'], 1)

    def test_duplicate_index(self):
        self.f.shard(); self.f.shards += copy.deepcopy(self.f.shards); self.reject()

    def test_orphan_file(self):
        self.f.write('runs/orphan/reservation.json', {}); self.reject()

    def test_missing_or_changed_inventory(self):
        manifest = self.f.manifest()
        path = self.f.root / m.BASE_FILES['source']
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'missing, changed or orphan'):
            m.merge(self.f.root, manifest)

    def test_missing_receipt(self):
        p, _ = self.f.shard(); (self.f.root / p / 'git-preservation-receipt.json').unlink(); self.reject()

    def test_tampered_raw_without_receipt_refresh(self):
        p, _ = self.f.shard(); self.f.write(p + '/stdout.log', {'changed': True}); self.reject()

    def test_missing_completed_core_even_with_receipt_refresh(self):
        p, _ = self.f.shard(); (self.f.root / p / 'child.json').unlink(); self.f.refresh(p); self.reject()

    def test_missing_reservation_identity(self):
        p, _ = self.f.shard(state='reservation_only')
        r = self.f.read(p + '/reservation.json'); del r['identity']
        self.f.write(p + '/reservation.json', r); self.f.refresh(p); self.reject()

    def test_wrong_process_identity(self):
        p, _ = self.f.shard(); r = self.f.read(p + '/process.json'); r['identity'] = 'f' * 64
        self.f.write(p + '/process.json', r); self.f.refresh(p); self.reject()

    def test_changed_run_code_split_retry(self):
        p, _ = self.f.shard(); original = self.f.read(p + '/run/run.json')
        for field, value in [('runner_sha256', 'f' * 64), ('ledger_code_sha256', 'f' * 64),
                             ('split', 'test'), ('automatic_retries', False), ('selected_records_count', True)]:
            with self.subTest(field=field):
                r = copy.deepcopy(original); r[field] = value
                self.f.write(p + '/run/run.json', r); self.f.refresh(p); self.reject()

    def test_actual_result_schema_and_model_identity(self):
        p, identity = self.f.shard(); path = p + '/run/attempts/' + identity + '.json'
        original = self.f.read(path)
        for field, value in [('returned_model', 'wrong-model'), ('revision', 'f' * 40), ('dtype', 'float16'), ('seed', 1)]:
            with self.subTest(field=field):
                attempt = copy.deepcopy(original); attempt['result'][field] = value
                self.f.write(path, attempt)
                ledger = self.f.read(p + '/run/local_ledger/ledger.json')
                ledger['attempts'][identity]['result'] = attempt['result']
                self.f.write(p + '/run/local_ledger/ledger.json', ledger); self.f.refresh(p); self.reject()

    def test_actual_input_tokens_must_match_preflight(self):
        p, identity = self.f.shard(); path = p + '/run/attempts/' + identity + '.json'
        a = self.f.read(path); a['result']['usage']['input_tokens'] = 201; self.f.write(path, a)
        ledger = self.f.read(p + '/run/local_ledger/ledger.json'); ledger['attempts'][identity]['result'] = a['result']
        self.f.write(p + '/run/local_ledger/ledger.json', ledger); self.f.refresh(p); self.reject()

    def test_changed_runtime(self):
        p, _ = self.f.shard(); self.f.write(p + '/run/runtime.json', {'python': 'changed', 'packages': {}})
        self.f.refresh(p); self.reject()

    def test_source_metadata_bound_beyond_prompt(self):
        p, identity = self.f.shard(); path = p + '/run/attempts/' + identity + '.json'
        a = self.f.read(path); a['record']['target_words'] += 1; self.f.write(path, a)
        run = self.f.read(p + '/run/run.json'); run['selected_records'] = [a['record']]
        self.f.write(p + '/run/run.json', run)
        ledger = self.f.read(p + '/run/local_ledger/ledger.json'); ledger['attempts'][identity]['metadata']['record'] = a['record']
        self.f.write(p + '/run/local_ledger/ledger.json', ledger); self.f.refresh(p); self.reject()

    def test_context_overflow_or_incomplete_denominator(self):
        contract = self.f.contract('qwen05'); path = contract['context_report_path']; original = self.f.read(path)
        for records in [original['records'][:99], original['records'] + original['records'][:1]]:
            r = copy.deepcopy(original); r['records'] = records; self.f.write(path, r); self.reject()
        r = copy.deepcopy(original); r['records'][0]['input_tokens'] = 1025; self.f.write(path, r); self.reject()

    def test_blocked_and_historical_rows_cannot_be_replayed(self):
        for order in [82, 91, 1]:
            with self.subTest(order=order):
                with tempfile.TemporaryDirectory() as tmp:
                    f = Fixture(tmp); f.shard('tinyllama11', order)
                    with self.assertRaises(ValueError): f.merge()

    def test_new_model_without_contract(self):
        self.f.shard(); self.f.contracts['qwen05'] = None; self.reject()

    def test_orphan_attempt_rejected(self):
        p, _ = self.f.shard(); self.f.write(p + '/run/attempts/' + 'f' * 64 + '.json', {})
        self.f.refresh(p); self.reject()

    def test_contract_cannot_substitute_model_revision(self):
        self.f.shard(); path = self.f.contracts['qwen05']; c = self.f.read(path); c['revision'] = 'f' * 40
        self.f.write(path, c); self.reject()

    def test_context_prompt_binding(self):
        c = self.f.contract('qwen05'); path = c['context_report_path']; r = self.f.read(path)
        r['records'][0]['prompt_sha256'] = 'f' * 64; self.f.write(path, r); self.reject()

    def test_symlink_evidence(self):
        (self.f.root / 'link').symlink_to(self.f.root / m.BASE_FILES['source']); self.reject()

    def test_duplicate_json_keys_and_nonfinite_numbers(self):
        for value in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}']:
            with self.subTest(value=value), self.assertRaises(ValueError): m.parse(value)

    def test_cli_requires_bound_manifest_and_external_output(self):
        manifest = Path(self.tmp.name) / 'manifest.json'
        manifest.write_text(json.dumps(self.f.manifest()))
        command = [sys.executable, str(SCRIPT), '--root', str(self.f.root), '--manifest', str(manifest),
                   '--expected-manifest-sha256', m.sha(manifest.read_bytes()), '--output']
        bad = subprocess.run(command + [str(self.f.root / 'result.json')], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn('output must be outside', bad.stderr)
        good = subprocess.run(command + [str(Path(self.tmp.name) / 'result.json')], capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)


if __name__ == '__main__':
    unittest.main()
