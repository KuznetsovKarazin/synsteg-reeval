"""Offline synthetic adapter tests. No actual tokenizer, model or HTTP calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qwen_adapter', ROOT/'scripts/other_model_single_record_v1.py')
adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
SOURCE = ROOT if (ROOT/adapter.FIRST).exists() else Path('/workspace/scratch/af7adce5db85/baseline92-continuation-v2/checkout')
METADATA = ROOT/'results/development-other-models-single-record-v1/preparation'


class OfflineAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qwen-adapter-synthetic-')
        self.root = Path(self.temp.name)
        for path in adapter.CORE_PINS:
            source = ROOT/path if (ROOT/path).exists() else SOURCE/path
            target = self.root/path; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        target = self.root/adapter.HELPER; shutil.copyfile(ROOT/adapter.HELPER, target)
        self.rows = adapter.verify_core(self.root)
        pre = self.root/adapter.PRE; pre.mkdir(parents=True)
        self.runtime = {'schema': 'synsteg-local-runtime-recovery-v1', 'python': 'SYNTHETIC TEST ONLY',
                        'packages': {'torch': '2.6.0+cpu', 'transformers': '4.57.6'}}
        adapter.write_json(pre/'started.json', {'synthetic_test_only': True})
        adapter.write_json(pre/'resources.json', {'synthetic_test_only': True})
        adapter.write_json(pre/'runtime.json', self.runtime)
        shutil.copyfile(METADATA/'config.json', pre/'config.json')
        self.context = {'schema': adapter.MERGER+'-context-report', 'model_id': adapter.MODEL,
            'revision': adapter.REVISION, 'source_sha256': adapter.CORE_PINS[adapter.PROMPTS],
            'model_config_sha256': adapter.sha(pre/'config.json'), 'model_context_limit': 32768,
            'records': [{'source_order': i, 'record_id': r['record_id'],
                'prompt_sha256': hashlib.sha256(r['prompt'].encode()).hexdigest(), 'input_tokens': 1000}
                for i, r in enumerate(self.rows, 1)]}
        adapter.write_json(pre/'all100-context.json', self.context)
        adapter.write_json(pre/'model-contract.json', adapter.expected_contract(self.root))
        adapter.write_bytes(self.root/adapter.PROMPT, adapter.job(self.root)[1])
        self.protocol_path = self.root/'experiments'/adapter.ID/'protocol.enabled.json'
        adapter.write_json(self.protocol_path, adapter.protocol(self.root, True))
        self.intent_path = self.protocol_path.with_name('intent.json')
        adapter.write_json(self.intent_path, adapter.intent(self.root, self.protocol_path))
        p = adapter.load(self.protocol_path)
        self.receipt_path = self.protocol_path.with_name('synthetic-activation.json')
        self.receipt = {'schema': adapter.ID+'-activation-readback-v1', 'readback_verified': True,
            'commit': '1'*40, 'identity': p['job']['identity'],
            'files_sha256': adapter.activation_pins(self.root, self.protocol_path, self.intent_path, p),
            'remote_output_scope_complete': True, 'remote_branches_and_prs_checked': True,
            'remote_qwen_current100_attempts': [], 'remote_qwen_active_intents': [p['job']['identity']]}
        adapter.write_json(self.receipt_path, self.receipt)

    def tearDown(self):
        self.temp.cleanup()

    def replace(self, path, value):
        path.write_text(json.dumps(value))

    def fixture_run(self):
        directory = self.root/adapter.SHARD
        runner, _ = adapter.preview(self.root, directory/'run')
        def fake(row, limit):
            return {'cover_text': 'SYNTHETIC FIXTURE; NOT A MODEL EXPERIMENT.', 'returned_model': adapter.MODEL,
                'revision': adapter.REVISION, 'dtype': 'float32',
                'seed': int(adapter.canonical([20261024, adapter.MODEL, row['record_id']])[:8], 16),
                'usage': {'input_tokens': 1000, 'output_tokens': 3}, 'hit_output_limit': False,
                'stop_reason': 'end_turn', 'elapsed_seconds': 0}
        with patch.dict(sys.modules, {'torch': None, 'transformers': None, 'tokenizers': None}):
            code, _ = runner.run(prompts=self.root/adapter.PROMPT, output=directory/'run', provider='local',
                model=adapter.MODEL, revision=adapter.REVISION, max_records=1, max_new_records=1,
                max_output_tokens=1024, dtype='float32', cpu_threads=6, seed=20261024,
                local_generator=fake, runtime_reader=lambda: self.runtime, execute=True)
        self.assertEqual(code, 0)
        return directory

    def test_pinned_original_source_order_and_grid(self):
        p = adapter.protocol(self.root)
        self.assertFalse(p['execution_enabled']); self.assertEqual(p['source_orders'], [1])
        self.assertEqual((p['full_denominator'], p['per_model_denominator']), (300, 100))
        self.assertEqual(p['model_id'], 'Qwen/Qwen2.5-0.5B-Instruct')
        self.assertEqual(p['settings'], adapter.SETTINGS)

    def test_activation_exact_readback(self):
        _, receipt = adapter.validate_activation(self.root, self.protocol_path, self.intent_path, self.receipt_path)
        self.assertEqual(receipt, self.receipt)

    def test_disabled_protocol_refused(self):
        p = adapter.load(self.protocol_path); p['execution_enabled'] = False; self.replace(self.protocol_path, p)
        with self.assertRaises(ValueError): adapter.intent(self.root, self.protocol_path)

    def test_context_all100_required(self):
        self.context['records'].pop(); self.replace(self.root/(adapter.PRE+'/all100-context.json'), self.context)
        with self.assertRaises(ValueError): adapter.preflight_pins(self.root)

    def test_model_specific_context_overflow_refused(self):
        self.context['records'][99]['input_tokens'] = 32768-1024+1
        self.replace(self.root/(adapter.PRE+'/all100-context.json'), self.context)
        with self.assertRaises(ValueError): adapter.preflight_pins(self.root)

    def test_context_order_mutation_refused(self):
        self.context['records'][1]['record_id'] = self.context['records'][0]['record_id']
        self.replace(self.root/(adapter.PRE+'/all100-context.json'), self.context)
        with self.assertRaises(ValueError): adapter.preflight_pins(self.root)

    def test_tiny_context_cannot_substitute(self):
        self.context['model_id'] = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        self.replace(self.root/(adapter.PRE+'/all100-context.json'), self.context)
        with self.assertRaises(ValueError): adapter.preflight_pins(self.root)

    def test_contract_dtype_mutation_refused(self):
        path = self.root/(adapter.PRE+'/model-contract.json'); value = adapter.load(path)
        value['dtype'] = 'bfloat16'; self.replace(path, value)
        with self.assertRaises(ValueError): adapter.preflight_pins(self.root)

    def test_activation_hash_mutation_refused(self):
        self.receipt['files_sha256'][adapter.RUNNER] = '0'*64; self.replace(self.receipt_path, self.receipt)
        with self.assertRaises(ValueError): adapter.validate_activation(self.root, self.protocol_path, self.intent_path, self.receipt_path)

    def test_incomplete_remote_scope_refused(self):
        self.receipt['remote_output_scope_complete'] = False; self.replace(self.receipt_path, self.receipt)
        with self.assertRaises(ValueError): adapter.validate_activation(self.root, self.protocol_path, self.intent_path, self.receipt_path)

    def test_prior_qwen_attempt_refused(self):
        self.receipt['remote_qwen_current100_attempts'] = [self.receipt['identity']]
        self.replace(self.receipt_path, self.receipt)
        with self.assertRaises(ValueError): adapter.validate_activation(self.root, self.protocol_path, self.intent_path, self.receipt_path)

    def test_extra_remote_intent_refused(self):
        self.receipt['remote_qwen_active_intents'].append('0'*64); self.replace(self.receipt_path, self.receipt)
        with self.assertRaises(ValueError): adapter.validate_activation(self.root, self.protocol_path, self.intent_path, self.receipt_path)

    def test_reserved_shard_cannot_replay(self):
        (self.root/adapter.SHARD).mkdir(parents=True)
        with self.assertRaises(ValueError): adapter.intent(self.root, self.protocol_path)

    def test_orphan_local_other_order_refused(self):
        (self.root/'runs'/adapter.ID/'qwen05/order002').mkdir(parents=True)
        with self.assertRaises(ValueError): adapter.intent(self.root, self.protocol_path)

    def test_symlink_refused(self):
        source = self.root/adapter.PROMPT; link = source.with_suffix('.link'); link.symlink_to(source)
        with self.assertRaises(ValueError): adapter.sha(link)

    def test_duplicate_json_refused(self):
        path = self.root/'duplicate.json'; path.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError): adapter.load(path)

    def test_missing_resources_refused_without_download(self):
        with self.assertRaises(ValueError): adapter.resources(self.root)

    def test_complete_synthetic_shard_accepted_no_nn(self):
        directory = self.fixture_run()
        self.assertEqual(adapter.inspect_shard(self.root, directory), {'status': 'completed', 'details': []})
        self.assertEqual(adapter.sha(directory/'run/runtime.json'), adapter.sha(self.root/(adapter.PRE+'/runtime.json')))

    def test_actual_input_count_must_match_model_preflight(self):
        directory = self.fixture_run(); identity = adapter.job(self.root)[0]['identity']
        path = directory/'run/attempts'/(identity+'.json'); attempt = adapter.load(path)
        ledgerpath = directory/'run/local_ledger/ledger.json'; ledger = adapter.load(ledgerpath)
        attempt['result']['usage']['input_tokens'] += 1
        ledger['attempts'][identity]['result'] = attempt['result']; self.replace(path, attempt); self.replace(ledgerpath, ledger)
        with self.assertRaises(ValueError): adapter.inspect_shard(self.root, directory)

    def test_empty_synthetic_cover_cannot_complete(self):
        directory = self.fixture_run(); identity = adapter.job(self.root)[0]['identity']
        path = directory/'run/attempts'/(identity+'.json'); attempt = adapter.load(path)
        ledgerpath = directory/'run/local_ledger/ledger.json'; ledger = adapter.load(ledgerpath)
        attempt['result']['cover_text'] = ' '; ledger['attempts'][identity]['result'] = attempt['result']
        self.replace(path, attempt); self.replace(ledgerpath, ledger)
        with self.assertRaises(ValueError): adapter.inspect_shard(self.root, directory)

    def test_partial_shard_stays_unknown(self):
        path = self.root/adapter.SHARD; path.mkdir(parents=True)
        self.assertEqual(adapter.inspect_shard(self.root, path)['status'], 'reserved_outcome_unknown')


if __name__ == '__main__':
    unittest.main()
