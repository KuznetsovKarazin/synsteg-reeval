"""Author tests with synthetic first/successor outputs; no model/tokenizer/HTTP."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT if (ROOT/'scripts/other_model_single_record_v1.py').exists() else Path('/workspace/scratch/af7adce5db85/baseline92-continuation-v2/checkout')
def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
q = import_file(ROOT/'scripts/qwen_sequential_development_v1.py', 'successor_author')
fixture = import_file(SOURCE/'tests/test_other_model_single_record_v1.py', 'first_author_fixture')


class SequentialTests(unittest.TestCase):
    def setUp(self):
        self.initial = fixture.OfflineAdapterTests(methodName='test_pinned_original_source_order_and_grid')
        self.initial.setUp(); self.addCleanup(self.initial.tearDown)
        self.root = self.initial.root
        shutil.copyfile(ROOT/q.HELPER, self.root/q.HELPER)
        self.first = q.support(self.root)
        self.initial.fixture_run()
        self.preserve(1)
        self.protocol = self.root/'experiments'/q.ID/'protocol.enabled.json'
        self.first.write_json(self.protocol, q.protocol(self.root, True))
        self.materialize(2)
        self.intent = self.protocol.with_name('intent002.json')
        self.first.write_json(self.intent, q.intent(self.root, self.protocol, 2))
        self.receipt = self.protocol.with_name('activation002.json')
        p = self.first.load(self.protocol); j = q.job_for(self.root, 2)[0]
        self.activation = {'schema': q.ID+'-activation-readback-v1', 'readback_verified': True,
            'commit': '2'*40, 'identity': j['identity'], 'source_order': 2,
            'files_sha256': q.activation_pins(self.root, self.protocol, self.intent, 2, p),
            'remote_output_scope_complete': True, 'remote_branches_and_prs_checked': True,
            'remote_qwen_attempted_source_orders': [1], 'remote_qwen_active_intents': [j['identity']],
            'predecessor_receipt_commit': '1'*40}
        self.first.write_json(self.receipt, self.activation)

    def materialize(self, order):
        job, content = q.job_for(self.root, order)
        self.first.write_bytes(self.root/job['prompt_path'], content)

    def replace(self, path, value):
        path.write_text(json.dumps(value))

    def preserve(self, order):
        f = self.first
        directory = self.root/f.SHARD if order == 1 else self.root/q.OUTPUT/f'order{order:03d}'
        job = f.job(self.root)[0] if order == 1 else q.job_for(self.root, order)[0]
        request, decision = f.guard_decision(self.root) if order == 1 else q.guard(self.root, order)
        prior = None if order == 1 else ('1'*40 if order == 2 else format(order-1, '040x'))
        intent_path = self.initial.intent_path if order == 1 else self.intent
        activation_path = self.initial.receipt_path if order == 1 else self.receipt
        bound = f.load(intent_path); activation = f.load(activation_path)
        reservation = {'experiment_id': f.ID if order == 1 else q.ID, 'model_alias': 'qwen05',
            'source_order': order, 'identity': job['identity'], 'record_id': job['record_id'],
            'status': 'reserved_outcome_unknown', 'automatic_retries': 0, 'guard_request_sha256': decision['request_sha256'],
            'protocol_sha256': bound['protocol_sha256'], 'intent_sha256': f.sha(intent_path),
            'activation_commit': activation['commit']}
        if prior:
            reservation['predecessor_receipt_commit'] = prior
        f.write_json(directory/'reservation.json', reservation)
        f.write_json(directory/'activation-receipt.json', activation)
        f.write_bytes(directory/'guard-request.json', f.canonical_bytes(request))
        f.write_json(directory/'guard-result.json', decision)
        f.write_bytes(directory/'bound-intent.json', intent_path.read_bytes())
        f.write_json(directory/'child.json', {'pid': -1, 'synthetic_fixture': True})
        f.write_bytes(directory/'stdout.log', b''); f.write_bytes(directory/'stderr.log', b'')
        f.write_json(directory/'process.json', {'status': 'completed_pending_remote_preservation',
            'exit_code': 0, 'timed_out': False, 'source_order': order, 'identity': job['identity'],
            'timeout_seconds': 1200, 'automatic_retries': 0, 'guard_request_sha256': decision['request_sha256']})
        util = f.module(self.root, f.UTIL, 'fixture_inventory')
        f.write_json(directory/'artifact-manifest.json', {'files_sha256': util.file_inventory(directory)})
        f.write_json(directory/'git-preservation-receipt.json', {'readback_verified': True,
            'commit': '1'*40 if order == 1 else format(order, '040x'), 'source_order': order,
            'files_sha256': util.file_inventory(directory), 'process_sha256': f.sha(directory/'process.json')})
        return directory

    def synthetic_successor(self, order=2):
        f = self.first; job = q.job_for(self.root, order)[0]
        directory = self.root/q.OUTPUT/f'order{order:03d}'
        runner, _ = q.preview(self.root, order, directory)
        count = f.load(self.root/(f.PRE+'/all100-context.json'))['records'][order-1]['input_tokens']
        def fake(row, limit):
            return {'cover_text': 'SYNTHETIC SUCCESSOR FIXTURE ONLY.', 'returned_model': f.MODEL,
                'revision': f.REVISION, 'dtype': 'float32', 'seed': int(f.canonical([20261024, f.MODEL, job['record_id']])[:8], 16),
                'usage': {'input_tokens': count, 'output_tokens': 3}, 'hit_output_limit': False,
                'stop_reason': 'end_turn', 'elapsed_seconds': 0}
        with patch.dict(sys.modules, {'torch': None, 'transformers': None, 'tokenizers': None}):
            code, _ = runner.run(prompts=self.root/job['prompt_path'], output=directory/'run', provider='local',
                model=f.MODEL, revision=f.REVISION, dtype='float32', cpu_threads=6, seed=20261024,
                max_records=1, max_new_records=1, max_output_tokens=1024, split='development',
                local_generator=fake, runtime_reader=lambda: self.initial.runtime, execute=True)
        self.assertEqual(code, 0)
        return directory

    def test_fixed_order2_through100_and_full_denominator(self):
        p = self.first.load(self.protocol)
        self.assertEqual(p['source_orders'], list(range(2, 101)))
        self.assertEqual((p['full_denominator'], p['model_denominator']), (300, 100))
        for n in [2, 100]:
            job, _ = q.job_for(self.root, n)
            self.assertEqual(job['record_id'], self.initial.rows[n-1]['record_id'])
        for invalid in [1, 101, True, 2.0, None]:
            with self.assertRaises(ValueError): q.job_for(self.root, invalid)

    def test_exact_readback_accepted(self):
        _, value = q.validate_activation(self.root, self.protocol, self.intent, self.receipt, 2)
        self.assertEqual(value, self.activation)

    def test_first_raw_mutation_blocks_protocol(self):
        (self.root/self.first.SHARD/'stdout.log').write_text('mutated')
        with self.assertRaises(ValueError): q.verify_protocol(self.root, self.protocol)

    def test_first_missing_receipt_blocks(self):
        (self.root/self.first.SHARD/'git-preservation-receipt.json').unlink()
        with self.assertRaises(FileNotFoundError): q.predecessor(self.root, 2)

    def test_changed_settings_block(self):
        p = self.first.load(self.protocol); p['settings']['top_p'] = .99; self.replace(self.protocol, p)
        with self.assertRaises(ValueError): q.verify_protocol(self.root, self.protocol)

    def test_equal_float_scientific_integer_refused(self):
        p = self.first.load(self.protocol); p['settings']['seed'] = 20261024.0; self.replace(self.protocol, p)
        with self.assertRaises(ValueError): q.verify_protocol(self.root, self.protocol)

    def test_disabled_protocol_blocks_intent(self):
        p = self.first.load(self.protocol); p['execution_enabled'] = False; self.replace(self.protocol, p)
        with self.assertRaises(ValueError): q.intent(self.root, self.protocol, 2)

    def test_cannot_skip_original_source2(self):
        with self.assertRaises(ValueError): q.predecessor(self.root, 3)

    def test_unknown_current_shard_is_consumed(self):
        (self.root/q.OUTPUT/'order002').mkdir(parents=True)
        with self.assertRaises(ValueError): q.predecessor(self.root, 2)
        with self.assertRaises(FileNotFoundError): q.predecessor(self.root, 3)

    def test_remote_future_intent_blocks(self):
        self.activation['remote_qwen_active_intents'].append(q.job_for(self.root, 55)[0]['identity'])
        self.replace(self.receipt, self.activation)
        with self.assertRaises(ValueError): q.validate_activation(self.root, self.protocol, self.intent, self.receipt, 2)

    def test_remote_attempt_gap_blocks(self):
        self.activation['remote_qwen_attempted_source_orders'] = []
        self.replace(self.receipt, self.activation)
        with self.assertRaises(ValueError): q.validate_activation(self.root, self.protocol, self.intent, self.receipt, 2)

    def test_boolean_remote_order_is_not_integer_source1(self):
        self.activation['remote_qwen_attempted_source_orders'] = [True]
        self.replace(self.receipt, self.activation)
        with self.assertRaises(ValueError): q.validate_activation(self.root, self.protocol, self.intent, self.receipt, 2)

    def test_remote_branch_census_required(self):
        self.activation['remote_branches_and_prs_checked'] = False; self.replace(self.receipt, self.activation)
        with self.assertRaises(ValueError): q.validate_activation(self.root, self.protocol, self.intent, self.receipt, 2)

    def test_guard_binds_new_id_order_and_identity(self):
        request, decision = q.guard(self.root, 2)
        self.assertEqual(request['experiment_id'], q.ID); self.assertEqual(request['source_order'], 2)
        self.assertEqual(request['intent_identity'], q.job_for(self.root, 2)[0]['identity'])
        import hashlib
        self.assertEqual(decision['request_sha256'], hashlib.sha256(self.first.canonical_bytes(request)).hexdigest())

    def test_completed_successor_advances_to3(self):
        directory = self.synthetic_successor()
        self.assertEqual(q.inspect_shard(self.root, directory, 2), {'status': 'completed', 'details': []})
        self.preserve(2)
        self.assertEqual(q.predecessor(self.root, 3)['commit'], format(2, '040x'))

    def test_different_row_input_count_cannot_complete(self):
        directory = self.synthetic_successor(); identity = q.job_for(self.root, 2)[0]['identity']
        p = directory/'run/attempts'/(identity+'.json'); attempt = self.first.load(p)
        l = directory/'run/local_ledger/ledger.json'; ledger = self.first.load(l)
        attempt['result']['usage']['input_tokens'] += 1; ledger['attempts'][identity]['result'] = attempt['result']
        self.replace(p, attempt); self.replace(l, ledger)
        with self.assertRaises(ValueError): q.inspect_shard(self.root, directory, 2)

    def test_orphan_attempt_refused(self):
        directory = self.synthetic_successor(); self.first.write_json(directory/'run/attempts/orphan.json', {})
        with self.assertRaises(ValueError): q.inspect_shard(self.root, directory, 2)

    def test_symlink_shard_refused(self):
        path = self.root/q.OUTPUT/'order002'; path.parent.mkdir(parents=True); path.symlink_to(self.root/self.first.SHARD)
        with self.assertRaises(ValueError): q.local_orders(self.root)


if __name__ == '__main__':
    unittest.main()
