"""Finite no-model integration checks; all synthetic writes use temporary roots."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


SOURCE_ROOT = Path(__file__).absolute().parents[1]
spec = importlib.util.spec_from_file_location('successor_v2_tested', SOURCE_ROOT/'scripts/baseline_single_record_recovery_v2.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SuccessorIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        names = [x['path'] for x in m.load(SOURCE_ROOT/m.INPUTS)['files']]
        names += [m.INPUTS, m.HELPER]
        names += [m.job_for(SOURCE_ROOT, order)[0]['prompt_path'] for order in m.ORDERS]
        for name in names:
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE_ROOT/name, target)
        self.root_patch = mock.patch.object(m, 'ROOT_PIN', str(self.root))
        self.root_patch.start()
        self.protocol = self.root/f'experiments/{m.ID}/protocol.test.json'
        self.intent = self.root/f'experiments/{m.ID}/intent.test.json'
        self.receipt = self.root/f'experiments/{m.ID}/receipt.test.json'

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def put(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

    def activation(self, enabled=True):
        protocol = m.expected_protocol(self.root, enabled)
        self.put(self.protocol, protocol)
        job = m.job_for(self.root, 92)[0]
        previous = m.predecessor(self.root, 92)
        self.put(self.intent, m.intent_for(self.root, self.protocol, job, previous['commit']))
        self.put(self.receipt, {
            'schema':m.ID+'-activation-readback-v1','readback_verified':True,'commit':'a'*40,
            'source_order':92,'identity':job['identity'],
            'files_sha256':m.readback_pins(self.root,self.protocol,self.intent,protocol),
            'remote_output_scope_complete':True,'remote_current_output_orders':[],
            'predecessor_receipt_commit':previous['commit']})

    def test_actual_source_closure_and_full_denominator(self):
        self.assertEqual(len(m.verify_inputs(self.root)['files']),57)
        result = m.projection(self.root)
        self.assertEqual(result['counts'], {'completed':89,'reserved_outcome_unknown':1,'not_attempted':210})
        self.assertEqual(len(result['records']),300)
        self.assertEqual(result['blocked_intents_separate_from_outcomes'][0]['source_order'],91)

    def test_all_nine_exact_jobs_and_guard_denies_no_launch_authority(self):
        for order in range(92,101):
            job, payload = m.job_for(self.root,order)
            self.assertEqual(payload,(self.root/job['prompt_path']).read_bytes())
            result = m.check_replay(self.root,job)
            self.assertEqual(result['status'],'not_denied_by_this_receipt')
            self.assertIs(result['launch_authorized'],False)
        for invalid in (True,82,91,101,92.0,'92'):
            with self.assertRaises(ValueError):
                m.job_for(self.root,invalid)

    def test_disabled_protocol_cannot_reserve_or_spawn(self):
        self.activation(enabled=False)
        with mock.patch.object(m.subprocess,'Popen') as launch, mock.patch.object(m,'verify_runtime') as runtime:
            with self.assertRaisesRegex(ValueError,'Source-only'):
                m.execute(self.root,self.protocol,self.intent,self.receipt,92)
            launch.assert_not_called()
            runtime.assert_not_called()
        self.assertFalse((self.root/'runs'/m.ID/'order092').exists())

    def test_forged_intent_rejected_even_with_recomputed_readback_map(self):
        self.activation()
        intent=m.load(self.intent);intent['source_order']=93;self.put(self.intent,intent)
        receipt=m.load(self.receipt)
        receipt['files_sha256']=m.readback_pins(self.root,self.protocol,self.intent,m.load(self.protocol))
        self.put(self.receipt,receipt)
        with mock.patch.object(m.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'Intent does not bind'):
                m.execute(self.root,self.protocol,self.intent,self.receipt,92)
            launch.assert_not_called()

    def test_spawn_failure_consumes_reservation_and_blocks_all_retry(self):
        self.activation()
        first=m.load(self.root/m.FIRST)
        with mock.patch.object(m,'verify_runtime',return_value=first), mock.patch.object(m.subprocess,'Popen',side_effect=OSError('synthetic no-child failure')) as launch:
            result=m.execute(self.root,self.protocol,self.intent,self.receipt,92)
            self.assertEqual(launch.call_count,1)
            self.assertEqual(result['status'],'stopped_requires_review_no_retry')
            with self.assertRaisesRegex(ValueError,'Consumed/orphan/missing'):
                m.execute(self.root,self.protocol,self.intent,self.receipt,92)
            self.assertEqual(launch.call_count,1)
            with self.assertRaises(ValueError):
                m.predecessor(self.root,93)
        shard=self.root/'runs'/m.ID/'order092'
        self.assertEqual(m.sha(shard/'guard-request.json'),m.load(shard/'guard-result.json')['request_sha256'])
        self.assertEqual(m.projection(self.root)['counts'],{'completed':89,'reserved_outcome_unknown':2,'not_attempted':209})

    def test_changed_scientific_source_fails_before_import_or_spawn(self):
        self.activation()
        with (self.root/m.OLD_HELPER).open('a') as stream: stream.write('\nraise RuntimeError("must never import")\n')
        with mock.patch.object(m.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'Source bytes changed'):
                m.execute(self.root,self.protocol,self.intent,self.receipt,92)
            launch.assert_not_called()

    def test_symlink_lock_fails_before_spawn(self):
        self.activation()
        target=self.root/'target.lock';target.write_text('')
        (self.root/'results/project-nn.lock').symlink_to(target)
        with mock.patch.object(m.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'Symlink'):
                m.execute(self.root,self.protocol,self.intent,self.receipt,92)
            launch.assert_not_called()


if __name__=='__main__':
    unittest.main()
