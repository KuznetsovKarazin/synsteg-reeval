#!/usr/bin/env python3
"""Independent prelaunch edge tests. No model or real child launch is permitted.

Fixture receipts are synthetic test data; they are never used as Git evidence.
The real source/input closure is checked separately before fixture mutation.
"""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

CHECKOUT = Path(__file__).resolve().parents[1]
if not (CHECKOUT / 'scripts/baseline_single_record_recovery_v2.py').is_file():
    CHECKOUT = CHECKOUT / 'checkout'
SCRIPT = CHECKOUT / 'scripts/baseline_single_record_recovery_v2.py'
spec = importlib.util.spec_from_file_location('successor_under_test', SCRIPT)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class SuccessorPrelaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s.verify_inputs(CHECKOUT)
        cls.inputs = s.load(CHECKOUT / s.INPUTS)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synsteg-independent-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'checkout'
        names = [item['path'] for item in self.inputs['files']]
        names += [s.INPUTS, s.HELPER]
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(CHECKOUT / name, target)
        self.patch_root = mock.patch.object(s, 'ROOT_PIN', str(self.root))
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.no_spawn = mock.patch.object(s.subprocess, 'Popen', side_effect=AssertionError('MODEL LAUNCH FORBIDDEN IN REVIEW TESTS'))
        self.blocked_spawn = self.no_spawn.start()
        self.addCleanup(self.no_spawn.stop)
        for order in s.ORDERS:
            job, payload = s.job_for(self.root, order)
            path = self.root / job['prompt_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        self.protocol_path = self.root / 'experiments' / s.ID / 'protocol.synthetic-test.json'
        write(self.protocol_path, s.expected_protocol(self.root, enabled=True))

    def activation(self, order=92):
        protocol = s.load(self.protocol_path)
        job = s.job_for(self.root, order)[0]
        previous = s.predecessor(self.root, order)
        intent_path = self.root / 'test-only' / f'intent-{order}.json'
        receipt_path = self.root / 'test-only' / f'receipt-{order}.json'
        write(intent_path, s.intent_for(self.root, self.protocol_path, job, previous['commit']))
        receipt = {
            'schema': s.ID + '-activation-readback-v1',
            'readback_verified': True, 'commit': 'a' * 40,
            'source_order': order, 'identity': job['identity'],
            'files_sha256': s.readback_pins(self.root, self.protocol_path, intent_path, protocol),
            'remote_output_scope_complete': True,
            'remote_current_output_orders': list(range(92, order)),
            'predecessor_receipt_commit': previous['commit'],
        }
        write(receipt_path, receipt)
        return intent_path, receipt_path

    def test_exact_source_closure_and_disabled_protocol(self):
        s.verify_inputs(self.root)
        disabled = s.expected_protocol(self.root, enabled=False)
        write(self.protocol_path, disabled)
        self.assertEqual(s.verify_protocol(self.root, self.protocol_path), disabled)
        with self.assertRaisesRegex(ValueError, 'not an activation'):
            s.validate_activation(self.root, self.protocol_path, None, None, 92)
        self.blocked_spawn.assert_not_called()

    def test_all_nine_jobs_bind_actual_prompt_identity_and_guard(self):
        rows = s.original_rows(self.root)
        old = s.module(self.root / s.OLD_HELPER, 'old_for_independent_test')
        identities = set()
        for order in range(92, 101):
            job, payload = s.job_for(self.root, order)
            self.assertEqual(json.loads(payload), rows[order - 1])
            self.assertEqual(job['identity'], old.canonical(['local', s.MODEL, rows[order - 1]['record_id']]))
            request = s.replay_request(job)
            self.assertEqual(request['source_order'], order)
            self.assertEqual(request['intent_identity'], job['identity'])
            decision = s.check_replay(self.root, job)
            self.assertEqual(decision['status'], 'not_denied_by_this_receipt')
            self.assertIs(decision['launch_authorized'], False)
            self.assertEqual(decision['request_sha256'], s.digest_bytes(s.canonical_bytes(request)))
            identities.add(job['identity'])
        self.assertEqual(len(identities), 9)

    def test_job_boundaries_and_type_rejections(self):
        for invalid in (None, True, False, 82, 90, 91, 101, 92.0, '92'):
            with self.subTest(order=invalid), self.assertRaises(ValueError):
                s.job_for(self.root, invalid)

    def test_guard_refuses_unknown82_and_saved_intent91(self):
        job = s.job_for(self.root, 92)[0]
        for blocked in (82, 91):
            altered = dict(job, source_order=blocked)
            with self.subTest(order=blocked), self.assertRaisesRegex(ValueError, 'denied'):
                s.check_replay(self.root, altered)
        altered = dict(job, identity='190860d03002e4cc73fd09b3b421847de7d9dda6143a0a359482fbd476703e92')
        with self.assertRaisesRegex(ValueError, 'denied'):
            s.check_replay(self.root, altered)

    def test_source_mutation_prevents_validation(self):
        path = self.root / s.GUARD
        path.write_bytes(path.read_bytes() + b'\n# synthetic mutation\n')
        with self.assertRaisesRegex(ValueError, 'Source bytes changed'):
            s.verify_protocol(self.root, self.protocol_path)
        self.blocked_spawn.assert_not_called()

    def test_protocol_and_prompt_mutations_rejected(self):
        original = s.load(self.protocol_path)
        for field, value in [('execution_enabled', 1), ('new_source_orders', list(range(91, 101))),
                             ('automatic_retries', 1), ('timeout_seconds_per_shard_including_model_load', 1201),
                             ('full_denominator', 299), ('output_relative', 'runs/old')]:
            with self.subTest(field=field):
                write(self.protocol_path, dict(original, **{field: value}))
                with self.assertRaises(ValueError):
                    s.verify_protocol(self.root, self.protocol_path)
        write(self.protocol_path, original)
        job = original['jobs'][0]
        with (self.root / job['prompt_path']).open('ab') as stream:
            stream.write(b' ')
        with self.assertRaisesRegex(ValueError, 'Prompt bytes changed'):
            s.verify_protocol(self.root, self.protocol_path)

    def test_full300_projection_preserves_every_original_row(self):
        base = s.load(self.root / s.BASE)
        projected = s.projection(self.root)
        self.assertEqual(projected['records'], base['records'])
        self.assertEqual(projected['counts'], {'completed': 89, 'not_attempted': 210, 'reserved_outcome_unknown': 1})
        self.assertEqual(len(projected['records']), 300)
        self.assertEqual(projected['model_calls_by_inspection'], 0)
        self.assertEqual(projected['blocked_intents_separate_from_outcomes'][0]['source_order'], 91)
        self.blocked_spawn.assert_not_called()

    def test_base_grid_counts_and_identity_corruption_rejected(self):
        original = s.load(self.root / s.BASE)
        rows = s.original_rows(self.root)
        mutations = []
        value = copy.deepcopy(original); value['records'].pop(); mutations.append(value)
        value = copy.deepcopy(original); value['records'][0] = copy.deepcopy(value['records'][1]); mutations.append(value)
        value = copy.deepcopy(original); value['records'][0]['record_id'] = 'wrong'; mutations.append(value)
        value = copy.deepcopy(original); value['records'][0]['status'] = 'completed'; mutations.append(value)
        for index, value in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                s.validate_base(value, rows)

    def test_empty_consumed_shard_is_unknown_and_never_retryable(self):
        shard = self.root / 'runs' / s.ID / 'order092'
        shard.mkdir(parents=True)
        projected = s.projection(self.root)
        self.assertEqual(projected['counts'], {'completed': 89, 'not_attempted': 209, 'reserved_outcome_unknown': 2})
        changed = [r for r in projected['records'] if r['model_alias'] == 'tinyllama11' and r['source_order'] == 92][0]
        self.assertEqual(changed['status'], 'reserved_outcome_unknown')
        with self.assertRaisesRegex(ValueError, 'no retry'):
            s.predecessor(self.root, 92)
        with self.assertRaises((ValueError, FileNotFoundError)):
            s.predecessor(self.root, 93)

    def test_missing_or_future_shards_stop_sequence(self):
        with self.assertRaisesRegex(ValueError, 'missing shard'):
            s.predecessor(self.root, 93)
        output = self.root / 'runs' / s.ID
        (output / 'order094').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'orphan'):
            s.predecessor(self.root, 92)
        shutil.rmtree(output / 'order094')
        (output / 'unexpected').mkdir()
        with self.assertRaisesRegex(ValueError, 'Unexpected output'):
            s.predecessor(self.root, 92)

    def test_previous90_real_evidence_passes_but_receipt_tamper_fails(self):
        receipt = s.predecessor(self.root, 92)
        self.assertEqual(receipt['source_order'], 90)
        path = self.root / s.PREVIOUS / 'git-preservation-receipt.json'
        value = s.load(path)
        value['files_sha256'].pop('process.json')
        write(path, value)
        with self.assertRaisesRegex(ValueError, 'receipt differs'):
            s.predecessor(self.root, 92)

    def test_dirty_previous_process_rejected_even_with_matching_inventory(self):
        directory = self.root / s.PREVIOUS
        old = s.module(self.root / s.OLD_HELPER, 'old_inventory_for_independent_test')
        protocol = s.load(self.root / s.OLD_PROTOCOL)
        job = protocol['jobs'][90 - 83]
        original = s.load(directory / 'process.json')
        for field, value in [('exit_code', 1), ('timed_out', True), ('interrupted', True), ('status', 'reserved_outcome_unknown')]:
            with self.subTest(field=field):
                write(directory / 'process.json', dict(original, **{field: value}))
                receipt = s.load(directory / 'git-preservation-receipt.json')
                receipt['process_sha256'] = s.sha(directory / 'process.json')
                receipt['files_sha256'] = {k: v for k, v in old.file_inventory(directory).items() if k != 'git-preservation-receipt.json'}
                write(directory / 'git-preservation-receipt.json', receipt)
                with self.assertRaisesRegex(ValueError, 'not cleanly terminal'):
                    s.verify_preserved(self.root, directory, job)

    def test_symlink_file_and_ancestor_refused(self):
        outside = Path(self.tmp.name) / 'outside'
        outside.mkdir()
        (outside / 'data.json').write_text('{}')
        link = self.root / 'test-link'
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            s.raw(link / 'data.json')
        direct = self.root / 'direct.json'
        direct.symlink_to(outside / 'data.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            s.load(direct)

    def test_activation_exact_job_binding_and_remote_scope(self):
        intent, receipt_path = self.activation()
        result = s.validate_activation(self.root, self.protocol_path, intent, receipt_path, 92)
        self.assertEqual(result[1]['source_order'], 92)
        original = s.load(receipt_path)
        for field, value in [('identity', '0' * 64), ('source_order', 93), ('remote_current_output_orders', [92]),
                             ('remote_output_scope_complete', False), ('readback_verified', False),
                             ('predecessor_receipt_commit', 'b' * 40), ('commit', 'not-a-commit')]:
            with self.subTest(field=field):
                write(receipt_path, dict(original, **{field: value}))
                with self.assertRaises(ValueError):
                    s.validate_activation(self.root, self.protocol_path, intent, receipt_path, 92)
        write(receipt_path, original)
        value = s.load(intent); value['source_order'] = 93; write(intent, value)
        with self.assertRaisesRegex(ValueError, 'Intent does not bind'):
            s.validate_activation(self.root, self.protocol_path, intent, receipt_path, 92)

    def test_spawn_failure_consumes_reservation_and_stops(self):
        intent, receipt = self.activation()
        first = s.load(self.root / s.FIRST)
        directory = self.root / 'runs' / s.ID / 'order092'
        def check_before_spawn(*args, **kwargs):
            for name in ('reservation.json', 'activation-receipt.json', 'bound-intent.json', 'guard-request.json', 'guard-result.json'):
                self.assertTrue((directory / name).is_file(), name + ' must exist before Popen')
            guard_result = s.load(directory / 'guard-result.json')
            self.assertEqual(s.sha(directory / 'guard-request.json'), guard_result['request_sha256'])
            self.assertTrue(kwargs['start_new_session'])
            raise RuntimeError('Synthetic child creation failure; no real process was launched')
        self.blocked_spawn.side_effect = check_before_spawn
        with mock.patch.object(s, 'verify_runtime', return_value=first):
            result = s.execute(self.root, self.protocol_path, intent, receipt, 92)
        self.blocked_spawn.assert_called_once()
        self.assertEqual(result['error_type'], 'RuntimeError')
        self.assertEqual(result['status'], 'stopped_requires_review_no_retry')
        self.assertTrue((directory / 'reservation.json').is_file())
        self.assertTrue((directory / 'process.json').is_file())
        self.assertEqual(s.load(directory / 'guard-request.json'), s.replay_request(s.job_for(self.root, 92)[0]))
        self.assertEqual(s.load(directory / 'reservation.json')['identity'], s.job_for(self.root, 92)[0]['identity'])
        with self.assertRaisesRegex(ValueError, 'no retry'):
            s.predecessor(self.root, 92)
        with self.assertRaises((ValueError, FileNotFoundError)):
            s.predecessor(self.root, 93)


if __name__ == '__main__':
    unittest.main(verbosity=2)
