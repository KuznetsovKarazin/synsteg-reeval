"""Independent edge tests using explicitly synthetic author fixture data.

No model/tokenizer/resource download or real child process is permitted. Fixture
generation injects the frozen runner's local_generator/runtime_reader seams.
"""
import importlib.util
import json
from pathlib import Path
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qwen_successor_fixture', ROOT / 'tests/test_qwen_sequential_development_v1.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
q = fixture.q


class IndependentSequentialTests(unittest.TestCase):
    def setUp(self):
        self.case = fixture.SequentialTests(methodName='test_completed_successor_advances_to3')
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.root, self.first = self.case.root, self.case.first
        patch = mock.patch.object(q.subprocess, 'Popen', side_effect=AssertionError('NO REAL CHILD IN REVIEW'))
        self.spawn = patch.start(); self.addCleanup(patch.stop)

    def replace(self, path, value):
        self.case.replace(path, value)

    def refresh_inventory(self, directory):
        util = self.first.module(self.root, self.first.UTIL, 'independent_inventory_fixture')
        raw = {p: v for p, v in util.file_inventory(directory).items()
               if p not in {'artifact-manifest.json', 'git-preservation-receipt.json'}}
        self.replace(directory / 'artifact-manifest.json', {'files_sha256': raw})
        receipt = self.first.load(directory / 'git-preservation-receipt.json')
        receipt['files_sha256'] = {p: v for p, v in util.file_inventory(directory).items()
                                  if p != 'git-preservation-receipt.json'}
        receipt['process_sha256'] = self.first.sha(directory / 'process.json')
        self.replace(directory / 'git-preservation-receipt.json', receipt)

    def completed_second(self):
        directory = self.case.synthetic_successor(2)
        self.case.preserve(2)
        q.verify_preserved(self.root, directory, 2)
        return directory

    def test_all99_jobs_bind_original_rows_and_unique_identities(self):
        identities = set()
        for order in range(2, 101):
            job, raw = q.job_for(self.root, order)
            row = self.case.initial.rows[order - 1]
            self.assertEqual(json.loads(raw), row)
            self.assertEqual(job['identity'], self.first.canonical(['local', self.first.MODEL, row['record_id']]))
            self.assertEqual(job['source_order'], order)
            identities.add(job['identity'])
        self.assertEqual(len(identities), 99)
        for order in (2, 50, 100):
            request, decision = q.guard(self.root, order)
            self.assertEqual(request['intent_identity'], q.job_for(self.root, order)[0]['identity'])
            self.assertEqual(request['source_order'], order)
            self.assertIs(decision['launch_authorized'], False)
            self.assertEqual(decision['request_sha256'], q.hashlib.sha256(self.first.canonical_bytes(request)).hexdigest())
        self.spawn.assert_not_called()

    def test_changed_first_adapter_rejected_before_import(self):
        path = self.root / q.FIRST_HELPER
        path.write_bytes(path.read_bytes() + b'\nraise RuntimeError("UNVERIFIED MODULE WAS IMPORTED")\n')
        with self.assertRaisesRegex(ValueError, 'Frozen first adapter changed'):
            q.support(self.root)
        self.spawn.assert_not_called()

    def test_preserved_reservation_metadata_must_match_completed_job(self):
        directory = self.completed_second()
        path = directory / 'reservation.json'
        original = self.first.load(path)
        for field, value in [('record_id', 'WRONG-RECORD'), ('model_alias', 'tinyllama11'), ('experiment_id', 'OTHER-EXPERIMENT')]:
            with self.subTest(field=field):
                self.replace(path, dict(original, **{field: value}))
                self.refresh_inventory(directory)
                with self.assertRaises(ValueError):
                    q.verify_preserved(self.root, directory, 2)
        self.replace(path, original)

    def test_preserved_bound_intent_semantics_cannot_be_rewritten(self):
        directory = self.completed_second()
        path = directory / 'bound-intent.json'
        reservation_path = directory / 'reservation.json'
        original = self.first.load(path)
        reservation = self.first.load(reservation_path)
        for field, value in [('record_id', 'WRONG-RECORD'), ('source_order', 3), ('identity', '0' * 64),
                             ('guard_request_sha256', '0' * 64), ('automatic_retries', 1)]:
            with self.subTest(field=field):
                self.replace(path, dict(original, **{field: value}))
                # Update the byte binding as well: semantic mismatch must still fail.
                changed = dict(reservation, intent_sha256=self.first.sha(path))
                self.replace(reservation_path, changed)
                self.refresh_inventory(directory)
                with self.assertRaises(ValueError):
                    q.verify_preserved(self.root, directory, 2)
        self.replace(path, original); self.replace(reservation_path, reservation)

    def test_preserved_activation_must_match_reservation(self):
        directory = self.completed_second()
        path = directory / 'activation-receipt.json'
        original = self.first.load(path)
        for field, value in [('identity', '0' * 64), ('source_order', 3), ('commit', 'b' * 40), ('readback_verified', False)]:
            with self.subTest(field=field):
                self.replace(path, dict(original, **{field: value}))
                self.refresh_inventory(directory)
                with self.assertRaises(ValueError):
                    q.verify_preserved(self.root, directory, 2)
        self.replace(path, original)

    def test_changed_per_record_seed_cannot_complete(self):
        directory = self.case.synthetic_successor(2)
        identity = q.job_for(self.root, 2)[0]['identity']
        path = directory / 'run/attempts' / (identity + '.json')
        ledger_path = directory / 'run/local_ledger/ledger.json'
        attempt = self.first.load(path); ledger = self.first.load(ledger_path)
        attempt['result']['seed'] += 1
        ledger['attempts'][identity]['result'] = attempt['result']
        self.replace(path, attempt); self.replace(ledger_path, ledger)
        with self.assertRaisesRegex(ValueError, 'Completed result differs'):
            q.inspect_shard(self.root, directory, 2)

    def test_unknown_or_failed_previous_never_advances(self):
        directory = self.root / q.OUTPUT / 'order002'
        directory.mkdir(parents=True)
        self.assertEqual(q.inspect_shard(self.root, directory, 2)['status'], 'reserved_outcome_unknown')
        with self.assertRaises(ValueError):
            q.predecessor(self.root, 2)
        with self.assertRaises((ValueError, FileNotFoundError)):
            q.predecessor(self.root, 3)
        self.spawn.assert_not_called()

    def test_guard_file_mutation_and_terminal_timeout_block_chain(self):
        directory = self.completed_second()
        path = directory / 'guard-request.json'
        original = path.read_bytes(); path.write_bytes(original + b' ')
        self.refresh_inventory(directory)
        with self.assertRaisesRegex(ValueError, 'guard evidence'):
            q.verify_preserved(self.root, directory, 2)
        path.write_bytes(original)
        process = self.first.load(directory / 'process.json'); process['timed_out'] = True
        self.replace(directory / 'process.json', process); self.refresh_inventory(directory)
        with self.assertRaisesRegex(ValueError, 'not clean terminal'):
            q.verify_preserved(self.root, directory, 2)

    def test_execute_reserves_exact_intent_before_failed_spawn_and_never_retries(self):
        directory = self.root / q.OUTPUT / 'order002'
        first = self.first
        manifest = first.load(self.root / first.FIRST)
        def stop_before_spawn(*args, **kwargs):
            for name in ('reservation.json', 'guard-request.json', 'guard-result.json', 'activation-receipt.json', 'bound-intent.json'):
                self.assertTrue((directory / name).is_file(), name)
            reservation = first.load(directory / 'reservation.json')
            decision = first.load(directory / 'guard-result.json')
            self.assertEqual(reservation['guard_request_sha256'], first.sha(directory / 'guard-request.json'))
            self.assertEqual(reservation['guard_request_sha256'], decision['request_sha256'])
            self.assertEqual((directory / 'bound-intent.json').read_bytes(), self.case.intent.read_bytes())
            self.assertEqual(reservation['intent_sha256'], first.sha(directory / 'bound-intent.json'))
            command = args[0]
            self.assertEqual(command[command.index('--prompts') + 1], str(self.root / q.job_for(self.root, 2)[0]['prompt_path']))
            self.assertEqual(command[command.index('--model') + 1], first.MODEL)
            self.assertEqual(command[command.index('--max-new-records') + 1], '1')
            self.assertTrue(kwargs['start_new_session'])
            raise RuntimeError('Synthetic child creation failure; no process was launched')
        self.spawn.side_effect = stop_before_spawn
        fake_runner = types.SimpleNamespace(local_runtime=lambda: self.case.initial.runtime)
        with mock.patch.object(q, 'ROOT_PIN', str(self.root)), \
             mock.patch.object(q, 'support', return_value=first), \
             mock.patch.object(first, 'verify_environment', return_value=manifest), \
             mock.patch.object(first, 'resources', return_value=first.load(self.root / first.PRE / 'resources.json')), \
             mock.patch.object(q, 'preview', return_value=(fake_runner, {})):
            # Rebind only the disposable fixture to its temporary execution root.
            protocol = q.protocol(self.root, True)
            self.replace(self.case.protocol, protocol)
            self.replace(self.case.intent, q.intent(self.root, self.case.protocol, 2))
            activation = dict(self.case.activation)
            activation['files_sha256'] = q.activation_pins(self.root, self.case.protocol, self.case.intent, 2, protocol)
            self.replace(self.case.receipt, activation)
            result = q.execute(self.root, self.case.protocol, self.case.intent, self.case.receipt, 2)
        self.spawn.assert_called_once()
        self.assertEqual(result['error_type'], 'RuntimeError')
        self.assertEqual(result['status'], 'stopped_requires_review_no_retry')
        self.assertEqual(result['observed_outcome']['status'], 'reserved_outcome_unknown')
        with self.assertRaises(ValueError):
            q.predecessor(self.root, 2)
        with self.assertRaises((ValueError, FileNotFoundError)):
            q.predecessor(self.root, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
