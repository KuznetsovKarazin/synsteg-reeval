#!/usr/bin/env python3
"""Independent Qwen adapter tests; all preflight data/receipts are synthetic.

No tokenizer/model/resource download or real process execution is permitted.
Run against the restored checkout or pass its path via SYNSTEG_REVIEW_INPUT_ROOT.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

OUTPUT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('SYNSTEG_REVIEW_INPUT_ROOT', '/workspace/scratch/af7adce5db85/baseline92-continuation-v2/checkout'))
if (OUTPUT / 'scripts/run_budget_generation.py').is_file():
    SOURCE = OUTPUT
spec = importlib.util.spec_from_file_location('independent_qwen_adapter', OUTPUT / 'scripts/other_model_single_record_v1.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


class IndependentQwenAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qwen-independent-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'checkout'
        for name in list(q.CORE_PINS) + [q.HELPER]:
            candidate = OUTPUT / name
            if not candidate.is_file():
                candidate = SOURCE / name
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, target)
        for target, value in [('ROOT_PIN', str(self.root))]:
            patch = mock.patch.object(q, target, value)
            patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(q.subprocess, 'Popen', side_effect=AssertionError('NO REAL CHILD IN INDEPENDENT REVIEW'))
        self.spawn = patch.start(); self.addCleanup(patch.stop)
        self.rows = q.verify_core(self.root)
        job, payload = q.job(self.root)
        (self.root / q.PROMPT).write_bytes(payload)
        self.runtime = {'schema': 'synsteg-local-runtime-recovery-v1', 'synthetic_fixture_only': True}
        self.resources = {'synthetic_fixture_only': True}
        pre = self.root / q.PRE
        write(pre / 'started.json', {'synthetic_fixture_only': True})
        write(pre / 'resources.json', self.resources)
        write(pre / 'runtime.json', self.runtime)
        config_candidates = [OUTPUT.parent / 'official-public-metadata/config.json',
                             SOURCE / q.PRE / 'config.json',
                             SOURCE / q.load(self.root / q.RESOURCE)['expected_snapshot_relative'] / 'config.json']
        config = next((p for p in config_candidates if p.is_file()), None)
        if config is None:
            raise RuntimeError('Review fixture needs public config.json bytes, not model weights')
        shutil.copyfile(config, pre / 'config.json')
        context = {'schema': q.MERGER + '-context-report', 'model_id': q.MODEL, 'revision': q.REVISION,
                   'source_sha256': q.CORE_PINS[q.PROMPTS], 'model_config_sha256': q.sha(pre / 'config.json'),
                   'model_context_limit': 32768, 'records': [
                       {'source_order': i + 1, 'record_id': row['record_id'],
                        'prompt_sha256': q.hashlib.sha256(row['prompt'].encode()).hexdigest(), 'input_tokens': 100}
                       for i, row in enumerate(self.rows)]}
        write(pre / 'all100-context.json', context)
        write(pre / 'model-contract.json', q.expected_contract(self.root))
        self.protocol_path = self.root / 'experiments' / q.ID / 'synthetic-protocol.json'
        write(self.protocol_path, q.protocol(self.root, enabled=True))

    def activation(self):
        intent_path = self.root / 'synthetic' / 'intent.json'
        receipt_path = self.root / 'synthetic' / 'receipt.json'
        value = q.verify_protocol(self.root, self.protocol_path)
        write(intent_path, q.intent(self.root, self.protocol_path))
        receipt = {'schema': q.ID + '-activation-readback-v1', 'readback_verified': True,
                   'commit': 'a' * 40, 'identity': value['job']['identity'],
                   'files_sha256': q.activation_pins(self.root, self.protocol_path, intent_path, value),
                   'remote_output_scope_complete': True, 'remote_qwen_current100_attempts': [],
                   'remote_qwen_active_intents': [value['job']['identity']],
                   'remote_branches_and_prs_checked': True}
        write(receipt_path, receipt)
        return intent_path, receipt_path

    def test_fixed_first_development_source_and_science(self):
        job, payload = q.job(self.root)
        self.assertEqual(len(self.rows), 100)
        self.assertEqual(json.loads(payload), self.rows[0])
        self.assertEqual(job['source_order'], 1)
        self.assertEqual(job['model_alias'], 'qwen05')
        self.assertEqual(job['identity'], q.canonical(['local', q.MODEL, self.rows[0]['record_id']]))
        _, plan = q.preview(self.root, self.root / 'preview-unused')
        self.assertEqual(plan['model'], q.MODEL)
        self.assertEqual(plan['revision'], q.REVISION)
        self.assertEqual(plan['selected_records_count'], 1)
        self.assertEqual(plan['selected_records'][0]['record_id'], self.rows[0]['record_id'])
        self.assertEqual(plan['settings'], q.SETTINGS)
        self.assertEqual(plan['automatic_retries'], 0)
        self.assertFalse((self.root / 'preview-unused').exists())
        self.spawn.assert_not_called()

    def test_missing_actual_preflight_blocks_protocol(self):
        (self.root / q.PRE / 'all100-context.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            q.protocol(self.root)
        self.spawn.assert_not_called()

    def test_full100_context_identity_limits_and_denominator(self):
        path = self.root / q.PRE / 'all100-context.json'
        original = q.load(path)
        mutations = []
        value = copy.deepcopy(original); value['records'].pop(); mutations.append(value)
        value = copy.deepcopy(original); value['records'][5]['source_order'] = 1; mutations.append(value)
        value = copy.deepcopy(original); value['records'][99]['record_id'] = 'test-secret'; mutations.append(value)
        value = copy.deepcopy(original); value['records'][0]['input_tokens'] = 32768 - 1024 + 1; mutations.append(value)
        value = copy.deepcopy(original); value['records'][0]['input_tokens'] = True; mutations.append(value)
        value = copy.deepcopy(original); value['records'][0]['input_tokens'] = 0; mutations.append(value)
        for i, value in enumerate(mutations):
            with self.subTest(mutation=i):
                write(path, value)
                with self.assertRaises(ValueError):
                    q.preflight_pins(self.root)

    def test_protocol_false_or_changed_selection_refused(self):
        write(self.protocol_path, q.protocol(self.root, enabled=False))
        with self.assertRaisesRegex(ValueError, 'Disabled'):
            q.intent(self.root, self.protocol_path)
        original = q.protocol(self.root, enabled=True)
        for field, value in [('execution_enabled', 1), ('source_orders', [2]), ('model_alias', 'falcon3_1b'),
                             ('max_new_requests', 2), ('automatic_retries', 1), ('revision', 'unfrozen')]:
            with self.subTest(field=field):
                write(self.protocol_path, dict(original, **{field: value}))
                with self.assertRaises(ValueError):
                    q.verify_protocol(self.root, self.protocol_path)

    def test_guard_binds_qwen_first_identity_and_denies_no_authority(self):
        request, decision = q.guard_decision(self.root)
        self.assertEqual(request['model_alias'], 'qwen05')
        self.assertEqual(request['source_order'], 1)
        self.assertEqual(request['intent_identity'], q.job(self.root)[0]['identity'])
        self.assertEqual(decision['status'], 'not_denied_by_this_receipt')
        self.assertIs(decision['launch_authorized'], False)

    def test_activation_requires_exact_remote_census_and_intent(self):
        intent, receipt_path = self.activation()
        q.validate_activation(self.root, self.protocol_path, intent, receipt_path)
        original = q.load(receipt_path)
        for field, value in [('remote_qwen_current100_attempts', ['existing']),
                             ('remote_qwen_active_intents', []), ('remote_output_scope_complete', False),
                             ('remote_branches_and_prs_checked', False), ('identity', '0' * 64),
                             ('files_sha256', {}), ('commit', 'not-a-commit')]:
            with self.subTest(field=field):
                write(receipt_path, dict(original, **{field: value}))
                with self.assertRaises(ValueError):
                    q.validate_activation(self.root, self.protocol_path, intent, receipt_path)
        write(receipt_path, original)
        value = q.load(intent); value['source_order'] = 2; write(intent, value)
        with self.assertRaises(ValueError):
            q.validate_activation(self.root, self.protocol_path, intent, receipt_path)

    def test_core_change_fails_before_preview_or_launch(self):
        path = self.root / q.RUNNER
        path.write_bytes(path.read_bytes() + b'\n# corruption\n')
        with self.assertRaisesRegex(ValueError, 'Pinned source'):
            q.execute(self.root, self.protocol_path, None, None)
        self.spawn.assert_not_called()

    def test_local_orphan_reservations_block_intent(self):
        output = self.root / 'runs' / q.ID
        (output / 'qwen05' / 'order002').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'Unexpected local'):
            q.intent(self.root, self.protocol_path)
        shutil.rmtree(output / 'qwen05')
        (output / 'falcon3_1b').mkdir()
        with self.assertRaisesRegex(ValueError, 'Unexpected local'):
            q.intent(self.root, self.protocol_path)

    def test_runtime_original_python_and_platform_required(self):
        first = q.load(self.root / q.FIRST)
        executable = self.root / first['python_relative']
        with mock.patch.object(q.sys, 'executable', str(executable)), \
             mock.patch.object(q.sys, 'prefix', str(executable.parent.parent)), \
             mock.patch.object(q.sys, 'version', first['python']), \
             mock.patch.object(q.platform, 'platform', return_value=first['platform']), \
             mock.patch.object(q.importlib.metadata, 'version', side_effect=lambda name: first['packages'][name]):
            q.verify_environment(self.root)
            with mock.patch.object(q.sys, 'version', 'different interpreter'), self.assertRaises(ValueError):
                q.verify_environment(self.root)
            with mock.patch.object(q.platform, 'platform', return_value='different platform'), self.assertRaises(ValueError):
                q.verify_environment(self.root)

    def test_inspection_refuses_symlink_root_and_ancestors(self):
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        link = self.root / 'linked-shard'
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            q.inspect_shard(self.root, link)
        nested = outside / 'nested'; nested.mkdir()
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            q.inspect_shard(self.root, link / 'nested')

    def test_spawn_failure_guard_bytes_reservation_and_no_retry(self):
        intent, receipt = self.activation()
        first = q.load(self.root / q.FIRST)
        directory = self.root / q.SHARD
        def stop_before_real_spawn(*args, **kwargs):
            for name in ('reservation.json', 'bound-intent.json', 'activation-receipt.json', 'guard-request.json', 'guard-result.json'):
                self.assertTrue((directory / name).is_file(), name)
            decision = q.load(directory / 'guard-result.json')
            self.assertEqual(q.sha(directory / 'guard-request.json'), decision['request_sha256'])
            self.assertEqual(q.load(directory / 'reservation.json')['guard_request_sha256'], decision['request_sha256'])
            command = args[0]
            for flag, expected in [('--model', q.MODEL), ('--revision', q.REVISION), ('--max-records', '1'),
                                   ('--max-new-records', '1'), ('--max-output-tokens', '1024')]:
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertTrue(kwargs['start_new_session'])
            raise RuntimeError('Synthetic child creation failure; no actual process')
        self.spawn.side_effect = stop_before_real_spawn
        fake_runner = types.SimpleNamespace(local_runtime=lambda: self.runtime)
        with mock.patch.object(q, 'verify_environment', return_value=first), \
             mock.patch.object(q, 'resources', return_value=self.resources), \
             mock.patch.object(q, 'preview', return_value=(fake_runner, {})):
            result = q.execute(self.root, self.protocol_path, intent, receipt)
        self.spawn.assert_called_once()
        self.assertEqual(result['error_type'], 'RuntimeError')
        self.assertEqual(result['status'], 'stopped_requires_review_no_retry')
        self.assertEqual(result['guard_request_sha256'], q.sha(directory / 'guard-request.json'))
        with self.assertRaisesRegex(ValueError, 'Consumed'):
            q.intent(self.root, self.protocol_path)
        self.assertEqual(q.inspect_shard(self.root, directory)['status'], 'reserved_outcome_unknown')


if __name__ == '__main__':
    unittest.main(verbosity=2)
