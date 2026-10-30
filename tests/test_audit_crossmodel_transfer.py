"""Synthetic boundary fixtures for the independent stdlib audit; no model/run data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('independent_crossmodel_audit', ROOT / 'scripts/audit_crossmodel_transfer.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def tiny_book():
    return {'version': 'fixed-codebook-v1', 'cover_sha256': 'fixture', 'seed': 'fixture',
            'max_bits_per_position': 1, 'min_group_size': 1, 'protect_capitalized': True,
            'candidate_filter_id': 'fixture', 'scorer_id': 'fixture',
            'separators': ['', '\t', '\n', ' ', ''],
            'positions': [{'original': 'calm', 'groups': [['calm'], ['quiet']], 'scores': [[1.0], [2.0]]},
                          {'original': 'bright', 'groups': [['bright'], ['light']], 'scores': [[1.0], [2.0]]},
                          {'original': 'plain', 'groups': [['plain'], ['simple']], 'scores': [[1.0], [2.0]]},
                          {'original': 'tail.', 'groups': [], 'scores': []}]}


class IndependentAuditFixtures(unittest.TestCase):
    def test_identity_comparison_rejects_tiny_numeric_mutation(self):
        checks = audit.Checks()
        checks.equal('arithmetic', 1.0, 1.0 + 1e-13)
        with self.assertRaises(AssertionError):
            checks.exact('identity', {'scores': [1.0]}, {'scores': [1.0 + 1e-13]})
        with self.assertRaises(AssertionError):
            checks.exact('integer_type', 1, 1.0)

    def test_interruption_at_marker_boundary_retains_missing_event_evidence(self):
        row = {'sample': {'id': 'a'}, 'started_at': 3, 'finished_at': 8,
               'status': 'failed', 'failure': 'interrupted_attempt_not_retried',
               'cover_text': None, 'variants': {}}
        marker = {'time_unix': 3, 'freeze_id': 'fixture'}
        bounds = {'freeze_id': 'fixture', 'frozen': 1, 'freeze_event': 2, 'generation': 10, 'opening': 11}
        end = [{'event': 'attempt_interrupted_not_retried', 'time_unix': 9}]
        result = audit.attempt_ledger(row, marker, [], end, [], bounds, audit.Checks())
        self.assertTrue(result['start_event_absent'])
        self.assertFalse(result['preparation_state_unavailable_after_interruption'])
        with self.assertRaisesRegex(AssertionError, 'start_event_count'):
            audit.attempt_ledger(row, marker, [], [{'event': 'attempt_completed', 'time_unix': 9}],
                                 [], bounds, audit.Checks())

    def test_interruption_after_preparation_preserves_unavailable_state(self):
        row = {'sample': {'id': 'a'}, 'started_at': 3, 'finished_at': 8,
               'status': 'failed', 'failure': 'interrupted_attempt_not_retried',
               'cover_text': None, 'variants': {}}
        marker = {'time_unix': 3, 'freeze_id': 'fixture'}
        bounds = {'freeze_id': 'fixture', 'frozen': 1, 'freeze_event': 2, 'generation': 10, 'opening': 11}
        start = [{'time_unix': 4}]
        end = [{'event': 'attempt_interrupted_not_retried', 'time_unix': 9}]
        prepared = [{'time_unix': 5, 'sha256': {v: 'a' * 64 for v in audit.VARIANTS}}]
        result = audit.attempt_ledger(row, marker, start, end, prepared, bounds, audit.Checks())
        self.assertTrue(result['preparation_state_unavailable_after_interruption'])
        with self.assertRaisesRegex(AssertionError, 'preparation_event_order'):
            audit.attempt_ledger(row, marker, [], end, prepared, bounds, audit.Checks())
        with self.assertRaisesRegex(AssertionError, 'preparation_event_count'):
            audit.attempt_ledger(row, marker, start, end, prepared * 2, bounds, audit.Checks())

    def test_generation_metadata_rejects_changed_seed_settings_or_length(self):
        model = {'model_id': 'fixture-model', 'revision': 'a' * 40, 'alias': 'test'}
        generation = {'max_new_tokens': 4, 'temperature': .8, 'top_p': .95, 'top_k': 50}
        template = {'chat_template': 'fixture', 'revision': model['revision'],
                    'generation_eos_token_id': 2, 'generation_pad_token_id': 2}
        metadata = {'model': model, 'seed': 7, 'dtype': 'bfloat16', 'device': 'cpu',
                    'add_special_tokens_after_template': False, 'output_token_ids': [3, 2],
                    'generated_tokens': 2, 'input_tokens': 9, 'hit_token_limit': False, 'seconds': 1.2,
                    'chat_template_sha256': audit.hashlib.sha256(b'fixture').hexdigest(),
                    'rendered_prompt_sha256': 'a' * 64, 'input_token_ids_sha256': 'b' * 64,
                    'eos_token_id': 2, 'pad_token_id': 2,
                    'effective_generation_config': {**generation, 'do_sample': True, 'repetition_penalty': 1.0,
                                                    'pad_token_id': 2, 'eos_token_id': 2}}
        row = {'sample': {'id': 'a', 'generation_seed': 7}, 'cover_text': 'fixture', 'failure': None,
               'generation_metadata': metadata}
        self.assertEqual(audit.inspect_generation_metadata(row, model, generation, template, audit.Checks())['generated_tokens'], 2)
        for field, value in [('seed', 8), ('generated_tokens', 3), ('chat_template_sha256', 'c' * 64)]:
            changed = copy.deepcopy(row)
            changed['generation_metadata'][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit.inspect_generation_metadata(changed, model, generation, template, audit.Checks())
        changed = copy.deepcopy(row)
        changed['generation_metadata']['effective_generation_config']['temperature'] = .8000000000001
        with self.assertRaises(AssertionError):
            audit.inspect_generation_metadata(changed, model, generation, template, audit.Checks())
        changed = copy.deepcopy(row)
        changed['generation_metadata']['pad_token_id'] = 3
        changed['generation_metadata']['effective_generation_config']['pad_token_id'] = 3
        with self.assertRaisesRegex(AssertionError, 'native_pad_frozen_identity'):
            audit.inspect_generation_metadata(changed, model, generation, template, audit.Checks())

    def test_decode_known_group_bits(self):
        bits, used = audit.independent_decode(tiny_book(), 'quiet\tbright\nplain tail.', 2)
        self.assertEqual(bits, '10')
        self.assertEqual([i for i, _, _ in used], [0, 1])

    def test_decode_rejects_layout_and_unused_changes(self):
        for text in ['quiet bright\nplain tail.', 'quiet\tbright\nsimple tail.', 'quiet\tbright\nplain changed.']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit.independent_decode(tiny_book(), text, 2)

    def test_decode_rejects_ambiguous_membership(self):
        book = tiny_book()
        book['positions'][0]['groups'][0].append('quiet')
        with self.assertRaisesRegex(ValueError, 'unique group'):
            audit.independent_decode(book, 'quiet\tbright\nplain tail.', 2)

    def test_encoder_order_is_recovered_from_sorted_storage(self):
        original = tiny_book()
        encoded = json.dumps(original, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        stored = json.loads(json.dumps(original, sort_keys=True))
        recovered = audit.ordered_key_bytes(stored)
        self.assertEqual(recovered, encoded)
        self.assertEqual(zlib.compress(recovered, 9), zlib.compress(encoded, 9))
        wrong = copy.deepcopy(stored)
        wrong['positions'][0]['unexpected'] = 1
        with self.assertRaisesRegex(ValueError, 'position fields'):
            audit.ordered_key_bytes(wrong)

    def test_auc_half_credit_ties_and_frozen_thresholds(self):
        rows = [{'label': 0, 'score': .1}, {'label': 0, 'score': .5},
                {'label': 1, 'score': .5}, {'label': 1, 'score': .9}]
        self.assertEqual(audit.prediction_metrics(rows, .5), {
            'accuracy': .75, 'auc': .875, 'fpr_at_development_threshold': .5,
            'tpr_at_development_threshold': 1.0})
        self.assertEqual(audit.prediction_metrics(rows, .6)['accuracy'], .75)
        self.assertEqual(audit.prediction_metrics(rows, .6)['fpr_at_development_threshold'], 0)
        with self.assertRaises(ValueError):
            audit.prediction_metrics([{'label': 1, 'score': float('nan')}], .5)

    def test_all_failures_keep_attempt_and_bit_denominators(self):
        rows = [{'sample': {'payload_bits': 8}, 'failure': 'generation_failed', 'variants': {}},
                {'sample': {'payload_bits': 16}, 'failure': None,
                 'variants': {'fixed_quality': {'metrics': None, 'error': 'capacity_failed'}}}]
        result = audit.reliability(rows, 'fixed_quality')
        self.assertEqual(result['attempted_samples'], 2)
        self.assertEqual(result['attempted_payload_bits'], 24)
        self.assertEqual(result['failed_samples'], 2)
        self.assertEqual(result['full_success_rate'], 0)
        self.assertIsNone(result['mean_successful_bpw_stego'])

    def test_incomplete_gate_does_not_read_generated_json(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with patch.object(audit, 'read', side_effect=AssertionError('Unexpected JSON read')):
                with self.assertRaisesRegex(AssertionError, 'completed_gate_files'):
                    audit.completed_headers(run, audit.Checks())

    def test_portable_repo_mapping_and_honest_missing_resource_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / 'new_clone'
            path = repo / 'scripts/crossmodel_quality.py'
            path.parent.mkdir(parents=True)
            path.write_text('fixture helper\n')
            missing = str(Path(directory) / 'absent_lexical_resource')
            frozen = {'resource_sha256': {'/old/clone/scripts/crossmodel_quality.py': audit.sha(path), missing: '0' * 64}}
            result = audit.resource_checks(repo, frozen, False, audit.Checks())
            self.assertEqual(len(result['verified_files']), 1)
            self.assertTrue(result['verified_files'][0]['repository_remapped'])
            self.assertFalse(result['all_declared_resource_files_verified'])
            self.assertEqual(len(result['unavailable_external_files']), 1)
            with self.assertRaisesRegex(AssertionError, 'required_resource_available'):
                audit.resource_checks(repo, frozen, True, audit.Checks())


if __name__ == '__main__':
    unittest.main()
