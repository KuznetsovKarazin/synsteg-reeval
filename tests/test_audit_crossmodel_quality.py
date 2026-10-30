"""Synthetic arithmetic/coverage fixtures; no model or experiment data is read."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SPEC = importlib.util.spec_from_file_location(
    'audit_crossmodel_transfer', ROOT / 'scripts/audit_crossmodel_transfer.py')
primary = importlib.util.module_from_spec(PRIMARY_SPEC)
PRIMARY_SPEC.loader.exec_module(primary)
SPEC = importlib.util.spec_from_file_location(
    'independent_crossmodel_quality_audit', ROOT / 'scripts/audit_crossmodel_quality.py')
audit = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {'audit_crossmodel_transfer': primary}):
    SPEC.loader.exec_module(audit)


def score_row(tokens=5, mean_nll=2.0):
    return {'score': math.exp(mean_nll), 'details': {
        'input_tokens': tokens, 'scored_input_tokens': tokens,
        'predicted_tokens': tokens - 1, 'truncated': False,
        'ppl_policy': 'sliding_window', 'sum_nll': mean_nll * (tokens - 1),
        'mean_nll': mean_nll}}


def measurement(ppl, predicted_tokens=1):
    return {'ppl': ppl, 'mean_nll': math.log(ppl),
            'predicted_tokens': predicted_tokens}


class IndependentQualityAuditFixtures(unittest.TestCase):
    def test_geometric_ratio_is_paired_document_mean_not_arithmetic_or_token_weighted(self):
        left = [measurement(2, 1), measurement(16, 999), measurement(4, 10)]
        right = [measurement(8, 1), measurement(8, 999), measurement(4, 10)]
        effect = audit.paired_effect(left, right)
        self.assertAlmostEqual(effect['geometric_mean_ppl_ratio'], .5 ** (1 / 3))
        self.assertAlmostEqual(effect['document_mean_nll_delta']['sum'], -math.log(2))
        self.assertEqual(effect['document_mean_nll_delta']['n'], 3)
        self.assertEqual(effect['document_mean_nll_delta']['median'], 0)
        self.assertEqual(effect['document_ppl_delta'], {
            'n': 3, 'sum': 2, 'mean': 2 / 3, 'median': 0, 'min': -6, 'max': 8})
        self.assertEqual([effect[k] for k in ('lower', 'equal_within_1e_12', 'higher')], [1, 1, 1])
        arithmetic_ratio = sum(a['ppl'] / b['ppl'] for a, b in zip(left, right)) / 3
        weighted_ratio = math.exp(sum((a['mean_nll'] - b['mean_nll']) * a['predicted_tokens']
                                     for a, b in zip(left, right)) / 1010)
        self.assertNotAlmostEqual(effect['geometric_mean_ppl_ratio'], arithmetic_ratio)
        self.assertNotAlmostEqual(effect['geometric_mean_ppl_ratio'], weighted_ratio)

    def test_reversing_arms_inverts_ratio_and_lower_higher_counts(self):
        left, right = [measurement(2), measurement(3)], [measurement(8), measurement(6)]
        forward, reverse = audit.paired_effect(left, right), audit.paired_effect(right, left)
        self.assertAlmostEqual(forward['geometric_mean_ppl_ratio'] * reverse['geometric_mean_ppl_ratio'], 1)
        self.assertEqual((forward['lower'], forward['higher']), (2, 0))
        self.assertEqual((reverse['lower'], reverse['higher']), (0, 2))

    def test_empty_pairs_retain_zero_counts_and_no_ratio(self):
        result = audit.paired_effect([], [])
        self.assertIsNone(result['geometric_mean_ppl_ratio'])
        self.assertEqual([result[k] for k in ('lower', 'equal_within_1e_12', 'higher')], [0, 0, 0])
        for field in ('document_mean_nll_delta', 'document_ppl_delta'):
            self.assertEqual(result[field], {'n': 0, 'sum': 0, 'mean': None,
                                            'median': None, 'min': None, 'max': None})

    def test_near_zero_counts_partition_all_pairs(self):
        deltas = [-2e-12, -.5e-12, 0, .5e-12, 2e-12]
        left = [{'mean_nll': 1 + delta, 'ppl': math.exp(1 + delta)} for delta in deltas]
        result = audit.paired_effect(left, [measurement(math.e) for _ in deltas])
        self.assertEqual([result[k] for k in ('lower', 'equal_within_1e_12', 'higher')], [1, 3, 1])
        self.assertEqual(sum(result[k] for k in ('lower', 'equal_within_1e_12', 'higher')), 5)

    def test_unpaired_or_inconsistent_nll_ppl_measurements_fail(self):
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            audit.paired_effect([measurement(2)], [])
        with self.assertRaisesRegex(ValueError, 'disagree'):
            audit.paired_effect([{'mean_nll': 1.0, 'ppl': 2.0}], [measurement(2)])

    def test_exact_n_minus_one_coverage_accepts_short_and_long_documents(self):
        for tokens in (2, 1025, 4099):
            with self.subTest(tokens=tokens):
                row = score_row(tokens)
                checked = audit.validate_score(row, audit.Checks())
                self.assertEqual(checked['input_tokens'], tokens)
                self.assertEqual(checked['predicted_tokens'], tokens - 1)
                self.assertEqual(checked['sum_nll'], 2 * (tokens - 1))

    def test_missing_duplicate_or_noninteger_targets_fail(self):
        for field, value in [('predicted_tokens', 3), ('predicted_tokens', 5),
                             ('predicted_tokens', 4.0), ('predicted_tokens', True),
                             ('input_tokens', 5.0), ('input_tokens', True),
                             ('scored_input_tokens', 5.0)]:
            with self.subTest(field=field, value=value):
                row = score_row()
                row['details'][field] = value
                with self.assertRaisesRegex(AssertionError, 'full_text_coverage'):
                    audit.validate_score(row, audit.Checks())

    def test_truncated_prefix_or_incomplete_input_coverage_fail(self):
        for field, value in [('truncated', True), ('truncated', 0), ('truncated', None),
                             ('scored_input_tokens', 4), ('ppl_policy', 'prefix')]:
            with self.subTest(field=field, value=value):
                row = score_row()
                row['details'][field] = value
                with self.assertRaisesRegex(AssertionError, 'full_text_coverage'):
                    audit.validate_score(row, audit.Checks())

    def test_nll_sum_and_ppl_arithmetic_must_both_match(self):
        for location, field, value in [('details', 'sum_nll', 8.5),
                                       ('details', 'mean_nll', 2.1),
                                       ('row', 'score', math.exp(2.1))]:
            with self.subTest(field=field):
                row = score_row()
                (row if location == 'row' else row['details'])[field] = value
                with self.assertRaises(AssertionError):
                    audit.validate_score(row, audit.Checks())

    def test_nonfinite_boolean_or_negative_scores_fail(self):
        for field in ('score', 'sum_nll', 'mean_nll'):
            for value in (float('nan'), float('inf'), True, -1):
                with self.subTest(field=field, value=value):
                    row = copy.deepcopy(score_row())
                    (row if field == 'score' else row['details'])[field] = value
                    with self.assertRaisesRegex(AssertionError, 'finite_score'):
                        audit.validate_score(row, audit.Checks())


if __name__ == '__main__':
    unittest.main()
