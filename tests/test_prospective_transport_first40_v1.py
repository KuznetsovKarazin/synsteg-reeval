"""Synthetic bookkeeping and mutation tests; no actual encode/decode/model calls."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RUN = module('first40_runner_test', 'scripts/run_prospective_development_transport_first40_v1.py')
AUDIT = module('first40_auditor_test', 'scripts/audit_prospective_development_transport_first40_v1.py')
P = RUN.parent(ROOT)


def fixture():
    prior, current = [], []
    for model in ('qwen05', 'tinyllama11', 'falcon3_1b'):
        for order in range(1, 101):
            row = {'model_alias': model, 'record_id': f'synthetic-{order}', 'source_group_id': f'synthetic-source-{order}', 'source_order': order, 'model': model, 'revision': 'fixture', 'target_words': 250, 'genre': 'synthetic', 'topic': 'synthetic', 'split': 'development', 'identity': f'{model}-{order}', 'prompt_sha256': 'fixture', 'status': 'completed' if model == 'tinyllama11' and order <= 30 else 'not_attempted', 'cover_text': 'Synthetic fixture only.' if model == 'tinyllama11' and order <= 30 else None, 'attempt_sha256': f'fixture-{order}' if model == 'tinyllama11' and order <= 30 else None}
            prior.append(row)
            updated = dict(row)
            if model == 'tinyllama11' and 31 <= order <= 40:
                updated.update(status='completed', cover_text='New synthetic fixture only.', attempt_sha256=f'fixture-{order}')
            current.append(updated)
    old_cells, new_cells = [], []
    public = sorted(AUDIT.PUBLIC_ARMS)
    for old, new in zip(prior, current):
        for n in range(51):
            is_public = n < 24
            arm = public[n % 4] if is_public else 'original_native'
            row = {'row_id': f"{old['identity']}-{n}", 'model_alias': old['model_alias'], 'record_id': old['record_id'], 'source_order': old['source_order'], 'arm': arm, 'encoder_called': True if is_public and old['status'] == 'completed' else None, 'execution_status': 'insufficient_capacity' if is_public and old['status'] == 'completed' else 'not_run_resource_blocked' if old['status'] == 'completed' else 'not_run_cover_unavailable', 'capacity_bits': 0 if is_public and old['status'] == 'completed' else None, 'full_exact': False if is_public and old['status'] == 'completed' else None}
            old_cells.append(row)
            if old['status'] != new['status']:
                new_cells.append(dict(row, encoder_called=True if is_public else None, execution_status='insufficient_capacity' if is_public else 'not_run_resource_blocked', capacity_bits=0 if is_public else None, full_exact=False if is_public else None))
    return prior, current, old_cells, new_cells


class First40SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.prior, self.current, self.old, self.delta = fixture()

    def test_only_fixed_new_pair_prefix_enters_evaluator(self):
        rows = RUN.new_records_only(P, self.current, self.prior)
        self.assertEqual([r['source_order'] for r in rows], list(range(31, 41)))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r['model_alias'] == 'tinyllama11' for r in rows))

    def test_old_cover_mutation_rejected(self):
        self.current[100]['cover_text'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'unchanged input'):
            RUN.new_records_only(P, self.current, self.prior)

    def test_new_identity_or_omission_rejected(self):
        self.current[130]['identity'] = 'different'
        with self.assertRaisesRegex(ValueError, 'metadata'):
            RUN.new_records_only(P, self.current, self.prior)
        with self.assertRaises(ValueError):
            RUN.new_records_only(P, self.current[:-1], self.prior)

    def test_new40_not_completed_rejected(self):
        self.current[139]['status'] = 'not_attempted'
        with self.assertRaises(ValueError):
            RUN.new_records_only(P, self.current, self.prior)

    def test_full15300_reuses720_exact_old_rows(self):
        before = copy.deepcopy(self.old)
        merged = RUN.merge_cells(P, self.old, self.delta)
        self.assertEqual(self.old, before)
        self.assertEqual(len(merged), 15300)
        old_public = {r['row_id']: P.canonical(r) for r in self.old if r['encoder_called'] is True}
        self.assertEqual(len(old_public), 720)
        self.assertEqual({r['row_id']: P.canonical(r) for r in merged if r['row_id'] in old_public}, old_public)
        self.assertEqual(sum(r['encoder_called'] is True for r in merged), 960)
        checked = AUDIT.validate_reuse(self.prior, self.current, self.old, merged, self.old, merged)
        self.assertEqual(checked['new_encoder_calls'], 240)

    def test_zero_and_null_stay_distinct(self):
        merged = RUN.merge_cells(P, self.old, self.delta)
        self.assertEqual(sum(r['capacity_bits'] == 0 for r in merged), 960)
        self.assertEqual(sum(r['capacity_bits'] is None for r in merged), 14340)
        self.assertEqual(sum(r['execution_status'] == 'not_run_cover_unavailable' for r in merged), 13260)
        self.assertEqual(sum(r['execution_status'] == 'not_run_resource_blocked' for r in merged), 1080)

    def test_omitted_or_old_delta_rejected(self):
        with self.assertRaises(ValueError):
            RUN.merge_cells(P, self.old, self.delta[:-1])
        self.delta[0]['source_order'] = 30
        with self.assertRaises(ValueError):
            RUN.merge_cells(P, self.old, self.delta)

    def test_old_attempt_cannot_be_overwritten(self):
        first_new_id = self.delta[0]['row_id']
        next(r for r in self.old if r['row_id'] == first_new_id)['encoder_called'] = True
        with self.assertRaises(ValueError):
            RUN.merge_cells(P, self.old, self.delta)

    def test_independent_audit_rejects_old_outcome_mutation(self):
        merged = copy.deepcopy(RUN.merge_cells(P, self.old, self.delta))
        next(r for r in merged if r['encoder_called'] is True)['full_exact'] = True
        with self.assertRaisesRegex(ValueError, 'old row'):
            AUDIT.validate_reuse(self.prior, self.current, self.old, merged, self.old, merged)

    def test_independent_audit_rejects_changed_before_receiver(self):
        merged = RUN.merge_cells(P, self.old, self.delta)
        before = copy.deepcopy(merged)
        before[0]['capacity_bits'] = 0
        with self.assertRaises(ValueError):
            AUDIT.validate_reuse(self.prior, self.current, self.old, merged, self.old, before)

    def test_analysis_error_counts_condition_without_invented_encoder_call(self):
        self.delta[0].update(encoder_called=False, execution_status='implementation_error', capacity_bits=None)
        merged = RUN.merge_cells(P, self.old, self.delta)
        checked = AUDIT.validate_reuse(self.prior, self.current, self.old, merged, self.old, merged)
        self.assertEqual(checked['new_encoder_calls'], 239)

    def test_single_encode_entrypoint_is_delta_only_and_no_model_import(self):
        tree = ast.parse((ROOT / 'scripts/run_prospective_development_transport_first40_v1.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'evaluate']
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[0], ast.Name)
        self.assertEqual(calls[0].args[0].id, 'delta')
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(any(x.name.split('.')[0] in {'torch', 'transformers', 'anthropic', 'openai'} for x in n.names) for n in imports))


if __name__ == '__main__':
    unittest.main()
