"""Synthetic bookkeeping and mutation tests; no actual encode/decode/model calls."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RUN = module('first80_runner_test', 'scripts/run_prospective_development_transport_first80_v1.py')
AUDIT = module('first80_auditor_test', 'scripts/audit_prospective_development_transport_first80_v1.py')
P = RUN.parent(ROOT)


def fixture():
    prior, current = [], []
    for model in ('qwen05', 'tinyllama11', 'falcon3_1b'):
        for order in range(1, 101):
            row = {'model_alias': model, 'record_id': f'synthetic-{order}', 'source_group_id': f'synthetic-source-{order}', 'source_order': order, 'model': model, 'revision': 'fixture', 'target_words': 250, 'genre': 'synthetic', 'topic': 'synthetic', 'split': 'development', 'identity': f'{model}-{order}', 'prompt_sha256': 'fixture', 'status': 'completed' if model == 'tinyllama11' and order <= 60 else 'not_attempted', 'cover_text': 'Synthetic fixture only.' if model == 'tinyllama11' and order <= 60 else None, 'attempt_sha256': f'fixture-{order}' if model == 'tinyllama11' and order <= 60 else None}
            prior.append(row)
            updated = dict(row)
            if model == 'tinyllama11' and 61 <= order <= 80:
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


class First80SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.prior, self.current, self.old, self.delta = fixture()

    def test_only_fixed_new_pair_prefix_enters_evaluator(self):
        rows = RUN.new_records_only(P, self.current, self.prior)
        self.assertEqual([r['source_order'] for r in rows], list(range(61, 81)))
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(r['model_alias'] == 'tinyllama11' for r in rows))

    def test_old_cover_mutation_rejected(self):
        self.current[100]['cover_text'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'unchanged input'):
            RUN.new_records_only(P, self.current, self.prior)

    def test_new_identity_or_omission_rejected(self):
        self.current[160]['identity'] = 'different'
        with self.assertRaisesRegex(ValueError, 'metadata'):
            RUN.new_records_only(P, self.current, self.prior)
        with self.assertRaises(ValueError):
            RUN.new_records_only(P, self.current[:-1], self.prior)

    def test_new80_not_completed_rejected(self):
        self.current[179]['status'] = 'not_attempted'
        with self.assertRaises(ValueError):
            RUN.new_records_only(P, self.current, self.prior)

    def test_full15300_reuses1440_exact_old_rows(self):
        before = copy.deepcopy(self.old)
        merged = RUN.merge_cells(P, self.old, self.delta)
        self.assertEqual(self.old, before)
        self.assertEqual(len(merged), 15300)
        old_public = {r['row_id']: P.canonical(r) for r in self.old if r['encoder_called'] is True}
        self.assertEqual(len(old_public), 1440)
        self.assertEqual({r['row_id']: P.canonical(r) for r in merged if r['row_id'] in old_public}, old_public)
        self.assertEqual(sum(r['encoder_called'] is True for r in merged), 1920)
        checked = AUDIT.validate_reuse(self.prior, self.current, self.old, merged, self.old, merged)
        self.assertEqual(checked['new_encoder_calls'], 480)

    def test_zero_and_null_stay_distinct(self):
        merged = RUN.merge_cells(P, self.old, self.delta)
        self.assertEqual(sum(r['capacity_bits'] == 0 for r in merged), 1920)
        self.assertEqual(sum(r['capacity_bits'] is None for r in merged), 13380)
        self.assertEqual(sum(r['execution_status'] == 'not_run_cover_unavailable' for r in merged), 11220)
        self.assertEqual(sum(r['execution_status'] == 'not_run_resource_blocked' for r in merged), 2160)

    def test_omitted_or_old_delta_rejected(self):
        with self.assertRaises(ValueError):
            RUN.merge_cells(P, self.old, self.delta[:-1])
        self.delta[0]['source_order'] = 60
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
        self.assertEqual(checked['new_encoder_calls'], 479)

    def test_single_encode_entrypoint_is_delta_only_and_no_model_import(self):
        tree = ast.parse((ROOT / 'scripts/run_prospective_development_transport_first80_v1.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'evaluate']
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[0], ast.Name)
        self.assertEqual(calls[0].args[0].id, 'delta')
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(any(x.name.split('.')[0] in {'torch', 'transformers', 'anthropic', 'openai'} for x in n.names) for n in imports))

    def test_actual_receiver_input_written_from_new_jobs_only(self):
        tree = ast.parse((ROOT / 'scripts/run_prospective_development_transport_first80_v1.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'write_rows' and len(n.args) == 2
                 and isinstance(n.args[0], ast.BinOp) and isinstance(n.args[0].left, ast.Name)
                 and n.args[0].left.id == 'recipient']
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[1], ast.Name)
        self.assertEqual(calls[0].args[1].id, 'new_jobs')

    def test_cumulative_recipient_reuses_prior150_without_replay(self):
        prior = [{'row_id': f'old-{i}', 'body': 'immutable'} for i in range(150)]
        new = [{'row_id': f'new-{i}', 'body': 'new'} for i in range(3)]
        outcomes = [{'row_id':r['row_id'], 'outcome':'synthetic'} for r in prior + new]
        cells = [{'row_id':r['row_id'], 'model_alias':'tinyllama11', 'record_id':'new-record'} for r in new]
        keys = {('tinyllama11','new-record')}
        self.assertEqual(AUDIT.validate_receiver_reuse(prior, outcomes[:150], new, outcomes[150:], prior+new, outcomes, keys, cells),3)
        changed=copy.deepcopy(outcomes);changed[0]['outcome']='changed'
        with self.assertRaisesRegex(ValueError,'exact prior'):
            AUDIT.validate_receiver_reuse(prior, outcomes[:150], new, outcomes[150:], prior+new, changed, keys, cells)
        with self.assertRaises(ValueError):
            AUDIT.validate_receiver_reuse(prior, outcomes[:150], prior+new, outcomes, prior+prior+new, outcomes[:150]+outcomes, keys, cells)
        with self.assertRaisesRegex(ValueError,'outside delta'):
            AUDIT.validate_receiver_reuse(prior, outcomes[:150], new, outcomes[150:], prior+new, outcomes, set(), cells)

    def test_cumulative_receiver_omissions_order_and_duplicates_refused(self):
        prior=[{'row_id':f'old-{i}'} for i in range(150)]
        new=[{'row_id':'new0'},{'row_id':'new1'}]
        cells=[{'row_id':r['row_id'],'model_alias':'tinyllama11','record_id':'new'} for r in new]
        keys={('tinyllama11','new')}
        for prior_jobs, prior_results, new_jobs, new_results, cumulative_jobs, cumulative_results in [
            (prior[:-1],prior[:-1],new,new,prior[:-1]+new,prior[:-1]+new),
            (prior,prior,new,new[::-1],prior+new,prior+new[::-1]),
            (prior,prior,new,new,prior+new[:-1],prior+new),
            (prior,prior,new+new[:1],new+new[:1],prior+new+new[:1],prior+new+new[:1])]:
            with self.assertRaises(ValueError):
                AUDIT.validate_receiver_reuse(prior_jobs,prior_results,new_jobs,new_results,cumulative_jobs,cumulative_results,keys,cells)

    def binding_fixture(self):
        dictionaries={v:{'version':'synthetic-'+v,'classes':[{'id':'fixture','members':['clear','plain']}]} for v in ('v1','v2')}
        class Dictionary:
            def __init__(self,value):self.sha256=AUDIT.digest(AUDIT.canonical(value))
        class Protocol:
            def __init__(self,**kw):self.sha256=AUDIT.digest(AUDIT.canonical(AUDIT.public_protocol(kw)))
        class Shared:pass
        Shared.Protocol=Protocol
        plan={'arms':{'fixed':[{'id':x} for x in sorted(AUDIT.FIXED_ARMS)],'public_arms':sorted(AUDIT.PUBLIC_ARMS)}}
        payload={'rows':[{'record_id':f'synthetic-{order}','primary_payload_length':length,'payload_bits':'0'*length,'secondary16_bits':'0'*16}
                         for order in range(1,101) for length in (128,256,512)]}
        # A non-ASCII record ensures source hashes use producer UTF-8, not oracle ASCII canonical form.
        records=copy.deepcopy(self.current);records[0]['topic']='синтетический пример'
        bound=RUN.input_bound_grid(P,records,payload,plan,Shared,{v:Dictionary(d) for v,d in dictionaries.items()})
        cells=[]
        pmap={(r['record_id'],r['primary_payload_length']):r for r in payload['rows']}
        for record in records:
            for length in (128,256,512):
                for grid in ('primary','secondary16'):
                    for arm in (['original_native'] if grid=='primary' else [])+[a['id'] for a in plan['arms']['fixed']]+plan['arms']['public_arms']:
                        cells.append(P.base_cell(record,pmap[(record['record_id'],length)],grid,arm,arm in AUDIT.PUBLIC_ARMS)[0])
        return bound,records,cells,dictionaries

    def test_pre_encode_binding_all15300_conditions(self):
        bound,records,cells,dictionaries=self.binding_fixture()
        actions=AUDIT.validate_input_bound_rows(bound,records,cells,dictionaries)
        self.assertEqual(actions,{'reuse_exact_prior_outcome':1440,'new_encode_once':480,'not_run_resource_blocked':2160,'not_run_cover_unavailable':11220})
        self.assertNotIn('capacity_bits',bound[0])
        self.assertNotIn('full_exact',bound[0])

    def test_input_binding_rejects_payload_metadata_resource_and_source_drift(self):
        bound,records,cells,dictionaries=self.binding_fixture()
        for field,value in [('payload_sha256','different'),('source_record_sha256','different'),('source_order',999),('planned_action','new_encode_once'),('dictionary_sha256','invented'),('unexpected_outcome',True)]:
            mutated=copy.deepcopy(bound);mutated[0][field]=value
            with self.assertRaises(ValueError):AUDIT.validate_input_bound_rows(mutated,records,cells,dictionaries)
        public=next(i for i,r in enumerate(bound) if r['arm'] in AUDIT.PUBLIC_ARMS)
        for field in ('dictionary_sha256','protocol_sha256'):
            mutated=copy.deepcopy(bound);mutated[public][field]='wrong'
            with self.assertRaises(ValueError):AUDIT.validate_input_bound_rows(mutated,records,cells,dictionaries)
        records[0]['topic']='changed source metadata'
        with self.assertRaisesRegex(ValueError,'full record hash'):AUDIT.validate_input_bound_rows(bound,records,cells,dictionaries)

    def test_input_binding_missing_duplicate_and_row_order_refused(self):
        bound,records,cells,dictionaries=self.binding_fixture()
        for mutated in (bound[:-1],bound[:-1]+bound[:1],[bound[1],bound[0]]+bound[2:]):
            with self.assertRaises(ValueError):AUDIT.validate_input_bound_rows(mutated,records,cells,dictionaries)

    def test_changed_prebound_file_refused_before_encoder_entrypoint(self):
        with tempfile.TemporaryDirectory(prefix='first80-guard-') as temp:
            root=Path(temp);directory=root/RUN.BINDING;directory.mkdir(parents=True)
            wanted={name:b'fixed synthetic binding' for name in RUN.BINDING_FILES}
            for name in RUN.BINDING_FILES:(directory/name).write_bytes(wanted[name])
            (directory/RUN.BINDING_FILES[0]).write_bytes(b'tampered synthetic binding')
            activation={'target':{'output':'never-created-output'}}
            old=(self.prior,self.old,self.old,[],[])
            dep=(None,None,None,None,None,None,None,None)
            with patch.object(RUN,'preflight',return_value=(P,dep,activation,old)), \
                 patch.object(RUN,'snapshot_records',return_value=(self.current,{},b'{}',root)), \
                 patch.object(RUN,'binding_material',return_value=wanted), \
                 patch.object(P,'evaluate',side_effect=AssertionError('encoder must remain untouched')) as encode:
                with self.assertRaisesRegex(ValueError,'Input binding differs'):
                    RUN.run(root,root,root/'never-created-output',root/'unused-receipt',root/'unused-intent')
                encode.assert_not_called()
            self.assertFalse((root/'never-created-output').exists())

    def test_independent_pure_oracle_algorithms_unchanged_from_frozen_first60(self):
        old=ast.parse((ROOT/'scripts/audit_prospective_development_transport_first60_v1.py').read_text())
        new=ast.parse((ROOT/'scripts/audit_prospective_development_transport_first80_v1.py').read_text())
        names={'canonical','digest','dictionary_canonical','public_protocol','public_analysis','hkdf_sha256','public_oracle','verify_canonical_members','cell_identity','validate_cell_grid','check_summary'}
        old_funcs={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef) and n.name in names}
        new_funcs={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef) and n.name in names}
        self.assertEqual(set(old_funcs),names)
        self.assertEqual(old_funcs,new_funcs)


if __name__ == '__main__':
    unittest.main()
