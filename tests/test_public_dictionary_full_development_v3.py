"""Synthetic source-bound capacity grids; no generation or transport calls."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
INPUTS = Path(os.environ.get('CAPACITY_V3_INPUTS',str(ROOT)))
spec=importlib.util.spec_from_file_location('capacity_v3',ROOT/'scripts/audit_public_dictionary_full_development_v3.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def plan():
    return dict(experiment_id=a.ID,scope='posthoc_mechanical_capacity_only',models=['tinyllama11','qwen05'],
                source_orders=list(range(1,101)),primary_payload_bits=a.PRIMARY,secondary_payload_bits=16,
                secondary_repeated_per_primary_parent=True,frames=a.FRAMES,min_group_size=1,max_bits_per_slot=2,
                resource_pins=a.PINS,runtime=a.RUNTIME)


class CapacityCases(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'evidence'
        for path in a.PINS:
            dst=self.root/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(INPUTS/path,dst)
        self.sources=[a.parse(line) for line in (self.root/a.SOURCE).read_bytes().splitlines() if line.strip()]
        self.grid=[]
        for alias in a.MODELS:
            for n,row in enumerate(self.sources,1):
                status='reserved_outcome_unknown' if alias=='tinyllama11' and n==82 else 'not_attempted'
                self.grid.append(dict(model_alias=alias,source_order=n,record_id=row['record_id'],status=status))
        self.index=[dict(model_alias=r['model_alias'],source_order=r['source_order'],record_id=r['record_id'],status=r['status'],attempt_path=None)
                    for r in self.grid if r['model_alias'] in plan()['models']]
        self.evidence=[];self.snapshot='fixtures/synthetic-mixed300.json';self.seal()

    def write(self,path,value):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')

    def complete(self,alias,order,cover):
        src=self.sources[order-1];identity=a.canonical(['local',a.MODELS[alias],src['record_id']])
        path=f'fixtures/attempts/{alias}/{identity}.json'
        self.write(path,dict(status='completed',provider='local',model=a.MODELS[alias],identity=identity,automatic_retries=0,
                             record={k:src[k] for k in ['genre','prompt_sha256','record_id','source_group_id','split','target_words','topic']},
                             result={'cover_text':cover,'returned_model':a.MODELS[alias],'revision':a.REVISIONS[alias],
                                     'dtype':'float32','seed':int(a.canonical([20261024,a.MODELS[alias],src['record_id']])[:8],16)}))
        if alias!='tinyllama11' or order>=92:
            self.evidence.append(dict(model_alias=alias,source_order=order,status='completed',attempt_sha256=a.sha((self.root/path).read_bytes()),cover_sha256=a.sha(cover.encode())))
        next(r for r in self.grid if r['model_alias']==alias and r['source_order']==order)['status']='completed'
        item=next(r for r in self.index if r['model_alias']==alias and r['source_order']==order)
        item.update(status='completed',attempt_path=path);self.seal();return path

    def seal(self):
        guard=a.parse((self.root/a.GUARD).read_bytes())
        self.write(self.snapshot,dict(schema='mixed-development-full300-merger-v2-result',snapshot_ref='f'*40,full_denominator=300,
                                     records=self.grid,counts=dict(a.Counter(r['status'] for r in self.grid)),scheduling_overlays=guard['blocked_baseline_intents'],new_shard_evidence=self.evidence))
        self.manifest=dict(schema=a.ID+'-input',actual_git_readback=True,snapshot_commit='f'*40,
                           files_sha256=a.inventory(self.root),snapshot_path=self.snapshot,cover_sources=self.index)

    def evaluate(self):return a.evaluate(self.root,self.manifest,plan())

    def reject(self):
        with self.assertRaises((ValueError,KeyError,TypeError)):self.evaluate()

    def test_all_missing_keeps_every_denominator_and_null(self):
        result=self.evaluate()
        self.assertEqual(len(result['dictionary_rows']),400)
        self.assertEqual(len(result['frame_conditions']),4800)
        for model in result['models'].values():
            self.assertEqual(model['cover_denominator'],100)
            self.assertEqual(model['available_saved_covers'],0)
            self.assertIsNone(model['v2_raw_bit_loss_fraction'])
        self.assertTrue(all(r['capacity_bits'] is None for r in result['dictionary_rows']))
        row=next(r for r in result['lengths'] if r['model_alias']=='tinyllama11' and r['source_order']==91)
        self.assertEqual(row['cover_status'],'not_attempted');self.assertEqual(row['scheduling_status'],'blocked_reserved_intent_evidence_gap')

    def test_measured_zero_is_distinct_from_unavailable(self):
        self.complete('qwen05',1,'zxqzxq qzxqzx.')
        result=self.evaluate();rows=[r for r in result['dictionary_rows'] if r['model_alias']=='qwen05' and r['source_order']==1]
        self.assertTrue(all(r['capacity_bits']==0 and r['capacity_status']=='measured' for r in rows))
        self.assertEqual(result['models']['qwen05']['dictionary_aggregates']['v1']['zero_capacity_covers'],1)
        self.assertEqual(result['models']['tinyllama11']['available_saved_covers'],0)

    def test_exact_frozen_analyze_text_counts(self):
        cover='Matter affair concern big large rapid quick dark matter.';self.complete('tinyllama11',1,cover)
        result=self.evaluate();core=a.core_module(self.root)
        for did in ['v1','v2']:
            dictionary=core.PublicDictionary.from_dict(a.parse((self.root/f'data/shared_dictionary_recovery_{did}.json').read_bytes()))
            expected=core.analyze_text(cover,dictionary,core.Protocol(min_group_size=1,max_bits_per_slot=2,framing='plain_length16'))
            row=next(r for r in result['dictionary_rows'] if r['model_alias']=='tinyllama11' and r['source_order']==1 and r['dictionary_id']==did)
            self.assertEqual(row['capacity_bits'],expected.capacity_bits)
            self.assertEqual(row['capacity_bits'],sum(s['width'] for s in row['slots']))

    def test_fixed_thresholds_and_secondary_parent_denominator(self):
        rows=a.frame_rows('tinyllama11',1,'v1',32,'measured')
        self.assertEqual(len(rows),12)
        secondary=[r for r in rows if r['payload_kind']=='secondary16' and r['framing']=='plain_length16']
        self.assertEqual(len(secondary),3);self.assertTrue(all(r['capacity_feasible'] is True for r in secondary))
        self.assertEqual({r['parent_primary_bits'] for r in secondary},{128,256,512})
        self.assertTrue(all(r['capacity_feasible'] is False for r in rows if r not in secondary))
        self.assertTrue(all(r['delivery_status']=='not_evaluated' for r in rows))

    def test_models_never_pooled(self):
        self.complete('tinyllama11',1,'large big good great.');self.complete('qwen05',1,'zxqzxq.')
        result=self.evaluate()
        self.assertEqual(set(result['models']),{'tinyllama11','qwen05'})
        self.assertNotIn('pooled_capacity',result)
        self.assertEqual(result['models']['tinyllama11']['available_saved_covers'],1)
        self.assertEqual(result['models']['qwen05']['available_saved_covers'],1)

    def test_missing_completed_attempt_rejected(self):
        path=self.complete('tinyllama11',1,'big large.');(self.root/path).unlink();self.seal();self.reject()

    def test_tampered_resource_rejected(self):
        p=self.root/a.CORE;p.write_bytes(p.read_bytes()+b'\n');self.seal();self.reject()

    def test_orphan_attempt_rejected(self):
        self.write('fixtures/orphan.json',{});self.seal();self.reject()

    def test_status_mismatch_rejected(self):
        self.index[0]['status']='completed';self.seal();self.reject()

    def test_wrong_record_metadata_rejected(self):
        path=self.complete('tinyllama11',1,'big large.');obj=a.parse((self.root/path).read_bytes());obj['record']['target_words']+=1
        self.write(path,obj);self.seal();self.reject()

    def test_missing_source_index_rejected(self):
        self.index.pop();self.seal();self.reject()

    def test_noncompleted_cannot_smuggle_measured_text(self):
        self.index[0]['attempt_path']='fixtures/wrong.json';self.write('fixtures/wrong.json',{});self.seal();self.reject()

    def test_82_91_remain_unreplayed(self):
        self.complete('tinyllama11',91,'big large.');self.reject()

    def test_duplicate_json_and_nonfinite_rejected(self):
        for raw in ['{"a":1,"a":2}','{"x":NaN}','{"x":1e999}']:
            with self.assertRaises(ValueError):a.parse(raw)

    def test_no_encoder_or_decoder_in_evaluation(self):
        self.complete('tinyllama11',1,'big large.')
        result=self.evaluate()
        for key in ['encoder_calls','decoder_calls','model_calls','tokenizer_calls','paid_calls']:self.assertEqual(result[key],0)
        self.assertTrue(all(r['source_fidelity']=='not_evaluated' for r in result['dictionary_rows']))

    def test_new_cover_must_match_final_snapshot_hash(self):
        path=self.complete('qwen05',1,'big large.');data=a.parse((self.root/path).read_bytes());data['result']['cover_text']='Changed text.'
        self.write(path,data);self.seal();self.reject()

    def test_snapshot_boolean_order_rejected(self):
        self.grid[0]['source_order']=True;self.seal();self.reject()


if __name__=='__main__':unittest.main()
