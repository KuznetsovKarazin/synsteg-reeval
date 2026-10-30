"""Offline adapter fixtures; synthetic new model outputs are never experiment data."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
MERGER_ROOT = Path(os.environ.get('MIXED_MERGER_CHECKOUT', str(HERE)))
spec = importlib.util.spec_from_file_location('builder', HERE/'scripts/build_mixed_development_snapshot_v1.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
m = b.merger_module(MERGER_ROOT/'scripts/merge_large_development_full300_v2.py')
spec = importlib.util.spec_from_file_location('merger_fixtures', MERGER_ROOT/'tests/test_mixed_development_full300_merger_v2.py')
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
ACTUAL = Path(os.environ.get('MIXED_TINY_CONTEXT_CHECKOUT', str(MERGER_ROOT)))


class BuilderCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.f = fixtures.Fixture(self.tmp.name)

    def qwen(self):
        old, identity = self.f.shard('qwen05', 1)
        new = b.QWEN_ROOT + '/qwen05/order001'
        (self.f.root/new).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(self.f.root/old, self.f.root/new)
        old_contract = self.f.read(self.f.contracts['qwen05'])
        new_contract = copy.deepcopy(old_contract)
        for field, name in [('runtime_path','runtime.json'),('model_config_path','config.json'),('context_report_path','all100-context.json')]:
            destination = b.QWEN_PRE+'/'+name
            target = self.f.root/destination; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(self.f.root/old_contract[field],target); new_contract[field]=destination
        self.f.write(b.QWEN_PRE+'/model-contract.json',new_contract)
        request = b.draft_request(m,self.f.root,'c'*40)
        request['remote_census']['readback_verified']=True
        request['remote_census']['branches_and_prs_checked']=True
        request['qwen_preflight']['commit']='d'*40
        return request,new,identity

    def reject(self, request):
        with self.assertRaises((ValueError,FileNotFoundError,KeyError)):
            b.stage(m,self.f.root,request)

    def test_qwen_full_bundle_build_and_frozen_merger(self):
        request,_,_=self.qwen(); raw=b.encoded(request); output=Path(self.tmp.name)/'bundle'
        result=b.build(m,self.f.root,raw,m.sha(raw),output)
        self.assertEqual(result['counts'],{'completed':90,'reserved_outcome_unknown':1,'not_attempted':209})
        actual=m.parse((output/'full300.json').read_bytes())
        self.assertFalse(actual['inherited_base_raw_reaudited'])
        self.assertEqual(actual['model_counts']['qwen05'],{'completed':1,'not_attempted':99})

    def test_draft_is_never_verified(self):
        self.qwen(); draft=b.draft_request(m,self.f.root,'c'*40)
        self.assertFalse(draft['remote_census']['readback_verified'])
        self.assertIsNone(draft['qwen_preflight']['commit']); self.reject(draft)

    def test_empty_receipt_map(self):
        request,_,_=self.qwen(); request['primary_receipts']={}; self.reject(request)

    def test_extra_local_reservation_not_in_request(self):
        request,_,_=self.qwen(); self.f.write(b.TINY_ROOT+'/order093/reservation.json',{})
        self.reject(request)

    def test_extra_censused_reservation_without_primary_receipt(self):
        request,_,_=self.qwen(); self.f.write(b.TINY_ROOT+'/order093/reservation.json',{})
        request['remote_census']['files_sha256']=b.census(m,self.f.root)[0]; self.reject(request)

    def test_missing_receipt(self):
        request,p,_=self.qwen(); (self.f.root/p/'git-preservation-receipt.json').unlink(); self.reject(request)

    def test_changed_primary_commit(self):
        request,p,_=self.qwen(); request['primary_receipts'][p]['commit']='e'*40; self.reject(request)

    def test_changed_raw_preserved_inventory(self):
        request,p,_=self.qwen(); self.f.write(p+'/stdout.log',{'changed':True})
        request['remote_census']['files_sha256']=b.census(m,self.f.root)[0]; self.reject(request)

    def test_changed_qwen_preflight(self):
        request,_,_=self.qwen(); self.f.write(b.QWEN_PRE+'/runtime.json',{})
        self.reject(request)

    def test_missing_terminal_cannot_be_optimistically_counted(self):
        request,p,_=self.qwen(); (self.f.root/p/'process.json').unlink(); self.f.refresh(p)
        request=b.draft_request(m,self.f.root,'c'*40); request['remote_census'].update(readback_verified=True,branches_and_prs_checked=True)
        request['qwen_preflight']['commit']='d'*40; self.reject(request)

    def test_supervisor_lock_recorded_not_evidence(self):
        request,_,_=self.qwen(); self.f.write(b.QWEN_ROOT+'/supervisor.lock',{'pid':'fixture'})
        payloads,_,provenance=b.stage(m,self.f.root,request)
        self.assertNotIn(b.QWEN_ROOT+'/supervisor.lock',payloads)
        self.assertIn(b.QWEN_ROOT+'/supervisor.lock',provenance['local_operational_files_excluded'])

    def test_orphan_file_outside_indexed_shard(self):
        request,_,_=self.qwen(); self.f.write(b.QWEN_ROOT+'/orphan.json',{})
        request['remote_census']['files_sha256']=b.census(m,self.f.root)[0]; self.reject(request)

    def test_scope_cannot_import_other_qwen_campaign(self):
        with self.assertRaises(ValueError):b.shard_identity(m,'runs/old-qwen/order001')
        with self.assertRaises(ValueError):b.shard_identity(m,b.TINY_ROOT+'/order091')
        with self.assertRaises(ValueError):b.shard_identity(m,b.QWEN_ROOT+'/qwen05/order002')

    def test_actual_tiny100_conversion_no_tokenizer(self):
        raw=(ACTUAL/b.TINY_CONTEXT).read_bytes()
        self.assertEqual(m.sha(raw),b.TINY_CONTEXT_SHA)
        result=b.tiny_context(m,raw,self.f.sources,b.TINY_CONFIG_SHA,2048)
        original=m.parse(raw)
        self.assertEqual(len(result['records']),100)
        self.assertEqual([r['input_tokens'] for r in result['records']], [r['input_tokens'] for r in original['rows']])
        self.assertEqual(min(r['input_tokens'] for r in result['records']),215)
        self.assertEqual(max(r['input_tokens'] for r in result['records']),346)

    def test_tiny_context_rejects_reordering_mismatch_and_bad_arithmetic(self):
        original=m.parse((ACTUAL/b.TINY_CONTEXT).read_bytes())
        for mutate in ['order','count','source','remaining']:
            with self.subTest(mutate=mutate):
                value=copy.deepcopy(original)
                if mutate=='order':value['rows'][0],value['rows'][1]=value['rows'][1],value['rows'][0]
                if mutate=='count':value['rows']=value['rows'][:-1]
                if mutate=='source':value['rows'][0]['record_id']='wrong'
                if mutate=='remaining':value['rows'][0]['remaining_after_requested_output']+=1
                with self.assertRaises(ValueError):b.tiny_context(m,b.encoded(value),self.f.sources,b.TINY_CONFIG_SHA,2048)

    def test_request_hash_and_output_protection(self):
        request,_,_=self.qwen(); raw=b.encoded(request)
        with self.assertRaises(ValueError):b.build(m,self.f.root,raw,'f'*64,Path(self.tmp.name)/'bad')
        with self.assertRaises(ValueError):b.build(m,self.f.root,raw,m.sha(raw),self.f.root/'bad')

    def test_changed_merger_hash_rejected(self):
        p=Path(self.tmp.name)/'wrong.py'; p.write_text('raise RuntimeError("must not execute")')
        with self.assertRaises(ValueError):b.merger_module(p)

    def test_failed_publication_leaves_no_partial_output(self):
        request,_,_=self.qwen(); raw=b.encoded(request); output=Path(self.tmp.name)/'bundle'
        with patch.object(b,'publish_new_directory',side_effect=OSError('injected publication failure')):
            with self.assertRaises(OSError):b.build(m,self.f.root,raw,m.sha(raw),output)
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.glob('mixed300-publish-*')),[])

    def test_atomic_publication_never_replaces_existing_directory(self):
        source=Path(self.tmp.name)/'staged'; source.mkdir()
        target=Path(self.tmp.name)/'existing'; target.mkdir()
        with self.assertRaises(OSError):b.publish_new_directory(source,target)
        self.assertTrue(source.is_dir()); self.assertTrue(target.is_dir())

    def test_output_symlink_ancestor_rejected(self):
        request,_,_=self.qwen(); raw=b.encoded(request)
        link=Path(self.tmp.name)/'link'; link.symlink_to(Path(self.tmp.name),target_is_directory=True)
        with self.assertRaises(ValueError):b.build(m,self.f.root,raw,m.sha(raw),link/'output')


if __name__=='__main__':unittest.main()
