"""Synthetic mutation checks only; no scientific requests or model imports.

The positive fixture transforms frozen order090 evidence into a synthetic92
record. It is deliberately not written into a real runs directory or reported
as generation evidence. The separate reviewed-source manifest is synthetic too.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/audit_baseline_single_record_recovery_v2.py'
spec = importlib.util.spec_from_file_location('independent_postrun_auditor', SCRIPT)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
SOURCE_ROOT = Path(os.environ.get('SYNSTEG_AUDIT_FIXTURE_SOURCE', Path(__file__).resolve().parents[1]))


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()


def inventory(path, excluded=()):
    return {p.relative_to(path).as_posix():sha(p) for p in sorted(path.rglob('*'))
            if p.is_file() and p.relative_to(path).as_posix() not in excluded}


def fixture(root):
    source = load(SOURCE_ROOT/A.INPUTS)
    paths = [i['path'] for i in source['files']] + [A.INPUTS,A.HELPER,A.PROPOSED]
    proposal = load(SOURCE_ROOT/A.PROPOSED)
    paths += [j['prompt_path'] for j in proposal['jobs']]
    for rel in paths:
        (root/rel).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE_ROOT/rel,root/rel)
    protocol = root/f'experiments/{A.ID}/protocol.enabled.synthetic.json'
    enabled = copy.deepcopy(proposal); enabled['execution_enabled'] = True
    save(protocol,enabled)
    review = root/'synthetic-review.json'
    save(review,{'schema':'baseline92-successor-independent-review-v1','status':'passed',
                 'wrapper_sha256':sha(root/A.HELPER),'proposed_protocol_sha256':sha(root/A.PROPOSED),
                 'synthetic_unit_fixture_only':True})
    job = proposal['jobs'][0]
    ident = job['identity']
    row = json.loads((root/job['prompt_path']).read_text())
    meta = {k:row[k] for k in ('record_id','source_group_id','split','topic','genre','target_words')}
    meta['prompt_sha256'] = sha_bytes(row['prompt'].encode())
    shard = root/f'runs/{A.ID}/order092'
    previous = root/'runs/baseline-single-record-recovery-v1/order090'
    shutil.copytree(previous/'run',shard/'run')
    run = load(shard/'run/run.json')
    run.update(prompts_sha256=job['prompt_file_sha256'],selected_records=[meta],selected_records_count=1)
    save(shard/'run/run.json',run)
    fingerprint = canon({'record':meta,'provider':'local','model':A.MODEL,'revision':A.REVISION,
        'dtype':run['dtype'],'settings':run['settings'],'pricing_sha256':None,'prompts_sha256':job['prompt_file_sha256'],
        'runtime_sha256':canon(load(shard/'run/runtime.json')),
        'implementation':{k:run[k] for k in ('runner_sha256','ledger_code_sha256','transport_code_sha256')}})
    ledger = load(shard/'run/local_ledger/ledger.json')
    entry = next(iter(ledger['attempts'].values()))
    old_attempt = next((shard/'run/attempts').iterdir())
    attempt = load(old_attempt); old_attempt.unlink()
    entry.update(identity=ident,request_fingerprint=fingerprint,metadata={'record':meta,'provider':'local','model':A.MODEL})
    entry['result']['seed'] = int(canon([20261024,A.MODEL,row['record_id']])[:8],16)
    entry['result']['cover_text'] = 'Synthetic fixture; never scientific evidence.'
    ledger['attempts'] = {ident:entry}
    attempt.update(identity=ident,request_fingerprint=fingerprint,record=meta,result=copy.deepcopy(entry['result']))
    save(shard/'run/local_ledger/ledger.json',ledger)
    save(shard/'run/attempts'/(ident+'.json'),attempt)
    summary = load(shard/'run/summary.json')
    summary['attempts'] = [{'identity':ident,'record_id':row['record_id'],'status':'completed'}]
    save(shard/'run/summary.json',summary)
    request = {'schema':'synsteg-continuation-replay-request-v1','experiment_id':A.ID,'model_alias':'tinyllama11',
               'source_order':92,'intent_identity':ident,'key_sha256':None,'nonces':[]}
    request_bytes = (json.dumps(request,sort_keys=True,ensure_ascii=True,separators=(',',':'))+'\n').encode()
    (shard/'guard-request.json').write_bytes(request_bytes)
    request_sha = sha_bytes(request_bytes)
    save(shard/'guard-result.json',{'schema':'continuation-replay-guard-v1-decision','status':'not_denied_by_this_receipt',
        'launch_authorized':False,'reasons':[],'guard_sha256':sha(root/A.DENIALS),
        'guard_source_commit':load(root/A.DENIALS)['source_commit'],'request_sha256':request_sha})
    prior = load(previous/'git-preservation-receipt.json')
    intent = {'schema':A.ID+'-intent','experiment_id':A.ID,'status':'reserved_intent_no_execution_claim',
        'source_order':92,'record_id':row['record_id'],'identity':ident,'source_snapshot_commit':A.SOURCE,
        'protocol_sha256':sha(protocol),'helper_sha256':sha(root/A.HELPER),'guard_request_sha256':request_sha,
        'predecessor_receipt_commit':prior['commit'],'max_new_requests':1,'automatic_retries':0}
    original_intent = root/'results/synthetic-only/intent092.json'
    save(original_intent,intent); save(shard/'bound-intent.json',intent)
    readback_paths = [i['path'] for i in source['files']] + [A.INPUTS,A.HELPER,protocol.relative_to(root).as_posix(),
        original_intent.relative_to(root).as_posix(),*[j['prompt_path'] for j in proposal['jobs']]]
    save(shard/'activation-receipt.json',{'schema':A.ID+'-activation-readback-v1','readback_verified':True,'commit':'a'*40,
        'source_order':92,'identity':ident,'files_sha256':{p:sha(root/p) for p in readback_paths},
        'remote_output_scope_complete':True,'remote_current_output_orders':[], 'predecessor_receipt_commit':prior['commit']})
    save(shard/'reservation.json',{'experiment_id':A.ID,'source_order':92,'record_id':row['record_id'],'identity':ident,
        'status':'reserved_outcome_unknown','protocol_sha256':sha(protocol),'activation_commit':'a'*40,
        'intent_sha256':sha(original_intent),'automatic_retries':0,'guard_request_sha256':request_sha,'time_unix':10.0})
    first = load(root/A.FIRST); execution_root = Path(proposal['execution_root'])
    command = [str(execution_root/first['python_relative']),*first['argv']]
    for flag,value in {'--prompts':str(execution_root/job['prompt_path']),
        '--output':str(execution_root/f'runs/{A.ID}/order092/run'),'--max-records':'1','--max-new-records':'1'}.items():
        command[command.index(flag)+1] = value
    save(shard/'process.json',{'schema':A.ID+'-process','source_order':92,'identity':ident,'command':command,
        'timeout_seconds':1200,'timed_out':False,'automatic_retries':0,'status':'completed_pending_remote_preservation',
        'started_at_unix':11.0,'guard_request_sha256':request_sha,'exit_code':0,'elapsed_seconds':120.0,
        'observed_outcome':{'status':'completed','details':[]}})
    save(shard/'child.json',{'pid':123,'pgid':123})
    (shard/'stdout.log').write_text('Synthetic fixture only.\n'); (shard/'stderr.log').write_text('')
    refresh_manifest(shard)
    return protocol,shard,review


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def refresh_manifest(shard):
    save(shard/'artifact-manifest.json',{'files_sha256':inventory(shard,{'artifact-manifest.json','git-preservation-receipt.json'})})


class PostrunMutationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-postrun-audit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol,self.shard,self.review = fixture(self.root)

    def audit(self,order=92,require_git=False):
        return A.audit(self.root,self.protocol,self.shard,order,self.review,require_git)

    def mutate(self,relative,update):
        path = self.shard/relative
        value = load(path); update(value); save(path,value); refresh_manifest(self.shard)

    def reject(self,label):
        result = self.audit()
        self.assertEqual(result['status'],'failed',result)
        self.assertIn(label,result['errors'])

    def test_valid_saved_fixture_without_live_pid_or_model(self):
        result = self.audit()
        self.assertEqual(result['status'],'passed',result)
        self.assertFalse(result['live_process_identity_assessed'])
        self.assertEqual(result['model_calls'],0)

    def test_guard_denied_orders_and_boolean_order(self):
        for order in (82,91,True,101):
            with self.subTest(order=order):
                self.assertEqual(self.audit(order)['status'],'failed')

    def test_wrong_seed_even_when_both_raw_objects_agree(self):
        ident = load(self.shard/'reservation.json')['identity']
        self.mutate('run/local_ledger/ledger.json',lambda x:x['attempts'][ident]['result'].update(seed=7))
        self.mutate('run/attempts/'+ident+'.json',lambda x:x['result'].update(seed=7))
        self.reject('result model revision dtype and seed')

    def test_scientific_settings_changed(self):
        self.mutate('run/run.json',lambda x:x['settings'].update(temperature=0.5))
        self.reject('run plan matches frozen science except exact shard metadata')

    def test_runtime_changed(self):
        self.mutate('run/runtime.json',lambda x:x.update(platform='different'))
        self.reject('declared runtime matches frozen runtime')

    def test_guard_binding_changed_even_when_inventory_refreshed(self):
        self.mutate('guard-result.json',lambda x:x.update(request_sha256='0'*64))
        self.reject('guard decision bound to actual request and pinned denials')

    def test_timeout_preserves_failure(self):
        self.mutate('process.json',lambda x:x.update(timed_out=True))
        self.reject('clean bounded terminal process')

    def test_boolean_exit_code_rejected(self):
        self.mutate('process.json',lambda x:x.update(exit_code=False))
        self.reject('clean bounded terminal process')

    def test_extra_attempt_rejected(self):
        save(self.shard/'run/attempts/other.json',{})
        refresh_manifest(self.shard)
        self.reject('one ledger identity and one raw attempt')

    def test_lost_raw_file_gives_failed_report(self):
        (self.shard/'stdout.log').unlink()
        self.reject('complete readable well-formed evidence')

    def test_raw_hash_drift_not_hidden(self):
        (self.shard/'stdout.log').write_text('changed after inventory')
        self.reject('all raw shard artifacts preserved exactly')

    def test_base91_reclassification_rejected(self):
        base = load(self.root/A.BASE)
        next(r for r in base['records'] if r['model_alias']=='tinyllama11' and r['source_order']==91)['status']='completed'
        save(self.root/A.BASE,base)
        self.reject('frozen source bytes: '+A.BASE)

    def test_receipt_required_cannot_be_omitted(self):
        self.assertEqual(self.audit(require_git=True)['status'],'failed')

    def test_exact_postrun_receipt_and_tamper(self):
        receipt = {'readback_verified':True,'commit':'b'*40,'source_order':92,
            'process_sha256':sha(self.shard/'process.json'),'files_sha256':inventory(self.shard)}
        save(self.shard/'git-preservation-receipt.json',receipt)
        self.assertEqual(self.audit(require_git=True)['status'],'passed')
        receipt['files_sha256']['stdout.log']='0'*64
        save(self.shard/'git-preservation-receipt.json',receipt)
        self.reject('postrun Git receipt exact saved inventory')

    def test_unreviewed_wrapper_rejected(self):
        review = load(self.review); review['wrapper_sha256']='0'*64; save(self.review,review)
        self.reject('independent passed source review pins')

    def test_duplicate_json_rejected(self):
        (self.shard/'child.json').write_text('{"pid":123,"pid":123,"pgid":123}')
        refresh_manifest(self.shard)
        self.reject('complete readable well-formed evidence')

    def test_symlink_raw_evidence_rejected(self):
        (self.shard/'stdout.log').unlink()
        (self.shard/'stdout.log').symlink_to(self.shard/'stderr.log')
        self.reject('complete readable well-formed evidence')


if __name__ == '__main__':
    unittest.main()
