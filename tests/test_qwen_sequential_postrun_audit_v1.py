"""Independent synthetic tests. No model/tokenizer/wrapper imports or Git calls."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

REPO = Path(os.environ.get('SYNSTEG_REFERENCE_ROOT', Path(__file__).resolve().parents[1]))
MODULE = Path(os.environ.get('SYNSTEG_AUDITOR', REPO/'scripts/audit_qwen_sequential_actual_v1.py'))
spec = importlib.util.spec_from_file_location('independent_successor_audit_under_test', MODULE)
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True)+'\n').encode()


def put(root, path, value):
    p = root/path; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(value if isinstance(value, bytes) else encode(value))


def read(root, path):
    return json.loads((root/path).read_bytes())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def filemap(root, directory, excludes=()):
    return {p.relative_to(root/directory).as_posix():digest(p.read_bytes())
            for p in sorted((root/directory).rglob('*')) if p.is_file()
            and p.relative_to(root/directory).as_posix() not in excludes}


def manifest(root, pins, ref):
    rows = []
    for path, expected in sorted(pins.items()):
        data = (root/path).read_bytes()
        assert digest(data) == expected
        rows.append(dict(path=path, sha256=expected, bytes=len(data),
                         git_blob_sha1=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    return dict(ref=ref, actual_git_readback=True, files=rows, synthetic_fixture=True)


class Fixture:
    """Saved source1 is copied as immutable input; successors are artificial JSON only."""
    def __init__(self, root):
        self.root=root
        self.proposal=read(REPO,A.PROPOSAL)
        self.rows=[json.loads(x) for x in (REPO/A.PROMPTS).read_bytes().splitlines() if x.strip()]
        required=set(self.proposal['core_pins'])|set(self.proposal['preflight_pins'])|set(self.proposal['first_preserved_shard_pins'])
        required.update([A.PROPOSAL,A.HELPER,A.FIRST_HELPER,A.TEMPLATE])
        for p in required: put(root,p,(REPO/p).read_bytes())
        self.protocol=f'experiments/{A.ID}/protocol.enabled.json'
        enabled=copy.deepcopy(self.proposal);enabled['execution_enabled']=True;put(root,self.protocol,enabled)

    def prefix(self,n): return A.FIRST_SHARD if n==1 else A.OUTPUT+f'/order{n:03d}'

    def add(self,n):
        root=self.root;row=self.rows[n-1];job=self.proposal['jobs'][n-2];ident=canonical(['local',A.MODEL,row['record_id']])
        prefix=self.prefix(n);prior=read(root,self.prefix(n-1)+'/git-preservation-receipt.json')
        prompt=job['prompt_path'];put(root,prompt,(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode())
        meta={k:row[k] for k in ['record_id','source_group_id','split','topic','genre','target_words']};meta['prompt_sha256']=digest(row['prompt'].encode())
        request=dict(schema='synsteg-continuation-replay-request-v1',experiment_id=A.ID,model_alias='qwen05',source_order=n,intent_identity=ident,key_sha256=None,nonces=[])
        request_bytes=(json.dumps(request,sort_keys=True,ensure_ascii=True,separators=(',',':'))+'\n').encode();guard_sha=digest(request_bytes)
        intent=f'experiments/{A.ID}/intent-order{n:03d}.json'
        bound=dict(schema=A.ID+'-intent',experiment_id=A.ID,source_order=n,model_alias='qwen05',identity=ident,record_id=row['record_id'],status='reserved_intent_no_execution_claim',protocol_sha256=digest((root/self.protocol).read_bytes()),helper_sha256=A.HELPER_SHA,predecessor_receipt_commit=prior['commit'],guard_request_sha256=guard_sha,automatic_retries=0,max_new_requests=1,lost_execution_identity_policy='Do not replay any saved intent after loss of execution identity.')
        put(root,intent,bound);put(root,prefix+'/bound-intent.json',(root/intent).read_bytes());put(root,prefix+'/guard-request.json',request_bytes)
        decision=read(root,A.FIRST_SHARD+'/guard-result.json');decision['request_sha256']=guard_sha;put(root,prefix+'/guard-result.json',decision)
        activation_commit=f'{n:040x}';raw_commit=f'{n+1000:040x}'
        reserve=dict(experiment_id=A.ID,scientific_execution_stratum=A.FIRST_ID,model_alias='qwen05',source_order=n,identity=ident,record_id=row['record_id'],status='reserved_outcome_unknown',automatic_retries=0,protocol_sha256=bound['protocol_sha256'],intent_sha256=digest((root/intent).read_bytes()),activation_commit=activation_commit,predecessor_receipt_commit=prior['commit'],guard_request_sha256=guard_sha,time_unix=1.0)
        put(root,prefix+'/reservation.json',reserve)
        template=read(root,A.TEMPLATE);runtime=read(root,A.PRE+'/runtime.json')
        run=copy.deepcopy(template);run.update(model=A.MODEL,revision=A.REVISION,prompts_sha256=digest((root/prompt).read_bytes()),selected_records=[meta],selected_records_count=1);put(root,prefix+'/run/run.json',run)
        put(root,prefix+'/run/runtime.json',(root/(A.PRE+'/runtime.json')).read_bytes())
        fingerprint=canonical(dict(record=meta,provider='local',model=A.MODEL,revision=A.REVISION,dtype='float32',settings=template['settings'],pricing_sha256=None,prompts_sha256=run['prompts_sha256'],runtime_sha256=canonical(runtime),implementation={k:template[k] for k in ['runner_sha256','ledger_code_sha256','transport_code_sha256']}))
        generated=dict(cover_text='Synthetic audit fixture only.',returned_model=A.MODEL,revision=A.REVISION,dtype='float32',seed=int(canonical([20261024,A.MODEL,row['record_id']])[:8],16),usage=dict(input_tokens=read(root,A.PRE+'/all100-context.json')['records'][n-1]['input_tokens'],output_tokens=7),hit_output_limit=False,stop_reason='end_turn')
        attempt=dict(schema='synsteg-budget-generation-attempt-v1',identity=ident,status='completed',result=generated,record=meta,provider='local',model=A.MODEL,request_fingerprint=fingerprint,automatic_retries=0)
        entry=dict(identity=ident,status='completed',result=generated,metadata=dict(record=meta,provider='local',model=A.MODEL),request_fingerprint=fingerprint,automatic_retries=0,accounted_nanodollars=0,reserved_nanodollars=0)
        ledger=dict(schema='synsteg-budget-ledger-recovery-v1',budget_nanodollars=20000000000,blocked_reason=None,attempts={ident:entry})
        put(root,prefix+'/run/attempts/'+ident+'.json',attempt);put(root,prefix+'/run/local_ledger/ledger.json',ledger)
        put(root,prefix+'/run/local_ledger/campaign.lock',b'0');put(root,prefix+'/run/run_guard/campaign.lock',b'0')
        summary=dict(status='selected_records_completed',exit_code=0,completed=1,requests_attempted_this_invocation=1,resumed_completed=0,selected_records_count=1,max_new_records_this_invocation=1,automatic_retries=0,attempts=[dict(identity=ident,record_id=row['record_id'],status='completed')],ledger=dict(attempt_count=1,unknown_or_failed_attempts=0))
        put(root,prefix+'/run/summary.json',summary)
        first=read(root,A.FIRST);execution_root=Path(self.proposal['execution_root']);command=[str(execution_root/first['python_relative']),*first['argv']]
        for flag,value in {'--model':A.MODEL,'--revision':A.REVISION,'--prompts':str(execution_root/prompt),'--output':str(execution_root/prefix/'run'),'--max-records':'1','--max-new-records':'1'}.items():command[command.index(flag)+1]=value
        process=dict(command=command,schema=A.ID+'-process',experiment_id=A.ID,scientific_execution_stratum=A.FIRST_ID,model_alias='qwen05',source_order=n,identity=ident,timeout_seconds=1200,exit_code=0,automatic_retries=0,timed_out=False,status='completed_pending_remote_preservation',observed_outcome=dict(status='completed',details=[]),elapsed_seconds=1.0,started_at_unix=2.0,guard_request_sha256=guard_sha)
        put(root,prefix+'/process.json',process);put(root,prefix+'/child.json',dict(pid=12345,pgid=12345));put(root,prefix+'/stdout.log',b'');put(root,prefix+'/stderr.log',b'')
        pins=dict(self.proposal['core_pins'])|self.proposal['preflight_pins']|self.proposal['first_preserved_shard_pins']
        for p in [A.HELPER,A.FIRST_HELPER,prompt,self.protocol,intent]:pins[p]=digest((root/p).read_bytes())
        for earlier in range(2,n):
            pre=self.prefix(earlier);inv=filemap(root,pre) if earlier==n-1 else {'git-preservation-receipt.json':digest((root/pre/'git-preservation-receipt.json').read_bytes())}
            pins.update({pre+'/'+p:h for p,h in inv.items()})
        activation=dict(schema=A.ID+'-activation-readback-v1',readback_verified=True,commit=activation_commit,identity=ident,source_order=n,files_sha256=pins,remote_output_scope_complete=True,remote_qwen_attempted_source_orders=list(range(1,n)),remote_qwen_active_intents=[ident],remote_branches_and_prs_checked=True,predecessor_receipt_commit=prior['commit'])
        put(root,prefix+'/activation-receipt.json',activation)
        activation_manifest=f'test-evidence/activation{n}.json';put(root,activation_manifest,manifest(root,pins,activation_commit))
        raw_manifest=f'test-evidence/raw{n}.json';self.close(n,raw_commit,raw_manifest)
        return dict(n=n,protocol=root/self.protocol,intent=root/intent,activation=root/activation_manifest,raw=root/raw_manifest)

    def close(self,n,raw_commit=None,raw_manifest=None):
        prefix=self.prefix(n);root=self.root
        if raw_commit is None:raw_commit=read(root,prefix+'/git-preservation-receipt.json')['commit']
        put(root,prefix+'/artifact-manifest.json',dict(files_sha256=filemap(root,prefix,{'artifact-manifest.json','git-preservation-receipt.json'})))
        inv=filemap(root,prefix,{'git-preservation-receipt.json'})
        put(root,prefix+'/git-preservation-receipt.json',dict(readback_verified=True,commit=raw_commit,source_order=n,process_sha256=inv['process.json'],files_sha256=inv))
        if raw_manifest:put(root,raw_manifest,manifest(root,{prefix+'/'+p:h for p,h in inv.items()},raw_commit))

    def audit(self,args):
        return A.audit(self.root,args['n'],args['protocol'],args['intent'],args['activation'],args['raw'],True)


class IndependentAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_json_duplicate_and_nonfinite_rejected(self):
        for raw in [b'{"a":1,"a":2}',b'{"x":{"a":1,"a":2}}',b'{"a":NaN}',b'{"a":Infinity}',b'{"a":-Infinity}']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):A.decode(raw)
    def test_integer_and_git_ref_types(self):
        self.assertTrue(A.integer(2,2));self.assertFalse(A.integer(True,1));self.assertFalse(A.integer(2.0,2))
        for ref in ['A'*40,'a'*39,'a'*41,'g'*40,True,None]:self.assertFalse(A.git_sha(ref))
    def test_symlink_bytes_and_inventory_rejected(self):
        put(self.root,'a',b'abc');(self.root/'link').symlink_to(self.root/'a')
        with self.assertRaises(ValueError):A.raw(self.root/'link')
        with self.assertRaises(ValueError):A.inventory(self.root)
    def test_provenance_positive_and_each_corruption_rejected(self):
        put(self.root,'a',b'abc');pins={'a':digest(b'abc')};good=manifest(self.root,pins,'a'*40)
        variants=[]
        for key,value in [('sha256','0'*64),('git_blob_sha1','0'*40),('bytes',2),('bytes',True)]:
            v=copy.deepcopy(good);v['files'][0][key]=value;variants.append(v)
        v=copy.deepcopy(good);v['files']*=2;variants.append(v)
        v=copy.deepcopy(good);v['files']=[];variants.append(v)
        v=copy.deepcopy(good);v['actual_git_readback']=False;variants.append(v)
        v=copy.deepcopy(good);v['ref']='b'*40;variants.append(v)
        for value,expected in [(good,True)]+[(v,False) for v in variants]:
            put(self.root,'manifest.json',value);checks=[]
            A.provenance(self.root,self.root/'manifest.json',pins,'a'*40,lambda _,ok:checks.append(bool(ok)))
            self.assertEqual(all(checks),expected)
    def test_provenance_parent_traversal_rejected(self):
        put(self.root,'manifest.json',dict(ref='a'*40,actual_git_readback=True,files=[dict(path='../outside',sha256='a'*64,bytes=1,git_blob_sha1='a'*40)]))
        with self.assertRaises(ValueError):A.provenance(self.root,self.root/'manifest.json',{'../outside':'a'*64},'a'*40,lambda *_:None)
    def test_invalid_orders_fail_without_scientific_reads(self):
        for order in [True,False,1,101,2.0,'2']:
            r=A.audit(self.root,order,self.root/'p',self.root/'i',self.root/'a',self.root/'r',True)
            self.assertEqual(r['status'],'failed');self.assertEqual(r['model_calls'],0)
    def test_synthetic_successor2_passes(self):
        f=Fixture(self.root);r=f.audit(f.add(2));self.assertEqual(r['status'],'passed',r['errors']);self.assertEqual(r['raw_files_checked'],17)
    def test_synthetic_successor3_with_preserved_predecessor_passes(self):
        f=Fixture(self.root);f.add(2);r=f.audit(f.add(3));self.assertEqual(r['status'],'passed',r['errors']);self.assertEqual(r['predecessors_checked'],2)
    def test_coherently_rehashed_orphan_predecessor_file_blocks_successor3(self):
        f=Fixture(self.root);f.add(2);put(self.root,f.prefix(2)+'/orphan.json',dict(synthetic=True));f.close(2)
        r=f.audit(f.add(3));self.assertEqual(r['status'],'failed');self.assertTrue(any('predecessor' in e for e in r['errors']),r['errors'])
    def test_coherently_rehashed_failed_predecessor_blocks_successor3(self):
        f=Fixture(self.root);f.add(2);prefix=f.prefix(2)
        ledger=read(self.root,prefix+'/run/local_ledger/ledger.json');identity=next(iter(ledger['attempts']))
        ledger['attempts'][identity]['status']='failed';put(self.root,prefix+'/run/local_ledger/ledger.json',ledger)
        attempt=read(self.root,prefix+'/run/attempts/'+identity+'.json');attempt['status']='failed';put(self.root,prefix+'/run/attempts/'+identity+'.json',attempt)
        f.close(2);args=f.add(3);r=f.audit(args)
        self.assertEqual(r['status'],'failed');self.assertTrue(any('predecessor' in e for e in r['errors']),r['errors'])
    def test_coherently_rehashed_current_seed_corruption_fails(self):
        f=Fixture(self.root);args=f.add(2);prefix=f.prefix(2)
        ledger=read(self.root,prefix+'/run/local_ledger/ledger.json');identity=next(iter(ledger['attempts']))
        ledger['attempts'][identity]['result']['seed']+=1;put(self.root,prefix+'/run/local_ledger/ledger.json',ledger)
        attempt=read(self.root,prefix+'/run/attempts/'+identity+'.json');attempt['result']['seed']+=1;put(self.root,prefix+'/run/attempts/'+identity+'.json',attempt)
        f.close(2,raw_manifest=args['raw'].relative_to(self.root).as_posix());r=f.audit(args)
        self.assertEqual(r['status'],'failed');self.assertIn('returned model and deterministic per-record seed',r['errors'])


if __name__=='__main__':unittest.main(verbosity=2)
