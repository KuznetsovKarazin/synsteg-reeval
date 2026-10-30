"""Synthetic saved metadata/transcripts only. No actual native/tokenizer/codec."""
import copy, hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from synsteg_audit import candidate_native_v2_independent_audit as a
from synsteg_audit import author_candidate_local_native_v2 as n
spec=importlib.util.spec_from_file_location('nativev2fixtures',ROOT/'tests/test_author_candidate_local_native_v2.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

def dump(path,value):path.write_bytes(a.canonical(value))
def native_files(root,directory,profile,pid):
    parent={'namespace_pids':[500,5],'pid_namespace':'pid:[1234]','proc_pid':500,'proc_ppid':499,'starttime_ticks':999}
    child={'namespace_pids':[1000+pid,pid],'pid_namespace':'pid:[1234]','proc_pid':1000+pid,'proc_ppid':500,'starttime_ticks':10000+pid}
    base=str(root/a.RESOURCE_ROOT)+'/'
    exe={'bytes':profile['resource_files'][a.BRIDGE]['bytes'],'sha256':profile['resource_files'][a.BRIDGE]['sha256'],
         'path':base+a.BRIDGE,'device':27,'inode':1234}
    capture={'schema':'framed-interval-public-v1-owned-process','popen_pid':pid,'parent':parent,'child':child,
        'executable':exe,'public_profile_sha256':a.sha(a.canonical(profile))}
    identity={'schema':'proc-namespace-identity-v2','before':{'identity':child,'executable':exe},'after':{'identity':child,'executable':exe},
        'parent':parent,'maps_sha256':'e'*64,'pidfd_identity':{'inode':5678,'namespace_pids':child['namespace_pids'],'proc_pid':child['proc_pid']},
        'pidfd_live_before_and_after_maps':True,'popen_pid':pid,'proc_visible_pid':child['proc_pid'],'scope':'synthetic test metadata; not actual process'}
    mapped={base+x:profile['resource_files'][x]['sha256'] for x in [a.BRIDGE,a.MODEL,a.CPU,'runtime/llama-b11791/libllama.so.0']}
    mapped.update(profile['system_libraries_sha256'])
    resident={'status':'passed','pid':pid,'mapped_files_sha256':mapped,'identity':identity,
        'scope':'owned live maps and current bytes; no hardware attestation'}
    resource={'schema':'framed-interval-public-v1-native-resource-check','verified_files':len(profile['resource_files']),
        'public_profile_sha256':a.sha(a.canonical(profile)),'model_calls':0,'tokenizer_calls':0,
        'scope':'current files/runtime only; no historical or resident attestation'}
    dump(directory/'resource-check.json',resource);dump(directory/'proc-identity-capture.json',capture)
    dump(directory/'native-ready.json',{'ready':profile['ready'],'resident':resident,'resource_profile_sha256':a.sha(a.canonical(profile)),
        'prompt_calls':0,'step_calls':0})
    (directory/'native.stderr.log').write_bytes(b'SYNTHETIC TEST: no process was launched\n')

class IndependentAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.tmp=Path(self.temp.name);self.calls=[]
    def prepare(self,select=None):
        root=self.tmp/'root';root.mkdir();closure=json.loads((ROOT/a.CLOSURE).read_bytes())
        for path in closure['files_sha256']:
            target=root/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,target)
        shutil.copyfile(ROOT/a.CLOSURE,root/a.CLOSURE)
        intent=root/a.BASE/'synthetic-intent.json';n.write_json(intent,n.execution_intent(root))
        activation=root/a.BASE/'synthetic-activation.json';n.write_json(activation,n.enabled_activation(root,intent))
        receipt=root/'synthetic-source-readback.json';files={**closure['files_sha256'],a.CLOSURE:a.PINNED[a.CLOSURE],
             intent.relative_to(root).as_posix():n.sha_file(intent),activation.relative_to(root).as_posix():n.sha_file(activation)}
        n.write_json(receipt,{'schema':a.ID+'-source-readback','source_commit':'a'*40,'activation_commit':'b'*40,'all_files_exact':True,'files_sha256':files})
        (root/'results').mkdir(exist_ok=True)
        def runner(root,profile,request,order,directory,commitment,**kwargs):
            def factory(root,profile,native,deadline):
                pid=200+len(self.calls);self.calls.append((request['chunk_order'],order));native_files(root,native,profile,pid)
                responses,fault=([f.abstain(request)],None) if select is None else select(request,order)
                bridge=f.MockBridge(responses,native,fault);bridge.process=SimpleNamespace(pid=pid,poll=lambda:-15);return bridge
            return n.run_attempt(root,profile,request,order,directory,commitment,**kwargs,bridge_factory=factory)
        n.run_campaign(root,root/a.OUTPUT,activation,intent,receipt,attempt_runner=runner)
        self.root=root;self.base=root/a.OUTPUT
        return root,self.base
    def audit(self):return a.audit_campaign(self.root,self.base)
    def mutate(self,path,change):
        p=self.base/path;x=json.loads(p.read_bytes());change(x);dump(p,x)
    def assert_unknown(self,result):
        self.assertEqual(result['assessment'],'evidence_UNKNOWN',result.get('issues'));self.assertIsNone(result['complete_verified_format_accepted_chunks'])
        self.assertEqual(result['full_cover_chunks'],109);self.assertEqual(result['requested_chunks'],2)
    def test_two_abstention_attempts_local_pending_git(self):
        self.prepare();r=self.audit();self.assertEqual(r['assessment'],'verified_local_evidence_pending_Git_readback',r['issues'])
        self.assertEqual(r['complete_verified_format_accepted_chunks'],2);self.assertFalse(r['post_save_Git_evidence_verified'])
        self.assertIsNone(r['candidate_usable_rows']);self.assertEqual(len(self.calls),2)
        self.assertTrue(all(x['native_evidence']['model_origin_authenticated'] is False for x in r['attempts']))
    def test_external_exact_receipt_accepts_and_does_not_change_runtime(self):
        self.prepare();local=self.audit();receipt={'schema':a.AUDIT+'-output-readback','output_commit':'c'*40,'all_files_exact':True,
            'campaign_files':local['campaign_files'],'terminal_pins':local['terminal_pins']}
        r=a.audit_campaign(self.root,self.base,external_receipt=receipt)
        self.assertEqual(r['assessment'],'verified_recorded_evidence_after_Git_readback',r['issues']);self.assertTrue(r['post_save_Git_evidence_verified'])
        self.assertEqual(r['runtime_status'],'complete_diagnostic_pending_Git_audit')
        receipt['terminal_pins']['chunk0-attempt0']='0'*64;self.assert_unknown(a.audit_campaign(self.root,self.base,external_receipt=receipt))
    def test_six_refusals_keep_full_denominator(self):
        self.prepare(lambda req,o:([b'```'],None));r=self.audit();self.assertEqual(r['assessment'],'verified_local_evidence_pending_Git_readback',r['issues'])
        self.assertEqual(r['runtime_reported_attempts'],6);self.assertEqual(len(r['attempts']),6);self.assertEqual(r['complete_verified_format_accepted_chunks'],0)
    def test_known_then_unknown_keeps_runtime_unknown_and_previous_acceptance(self):
        self.prepare(lambda req,o:([f.abstain(req)],None if req['chunk_order']==0 else 'step_error'));r=self.audit()
        self.assertEqual(r['assessment'],'runtime_UNKNOWN',r['issues']);self.assertEqual(r['runtime_status'],'unknown_terminal')
        self.assertEqual(r['observed_verified_format_accepted_attempts'],1);self.assertIsNone(r['complete_verified_format_accepted_chunks'])
        self.assertTrue(r['attempts'][1]['pending_prepared'])
    def test_resource_refusal_is_known_no_semantic_success(self):
        self.prepare(lambda req,o:([b'{"x":'+b'['*40],None));r=self.audit()
        self.assertEqual(r['assessment'],'verified_local_evidence_pending_Git_readback',r['issues'])
        self.assertEqual(r['runtime_status'],'parser_resource_refusal_terminal');self.assertEqual(r['runtime_reported_attempts'],1)
        self.assertEqual(r['complete_verified_format_accepted_chunks'],0)
    def test_lost_result_retains_attempt_inventory(self):
        self.prepare();(self.base/'result.json').unlink();r=self.audit();self.assert_unknown(r)
        self.assertEqual(r['reservation_files_present'],2);self.assertEqual(len(r['attempt_directories_present']),2);self.assertEqual(len(r['attempts']),2)
    def test_truncated_journal_keeps_original_runtime_claim(self):
        self.prepare();p=self.base/'chunk0-attempt0/sealed/journal.jsonl';p.write_bytes(p.read_bytes()[:-7]);r=self.audit();self.assert_unknown(r)
        self.assertEqual(r['attempts'][0]['runtime_outcome'],'format_accepted');self.assertEqual(r['attempts'][0]['evidence_status'],'evidence_UNKNOWN')
    def test_wrong_derived_campaign_count(self):
        self.prepare();self.mutate('result.json',lambda x:x.__setitem__('attempts_executed',True));self.assert_unknown(self.audit())
    def test_hidden_attempt_or_reservation_is_rejected(self):
        self.prepare();p=self.base/'reservations/chunk1-attempt2.json';p.write_bytes(b'{}');self.assert_unknown(self.audit())
    def test_unknown_native_file_not_filtered(self):
        self.prepare();p=self.base/'chunk0-attempt0/native/extra.raw';p.write_bytes(b'partial');r=self.audit();self.assert_unknown(r)
        self.assertIn('chunk0-attempt0/native/extra.raw',r['campaign_files'])
    def test_known_attempt_extra_root_file_is_rejected(self):
        self.prepare();(self.base/'chunk0-attempt0/unregistered.raw').write_bytes(b'old');self.assert_unknown(self.audit())
    def test_native_resource_and_process_faults(self):
        changes=[('resource-check.json',lambda x:x.__setitem__('verified_files',40)),
            ('native-ready.json',lambda x:x['resident']['mapped_files_sha256'].__setitem__(next(iter(x['resident']['mapped_files_sha256'])),'0'*64)),
            ('native-ready.json',lambda x:x['resident']['identity'].__setitem__('pidfd_live_before_and_after_maps',False)),
            ('native-ready.json',lambda x:x['resident']['identity']['after']['identity'].__setitem__('starttime_ticks',1)),
            ('native-ready.json',lambda x:x['ready'].__setitem__('vocabulary_size',1)),
            ('proc-identity-capture.json',lambda x:x['child'].__setitem__('proc_ppid',999)),
            ('proc-identity-capture.json',lambda x:x['executable'].__setitem__('sha256','0'*64))]
        self.prepare()
        for name,change in changes:
            with self.subTest(name=name):
                p=self.base/'chunk0-attempt0/native'/name;before=p.read_bytes();self.mutate('chunk0-attempt0/native/'+name,change)
                self.assert_unknown(self.audit());p.write_bytes(before)
    def test_saved_prompt_identity_is_checked(self):
        self.prepare();self.mutate('chunk0-attempt0/tokenized-prompt.json',lambda x:x['prompt_ids'].append(1));self.assert_unknown(self.audit())
    def test_output_readback_inventory_type_strict(self):
        self.prepare();r=self.audit();receipt={'schema':a.AUDIT+'-output-readback','output_commit':'c'*40,'all_files_exact':True,
            'campaign_files':r['campaign_files'],'terminal_pins':r['terminal_pins']}
        receipt['campaign_files']['result.json']['bytes']=float(receipt['campaign_files']['result.json']['bytes'])
        self.assert_unknown(a.audit_campaign(self.root,self.base,external_receipt=receipt))
    def test_duplicate_json_rejected(self):
        self.prepare();p=self.base/'result.json';p.write_bytes(b'{"schema":"bad",'+p.read_bytes()[1:]);self.assert_unknown(self.audit())
    def test_symlink_campaign_member_rejected(self):
        self.prepare();p=self.base/'chunk0-attempt0/observed-response.raw';target=self.tmp/'held';target.write_bytes(p.read_bytes());p.unlink();p.symlink_to(target)
        with self.assertRaises(ValueError):self.audit()
    def test_fifo_campaign_member_rejected_without_read(self):
        self.prepare();p=self.base/'unknown.fifo';os.mkfifo(p)
        code='import sys\nsys.path.insert(0,'+repr(str(ROOT/'src'))+')\nfrom synsteg_audit.candidate_native_v2_independent_audit import snapshot\ntry:snapshot('+repr(str(self.base))+')\nexcept ValueError:sys.exit(0)\nsys.exit(9)\n'
        r=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,timeout=2);self.assertEqual(r.returncode,0,r.stderr)
    def test_source_file_mutation_blocks_acceptance(self):
        self.prepare();p=self.root/'src/synsteg_audit/author_candidate_local_native_v2.py';p.write_bytes(p.read_bytes()+b'\n');self.assert_unknown(self.audit())
    def test_active_nn_lease_blocks_read(self):
        import fcntl
        self.prepare();fd=os.open(self.root/'results/project-nn.lock',os.O_RDWR)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):self.audit()
        finally:os.close(fd)
    def test_malformed_attempt_list_retained_as_unknown(self):
        self.prepare();self.mutate('result.json',lambda x:x['chunks'][0]['attempts'].__setitem__(0,[]));self.assert_unknown(self.audit())
    def test_non_object_campaign_result_keeps_evidence_unknown_report(self):
        self.prepare()
        for bad in [[],None,1,True,"claimed complete"]:
            with self.subTest(bad=bad):
                dump(self.base/'result.json',bad);r=self.audit();self.assert_unknown(r)
                self.assertEqual(len(r['attempts']),2)
    def test_non_object_commitment_returns_evidence_unknown(self):
        self.prepare()
        for bad in [[],None,1,True,"bad"]:
            with self.subTest(bad=bad):
                dump(self.base/'commitments/chunk0-attempt0.json',bad);r=self.audit();self.assert_unknown(r)
                self.assertEqual(r['attempts'][0]['runtime_outcome'],'format_accepted')
    def test_independent_journal_structural_mutations_after_rehash(self):
        self.prepare();data,_=a.snapshot(self.base);request=f.REQUESTS[0];expected=a.binding(f.PROFILE,request,0,a.PINNED[a.PROTOCOL])
        rows=[json.loads(line) for line in data['chunk0-attempt0/sealed/journal.jsonl'].splitlines()]
        def repack(seq):
            previous='0'*64;out=[]
            for i,row in enumerate(seq):
                body={'sequence':i,'previous_sha256':previous,'event':row['event']};previous=a.sha(a.canonical(body))
                out.append(a.canonical({**body,'event_sha256':previous})+b'\n')
            return b''.join(out)
        mutations=[lambda x:x[1]['event'].__setitem__('prompt_ids',[True]),
            lambda x:x[2]['event'].__setitem__('evaluated_tokens',True),
            lambda x:x[3]['event'].__setitem__('token_index',True),
            lambda x:x[3]['event'].__setitem__('sample_evidence',{**x[3]['event']['sample_evidence'],'seed_hex':'0'*64}),
            lambda x:x[4]['event'].__setitem__('evaluated_tokens',4),
            lambda x:x[-1]['event'].__setitem__('response_bytes',True),
            lambda x:x[-1]['event'].__setitem__('stop_label','token_cap'),
            lambda x:x.__setitem__(2,copy.deepcopy(x[0])),
            lambda x:x.pop(4),lambda x:x.append(copy.deepcopy(x[-1]))]
        # Updating the outer chain does not make malformed acknowledgements,
        # typed counters, sample evidence or terminal events acceptable.
        for mutate in mutations:
            changed=copy.deepcopy(rows);mutate(changed)
            with self.subTest(mutation=str(mutate)):
                with self.assertRaises(ValueError):a.journal(repack(changed),expected,f.PROFILE,request,0,complete=True)
        partial=b'\n'.join(data['chunk0-attempt0/sealed/journal.jsonl'].splitlines()[:4])+b'\n'
        got=a.journal(partial,expected,f.PROFILE,request,0,complete=False)
        self.assertIsNotNone(got['pending']);self.assertEqual(len(got['ids']),0)
        with self.assertRaises(ValueError):a.journal(partial,expected,f.PROFILE,request,0,complete=True)
    def test_no_imports_of_native_production_or_old_verifier(self):
        import ast
        source=ast.parse((ROOT/'src/synsteg_audit/candidate_native_v2_independent_audit.py').read_text())
        modules=[n.module for n in ast.walk(source) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(m and ('author_candidate_local_native' in m or 'candidate_journal_guard' in m or 'framed_interval' in m or 'seeded_pair' in m) for m in modules))

if __name__=='__main__':unittest.main(verbosity=2)
