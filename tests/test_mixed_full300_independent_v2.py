"""Independent adversarial fixtures; no model or tokenizer is imported."""
import copy, hashlib, importlib.util, json, pathlib, shutil, subprocess, sys, tempfile, unittest
HERE=pathlib.Path(__file__).resolve().parent
SOURCE=HERE.parent/'output/scripts/merge_large_development_full300_v2.py'
if not SOURCE.exists(): SOURCE=HERE.parent/'scripts/merge_large_development_full300_v2.py'
INPUTS=HERE/'inputs' if (HERE/'inputs').exists() else HERE.parent
spec=importlib.util.spec_from_file_location('reviewed_merger_v2', SOURCE)
merger=importlib.util.module_from_spec(spec);spec.loader.exec_module(merger)
BASE_FILES=list(merger.BASE_FILES.values())
def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return digest(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode())

def put(root,path,value):
 p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_bytes((json.dumps(value,sort_keys=True,indent=2)+'\n').encode() if not isinstance(value,bytes) else value)
 return p

def inputs(root):
 for path in BASE_FILES:
  p=root/path;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(INPUTS/path,p)
 return {'schema':merger.ID+'-input','campaign':'large-campaign-development-v2','base_ref':'4050291345c43776e9c93528b6afe292c062ba7a','snapshot_ref':'a'*40,'files_sha256':{},'model_contracts':{'tinyllama11':None,'qwen05':None,'falcon3_1b':None},'new_shards':[],'reservation_scope':{'snapshot_ref':'a'*40,'index_complete_for_supplied_roots':True,'roots':[],'files':[]}}

def refresh(root,m):
 m['files_sha256']={str(p.relative_to(root)):digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
 m['reservation_scope']['files']=sorted(p for p in m['files_sha256'] if any(p.startswith(v+'/') for v in m['reservation_scope']['roots']))

def rows(root):return [json.loads(v) for v in (root/merger.BASE_FILES['source']).read_bytes().splitlines() if v.strip()]

def add(root,m,alias='qwen05',order=1,status='completed',limit=4096,input_tokens=1200,terminal=True):
 model,rev=merger.MODELS[alias];src=rows(root)[order-1]
 runtime={'schema':'synthetic-runtime-not-an-experiment','python':'fixture-3.12','packages':{'torch':'fixture-only'}}
 meta='contracts/'+alias;runtime_path=meta+'/runtime.json';config_path=meta+'/config.json';context_path=meta+'/context.json';contract_path=meta+'/contract.json'
 put(root,runtime_path,runtime);put(root,config_path,{'max_position_embeddings':limit})
 put(root,context_path,{'schema':merger.ID+'-context-report','model_id':model,'revision':rev,'source_sha256':digest((root/merger.BASE_FILES['source']).read_bytes()),'model_config_sha256':digest((root/config_path).read_bytes()),'model_context_limit':limit,'records':[{'source_order':i,'record_id':r['record_id'],'prompt_sha256':r['prompt_sha256'],'input_tokens':input_tokens} for i,r in enumerate(rows(root),1)]})
 put(root,contract_path,{'schema':merger.ID+'-model-contract','model_alias':alias,'model_id':model,'revision':rev,'dtype':'float32','settings':copy.deepcopy(merger.SETTINGS),'runtime_path':runtime_path,'runtime_sha256':digest((root/runtime_path).read_bytes()),'context_report_path':context_path,'model_config_path':config_path,'execution_stratum':'synthetic-integration-fixture-not-real-model','timeout_seconds_including_load':1200})
 m['model_contracts'][alias]=contract_path
 scope='runs/fixture-'+alias;prefix=scope+f'/order{order:03d}'
 if scope not in m['reservation_scope']['roots']:m['reservation_scope']['roots'].append(scope)
 m['new_shards'].append({'model_alias':alias,'source_order':order,'path':prefix})
 identity=canonical(['local',model,src['record_id']]);record={k:src[k] for k in ['genre','prompt_sha256','record_id','source_group_id','split','target_words','topic']}
 put(root,prefix+'/reservation.json',{'schema':'synthetic-reservation','model_alias':alias,'source_order':order,'record_id':src['record_id'],'identity':identity,'status':'reserved_outcome_unknown','automatic_retries':0})
 if status!='reservation_only':
  result={'cover_text':'Synthetic test text only.','dtype':'float32','elapsed_seconds':1.0,'hit_output_limit':False,'returned_model':model,'revision':rev,'seed':int(canonical([20261024,model,src['record_id']])[:8],16),'stop_reason':'end_turn','usage':{'input_tokens':input_tokens,'output_tokens':9}} if status=='completed' else None
  attempt={'schema':'synsteg-budget-generation-attempt-v1','automatic_retries':0,'identity':identity,'model':model,'provider':'local','record':record,'status':status,'result':result}
  put(root,prefix+'/run/attempts/'+identity+'.json',attempt)
  put(root,prefix+'/run/run.json',{'schema':'synsteg-budget-generation-run-v1','automatic_retries':0,'dtype':'float32','model':model,'revision':rev,'provider':'local','runner_sha256':'4d5ab74c78c58214c1b5e11a346b4339eb0c6d08e713e8909d0d5fd9947c7e02','ledger_code_sha256':'c04ca4e3141a9b9873054ab71c38545705353c8609c12ec0536eaabcd2e0b19b','settings':copy.deepcopy(merger.SETTINGS),'selected_records_count':1,'selected_records':[record],'split':'development'})
  put(root,prefix+'/run/runtime.json',runtime)
  put(root,prefix+'/run/local_ledger/ledger.json',{'attempts':{identity:{'status':status,'automatic_retries':0,'result':result,'metadata':{'record':record}}}})
  put(root,prefix+'/activation-receipt.json',{'synthetic':True});put(root,prefix+'/child.json',{'pid':123})
  put(root,prefix+'/stdout.log',b'fixture only\n');put(root,prefix+'/stderr.log',b'')
  put(root,prefix+'/run/summary.json',{'synthetic':True});put(root,prefix+'/run/local_ledger/campaign.lock',b'0');put(root,prefix+'/run/run_guard/campaign.lock',b'0')
  if terminal:put(root,prefix+'/process.json',{'source_order':order,'identity':identity,'record_id':src['record_id'],'status':'completed_pending_remote_preservation' if status=='completed' else 'stopped_requires_review_no_retry','exit_code':0 if status=='completed' else 1,'timed_out':False,'timeout_seconds':1200,'automatic_retries':0})
 return prefix,identity

def seal(root,m,prefix):
 raw={str(p.relative_to(root/prefix)):digest(p.read_bytes()) for p in (root/prefix).rglob('*') if p.is_file() and p.name not in ('git-preservation-receipt.json','artifact-manifest.json')}
 if 'run/run.json' in raw:put(root,prefix+'/artifact-manifest.json',{'files_sha256':raw})
 raw={str(p.relative_to(root/prefix)):digest(p.read_bytes()) for p in (root/prefix).rglob('*') if p.is_file() and p.name!='git-preservation-receipt.json'}
 receipt={'readback_verified':True,'commit':'b'*40,'files_sha256':raw}
 if 'process.json'in raw:receipt['process_sha256']=raw['process.json']
 put(root,prefix+'/git-preservation-receipt.json',receipt);refresh(root,m)

class IndependentCases(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)/'root';self.root.mkdir();self.m=inputs(self.root)
 def tearDown(self):self.tmp.cleanup()
 def result(self):refresh(self.root,self.m);return merger.merge(self.root,self.m)
 def make(self,**kw):p,i=add(self.root,self.m,**kw);seal(self.root,self.m,p);return p,i
 def reject(self):
  with self.assertRaises((ValueError,KeyError)):self.result()
 def edit(self,path,fn):
  j=json.loads((self.root/path).read_bytes());fn(j);put(self.root,path,j)
 def test_base_300_and_blocked91_not_promoted(self):
  r=self.result();self.assertEqual(r['counts'],{'completed':89,'reserved_outcome_unknown':1,'not_attempted':210});self.assertEqual(len(r['records']),300);self.assertEqual(r['scheduling_overlays'][0]['source_order'],91);self.assertEqual([x['status'] for x in r['records'] if x['model_alias']=='tinyllama11' and x['source_order']==91],['not_attempted'])
 def test_qwen_actual_usage_schema_completed(self):
  self.make();self.assertEqual(self.result()['model_counts']['qwen05'],{'completed':1,'not_attempted':99})
 def test_falcon_unknown_preserves_denominator(self):
  self.make(alias='falcon3_1b',status='reservation_only');r=self.result();self.assertEqual(r['counts']['reserved_outcome_unknown'],2);self.assertEqual(r['counts']['not_attempted'],209)
 def test_explicit_failed_is_neither_unknown_nor_success(self):
  self.make(status='failed');self.assertEqual(self.result()['counts']['failed'],1)
 def test_missing_terminal_cannot_be_completed(self):
  self.make(terminal=False);r=self.result();self.assertEqual(r['counts']['completed'],89);self.assertEqual(r['counts']['reserved_outcome_unknown'],2)
 def test_qwen_model_specific_context_above_tiny_limit(self):
  self.make(input_tokens=2500,limit=4096);self.assertEqual(self.result()['counts']['completed'],90)
 def test_same_input_overflows_smaller_context(self):
  self.make(input_tokens=2500,limit=2048);self.reject()
 def test_missing_reservation_identity_rejected(self):
  p,_=self.make(status='reservation_only');self.edit(p+'/reservation.json',lambda j:j.pop('identity'));seal(self.root,self.m,p);self.reject()
 def test_modified_scientific_runner_rejected(self):
  p,_=self.make();self.edit(p+'/run/run.json',lambda j:j.update(runner_sha256='0'*64));seal(self.root,self.m,p);self.reject()
 def test_changed_result_seed_rejected(self):
  p,i=self.make();ap=p+'/run/attempts/'+i+'.json';self.edit(ap,lambda j:j['result'].update(seed=123));a=json.loads((self.root/ap).read_text());self.edit(p+'/run/local_ledger/ledger.json',lambda j:j['attempts'][i].update(result=a['result']));seal(self.root,self.m,p);self.reject()
 def test_wrong_runtime_rejected_even_if_receipt_updated(self):
  p,_=self.make();self.edit(p+'/run/runtime.json',lambda j:j.update(python='different'));seal(self.root,self.m,p);self.reject()
 def test_missing_raw_byte_does_not_become_unknown(self):
  p,_=self.make();(self.root/p/'run/runtime.json').unlink();refresh(self.root,self.m);self.reject()
 def test_missing_receipt_not_counted_as_unattempted(self):
  p,_=self.make(status='reservation_only');(self.root/p/'git-preservation-receipt.json').unlink();self.reject()
 def test_duplicate_cell_rejected(self):
  self.make();self.m['new_shards'].append(copy.deepcopy(self.m['new_shards'][0]));self.reject()
 def test_orphan_reservation_rejected(self):
  self.make();self.m['new_shards']=[];self.reject()
 def test_unknown82_cannot_be_overwritten(self):
  self.make(alias='tinyllama11',order=82,status='reservation_only');self.reject()
 def test_blocked91_cannot_be_imported(self):
  self.make(alias='tinyllama11',order=91,status='reservation_only');self.reject()
 def test_old_completed90_cannot_be_replaced(self):
  self.make(alias='tinyllama11',order=90);self.reject()
 def test_reserved_boolean_order_rejected(self):
  self.make();self.m['new_shards'][0]['source_order']=True;self.reject()
 def test_changed_base_even_updated_manifest_rejected(self):
  p=merger.BASE_FILES['base'];self.edit(p,lambda j:j['records'][0].update(status='completed'));self.reject()
 def test_two_new_arms_change_all_300_denominators(self):
  self.make();self.make(alias='falcon3_1b',status='reservation_only');r=self.result()
  self.assertEqual(r['counts'],{'completed':90,'reserved_outcome_unknown':2,'not_attempted':208});self.assertEqual(r['model_denominators'],{'tinyllama11':100,'qwen05':100,'falcon3_1b':100})
 def test_missing_completed_core_even_resigned_rejected(self):
  p,_=self.make();(self.root/p/'activation-receipt.json').unlink();seal(self.root,self.m,p);self.reject()
 def test_unverified_preservation_receipt_rejected(self):
  p,_=self.make();self.edit(p+'/git-preservation-receipt.json',lambda j:j.update(readback_verified=False));self.reject()
 def test_complete_result_context_mismatch_rejected(self):
  p,i=self.make();ap=p+'/run/attempts/'+i+'.json';self.edit(ap,lambda j:j['result']['usage'].update(input_tokens=1199));a=json.loads((self.root/ap).read_bytes());self.edit(p+'/run/local_ledger/ledger.json',lambda j:j['attempts'][i].update(result=a['result']));seal(self.root,self.m,p);self.reject()
 def test_context_report_cannot_rebind_source(self):
  self.make();self.edit('contracts/qwen05/context.json',lambda j:j['records'][0].update(prompt_sha256='0'*64));self.reject()
 def test_selected_metadata_tamper_rejected(self):
  p,i=self.make();ap=p+'/run/attempts/'+i+'.json';self.edit(ap,lambda j:j['record'].update(target_words=999));a=json.loads((self.root/ap).read_bytes());self.edit(p+'/run/run.json',lambda j:j.update(selected_records=[a['record']]));self.edit(p+'/run/local_ledger/ledger.json',lambda j:j['attempts'][i]['metadata'].update(record=a['record']));seal(self.root,self.m,p);self.reject()
 def test_completed_with_timed_out_process_rejected(self):
  p,_=self.make();self.edit(p+'/process.json',lambda j:j.update(timed_out=True));seal(self.root,self.m,p);self.reject()
 def cli(self,output,wrong_pin=False):
  refresh(self.root,self.m);manifest=put(pathlib.Path(self.tmp.name),'manifest.json',self.m)
  return subprocess.run([sys.executable,str(SOURCE),'--root',str(self.root),'--manifest',str(manifest),'--expected-manifest-sha256','0'*64 if wrong_pin else digest(manifest.read_bytes()),'--output',str(output)],capture_output=True,text=True)
 def test_cli_output_cannot_mutate_bound_input_root(self):
  before=dict(self.m['files_sha256']);r=self.cli(self.root/'result.json');self.assertNotEqual(r.returncode,0);self.assertFalse((self.root/'result.json').exists());self.assertIn('output must be outside input root',r.stderr)
 def test_cli_wrong_manifest_pin_creates_no_output(self):
  output=pathlib.Path(self.tmp.name)/'out.json';r=self.cli(output,wrong_pin=True);self.assertNotEqual(r.returncode,0);self.assertFalse(output.exists())
 def test_cli_actual_base_has_input_and_merger_digest(self):
  output=pathlib.Path(self.tmp.name)/'out.json';r=self.cli(output);self.assertEqual(r.returncode,0,r.stderr);j=json.loads(output.read_bytes());self.assertEqual(j['merger_sha256'],digest(SOURCE.read_bytes()));self.assertEqual(j['input_manifest_sha256'],digest((pathlib.Path(self.tmp.name)/'manifest.json').read_bytes()));self.assertEqual(j['counts'],{'completed':89,'reserved_outcome_unknown':1,'not_attempted':210})
 def test_duplicate_json_and_nonfinite_rejected(self):
  for raw in [b'{"x":1,"x":2}',b'{"x":NaN}']:
   with self.assertRaises(ValueError):merger.parse(raw)
 def test_symlink_ancestor_refused(self):
  refresh(self.root,self.m);link=pathlib.Path(self.tmp.name)/'linked';link.symlink_to(self.root,target_is_directory=True)
  with self.assertRaises(ValueError):merger.merge(link,self.m)

if __name__=='__main__':unittest.main(verbosity=2)
