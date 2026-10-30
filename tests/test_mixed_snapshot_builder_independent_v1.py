"""Independent offline adapter/builder tests. Synthetic shards are not experiments."""
import copy, hashlib, importlib.util, json, pathlib, shutil, tempfile, unittest
from unittest import mock
HERE=pathlib.Path(__file__).resolve().parent
WORK=HERE.parent.parent
BUILDER=HERE.parent/'output/scripts/build_mixed_development_snapshot_v1.py'
MERGER=WORK/'mixed300-merger-restored-v2/output/scripts/merge_large_development_full300_v2.py'
PREVIOUS=WORK/'mixed300-merger-restored-v2/review/test_merger_independent.py'
ACTUAL=WORK/'baseline92-continuation-v2/checkout'
if not BUILDER.exists():
 BUILDER=HERE.parent/'scripts/build_mixed_development_snapshot_v1.py';MERGER=HERE.parent/'scripts/merge_large_development_full300_v2.py';PREVIOUS=HERE/'test_mixed_full300_independent_v2.py';ACTUAL=HERE.parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
b=load('builder_under_independent_review',BUILDER);m=b.merger_module(MERGER);f=load('independent_shard_builder',PREVIOUS)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def parsed(path):return json.loads(path.read_bytes())
def write(root,path,value):return f.put(root,path,value)

class BuilderIndependent(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)/'input';self.root.mkdir();self.old=f.inputs(self.root);self.output=pathlib.Path(self.tmp.name)/'bundle'
  for path in [b.TINY_CONTEXT,b.TINY_CONFIG]:
   dest=self.root/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile((ACTUAL/path) if (ACTUAL/path).exists() else HERE/'fixtures/mixed-builder-tiny-config-v1.json',dest)
  self.original_context=parsed(self.root/b.TINY_CONTEXT)
  self.sources=f.rows(self.root)
  self.prefix,self.identity=self.add_tiny()
  self.request=b.draft_request(m,self.root,'c'*40)
  self.request['remote_census'].update(readback_verified=True,branches_and_prs_checked=True)
 def tearDown(self):self.tmp.cleanup()
 def add_tiny(self,order=92):
  n=self.original_context['rows'][order-1]['input_tokens'];old,identity=f.add(self.root,self.old,alias='tinyllama11',order=order,input_tokens=n,limit=2048)
  prefix=b.TINY_ROOT+f'/order{order:03d}';target=self.root/prefix;target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(self.root/old),str(target))
  runtime=ACTUAL/'runs/baseline-single-record-recovery-v2/order092/run/runtime.json'
  if not runtime.exists(): runtime=HERE.parent/'runs/baseline-single-record-recovery-v1/order090/run/runtime.json'
  shutil.copyfile(runtime,target/'run/runtime.json');f.seal(self.root,self.old,prefix)
  return prefix,identity
 def build(self):
  raw=b.encoded(self.request);return b.build(m,self.root,raw,hashlib.sha256(raw).hexdigest(),self.output)
 def context(self,value=None):return b.tiny_context(m,b.encoded(value or self.original_context),self.sources,b.TINY_CONFIG_SHA,2048)
 def refresh_census(self):self.request['remote_census']['files_sha256']=b.census(m,self.root)[0]
 def reject_build(self):
  with self.assertRaises((ValueError,FileNotFoundError)):self.build()
  self.assertFalse(self.output.exists())
 def add_qwen(self):
  old,identity=f.add(self.root,self.old,alias='qwen05',order=1,input_tokens=1200,limit=4096)
  prefix=b.QWEN_ROOT+'/qwen05/order001';target=self.root/prefix;target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(self.root/old),str(target));f.seal(self.root,self.old,prefix)
  paths={'runtime_path':'runtime.json','context_report_path':'all100-context.json','model_config_path':'config.json'}
  contract=parsed(self.root/'contracts/qwen05/contract.json')
  for key,name in paths.items():
   source=self.root/contract[key];new=b.QWEN_PRE+'/'+name;write(self.root,new,source.read_bytes());contract[key]=new
  write(self.root,b.QWEN_PRE+'/model-contract.json',contract)
  self.request=b.draft_request(m,self.root,'c'*40);self.request['remote_census'].update(readback_verified=True,branches_and_prs_checked=True)
 def test_two_arm_synthetic_bundle_uses_qwen_preflight(self):
  self.add_qwen();self.request['qwen_preflight']['commit']='e'*40;self.build();result=parsed(self.output/'full300.json');self.assertEqual(result['counts'],{'completed':91,'reserved_outcome_unknown':1,'not_attempted':208});self.assertEqual(result['model_counts']['qwen05'],{'completed':1,'not_attempted':99})
 def test_qwen_without_preflight_commit_rejected(self):
  self.add_qwen();self.assertIsNone(self.request['qwen_preflight']['commit']);self.reject_build()
 def test_interrupted_publication_leaves_no_bundle(self):
  with mock.patch.object(b,'publish_new_directory',side_effect=OSError('injected publication failure')):
   with self.assertRaises(OSError):self.build()
  self.assertFalse(self.output.exists());self.assertEqual(list(pathlib.Path(self.tmp.name).glob('mixed300-publish-*')),[])
 def test_output_ancestor_symlink_rejected(self):
  real=pathlib.Path(self.tmp.name)/'real';real.mkdir();link=pathlib.Path(self.tmp.name)/'link';link.symlink_to(real,target_is_directory=True);self.output=link/'bundle';self.reject_build()
 def test_new_reservation_during_staging_rejected(self):
  original=b.bound_read;injected=False
  def changing_read(*args,**kwargs):
   nonlocal injected
   data=original(*args,**kwargs)
   if not injected:
    injected=True;write(self.root,b.TINY_ROOT+'/order093/reservation.json',{'status':'reserved_outcome_unknown'})
   return data
  with mock.patch.object(b,'bound_read',side_effect=changing_read):self.reject_build()
 def test_actual_100_context_mapping_and_token_oracle(self):
  result=self.context();self.assertEqual(len(result['records']),100);self.assertEqual(result['model_context_limit'],2048)
  for converted,original,src in zip(result['records'],self.original_context['rows'],self.sources):
   self.assertEqual(converted,{'source_order':original['source_order'],'record_id':src['record_id'],'prompt_sha256':src['prompt_sha256'],'input_tokens':original['input_tokens']})
  self.assertEqual((min(x['input_tokens']for x in result['records']),max(x['input_tokens']for x in result['records'])),(215,346))
 def test_adapter_duplicate_order_rejected(self):
  value=copy.deepcopy(self.original_context);value['rows'][1]['source_order']=1
  with self.assertRaises(ValueError):self.context(value)
 def test_adapter_reordered_rows_rejected(self):
  value=copy.deepcopy(self.original_context);value['rows'][0],value['rows'][1]=value['rows'][1],value['rows'][0]
  with self.assertRaises(ValueError):self.context(value)
 def test_adapter_wrong_remaining_arithmetic_rejected(self):
  value=copy.deepcopy(self.original_context);value['rows'][0]['remaining_after_requested_output']+=1
  with self.assertRaises(ValueError):self.context(value)
 def test_adapter_boolean_count_rejected(self):
  value=copy.deepcopy(self.original_context);value['rows'][0]['input_tokens']=True
  with self.assertRaises(ValueError):self.context(value)
 def test_adapter_changed_source_prompt_rejected(self):
  value=copy.deepcopy(self.original_context);value['rows'][0]['prompt_sha256']='0'*64
  with self.assertRaises(ValueError):self.context(value)
 def test_tiny92_synthetic_bundle_validates_frozen_merger(self):
  self.build();result=parsed(self.output/'full300.json');self.assertEqual(result['counts'],{'completed':90,'reserved_outcome_unknown':1,'not_attempted':209});manifest=parsed(self.output/'input-manifest.json');self.assertEqual(m.merge(self.output/'evidence',manifest)['counts'],result['counts']);self.assertEqual(result['merger_sha256'],digest(MERGER));self.assertEqual(parsed(self.output/'build-provenance.json')['actual_new_shard_count'],1)
 def test_draft_cannot_build_without_remote_attestation(self):
  self.request=b.draft_request(m,self.root,'c'*40);self.assertIs(self.request['remote_census']['readback_verified'],False);self.assertIs(self.request['remote_census']['branches_and_prs_checked'],False);self.reject_build()
 def test_missing_receipt_never_synthesized(self):
  (self.root/self.prefix/'git-preservation-receipt.json').unlink();self.refresh_census();self.reject_build()
 def test_extra_local_reservation_stale_census_rejected(self):
  write(self.root,b.TINY_ROOT+'/order093/reservation.json',{'status':'reserved_outcome_unknown'});self.reject_build()
 def test_extra_local_reservation_even_updated_census_rejected(self):
  write(self.root,b.TINY_ROOT+'/order093/reservation.json',{'status':'reserved_outcome_unknown'});self.refresh_census();self.reject_build()
 def test_orphan_raw_file_even_updated_census_rejected(self):
  write(self.root,b.TINY_ROOT+'/unindexed.log',b'not indexed\n');self.refresh_census();self.reject_build()
 def test_tampered_raw_even_updated_census_rejected(self):
  write(self.root,self.prefix+'/stdout.log',b'changed\n');self.refresh_census();self.reject_build()
 def test_forged_primary_commit_binding_rejected(self):
  self.request['primary_receipts'][self.prefix]['commit']='d'*40;self.reject_build()
 def test_unpreserved_terminal_cannot_build(self):
  (self.root/self.prefix/'process.json').unlink();f.seal(self.root,self.old,self.prefix);self.request=b.draft_request(m,self.root,'c'*40);self.request['remote_census'].update(readback_verified=True,branches_and_prs_checked=True);self.reject_build()
 def test_wrong_request_pin_rejected_without_output(self):
  with self.assertRaises(ValueError):b.build(m,self.root,b.encoded(self.request),'0'*64,self.output)
  self.assertFalse(self.output.exists())
 def test_source_input_symlink_rejected(self):
  path=self.root/b.TINY_CONTEXT;raw=path.read_bytes();path.unlink();target=pathlib.Path(self.tmp.name)/'other.json';target.write_bytes(raw);path.symlink_to(target);self.reject_build()
 def test_output_inside_source_root_rejected(self):
  self.output=self.root/'new-bundle';self.reject_build()
 def test_existing_destination_preserved(self):
  self.output.mkdir();(self.output/'sentinel').write_bytes(b'keep')
  with self.assertRaises(ValueError):self.build()
  self.assertEqual((self.output/'sentinel').read_bytes(),b'keep')
 def test_known_supervisor_lock_excluded_and_reported(self):
  write(self.root,b.TINY_ROOT+'/supervisor.lock',b'')
  self.build();provenance=parsed(self.output/'build-provenance.json');self.assertIn(b.TINY_ROOT+'/supervisor.lock',provenance['local_operational_files_excluded']);self.assertNotIn(b.TINY_ROOT+'/supervisor.lock',parsed(self.output/'input-manifest.json')['files_sha256'])

if __name__=='__main__':unittest.main(verbosity=2)
