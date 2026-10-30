"""Independent synthetic preflight tests; no actual capacity aggregates are run."""
import collections,copy,hashlib,importlib.util,json,pathlib,shutil,tempfile,unittest
HERE=pathlib.Path(__file__).resolve().parent
AUDIT=HERE.parent/'output/scripts/audit_public_dictionary_full_development_v3.py'
ORACLE=HERE/'check_full_development_capacity_independent_v3.py'
INPUTS=HERE.parent/'inputs'
PLAN=HERE.parent/'output/experiments/public-dictionary-full-development-v3/plan.json'
if not AUDIT.exists():
 AUDIT=HERE.parent/'scripts/audit_public_dictionary_full_development_v3.py';ORACLE=HERE.parent/'scripts/check_full_development_capacity_independent_v3.py';INPUTS=HERE.parent;PLAN=HERE.parent/'experiments/public-dictionary-full-development-v3/plan.json'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
a=module('capacity_author_under_review',AUDIT);o=module('independent_capacity_oracle_v3',ORACLE)
def put(root,path,value):
 p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n');return p
def load(path):return json.loads(path.read_bytes())

class IndependentCapacity(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)/'inputs';self.root.mkdir();self.plan=load(PLAN)
  for path in a.PINS:
   dest=self.root/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(INPUTS/path,dest)
  self.sources=[json.loads(line)for line in(self.root/a.SOURCE).read_bytes().splitlines()if line.strip()]
  models={'tinyllama11':('TinyLlama/TinyLlama-1.1B-Chat-v1.0','fe8a4ea1ffedaf415f4da2f062534de366a451e6'),'qwen05':('Qwen/Qwen2.5-0.5B-Instruct','7ae557604adf67be50417f59c2c2f167def9a775'),'falcon3_1b':('tiiuae/Falcon3-1B-Instruct','28ba2251970a01dd1edc7ba7dad2eb71216ccfdf')}
  records=[];indexed=[];evidence=[]
  token=load(self.root/'data/shared_dictionary_recovery_v2.json')['classes'][0]['words'][0]
  for alias,(model,rev)in models.items():
   for n,src in enumerate(self.sources,1):
    completed=alias in ['tinyllama11','qwen05']and n==1
    status='completed'if completed else'reserved_outcome_unknown'if alias=='tinyllama11'and n==82 else'not_attempted'
    records.append({'model_alias':alias,'source_order':n,'record_id':src['record_id'],'status':status})
    if alias=='falcon3_1b':continue
    path=None
    if completed:
     path=f'fixtures/synthetic-{alias}-order001.json';identity=o.canonical(['local',model,src['record_id']]);cover=(' '.join([token]*40))if alias=='tinyllama11'else'1234 @@'
     attempt={'status':'completed','provider':'local','model':model,'identity':identity,'automatic_retries':0,'record':{k:src[k]for k in ['genre','prompt_sha256','record_id','source_group_id','split','target_words','topic']},'result':{'cover_text':cover,'returned_model':model,'revision':rev,'dtype':'float32','seed':int(o.canonical([20261024,model,src['record_id']])[:8],16)}}
     p=put(self.root,path,attempt)
     if alias=='qwen05':evidence.append({'model_alias':alias,'source_order':n,'status':'completed','attempt_sha256':o.sha(p.read_bytes()),'cover_sha256':o.sha(cover.encode())})
    indexed.append({'model_alias':alias,'source_order':n,'record_id':src['record_id'],'status':status,'attempt_path':path})
  snapshot={'schema':'mixed-development-full300-merger-v2-result','snapshot_ref':'b'*40,'full_denominator':300,'records':records,'counts':dict(collections.Counter(x['status']for x in records)),'scheduling_overlays':load(self.root/a.GUARD)['blocked_baseline_intents'],'new_shard_evidence':evidence}
  put(self.root,'fixtures/synthetic-full300.json',snapshot)
  self.manifest={'schema':a.ID+'-input','actual_git_readback':True,'snapshot_commit':'b'*40,'snapshot_path':'fixtures/synthetic-full300.json','cover_sources':indexed,'files_sha256':{}}
  self.refresh()
 def tearDown(self):self.tmp.cleanup()
 def refresh(self):self.manifest['files_sha256']={str(p.relative_to(self.root)):o.sha(p.read_bytes())for p in self.root.rglob('*')if p.is_file()}
 def result(self):return a.evaluate(self.root,self.manifest,self.plan)
 def edit(self,path,fn):j=load(self.root/path);fn(j);put(self.root,path,j);self.refresh()
 def reject_input(self):
  with self.assertRaises((ValueError,KeyError)):a.validate_inputs(self.root,self.manifest,self.plan)
 def test_independent_oracle_accepts_complete_synthetic_grid(self):
  r=self.result();check=o.check(self.root,self.manifest,r);self.assertEqual(check['dictionary_rows_checked'],400);self.assertEqual(check['frame_conditions_checked'],4800);self.assertEqual(r['models']['qwen05']['dictionary_aggregates']['v2']['zero_capacity_covers'],1);self.assertEqual(r['models']['tinyllama11']['unavailable_covers'],99)
 def test_character_case_punctuation_compound_and_affix_rules(self):
  text='quick QUICK Quick qUiCk quick, «quick» _quick quick-slow 7quick équick quick\tquick';look={'quick':('q',4)};r=o.analyze(text,look)
  self.assertEqual([s['token_index']for s in r['slots']],[0,1,2,4,5,10,11]);self.assertEqual([s['case_style']for s in r['slots']],['lower','upper','title','lower','lower','lower','lower']);self.assertEqual(r['capacity_bits'],14)
 def test_dictionary_size_width_floor_and_cap(self):
  look={strchr:(strchr,size)for strchr,size in [('aa',2),('bb',3),('cc',4),('dd',8)]};self.assertEqual([x['width']for x in o.analyze('aa bb cc dd',look)['slots']],[1,1,2,2])
 def test_same_class_same_case_preserves_normalization(self):
  lookup={'quick':('q',2),'rapid':('q',2)};self.assertEqual(o.analyze('A quick, end.',lookup)['normalization_sha256'],o.analyze('A rapid, end.',lookup)['normalization_sha256'])
 def test_capacity_number_tampering_detected(self):
  r=self.result();r['dictionary_rows'][0]['capacity_bits']+=1
  with self.assertRaises(ValueError):o.check(self.root,self.manifest,r)
 def test_slot_span_tampering_detected(self):
  r=self.result();row=next(x for x in r['dictionary_rows']if x['slots']);row['slots'][0]['end']+=1
  with self.assertRaises(ValueError):o.check(self.root,self.manifest,r)
 def test_unknown_capacity_cannot_be_changed_to_zero(self):
  r=self.result();row=next(x for x in r['dictionary_rows']if x['model_alias']=='tinyllama11'and x['source_order']==82);row['capacity_bits']=0
  with self.assertRaises(ValueError):o.check(self.root,self.manifest,r)
 def test_repeated_secondary_parent_identity_is_checked(self):
  r=self.result();row=next(x for x in r['frame_conditions']if x['payload_kind']=='secondary16'and x['parent_primary_bits']==256);row['parent_primary_bits']=128
  with self.assertRaises(ValueError):o.check(self.root,self.manifest,r)
 def test_feasibility_flag_tampering_detected(self):
  r=self.result();r['frame_conditions'][0]['capacity_feasible']=not r['frame_conditions'][0]['capacity_feasible']
  with self.assertRaises(ValueError):o.check(self.root,self.manifest,r)
 def test_missing_null_field_is_not_valid_null(self):
  r=self.result();row=next(x for x in r['dictionary_rows']if x['source_order']==82);row.pop('capacity_bits')
  with self.assertRaises(ValueError):o.check(self.root,self.manifest,r)
 def test_boolean_zero_is_not_measured_integer_zero(self):
  r=self.result();row=next(x for x in r['dictionary_rows']if x['model_alias']=='qwen05'and x['source_order']==1);row['capacity_bits']=False
  with self.assertRaises(ValueError):o.check(self.root,self.manifest,r)
 def test_pooled_model_summary_rejected(self):
  r=self.result();r['models']['pooled']={}
  with self.assertRaises(ValueError):o.check(self.root,self.manifest,r)
 def test_new_shard_snapshot_hash_binding_enforced(self):
  self.edit('fixtures/synthetic-qwen05-order001.json',lambda j:j['result'].update(cover_text='different synthetic text'));self.reject_input()
 def test_snapshot_boolean_order_rejected(self):
  self.edit('fixtures/synthetic-full300.json',lambda j:j['records'][0].update(source_order=True));self.reject_input()
 def test_wrong_returned_revision_rejected(self):
  self.edit('fixtures/synthetic-tinyllama11-order001.json',lambda j:j['result'].update(revision='0'*40));self.reject_input()
 def test_wrong_record_seed_rejected(self):
  self.edit('fixtures/synthetic-tinyllama11-order001.json',lambda j:j['result'].update(seed=17));self.reject_input()
 def test_missing_completed_attempt_rejected(self):
  (self.root/'fixtures/synthetic-tinyllama11-order001.json').unlink();self.refresh();self.reject_input()
 def test_orphan_cover_file_rejected(self):
  put(self.root,'fixtures/orphan.json',{'synthetic':True});self.refresh();self.reject_input()
 def test_dictionary_change_rejected_even_rehashed(self):
  self.edit('data/shared_dictionary_recovery_v2.json',lambda j:j.update(version='tampered-fixture'));self.reject_input()

if __name__=='__main__':unittest.main(verbosity=2)
