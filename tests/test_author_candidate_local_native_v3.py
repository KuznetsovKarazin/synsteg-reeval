"""Pure mock integration/fault tests; never creates a native bridge."""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from synsteg_audit import author_candidate_local_native_v3 as n
REQUESTS=json.loads((ROOT/n.REQUESTS).read_bytes())['requests']
PROFILE=json.loads((ROOT/n.PROFILE).read_bytes())
def abstain(r):return json.dumps({'words':[{w:[]} for w in r['words']]},separators=(',',':')).encode()
class MockBridge:
 def __init__(self,responses,directory,fault=None):
  self.responses=responses;self.directory=directory;self.fault=fault;self.evaluated=0
  self.vocabulary_size=PROFILE['ready']['vocabulary_size'];self.context_tokens=2048;self.process=None
  self.calls={k:0 for k in ('tokenize','start','distribution','render','step','close')}
 def tokenize(self,*_):
  self.calls['tokenize']+=1
  if self.fault=='tokenize_mutates':self.evaluated+=1
  return [50]*2048 if self.fault=='oversized_prompt' else [50,51]
 def start(self,ids):
  self.calls['start']+=1
  if self.fault=='prompt_error':raise RuntimeError('mock prompt failure')
  self.evaluated=len(ids)+int(self.fault=='prompt_bad_counter')
 def distribution(self):
  self.calls['distribution']+=1
  if self.fault=='dist_error':raise RuntimeError('mock distribution failure')
  if self.fault=='dist_mutates':self.evaluated+=1
  rows=[(100+i,struct.pack('>f',-float(i)/10).hex()) for i in range(50)]
  if self.fault=='duplicate_law':rows[-1]=rows[0]
  if self.fault=='nan_law':rows[0]=(100,'7fc00000')
  if self.fault=='masked_law':rows[0]=(PROFILE['special_ids'][0],'00000000')
  if self.fault=='zero_weights':rows=[(100,'00000000')]+[(101+i,'c47a0000') for i in range(49)]
  return rows
 def render(self,ids):
  self.calls['render']+=1
  if self.fault=='render_mutates':self.evaluated+=1
  if self.fault=='render_error':raise RuntimeError('mock render failure')
  if self.fault=='nonprefix' and len(ids)>1:return b'bad-prefix'
  if self.fault=='empty_piece':return b''
  if self.fault=='huge_piece' and ids[-1]==100:return b' '*262145
  if self.responses=='spaces':return b' '*len(ids)
  return self.responses[min(len(ids)-1,len(self.responses)-1)]
 def step(self,token):
  self.calls['step']+=1
  if self.fault=='step_error':raise RuntimeError('mock STEP failure')
  self.evaluated+=2 if self.fault=='step_bad_counter' else 1
 def close(self):
  self.calls['close']+=1
  if self.fault=='cleanup_error':raise RuntimeError('mock cleanup failure')
  if self.fault=='truncate_before_seal':
   p=self.directory.parent/'sealed/journal.jsonl';p.write_bytes(p.read_bytes().splitlines(keepends=True)[0])
class NativeV3Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.tmp=Path(self.temp.name);self.instances=[]
 def factory(self,responses,fault=None):
  def create(root,profile,directory,deadline):
   bridge=MockBridge(responses,directory,fault);self.instances.append(bridge);return bridge
  return create
 def attempt(self,responses=None,fault=None,request=None,name='attempt'):
  request=request or REQUESTS[0];responses=responses if responses is not None else [abstain(request)]
  d=self.tmp/name;c=self.tmp/(name+'.commitment.json')
  r=n.run_attempt(ROOT,PROFILE,request,0,d,c,protocol_sha256='9'*64,bridge_factory=self.factory(responses,fault))
  return r,d,c
 def verify(self,r,d,c,request=None):return n.verify_attempt_outcome(PROFILE,request or REQUESTS[0],0,d,c,'9'*64,r)
 def fixture(self):
  root=self.tmp/'root';root.mkdir()
  for rel in n.REQUIRED_CLOSURE:
   p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
  closure={'schema':n.ID+'-source-closure','files_sha256':{p:n.sha_file(root/p) for p in n.REQUIRED_CLOSURE}}
  n.write_json(root/n.CLOSURE,closure)
  intent=root/n.BASE/'test-intent.json';n.write_json(intent,n.execution_intent(root))
  activation=root/n.BASE/'test-activation.json';n.write_json(activation,n.enabled_activation(root,intent))
  files={**closure['files_sha256'],n.CLOSURE:n.sha_file(root/n.CLOSURE),intent.relative_to(root).as_posix():n.sha_file(intent),activation.relative_to(root).as_posix():n.sha_file(activation)}
  receipt=root/'test-readback.json';n.write_json(receipt,{'schema':n.ID+'-source-readback','source_commit':'a'*40,'activation_commit':'b'*40,'all_files_exact':True,'files_sha256':files})
  (root/'results').mkdir(exist_ok=True)
  return root,root/n.OUTPUT,activation,intent,receipt
 def runner(self,kind='accepted',fault=None):
  def run(root,profile,request,order,d,c,**kw):
   responses=[abstain(request)] if kind=='accepted' else [b'```']
   return n.run_attempt(root,profile,request,order,d,c,**kw,bridge_factory=self.factory(responses,fault))
  return run
 def test_abstention_format_only(self):
  r,d,c=self.attempt();self.assertEqual(r['outcome'],'format_accepted');self.assertEqual(r['format_result']['abstained_word_rows'],5)
  self.assertIsNone(r['capacity_bits']);self.assertEqual(self.verify(r,d,c),r);self.assertEqual(r['render_calls_invoked'],50)
 def test_nonempty_not_semantic_certificate(self):
  rows=[{w:[]} for w in REQUESTS[0]['words']];rows[0][REQUESTS[0]['words'][0]]=['candidate'+str(i) for i in range(8)]
  r,d,c=self.attempt([json.dumps({'words':rows}).encode()]);self.assertEqual(r['format_result']['candidate_strings_returned'],8)
  self.assertFalse(r['semantic_equivalence_verified']);self.verify(r,d,c)
 def test_initial_empty_support_has_prompt_and_no_step(self):
  r,d,c=self.attempt([b'```']);self.assertEqual(r['outcome'],'empty_support');self.assertEqual(r['step_calls_invoked'],0)
  self.assertEqual(r['distribution_calls_invoked'],1);self.assertTrue(r['prompt_committed']);self.verify(r,d,c)
 def test_empty_support_after_prefix(self):
  r,d,c=self.attempt([b'{',b'{bad']);self.assertEqual(r['outcome'],'empty_support');self.assertEqual(r['step_calls_invoked'],1)
  self.assertEqual(r['distribution_calls_invoked'],2);self.verify(r,d,c)
 def test_whole_token_with_trailing_junk_never_selected(self):
  r,d,c=self.attempt([abstain(REQUESTS[0])+b'trailing']);self.assertEqual(r['outcome'],'empty_support');self.assertEqual(r['raw_bytes'],0);self.verify(r,d,c)
 def test_mixed_support_masks_invalid_piece(self):
  base=self.factory([b'```'])
  def factory(*args):
   bridge=base(*args);orig=bridge.render
   bridge.render=lambda ids: abstain(REQUESTS[0]) if ids[-1]==100 else orig(ids)
   return bridge
  d=self.tmp/'attempt';c=self.tmp/'commitment.json'
  r=n.run_attempt(ROOT,PROFILE,REQUESTS[0],0,d,c,protocol_sha256='9'*64,bridge_factory=factory)
  self.assertEqual(r['outcome'],'format_accepted');self.assertEqual(r['generated_ids'],[100]);self.verify(r,d,c)
 def test_zero_mass_rows_are_retained_but_not_supplied(self):
  r,d,c=self.attempt(fault='zero_weights');self.verify(r,d,c)
  events=[json.loads(x)['event'] for x in (d/'sealed/journal.jsonl').read_bytes().splitlines()]
  e=next(e for e in events if e['kind']=='distribution_committed')['sample_evidence']
  self.assertEqual(len(e['zero_mass_token_ids']),49);self.assertEqual(len(e['conditional_law']['accepted']),1)
 def test_resource_refusal_can_be_zero_step(self):
  r,d,c=self.attempt([b'```'],fault='huge_piece');self.assertEqual(r['outcome'],'resource_refusal');self.assertEqual(r['step_calls_invoked'],0);self.verify(r,d,c)
 def test_cap_has_no_extra_distribution(self):
  r,d,c=self.attempt('spaces');self.assertEqual(r['outcome'],'token_cap_refusal');self.assertEqual(r['step_calls_committed'],512)
  self.assertEqual(r['distribution_calls_invoked'],512);self.verify(r,d,c)
 def test_native_faults_unknown(self):
  for i,f in enumerate(('tokenize_mutates','oversized_prompt','prompt_error','prompt_bad_counter','dist_error','dist_mutates','duplicate_law','nan_law','masked_law','render_mutates','render_error','empty_piece','step_error','step_bad_counter','cleanup_error','truncate_before_seal')):
   with self.subTest(fault=f):
    r,d,c=self.attempt(fault=f,name='fault'+str(i));self.assertEqual(r['outcome'],'unknown');self.assertIsNone(r['format_result']);self.assertFalse(c.exists());self.assertLessEqual(self.instances[-1].calls['step'],1)
 def test_nonprefix_render_unknown(self):
  r,d,c=self.attempt([b'{',b'{"'],fault='nonprefix');self.assertEqual(r['outcome'],'unknown');self.assertTrue(r['pending_distribution_unknown']);self.assertEqual(r['step_calls_committed'],1)
 def test_pending_step_unknown(self):
  r,d,c=self.attempt(fault='step_error');self.assertTrue(r['pending_step_unknown']);self.assertEqual(r['step_calls_invoked'],1);self.assertEqual(r['step_calls_committed'],0)
 def test_changed_response_rejected(self):
  r,d,c=self.attempt();(d/'sealed/response.raw').write_bytes(b'');self.assertRaises(ValueError,self.verify,r,d,c)
 def test_changed_result_rejected(self):
  r,d,c=self.attempt();r['distribution_calls_invoked']=99;self.assertRaises(ValueError,self.verify,r,d,c)
 def test_truncated_journal_rejected(self):
  r,d,c=self.attempt();p=d/'sealed/journal.jsonl';p.write_bytes(p.read_bytes()[:-1]);self.assertRaises(ValueError,self.verify,r,d,c)
 def test_symlink_commitment_rejected(self):
  r,d,c=self.attempt();backup=self.tmp/'backup';c.rename(backup);c.symlink_to(backup);self.assertRaises(OSError,self.verify,r,d,c)
 def test_existing_attempt_refused(self):
  self.attempt();self.assertRaises(FileExistsError,self.attempt);self.assertEqual(len(self.instances),1)
 def test_nonzero_attempt_refused(self):
  with self.assertRaises(ValueError):n.seed_hex(REQUESTS[0]['request_sha256'],1)
 def test_campaign_accepts_two_fixed_chunks(self):
  args=self.fixture();r=n.run_campaign(*args,attempt_runner=self.runner());self.assertEqual(r['attempts_executed'],2);self.assertEqual(r['complete_campaign_format_accepted_chunks'],2)
 def test_campaign_empty_support_no_retry(self):
  args=self.fixture();r=n.run_campaign(*args,attempt_runner=self.runner('empty'));self.assertEqual(r['attempts_executed'],2);self.assertEqual([len(x['attempts']) for x in r['chunks']],[1,1]);self.assertEqual(r['complete_campaign_format_accepted_chunks'],0)
 def test_campaign_unknown_stops_second(self):
  args=self.fixture();r=n.run_campaign(*args,attempt_runner=self.runner(fault='step_error'));self.assertEqual(r['attempts_executed'],1);self.assertEqual(r['chunks'][1]['status'],'not_attempted');self.assertIsNone(r['complete_campaign_format_accepted_chunks']);self.assertIsNone(r['complete_campaign_format_accepted_chunks'])
 def test_campaign_resource_stops_second(self):
  args=self.fixture();r=n.run_campaign(*args,attempt_runner=self.runner('empty','huge_piece'));self.assertEqual(r['attempts_executed'],1);self.assertEqual(r['chunks'][1]['status'],'not_attempted');self.assertIsNone(r['complete_campaign_format_accepted_chunks'])
 def test_campaign_existing_output_refused(self):
  args=self.fixture();n.run_campaign(*args,attempt_runner=self.runner());self.assertRaises(FileExistsError,n.run_campaign,*args,attempt_runner=self.runner());self.assertEqual(len(self.instances),2)
 def test_disabled_activation_refused(self):
  args=list(self.fixture());args[2]=args[0]/n.DEFAULT_ACTIVATION;self.assertRaises(ValueError,n.run_campaign,*args,attempt_runner=self.runner());self.assertEqual(self.instances,[])
 def test_changed_source_gate_refused(self):
  args=self.fixture();p=args[0]/'src/synsteg_audit/author_candidate_schema_v3.py';p.write_bytes(p.read_bytes()+b'\n');self.assertRaises(ValueError,n.run_campaign,*args,attempt_runner=self.runner());self.assertEqual(self.instances,[])
 def test_project_lock_blocks_no_model(self):
  args=self.fixture();fd=os.open(args[0]/'results/project-nn.lock',os.O_RDWR|os.O_CREAT,0o600)
  try:
   fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);self.assertRaises(BlockingIOError,n.run_campaign,*args,attempt_runner=self.runner());self.assertEqual(self.instances,[])
  finally:os.close(fd)
if __name__=='__main__':unittest.main()
