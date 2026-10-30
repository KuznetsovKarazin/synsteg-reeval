"""Synthetic Git-proof gate checks; no package/native imports or model calls."""
import importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
CHECKOUT=Path(os.environ.get('CONTEXT4_CHECKOUT',str(HERE.parent/'checkout'))).resolve()
spec=importlib.util.spec_from_file_location('context4_gate_target',CHECKOUT/'scripts/run_public_context4_v1.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def replace(path,value):path.write_bytes(h.canonical(value))
class GateTests(unittest.TestCase):
 def setup_tree(self,root,stage,mutate=None):
  for name in h.REQUIRED_SOURCE:h.write(root/name,(CHECKOUT/name).read_bytes())
  p=root/h.EXP/'protocol.proposed.json';protocol=h.load(p);protocol['execution_root']=str(root);replace(p,protocol)
  if stage=='execute':
   texts=h.load(root/h.EXP/'selected-context-texts.json')['rows'];contexts=[]
   for i,row in enumerate(texts):
    c={'context_id':row['context_id'],'prompt_ids':[i+1]};contexts.append(c)
   if mutate=='context':contexts[0]['context_id']='wrong-source-binding'
   for i,c in enumerate(contexts):
    d=root/h.OUT/f'tokenization/c{i:02d}'
    h.put(d/'public-context.json',c)
    h.put(d/'tokenization.json',{'status':'completed','tokenizer_calls':1,'prompt_calls':0,'step_calls':0,'prompt_ids':c['prompt_ids']})
    h.put(d/'supervisor-result.json',{'status':'completed_child'})
   h.put(root/h.OUT/'tokenization/contexts.json',{'count':4,'contexts':contexts})
  hashes={name:h.digest((root/name).read_bytes()) for name in h.REQUIRED_SOURCE}
  if mutate=='closure':hashes.pop(h.OLD_PILOT)
  closure={'schema':h.ID+'-source-closure','files_sha256':hashes};h.put(root/h.EXP/'source-closure.json',closure)
  activation=root/h.EXP/('activation-'+stage+'.json')
  h.put(activation,{'schema':h.ID+'-activation','stage':stage,'enabled':True,'proposal_sha256':h.digest(p.read_bytes()),'closure_sha256':h.digest((root/h.EXP/'source-closure.json').read_bytes()),'automatic_retries':0})
  expected={**hashes,str(activation.relative_to(root)):h.digest(activation.read_bytes()),h.EXP+'/source-closure.json':h.digest((root/h.EXP/'source-closure.json').read_bytes())}
  if stage=='execute':expected.update({h.OUT+'/tokenization/'+p:s for p,s in h.inventory(root/h.OUT/'tokenization').items()})
  proof=root/'synthetic-proof.json';h.put(proof,{'schema':h.ID+'-readback','stage':stage,'all_files_exact':True,'commit':'a'*40,'files_sha256':expected})
  if mutate=='source':(root/h.OLD_PILOT).write_bytes(b'changed after synthetic readback')
  return activation,proof
 def invoke(self,root,stage,mutate=None):
  a,p=self.setup_tree(root,stage,mutate)
  with patch.object(h,'__file__',str(root/h.HELPER)):return h.gate(root,stage,a,p)
 def test_tokenization_gate_accepts_complete_synthetic_contract(self):
  with tempfile.TemporaryDirectory() as d:self.assertEqual(len(self.invoke(Path(d),'tokenize')[1]),4)
 def test_execute_gate_accepts_complete_synthetic_contract(self):
  with tempfile.TemporaryDirectory() as d:self.assertEqual(len(self.invoke(Path(d),'execute')[1]),4)
 def test_missing_closure_path_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'complete exact source closure'):self.invoke(Path(d),'tokenize','closure')
 def test_changed_source_after_readback_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'source changed'):self.invoke(Path(d),'tokenize','source')
 def test_context_source_mapping_rejected_even_with_consistent_hashes(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'source mapping differs'):self.invoke(Path(d),'execute','context')
if __name__=='__main__':unittest.main(verbosity=2)
