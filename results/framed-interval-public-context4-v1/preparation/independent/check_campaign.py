"""Synthetic state-machine checks; all model/resource/process operations replaced."""
import importlib.util,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
CHECKOUT=Path(os.environ.get('CONTEXT4_CHECKOUT',str(HERE.parent/'checkout'))).resolve()
SOURCE=CHECKOUT/'scripts/run_public_context4_v1.py'
spec=importlib.util.spec_from_file_location('context4_review_target',SOURCE)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

class CampaignTests(unittest.TestCase):
 def setup_campaign(self,root):
  h.put(root/h.PROFILE,{})
  h.put(root/h.SPEC,{})
  contexts=[{'context_id':f'c{i}','prompt_ids':[i+1]} for i in range(4)]
  h.put(root/h.OUT/'tokenization/contexts.json',{'contexts':contexts})
  return [{'context_id':f'c{i}'} for i in range(4)]
 def simulation(self,root,texts,mode):
  calls=[]
  core=SimpleNamespace(validate_public_context=lambda *a:None,
   make_envelope=lambda spec,ctx,nonce,**kwargs:{'context_id':ctx['context_id'],'nonce_hex':nonce.hex(),**{k:(v.hex() if isinstance(v,bytes) else v) for k,v in kwargs.items()}})
  def supervise(log,unused,seconds):
   self.assertEqual(seconds,300)
   argv=h.load(log/'command.json')['argv'];role=argv[argv.index('--role')+1];out=Path(argv[argv.index('--output')+1])
   calls.append((role,str(log)))
   if mode=='unknown':return {'status':'unknown'}
   out.mkdir()
   status=('authenticated_delivery' if role=='receive' else ('threshold_pending_receiver' if '--payload-file' in argv else 'ordinary_threshold_generated'))
   if mode=='backend':status='backend_error'
   if mode=='missing':status=None
   code={'status':status,'authenticated_delivery':role=='receive','payload_hex':bytes(range(16)).hex() if role=='receive' else None}
   record={'status':'returned','role':role,'execution_kind':'native','model_process_attempted':True,'core':code}
   if mode=='worker_error':record['synthetic_error']='must stop even with returned status'
   h.put(out/'worker-result.json',record)
   h.write(out/'committed-wire.bin',b'synthetic only')
   return {'status':'completed_child'}
  def collect(root,directory,observed,role,wire):
   result={'authenticated_delivery':role=='receive','reconciliation':{'wire_available':True,'unknown_native_step':False,'torn_last_record':False}}
   if mode=='evidence' and role=='receive':result.update(authenticated_delivery=False,delivery_evidence_error='synthetic wire mismatch')
   return result
  with patch.object(h,'imports',return_value=(core,None,None,supervise)),patch.object(h,'install_receiver',return_value={'synthetic':True}),patch.object(h,'collect',side_effect=collect):
   result=h.execute(root,texts)
  return result,calls,h.load(root/h.OUT/'generation/result.json')
 def test_fixed_fixtures_no_nonce_reuse(self):
  rows=h.fixtures();self.assertEqual(len(rows),8)
  self.assertEqual(len({(x['key_hex'],x['nonce_hex']) for x in rows}),8)
  self.assertEqual(len({x['key_hex'] for x in rows}),1)
  self.assertEqual([x['case_id'] for x in rows],[f'o{o:03d}-{a}' for o in [1,4,7,10] for a in ('stego','ordinary')])
  self.assertTrue(all(x['payload_hex']==(bytes(range(16)).hex() if x['arm']=='stego' else None) for x in rows))
 def test_full_twelve_endpoint_flow(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);texts=self.setup_campaign(root);ok,calls,result=self.simulation(root,texts,'success')
   self.assertTrue(ok);self.assertEqual(len(calls),12);self.assertEqual(result['exact_deliveries'],4)
   self.assertEqual(sum(r=='receive' for r,_ in calls),4)
   with self.assertRaises(FileExistsError):self.simulation(root,texts,'success')
 def test_unknown_stops_first_retains_full_denominator(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ok,calls,r=self.simulation(root,self.setup_campaign(root),'unknown')
   self.assertFalse(ok);self.assertEqual(len(calls),1);self.assertEqual(r['planned_endpoints'],12)
   self.assertEqual(r['payload_delivery_denominator'],4);self.assertEqual(len(r['cases']),8)
   self.assertTrue(all(x['exact_payload'] is None for x in r['cases']))
 def test_backend_error_stops(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ok,calls,r=self.simulation(root,self.setup_campaign(root),'backend')
   self.assertFalse(ok);self.assertEqual(len(calls),1)
 def test_missing_status_stops(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ok,calls,r=self.simulation(root,self.setup_campaign(root),'missing')
   self.assertFalse(ok);self.assertEqual(len(calls),1)

 def test_delivery_evidence_error_stops(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ok,calls,r=self.simulation(root,self.setup_campaign(root),'evidence')
   self.assertFalse(ok);self.assertEqual(len(calls),2);self.assertEqual(r['exact_deliveries'],0)
 def test_worker_error_field_stops(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ok,calls,r=self.simulation(root,self.setup_campaign(root),'worker_error')
   self.assertFalse(ok);self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main(verbosity=2)
