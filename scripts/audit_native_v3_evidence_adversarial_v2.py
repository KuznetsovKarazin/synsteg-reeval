#!/usr/bin/env python3
import argparse,copy,hashlib,importlib.util,json,os,shutil,sys,tempfile,time
from pathlib import Path
parser=argparse.ArgumentParser(description='Independent synthetic adversarial validation of native-v3 evidence auditor; no native/model calls')
parser.add_argument('--auditor',type=Path,required=True)
parser.add_argument('--fixtures',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
source=args.auditor.resolve();BASE=source.parent;OUT=args.output.resolve();FIXTURES=args.fixtures.resolve()
oracle=BASE/'probe_oracle.py'
fd=os.open(OUT,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
snapshot=OUT.with_suffix('.auditor.py');
with snapshot.open('xb') as f:f.write(source.read_bytes())
oraclesnapshot=OUT.with_suffix('.oracle.py');
with oraclesnapshot.open('xb') as f:f.write(oracle.read_bytes())
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('audit_under_test',source);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
canon=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
hashraw=lambda x:hashlib.sha256(x).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,x):p.write_bytes(canon(x))
checks=[]
runrel='results/author-candidate-local-native-v3/native-diagnostic-v3'
def test(name,mutation=None,fixture='complete-None',should_pass=False):
 with tempfile.TemporaryDirectory(prefix='audit-adversary-') as d:
  r=Path(d)/'root';shutil.copytree(FIXTURES/fixture,r);run=r/runrel
  if mutation:mutation(run)
  audit=a.Auditor(r)
  try:audit.campaign(run);accepted=True;error=None
  except BaseException as e:accepted=False;error=type(e).__name__+': '+str(e)
  checks.append({'name':name,'passed':accepted==should_pass,'accepted':accepted,'expected_accept':should_pass,'error':error,'auditor_checks':audit.checks})
def alter_result(run,key,value):
 p=run/'chunk0-attempt0/attempt-result.json';doc=read(p);doc[key]=value;write(p,doc)
 c=read(run/'result.json');c['chunks'][0]['attempts'][0]=doc;write(run/'result.json',c)
def extra_attempt(run):shutil.copytree(run/'chunk0-attempt0',run/'chunk0-attempt1')
def extra_commitment(run):shutil.copyfile(run/'commitments/chunk0-attempt0.json',run/'commitments/chunk0-attempt1.json')
def alter_probe(run,kind):
 p=run/'chunk0-attempt0/sealed/journal.jsonl';rows=[readline for readline in map(json.loads,p.read_bytes().splitlines())]
 events=[r['event'] for r in rows]
 e=next(e for e in events if e['kind']=='distribution_committed')['sample_evidence']
 if kind=='weight':e['weights'][0]['weight']+=1
 elif kind=='selection':e['selection']['token_id']+=1
 elif kind=='piece':e['contextual_pieces'][0]['piece_hex']='21'
 elif kind=='extra_field':next(e for e in events if e['kind']=='distribution_prepared')['unexpected_field']=True
 elif kind=='boolean_index':next(e for e in events if e['kind']=='distribution_prepared')['token_index']=False
 # Recompute internal prepare acknowledgements and every chain hash, terminal
 # manifest/member hashes and external commitment. This deliberately removes
 # trivial stale-hash defenses and exercises independent semantic replay.
 previous='0'*64;new=[];mapped={}
 for old,event in zip(rows,events):
  if 'prepared_sha256' in event:event['prepared_sha256']=mapped[event['prepared_sha256']]
  body={'sequence':old['sequence'],'previous_sha256':previous,'event':event};previous=hashraw(canon(body));mapped[old['event_sha256']]=previous
  new.append({**body,'event_sha256':previous})
 p.write_bytes(b''.join(canon(row)+b'\n' for row in new))
 sealed=p.parent;terminal=read(sealed/'terminal.json')
 local=read(sealed/'result.json');local.update(journal_bytes=p.stat().st_size,journal_sha256=hashraw(p.read_bytes()),journal_head_sha256=previous);write(sealed/'result.json',local)
 terminal['journal_head_sha256']=previous
 for name in terminal['files']:
  data=(sealed/name).read_bytes();terminal['files'][name]={'bytes':len(data),'sha256':hashraw(data)}
 write(sealed/'terminal.json',terminal);termsha=hashraw((sealed/'terminal.json').read_bytes())
 cp=run/'commitments/chunk0-attempt0.json';commit=read(cp);commit['terminal_sha256']=termsha;write(cp,commit)
 result=read(run/'chunk0-attempt0/attempt-result.json');result['terminal_sha256']=termsha;result['commitment_sha256']=hashraw(cp.read_bytes());write(run/'chunk0-attempt0/attempt-result.json',result)
 campaign=read(run/'result.json');campaign['chunks'][0]['attempts'][0]=result;write(run/'result.json',campaign)
for f in ['complete-None','empty-None','later_empty-None','complete-step_throw']:test('valid_'+f,fixture=f,should_pass=True)
for k,v in [('step_calls_committed',99),('distribution_calls_invoked',99),('render_calls_invoked',49),('prompt_token_count',99),('pending_step_unknown',True),('capacity_bits',1)]:test('reject_result_'+k,lambda r,k=k,v=v:alter_result(r,k,v))
test('reject_missing_reservation',lambda r:(r/'reservations/chunk0-attempt0.json').unlink())
test('reject_extra_reserved_attempt',lambda r:shutil.copyfile(r/'reservations/chunk0-attempt0.json',r/'reservations/chunk0-attempt1.json'))
test('reject_extra_attempt_directory',extra_attempt)
test('reject_extra_commitment',extra_commitment)
for kind in ['weight','selection','piece','extra_field','boolean_index']:test('reject_resealed_'+kind,lambda r,kind=kind:alter_probe(r,kind))
report={'schema':'native-v3-evidence-auditor-adversarial-review-v2','auditor_sha256':hashraw(snapshot.read_bytes()),'oracle_sha256':hashraw(oraclesnapshot.read_bytes()),'checks':checks,'passed':sum(c['passed'] for c in checks),'total':len(checks),'model_calls':0,'native_calls':0,'tokenizer_calls':0,'scope':'Synthetic saved fixtures and mutations, not native data','completed_at_unix':time.time()}
with os.fdopen(fd,'wb') as f:f.write(json.dumps(report,indent=2).encode()+b'\n')
print(json.dumps({'passed':report['passed'],'total':len(checks),'failures':[c for c in checks if not c['passed']]}))
