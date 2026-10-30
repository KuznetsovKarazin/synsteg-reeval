#!/usr/bin/env python3
"""Offline independent preparation binding and no-overwrite fault checks."""
from pathlib import Path
import contextlib,hashlib,importlib.util,io,json,os,shutil,sys,tempfile
from unittest.mock import patch

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'results/author-candidate-schema-v3-external-review-v1/reproduced-prepare-review.json'
if OUT.exists() or OUT.is_symlink():raise FileExistsError('refuse existing audit output')
HELPER=ROOT/'scripts/prepare_author_candidate_schema_v3.py'
spec=importlib.util.spec_from_file_location('prepare_under_review',HELPER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def ck(name,value):checks.append({'name':name,'pass':bool(value)})
def refuses(fn):
 try:fn();return False
 except (ValueError,OSError):return True

def run():
 with contextlib.redirect_stdout(io.StringIO()):m.main()

with tempfile.TemporaryDirectory() as td:
 root=Path(td)/'repo';root.mkdir();m.ROOT=root;m.BASE=root/'experiments/author-candidate-schema-v3'
 for p in [m.V2_REQUESTS,'src/synsteg_audit/author_candidate_format_v2.py']:
  target=root/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,target)
 run();outputs=sorted(m.BASE.glob('*.json'));raw={p.name:p.read_bytes() for p in outputs}
 ck('three_bound_outputs',len(outputs)==3)
 ck('prepared_matches_committed_intended_bytes',all(raw[n]==(ROOT/'experiments/author-candidate-schema-v3'/n).read_bytes() for n in raw))
 ck('activation_disabled',json.loads(raw['activation.disabled.json'])['enabled'] is False)
 ck('new_native_calls_zero',json.loads(raw['protocol.json'])['model_calls']==0)
 before={p.name:p.stat().st_mtime_ns for p in outputs};run()
 ck('identical_existing_outputs_not_rewritten',all(p.read_bytes()==raw[p.name] and p.stat().st_mtime_ns==before[p.name] for p in outputs))
 for p in [m.V2_REQUESTS,'src/synsteg_audit/author_candidate_format_v2.py']:
  target=root/p;old=target.read_bytes();target.write_bytes(old+b'\n')
  ck('input_binding_'+Path(p).name,refuses(run));target.write_bytes(old)
  ck('binding_refusal_did_not_touch_outputs_'+Path(p).name,all(p.read_bytes()==raw[p.name] for p in outputs))
 dest=root/'different.json';dest.write_bytes(b'other-author-content')
 ck('different_file_refused',refuses(lambda:m.save(dest,{'test':1})))
 ck('different_file_preserved',dest.read_bytes()==b'other-author-content')
 with patch.object(Path,'exists',return_value=False):
  ck('creation_race_refused',refuses(lambda:m.save(dest,{'test':1})))
 ck('creation_race_content_preserved',dest.read_bytes()==b'other-author-content')
 for name,target in [('existing_symlink',dest),('dangling_symlink',root/'nonexistent-target')]:
  link=root/name;link.symlink_to(target)
  ck(name+'_refused',refuses(lambda:m.save(link,{'test':1})))
  ck(name+'_unchanged',link.is_symlink() and os.readlink(link)==str(target))
 fifo=root/'pipe';os.mkfifo(fifo)
 ck('fifo_refused_without_blocking',refuses(lambda:m.save(fifo,{'test':1})))
 directory=root/'directory';directory.mkdir()
 ck('directory_refused',refuses(lambda:m.save(directory,{'test':1})))

report={'schema':'candidate-schema-v3-prepare-external-review-v1','helper_sha256':hashlib.sha256(HELPER.read_bytes()).hexdigest(),'test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':len(checks),'passed':sum(c['pass'] for c in checks),'failed':sum(not c['pass'] for c in checks),'model_calls':0,'details':checks,'scope':'Local output nonreplacement and frozen input binding; no claim of OS immutability or malicious concurrent writer protection.'}
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('x',encoding='utf-8') as f:f.write(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2));raise SystemExit(bool(report['failed']))
