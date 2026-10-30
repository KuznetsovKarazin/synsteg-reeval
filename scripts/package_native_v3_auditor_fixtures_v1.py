#!/usr/bin/env python3
"""Archive four synthetic auditor fixtures; no model/native calls."""
import argparse,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
files={};args.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for name in ['complete-None','empty-None','later_empty-None','complete-step_throw']:
  for path in sorted((args.fixtures/name).rglob('*')):
   if path.is_symlink():raise ValueError('fixture symlink')
   if path.is_file() and '__pycache__' not in path.parts:
    rel=path.relative_to(args.fixtures).as_posix();raw=path.read_bytes();info=zipfile.ZipInfo(rel,(1980,1,1,0,0,0));info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,raw)
    files[rel]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
data=args.output.read_bytes();parts=[]
for index,start in enumerate(range(0,len(data),196608)):
 raw=data[start:start+196608];part=args.output.with_name(args.output.name+f'.part{index:03d}')
 with part.open('xb') as f:f.write(raw)
 parts.append({'path':part.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
manifest={'schema':'native-v3-evidence-auditor-adversarial-synthetic-fixtures-v1','archive':args.output.name,'archive_bytes':len(data),'archive_sha256':hashlib.sha256(data).hexdigest(),'file_count':len(files),'files':files,'parts':parts,'native_calls':0,'scope':'Synthetic saved fixtures from mocks, not production results; temporary fixture origin paths retained'}
manifestpath=args.output.with_suffix('.manifest.json')
with manifestpath.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps({'archive_bytes':len(data),'file_count':len(files),'parts':len(parts),'manifest':str(manifestpath)}))
