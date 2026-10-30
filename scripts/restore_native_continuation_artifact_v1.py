#!/usr/bin/env python3
"""Verify and restore one manifest-bound continuation ZIP without overwriting differences."""
import argparse,hashlib,json,zipfile
from pathlib import Path,PurePosixPath

def restore(manifest,destination):
    root=Path(__file__).resolve().parents[1]
    m=json.loads(manifest.read_text());a=m['archive'];archive=root/a['path']
    if archive.stat().st_size!=a['bytes'] or hashlib.sha256(archive.read_bytes()).hexdigest()!=a['sha256']: raise ValueError('archive identity differs')
    dest=destination.resolve();pending=[]
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(m['files']): raise ValueError('archive member set differs')
        for name,row in m['files'].items():
            rel=PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or str(rel)!=name: raise ValueError('unsafe member path')
            data=z.read(name)
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']: raise ValueError('member identity differs: '+name)
            target=dest/name
            if target.is_symlink() or not target.resolve().is_relative_to(dest): raise ValueError('unsafe destination')
            if target.exists():
                if not target.is_file() or target.read_bytes()!=data: raise ValueError('existing different file: '+name)
            else: pending.append((target,data))
        for target,data in pending:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as out: out.write(data)
    return {'verified_members':len(names),'written_members':len(pending),'model_calls':0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();print(json.dumps(restore(a.manifest,a.destination),sort_keys=True))
