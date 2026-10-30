#!/usr/bin/env python3
"""Restore the exact 123 baseline32 primary files; never run models or overwrite differences."""
import argparse, hashlib, json, zipfile
from pathlib import Path, PurePosixPath

def main():
    p=argparse.ArgumentParser(); p.add_argument('--destination', type=Path); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    base=root/'results/baseline32-recovery-v1'
    m=json.loads((base/'manifest.json').read_text())
    archive=root/m['archive']['path']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==m['archive']['sha256']
    dest=(a.destination or root).resolve()
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist()))==123
        assert set(z.namelist())==set(m['files'])
        payload=[]
        for name, item in m['files'].items():
            rel=PurePosixPath(name)
            assert not rel.is_absolute() and '..' not in rel.parts
            data=z.read(name)
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
            target=dest/name
            assert target.resolve().is_relative_to(dest)
            assert not target.is_symlink()
            if target.exists():
                assert target.is_file() and target.read_bytes()==data, name
            else: payload.append((target,data))
        for target,data in payload:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f: f.write(data)
    print(json.dumps({'verified':123,'written':len(payload),'model_calls':0}))
if __name__=='__main__': main()
