#!/usr/bin/env python3
"""Lossless JSONL packaging/reconstruction for a complete capacity result."""
import argparse
import hashlib
import json
from pathlib import Path

SCHEMA='public-dictionary-full-development-v3-lossless-package'
TABLES={'lengths':200,'dictionary_rows':400,'frame_conditions':4800}
CHUNK_ROWS={'lengths':100,'dictionary_rows':100,'frame_conditions':600}


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()


def read_json(path):
    def unique(pairs):
        value={}
        for k,v in pairs:
            require(k not in value,'duplicate JSON key');value[k]=v
        return value
    return json.loads(path.read_bytes(),object_pairs_hook=unique)


def safe_path(root,name):
    relative=Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and str(relative)==name,'unsafe package path')
    target=root/relative
    require(all(not p.is_symlink() for p in (target,*target.parents)),'linked package file/ancestor')
    return target


def reconstruct(root,manifest):
    require(manifest['schema']==SCHEMA and set(manifest['tables'])==set(TABLES),'wrong package schema/tables')
    metadata=manifest['metadata'];path=safe_path(root,metadata['path']);raw=path.read_bytes()
    require(len(raw)==metadata['bytes'] and sha(raw)==metadata['sha256'],'metadata byte mismatch')
    result=read_json(path);require(not set(TABLES)&set(result),'table duplicates in metadata')
    used={metadata['path']}
    for field,expected in TABLES.items():
        table=manifest['tables'][field];rows=[]
        require(table['rows']==expected,'table denominator differs')
        for part in table['parts']:
            require(part['path'] not in used,'duplicate package part');used.add(part['path'])
            raw=safe_path(root,part['path']).read_bytes()
            require(len(raw)==part['bytes'] and sha(raw)==part['sha256'],'table part byte mismatch')
            lines=raw.splitlines();require(len(lines)==part['rows'] and all(lines),'table part row mismatch')
            rows.extend(json.loads(line) for line in lines)
        require(len(rows)==expected and sha(canonical(rows))==table['canonical_sha256'],'table order/content mismatch')
        result[field]=rows
    require(sha(canonical(result))==manifest['canonical_result_sha256'],'reconstructed result differs')
    return result


def pack(result,output):
    require(result['schema']=='public-dictionary-full-development-v3-result','wrong scientific result')
    require(not output.exists() and all(not p.is_symlink() for p in (output,*output.parents)),'package output exists/linked')
    require(all(isinstance(result[k],list) and len(result[k])==v for k,v in TABLES.items()),'scientific result rows differ')
    output.mkdir(parents=True,exist_ok=False)
    metadata={k:v for k,v in result.items() if k not in TABLES};raw=canonical(metadata)+b'\n';(output/'result-metadata.json').write_bytes(raw)
    manifest={'schema':SCHEMA,'canonical_result_sha256':sha(canonical(result)),
              'metadata':{'path':'result-metadata.json','bytes':len(raw),'sha256':sha(raw)},'tables':{},
              'scope':'Every field and ordered row retained; only whitespace/object-key order may differ from the original pretty JSON.'}
    for field,expected in TABLES.items():
        rows=result[field];parts=[]
        for start in range(0,expected,CHUNK_ROWS[field]):
            subset=rows[start:start+CHUNK_ROWS[field]];name=f'{field}-{start:04d}-{start+len(subset)-1:04d}.jsonl'
            raw=b''.join(canonical(row)+b'\n' for row in subset);(output/name).write_bytes(raw)
            parts.append({'path':name,'bytes':len(raw),'sha256':sha(raw),'rows':len(subset)})
        manifest['tables'][field]={'rows':len(rows),'canonical_sha256':sha(canonical(rows)),'parts':parts}
    require(reconstruct(output,manifest)==result,'lossless self-check failed')
    (output/'package-manifest.json').write_bytes(canonical(manifest)+b'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('pack');p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('reconstruct');p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--expected-manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='pack':
        result=pack(read_json(args.result),args.output)
        print(json.dumps({'canonical_result_sha256':result['canonical_result_sha256'],'tables':{k:v['rows'] for k,v in result['tables'].items()}}))
    else:
        require(sha(args.manifest.read_bytes())==args.expected_manifest_sha256,'unbound package manifest')
        result=reconstruct(args.root,read_json(args.manifest));require(not args.output.exists(),'reconstruction output exists')
        require(not args.output.resolve().is_relative_to(args.root.resolve()),'reconstruction output must be outside package')
        require(all(not p.is_symlink() for p in (args.output,*args.output.parents)),'linked output path')
        with args.output.open('xb') as stream:stream.write(canonical(result)+b'\n')
        print(json.dumps({'canonical_result_sha256':sha(canonical(result)),'lossless':True}))


if __name__=='__main__':main()
