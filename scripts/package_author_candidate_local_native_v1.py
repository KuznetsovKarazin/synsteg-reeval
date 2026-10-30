#!/usr/bin/env python3
"""Preserve all candidate-diagnostic bytes in<=192KiB Git-fetchable ZIP parts.

No models, tokenizers or experiment reruns. Includes refusals, UNKNOWN, stderr,
empty files and .writing files. Sources are read-only; outputs are exclusive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE='results/author-candidate-local-native-v1'
ARTIFACTS='results/author-candidate-local-native-v1-artifacts/native-diagnostic-v1'
PART_BYTES=192*1024


def sha(data):return hashlib.sha256(data).hexdigest()

def json_bytes(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()

def put(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as out:out.write(data)


def relative(value):
    if not isinstance(value,str) or not value or '\\' in value:raise ValueError('unsafe path')
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=value:raise ValueError('unsafe path')
    return p


def inventory(root,base):
    root=Path(root).resolve(strict=True);base_path=root/relative(base)
    if base_path.is_symlink() or not base_path.is_dir():raise ValueError('primary directory missing or symlinked')
    out={}
    for p in sorted(base_path.rglob('*')):
        if p.is_symlink():raise ValueError('primary symlink refused')
        if p.is_dir():continue
        if not p.is_file() or root not in p.resolve().parents:raise ValueError('nonregular primary')
        data=p.read_bytes();name=p.relative_to(root).as_posix()
        out[name]={'bytes':len(data),'sha256':sha(data)}
    if not out:raise ValueError('empty primary tree')
    return out


def directories(root,base):
    root=Path(root).resolve(strict=True);base_path=root/relative(base)
    paths=[base_path,*sorted(p for p in base_path.rglob('*') if p.is_dir())]
    if any(p.is_symlink() for p in paths):raise ValueError('symlinked primary directory')
    return [p.relative_to(root).as_posix() for p in paths]


def read_archive(index_path):
    index_path=Path(index_path);index=json.loads(index_path.read_text());chunks=[]
    if index.get('schema')!='author-candidate-local-native-v1-artifacts':raise ValueError('artifact schema differs')
    if index['part_bytes']!=PART_BYTES:raise ValueError('part bound differs')
    expected_names=[f'full.zip.part{i:03d}' for i in range(len(index['parts']))]
    if not expected_names or [x['name'] for x in index['parts']]!=expected_names:raise ValueError('part order differs')
    for row in index['parts']:
        p=index_path.parent/relative(row['name'])
        if p.is_symlink():raise ValueError('symlinked part')
        data=p.read_bytes()
        if len(data)>PART_BYTES or len(data)!=row['bytes'] or sha(data)!=row['sha256']:
            raise ValueError('part bytes differ')
        chunks.append(data)
    raw=b''.join(chunks)
    if len(raw)!=index['archive']['bytes'] or sha(raw)!=index['archive']['sha256']:
        raise ValueError('full ZIP identity differs')
    return index,raw


def restore(index_path,destination):
    import io
    index,raw=read_archive(index_path);destination=Path(destination)
    if any(p.is_symlink() for p in [destination,*destination.absolute().parents]):raise ValueError('symlinked destination/ancestor')
    dest=destination.resolve();pending=[]
    directory_names=index.get('directories',[])
    if len(directory_names)!=len(set(directory_names)):raise ValueError('duplicate directory metadata')
    if set(directory_names)&set(index['files']):raise ValueError('directory/file metadata conflict')
    pending_directories=[]
    for name in directory_names:
        rel=relative(name);current=dest
        for part in rel.parts:
            current=current/part
            if current.is_symlink():raise ValueError('symlinked destination directory')
        target=dest/rel
        if target.exists() and not target.is_dir():raise FileExistsError('directory conflicts with file')
        pending_directories.append(target)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(index['files']):raise ValueError('member set differs')
        for name in names:
            rel=relative(name);row=index['files'][name];data=z.read(name)
            if len(data)!=row['bytes'] or sha(data)!=row['sha256']:raise ValueError('member differs: '+name)
            target=dest/rel
            current=dest
            if any(p.is_symlink() for p in [destination,*destination.absolute().parents]):raise ValueError('symlinked destination ancestor')
            for part in rel.parts:
                current=current/part
                if current.is_symlink():raise ValueError('symlinked destination member')
            if target.exists():
                if not target.is_file() or target.read_bytes()!=data:raise FileExistsError('existing differing member: '+name)
            else:pending.append((target,data))
        for target in pending_directories:target.mkdir(parents=True,exist_ok=True)
        for target,data in pending:put(target,data)
    return {'status':'passed','files_verified':len(names),'files_written':len(pending),
            'archive_sha256':index['archive']['sha256'],'model_calls':0,'tokenizer_calls':0}


def package(root,base,output,*,allow_incomplete=False):
    root=Path(root).resolve(strict=True);output=Path(output).resolve()
    primary=(root/relative(base)).resolve(strict=True)
    if output==primary or primary in output.parents:raise ValueError('artifact output cannot be inside primary')
    if output.exists():raise FileExistsError('exclusive artifact output required')
    completion=primary/'native-diagnostic-v1/result.json'
    if not allow_incomplete and not completion.is_file():raise ValueError('terminal result absent; explicit incomplete preservation required')
    files=inventory(root,base)
    directory_names=directories(root,base)
    output.mkdir(parents=True)
    archive=output/'full.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_LZMA) as z:
        for name,row in files.items():
            data=(root/name).read_bytes()
            if len(data)!=row['bytes'] or sha(data)!=row['sha256']:raise ValueError('primary changed during package')
            info=zipfile.ZipInfo(name,date_time=(2026,10,25,0,0,0));info.compress_type=zipfile.ZIP_LZMA
            info.external_attr=0o100644<<16;z.writestr(info,data)
    if inventory(root,base)!=files or directories(root,base)!=directory_names:raise ValueError('primary changed during archive creation')
    raw=archive.read_bytes();parts=[]
    for i,start in enumerate(range(0,len(raw),PART_BYTES)):
        data=raw[start:start+PART_BYTES];name=f'full.zip.part{i:03d}'
        put(output/name,data);parts.append({'name':name,'bytes':len(data),'sha256':sha(data)})
    readable=[]
    for name,row in files.items():
        if not name.endswith('/response.raw'):continue
        data=(root/name).read_bytes()
        if len(data)!=row['bytes'] or sha(data)!=row['sha256']:raise ValueError('primary response changed before readable copy')
        attempt=Path(name).parent.name
        try:data.decode('utf-8');suffix='.txt';copy=data;encoding='utf-8 exact bytes'
        except UnicodeDecodeError:suffix='.hex.txt';copy=(data.hex()+'\n').encode();encoding='lossless hexadecimal; source not valid UTF-8'
        target='readable/'+attempt+'-response'+suffix;put(output/target,copy)
        readable.append({'path':target,'source_path':name,'source_sha256':row['sha256'],
                         'bytes':len(copy),'sha256':sha(copy),'encoding':encoding})
    index={'schema':'author-candidate-local-native-v1-artifacts','source_base':base,
        'archive':{'name':'full.zip','bytes':len(raw),'sha256':sha(raw),'compression':'ZIP_LZMA',
                   'full_zip_local_only_upload_parts':True},'part_bytes':PART_BYTES,'parts':parts,
        'files':files,'primary_files':len(files),'primary_bytes':sum(x['bytes'] for x in files.values()),
        'directories':directory_names,'empty_reserved_directories_preserved':True,
        'includes_failed_unknown_and_writing_files':True,'terminal_result_present':completion.is_file(),
        'allow_incomplete_requested':allow_incomplete,'readable_responses':readable,
        'logical_requests':2,'physical_attempt_upper_bound':6,'full_cover_required_chunks':109,
        'full_cover_candidate_resource_complete':False,'model_calls':0,'tokenizer_calls':0}
    put(output/'index.json',json_bytes(index))
    with tempfile.TemporaryDirectory(prefix='candidate-fresh-restore-') as tmp:
        fresh=restore(output/'index.json',Path(tmp))
        fresh_files=inventory(Path(tmp),base)
        if fresh_files!=files or directories(Path(tmp),base)!=directory_names:raise ValueError('fresh restoration inventory differs')
    put(output/'fresh-restore.json',json_bytes(fresh))
    uploads=['index.json','fresh-restore.json']+[x['name'] for x in parts]+[x['path'] for x in readable]
    listing={'schema':'author-candidate-local-native-v1-upload-list','files':[
        {'path':str((output/name).relative_to(root)),'bytes':(output/name).stat().st_size,
         'sha256':sha((output/name).read_bytes())} for name in uploads],
        'do_not_upload':['full.zip'],'maximum_binary_part_bytes':PART_BYTES,
        'full_primary_files':len(files),'model_calls':0}
    put(output/'upload-files.json',json_bytes(listing))
    return {'status':'passed','primary_files':len(files),'archive_bytes':len(raw),
        'parts':len(parts),'largest_part':max(x['bytes'] for x in parts),'readable_responses':len(readable),
        'upload_list':str(output/'upload-files.json'),'model_calls':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('package');a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--base',default=BASE)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--allow-incomplete',action='store_true')
    a=sub.add_parser('restore');a.add_argument('--index',type=Path,required=True);a.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    result=package(a.root,a.base,a.output,allow_incomplete=a.allow_incomplete) if a.command=='package' else restore(a.index,a.destination)
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
