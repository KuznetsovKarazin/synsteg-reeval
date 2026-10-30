#!/usr/bin/env python3
"""File-only orchestration; root supplies actual Git readbacks and fresh census.

No model, tokenizer, subprocess, network or Git calls. Scientific wrapper unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

ROOT = Path('/workspace/scratch/af7adce5db85/baseline92-continuation-v2/checkout')
WRAPPER = ROOT/'scripts/qwen_sequential_development_v1.py'
WRAPPER_SHA = '12492e8411eeec4d0e0d4f0a8c2e969b2fb055045213bb5c8d88a29a14c19e0e'
PROTOCOL = ROOT/'experiments/development-qwen-sequential-v1/protocol.enabled.json'
STAGE = 'qwen-sequential-stage-20261028-v1'
OUT = ROOT.parent/STAGE
ORDERS = list(range(2,11))

def require(ok, message):
    if not ok: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text())

def write(path, value):
    path=Path(path); payload=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    require(not any(p.is_symlink() for p in (path,*path.parents)), 'symlink')
    if path.exists(): require(path.read_bytes()==payload, 'refuse overwrite '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f: f.write(payload)

def module():
    require(hashlib.sha256(WRAPPER.read_bytes()).hexdigest()==WRAPPER_SHA,'wrapper changed')
    spec=importlib.util.spec_from_file_location('frozen_qwen_sequential_stage',WRAPPER)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def entry(path):
    path=Path(path); b=path.read_bytes()
    return {'path':str(path.relative_to(ROOT)),'local_path':str(path),'bytes':len(b),
            'sha256':hashlib.sha256(b).hexdigest(),
            'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()}

def pins_entries(pins):
    entries=[entry(ROOT/p) for p in sorted(pins)]
    require({e['path']:e['sha256'] for e in entries}==pins,'local pins differ')
    return entries

def manifest(path, expected):
    value=read(path)
    require(value.get('actual_git_readback') is True and len(value.get('ref',''))==40,'actual readback required')
    require({f['path']:f['sha256'] for f in value['files']}==expected,'readback pin set differs')
    for f in value['files']:
        local=entry(ROOT/f['path'])
        require(all(f[k]==local[k] for k in ('sha256','bytes','git_blob_sha1')),'readback/local mismatch')
    return value

def prepare(order):
    require(order in ORDERS,'outside declared stage')
    m=module(); value=m.verify_protocol(ROOT,PROTOCOL);require(value['execution_enabled'],'disabled')
    previous=m.predecessor(ROOT,order);job,content=m.job_for(ROOT,order);m.guard(ROOT,order)
    prompt=ROOT/job['prompt_path']
    if prompt.exists(): require(prompt.read_bytes()==content,'prompt differs')
    else:
        prompt.parent.mkdir(parents=True,exist_ok=True)
        with prompt.open('xb') as f:f.write(content)
    intent=ROOT/f'experiments/{m.ID}/intents/order{order:03d}.json'
    write(intent,m.intent(ROOT,PROTOCOL,order))
    pins=m.activation_pins(ROOT,PROTOCOL,intent,order,value)
    activation=OUT/f'activation{order:03d}.json'
    argv=[str(ROOT/'runtime/large-campaign-development-v2/venv/bin/python'),str(WRAPPER),
          'execute','--root',str(ROOT),'--protocol',str(PROTOCOL),'--intent',str(intent),
          '--receipt',str(activation),'--source-order',str(order)]
    plan={'schema':STAGE+'-activation-plan','source_order':order,'identity':job['identity'],
          'files':[entry(prompt),entry(intent)],'pins':pins,'readback_files':pins_entries(pins),
          'predecessor_receipt_commit':previous['commit'],'intent_path':str(intent),
          'activation_path':str(activation),'execute_argv':argv,'execute_shell':shlex.join(argv)}
    path=OUT/f'prepare{order:03d}.json';write(path,plan);return {'plan':str(path),'source_order':order}

def activation(order, readback, census):
    m=module();p=read(OUT/f'prepare{order:03d}.json');rb=manifest(readback,p['pins']);c=read(census)
    require(c['commit']==rb['ref'] and c['remote_output_scope_complete'] is True
            and c['remote_branches_and_prs_checked'] is True
            and c['remote_qwen_attempted_source_orders']==list(range(1,order))
            and c['remote_qwen_active_intents']==[p['identity']],'fresh remote census differs')
    r={'schema':m.ID+'-activation-readback-v1','readback_verified':True,'commit':rb['ref'],
       'identity':p['identity'],'source_order':order,'files_sha256':p['pins'],
       'remote_output_scope_complete':True,'remote_branches_and_prs_checked':True,
       'remote_qwen_attempted_source_orders':list(range(1,order)),
       'remote_qwen_active_intents':[p['identity']],
       'predecessor_receipt_commit':p['predecessor_receipt_commit']}
    write(p['activation_path'],r)
    m.validate_activation(ROOT,PROTOCOL,Path(p['intent_path']),Path(p['activation_path']),order)
    return {'activation':p['activation_path'],'validated':True,'execute_shell':p['execute_shell']}

def terminal(order):
    m=module();d=ROOT/m.OUTPUT/f'order{order:03d}';p=read(d/'process.json')
    require(p['source_order']==order and p['status'] in ['completed_pending_remote_preservation',
            'stopped_requires_review_no_retry','integrity_review_required_no_retry'],'no terminal record')
    files=[entry(x) for x in sorted(d.rglob('*')) if x.is_file() and x.name!='git-preservation-receipt.json']
    raw={str(Path(e['local_path']).relative_to(d)):e['sha256'] for e in files}
    require(read(d/'artifact-manifest.json')['files_sha256']=={k:v for k,v in raw.items() if k!='artifact-manifest.json'},'raw inventory mismatch')
    plan={'schema':STAGE+'-terminal-plan','source_order':order,'files':files,
          'raw_files_sha256':{e['path']:e['sha256'] for e in files},'process_status':p['status']}
    path=OUT/f'terminal{order:03d}.json';write(path,plan);return {'plan':str(path),'status':p['status'],'files':len(files)}

def preserve(order, readback):
    m=module();d=ROOT/m.OUTPUT/f'order{order:03d}';p=read(OUT/f'terminal{order:03d}.json')
    rb=manifest(readback,p['raw_files_sha256']);raw={str(Path(e['local_path']).relative_to(d)):e['sha256'] for e in p['files']}
    receipt={'readback_verified':True,'commit':rb['ref'],'source_order':order,
             'process_sha256':raw['process.json'],'files_sha256':raw,
             'scope':'All raw bytes matched actual Git readback; independent postrun audit separate.'}
    write(d/'git-preservation-receipt.json',receipt)
    m.verify_preserved(ROOT,d,order)
    path=OUT/f'preservation{order:03d}.json';write(path,{'files':[entry(d/'git-preservation-receipt.json')]})
    return {'plan':str(path),'predecessor_verified':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','activation','terminal','preserve'])
    p.add_argument('--source-order',type=int,required=True);p.add_argument('--readback',type=Path);p.add_argument('--census',type=Path)
    a=p.parse_args();require(a.source_order in ORDERS,'outside stage')
    if a.operation=='prepare':r=prepare(a.source_order)
    elif a.operation=='activation':r=activation(a.source_order,a.readback,a.census)
    elif a.operation=='terminal':r=terminal(a.source_order)
    else:r=preserve(a.source_order,a.readback)
    print(json.dumps(r))

if __name__=='__main__':main()
