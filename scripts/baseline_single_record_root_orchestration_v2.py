#!/usr/bin/env python3
"""Root-only file orchestration. No Git, subprocess, tokenizer or model calls.

The root caller supplies real remote readback evidence; this helper can validate
its contents against local pins but cannot authenticate the Git transaction.
Frozen scientific inputs and the frozen execution wrapper are never changed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex

ROOT = Path('/workspace/scratch/af7adce5db85/baseline92-continuation-v2/checkout')
WRAPPER = ROOT/'scripts/baseline_single_record_recovery_v2.py'
WRAPPER_SHA = '89e94639fae9fef70e1b2662fccb92c73319234db55b237b246b6ba89fafd8bf'
PROTOCOL = ROOT/'experiments/baseline-single-record-recovery-v2/protocol.enabled.json'


def check(ok, message):
    if not ok:
        raise ValueError(message)


def wrapper():
    check(hashlib.sha256(WRAPPER.read_bytes()).hexdigest() == WRAPPER_SHA, 'Frozen wrapper changed')
    spec = importlib.util.spec_from_file_location('frozen_successor_root_support',WRAPPER)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()


def write_once(m,path,value):
    payload=json_bytes(value)
    m.reject_links(path)
    if path.exists():
        check(m.raw(path)==payload,'Existing output differs; refuse overwrite: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        m.write_bytes_new(path,payload)


def entry(m,path,with_content=True):
    payload=m.raw(path)
    result={'path':str(path.relative_to(ROOT)),'local_path':str(path),
            'sha256':m.digest_bytes(payload),'bytes':len(payload)}
    if with_content:
        result['content']=payload.decode('utf-8')
    return result


def common(order):
    m=wrapper()
    m.require(type(order) is int and order in m.ORDERS,'Only original92..100')
    protocol=m.verify_protocol(ROOT,PROTOCOL)
    m.require(protocol['execution_enabled'] is True,'Disabled protocol')
    job=m.job_for(ROOT,order)[0]
    return m,protocol,job


def intent_path(m,order):
    return ROOT/f'experiments/{m.ID}/intents/order{order:03d}.json'


def prepare_intent(order,out):
    m,protocol,job=common(order)
    m.check_replay(ROOT,job)
    previous=m.predecessor(ROOT,order)
    intent=intent_path(m,order)
    write_once(m,intent,m.intent_for(ROOT,PROTOCOL,job,previous['commit']))
    pins=m.readback_pins(ROOT,PROTOCOL,intent,protocol)
    receipt=out/f'activation{order:03d}-receipt.json'
    first=m.load(ROOT/m.FIRST)
    command=[str(ROOT/first['python_relative']),str(WRAPPER),'execute','--root',str(ROOT),
             '--protocol',str(PROTOCOL),'--intent',str(intent),'--receipt',str(receipt),
             '--source-order',str(order)]
    plan={'schema':m.ID+'-root-activation-plan','source_order':order,'identity':job['identity'],
          'files':[entry(m,PROTOCOL),entry(m,intent)],
          'readback_files':[entry(m,ROOT/name) for name in sorted(pins)],
          'readback_files_sha256':pins,'remote_current_output_orders_required':list(range(92,order)),
          'predecessor_receipt_commit':previous['commit'],
          'external_requirement':'Root must save files, read all pinned Git contents at the actual commit, and inspect complete remote output scope. Local content is not remote proof.',
          'activation_receipt_output':str(receipt),'execute_argv':command,'execute_shell':shlex.join(command)}
    path=out/f'activation{order:03d}-plan.json'
    write_once(m,path,plan)
    return {'plan':str(path),'source_order':order,'readback_files':len(pins),'execute_shell':plan['execute_shell']}


def actual_readback(m,path,commit,expected):
    check(type(commit) is str and re.fullmatch('[0-9a-f]{40}',commit),'Invalid actual Git commit')
    value=m.load(path)
    check(value.get('commit')==commit and value.get('readback_verified') is True,
          'Root must supply actual matching Git readback attestation')
    files=value.get('files_sha256')
    check(type(files) is dict and all(files.get(name)==digest for name,digest in expected.items()),
          'Remote file digest missing/different from exact local bytes')
    return value


def activation_receipt(order,out,commit,readback):
    m,protocol,job=common(order)
    m.check_replay(ROOT,job)
    previous=m.predecessor(ROOT,order)
    intent=intent_path(m,order)
    check(m.load(intent)==m.intent_for(ROOT,PROTOCOL,job,previous['commit']),'Intent changed')
    pins=m.readback_pins(ROOT,PROTOCOL,intent,protocol)
    remote=actual_readback(m,readback,commit,pins)
    check(remote.get('remote_output_scope_complete') is True
          and remote.get('remote_current_output_orders')==list(range(92,order)),
          'Need complete fresh remote output/reservation scope')
    receipt={'schema':m.ID+'-activation-readback-v1','readback_verified':True,'commit':commit,
             'source_order':order,'identity':job['identity'],'files_sha256':pins,
             'remote_output_scope_complete':True,'remote_current_output_orders':list(range(92,order)),
             'predecessor_receipt_commit':previous['commit']}
    path=out/f'activation{order:03d}-receipt.json'
    write_once(m,path,receipt)
    m.validate_activation(ROOT,PROTOCOL,intent,path,order)
    return {'activation_receipt':str(path),'validated':True,'source_order':order}


def terminal_inventory(m,order):
    directory=ROOT/'runs'/m.ID/f'order{order:03d}'
    m.reject_links(directory)
    process=m.load(directory/'process.json')
    check(process.get('source_order')==order and process.get('status') in
          ('completed_pending_remote_preservation','stopped_requires_review_no_retry','integrity_review_required_no_retry'),
          'Terminal process record required; do not snapshot a running shard')
    old=m.module(ROOT/m.OLD_HELPER,'frozen_inventory_root_support')
    inventory={name:value for name,value in old.file_inventory(directory).items()
               if name!='git-preservation-receipt.json'}
    manifest=m.load(directory/'artifact-manifest.json')
    expected={name:value for name,value in inventory.items() if name!='artifact-manifest.json'}
    check(manifest=={'files_sha256':expected},'Raw artifact manifest differs; stop for independent review')
    return directory,process,inventory


def terminal_plan(order,out):
    m,protocol,job=common(order)
    directory,process,inventory=terminal_inventory(m,order)
    target=ROOT/'results'/f'{m.ID}-root'
    projection_path=target/f'full300-after{order:03d}.json'
    projection_error=None
    try:
        projection=m.projection(ROOT)
        write_once(m,projection_path,projection)
    except Exception as exc:
        projection_error={'error_type':type(exc).__name__,'message':str(exc)}
        projection_path=target/f'full300-after{order:03d}.error.json'
        write_once(m,projection_path,projection_error)
    check(process.get('identity')==job['identity'],'Terminal process identity differs')
    files=[entry(m,directory/name) for name in sorted(inventory)]+[entry(m,projection_path)]
    plan={'schema':m.ID+'-root-terminal-save-plan','source_order':order,'identity':job['identity'],
          'files':files,'raw_files_sha256':{str((directory/name).relative_to(ROOT)):digest for name,digest in inventory.items()},
          'process_status':process['status'],'projection_error':projection_error,
          'next_shard_allowed_by_this_plan':False,
          'requirement':'Save complete raw files, actual Git readback, independent postrun audit and preservation receipt before considering another order. Failure/unknown stops.'}
    path=out/f'terminal{order:03d}-save-plan.json'
    write_once(m,path,plan)
    return {'plan':str(path),'files':len(files),'process_status':process['status'],'projection_path':str(projection_path),
            'projection_error':projection_error}


def preservation_receipt(order,out,commit,readback):
    m,protocol,job=common(order)
    directory,process,inventory=terminal_inventory(m,order)
    expected={str((directory/name).relative_to(ROOT)):digest for name,digest in inventory.items()}
    actual_readback(m,readback,commit,expected)
    receipt={'readback_verified':True,'commit':commit,'source_order':order,
             'process_sha256':m.sha(directory/'process.json'),'files_sha256':inventory,
             'scope':'All raw shard files matched actual Git readback supplied by root; independent scientific audit is separate.'}
    path=directory/'git-preservation-receipt.json'
    write_once(m,path,receipt)
    clean=False
    try:
        m.verify_preserved(ROOT,directory,job)
        clean=True
    except Exception:
        pass
    plan={'schema':m.ID+'-root-receipt-save-plan','files':[entry(m,path)],
          'source_order':order,'clean_predecessor_validation':clean,
          'requirement':'Root must also save this receipt to Git and read it back before next activation. No automatic next launch.'}
    plan_path=out/f'preservation{order:03d}-save-plan.json'
    write_once(m,plan_path,plan)
    return {'receipt':str(path),'save_plan':str(plan_path),'clean_predecessor_validation':clean}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['prepare-intent','activation-receipt','terminal-plan','preservation-receipt'])
    parser.add_argument('--source-order',type=int,required=True)
    parser.add_argument('--out',type=Path,default=ROOT.parent/'root-stage-plans')
    parser.add_argument('--git-commit')
    parser.add_argument('--actual-readback',type=Path)
    args=parser.parse_args()
    check(args.out.is_absolute(),'Use absolute plan directory')
    if args.operation in ('prepare-intent','terminal-plan'):
        result={'prepare-intent':prepare_intent,'terminal-plan':terminal_plan}[args.operation](args.source_order,args.out)
    else:
        check(args.actual_readback is not None and args.git_commit is not None,'Actual readback and Git commit required')
        result={'activation-receipt':activation_receipt,'preservation-receipt':preservation_receipt}[args.operation](args.source_order,args.out,args.git_commit,args.actual_readback)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
