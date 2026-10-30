#!/usr/bin/env python3
"""Independent saved-evidence audit for Qwen development orders 2..100.

Stdlib only. Does not import either execution wrapper, run a model/tokenizer,
contact GitHub, alter a run, or assess text quality. Git manifests must be made
from actual external readback by the execution owner, not from local assertions.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

ID = 'development-qwen-sequential-v1'
FIRST_ID = 'development-other-models-single-record-v1'
AUDIT_ID = 'qwen-sequential-postrun-audit-v1'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
PROPOSAL = 'experiments/'+ID+'/protocol.proposed.json'
PROPOSAL_SHA = '86aaeca60dc6fd67341d8a047338c9444e0ec75a015868683e51e33b2e0a71aa'
HELPER = 'scripts/qwen_sequential_development_v1.py'
HELPER_SHA = '12492e8411eeec4d0e0d4f0a8c2e969b2fb055045213bb5c8d88a29a14c19e0e'
FIRST_HELPER = 'scripts/other_model_single_record_v1.py'
FIRST_HELPER_SHA = 'cfc5e284381964db30037931ff9a7091f4fc7e28618b1f0da3374e369eeadcdd'
PROMPTS = 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
PROMPTS_SHA = '3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1'
FIRST = 'experiments/large-campaign-development-v2/execution-manifest.json'
PRE = 'results/'+FIRST_ID+'/qwen05-preflight'
FIRST_SHARD = 'runs/'+FIRST_ID+'/qwen05/order001'
TEMPLATE = 'runs/baseline-single-record-recovery-v1/order090/run/run.json'
TEMPLATE_SHA = '8a6cd75473a9ec61d2f2c82da46e60b45f379361269b5e335923470c16f5ab16'
RUNTIME_SHA = 'b9902f9a50d45933b0797788814ce9793347197c2c4b52bd517a0e0d8427f03a'
DENIALS = 'results/current-git-status-audit-20261028-v1/no-replay-guard.json'
OUTPUT = 'runs/'+ID


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()


def canonical(value):
    return digest(canonical_bytes(value))


def raw(path):
    path=Path(path)
    if any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file():
        raise ValueError('Missing regular file or symlink: '+str(path))
    return path.read_bytes()


def decode(data):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise ValueError('Duplicate JSON key')
            result[k]=v
        return result
    def bad(_): raise ValueError('Nonfinite JSON number')
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad)


def inventory(directory,excluded=()):
    directory=Path(directory)
    if directory.is_symlink() or any(p.is_symlink() for p in directory.parents):
        raise ValueError('Symlink directory')
    result={}
    for p in sorted(directory.rglob('*')):
        if p.is_symlink(): raise ValueError('Symlink inventory member')
        if p.is_file() and p.relative_to(directory).as_posix() not in excluded:
            result[p.relative_to(directory).as_posix()]=digest(raw(p))
    return result


def integer(value,expected=None):
    return type(value) is int and (expected is None or value==expected)


def git_sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{40}',value) is not None


def provenance(root,manifest,expected_pins,expected_ref,check):
    value=decode(raw(manifest))
    check('actual Git readback declaration: '+str(manifest),value.get('actual_git_readback') is True
          and git_sha(value.get('ref')) and value.get('ref')==expected_ref)
    rows=value['files']
    check('exact Git file set: '+str(manifest),isinstance(rows,list)
          and len(rows)==len(expected_pins) and len({r['path'] for r in rows})==len(rows)
          and {r['path'] for r in rows}==set(expected_pins))
    for item in rows:
        relative=Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe provenance path')
        data=raw(root/relative)
        check('Git readback bytes: '+item['path'],digest(data)==item['sha256']==expected_pins.get(item['path'])
              and integer(item['bytes'],len(data)) and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_sha1'])
    return value


def audit(root,order,protocol_path,intent_path,activation_manifest,raw_manifest,require_git_receipt=False):
    root=Path(root).absolute();protocol_path=Path(protocol_path).absolute();intent_path=Path(intent_path).absolute()
    checks=[];errors=[];inputs={}
    def check(label,condition):
        checks.append(label)
        if not condition: errors.append(label)
    def rb(path):
        data=raw(path);inputs[str(path)]=digest(data);return data
    def read(path): return decode(rb(path))
    result={'schema':AUDIT_ID,'source_order':order,'model_alias':'qwen05','model_id':MODEL,'revision':REVISION,
            'model_calls':0,'tokenizer_calls':0,'encoder_calls':0,'decoder_calls':0,'git_network_calls':0,
            'quality_assessed':False,'git_receipt_required':require_git_receipt,
            'limitations':['Saved-evidence integrity only; no factuality, semantic capacity, bit delivery or physical PID continuity claim.',
                           'Actual remote readback and branch/PR census must be performed externally; a locally forged readback declaration cannot authenticate GitHub.',
                           'Predecessor chain inventories and declared identities are checked; old neural computations are not repeated.']}
    try:
        if not integer(order) or order not in range(2,101): raise ValueError('Only integer development orders2..100')
        if not protocol_path.is_relative_to(root) or not intent_path.is_relative_to(root): raise ValueError('Activation paths outside root')
        protocol_rel=protocol_path.relative_to(root).as_posix();intent_rel=intent_path.relative_to(root).as_posix()
        proposal=read(root/PROPOSAL);protocol=read(protocol_path)
        check('frozen reviewed proposal/helper source',digest(raw(root/PROPOSAL))==PROPOSAL_SHA
              and digest(rb(root/HELPER))==HELPER_SHA and digest(rb(root/FIRST_HELPER))==FIRST_HELPER_SHA)
        expected_protocol=copy.deepcopy(proposal);expected_protocol['execution_enabled']=True
        check('only reviewed execution flag enabled',canonical_bytes(protocol)==canonical_bytes(expected_protocol)
              and protocol['execution_enabled'] is True and protocol['source_orders']==list(range(2,101)))
        allpins=dict(proposal['core_pins'])|proposal['preflight_pins']|proposal['first_preserved_shard_pins']
        for path,expected in allpins.items(): check('frozen pinned input: '+path,digest(rb(root/path))==expected)
        check('frozen full100 source and run template',digest(rb(root/PROMPTS))==PROMPTS_SHA and digest(rb(root/TEMPLATE))==TEMPLATE_SHA)
        rows=[decode(line) for line in raw(root/PROMPTS).splitlines() if line.strip()]
        check('100 unique development records',len(rows)==100 and len({r['record_id'] for r in rows})==100
              and len({r['source_group_id'] for r in rows})==100 and all(r['split']=='development' for r in rows))
        row=rows[order-1];job=proposal['jobs'][order-2];ident=canonical(['local',MODEL,row['record_id']])
        prompt_path=f'experiments/{ID}/prompts/order{order:03d}.jsonl'
        expected_prompt=(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode()
        check('exact current prompt and frozen job',rb(root/prompt_path)==expected_prompt and job=={
            'model_alias':'qwen05','source_order':order,'record_id':row['record_id'],'identity':ident,'prompt_path':prompt_path,
            'prompt_file_sha256':digest(expected_prompt),'original_prompt_sha256':digest(row['prompt'].encode())})
        meta={k:row[k] for k in ['record_id','source_group_id','split','topic','genre','target_words']};meta['prompt_sha256']=digest(row['prompt'].encode())
        shard=root/OUTPUT/f'order{order:03d}';raw_files=inventory(shard,{'git-preservation-receipt.json'})
        names={'activation-receipt.json','artifact-manifest.json','child.json','process.json','reservation.json','stdout.log','stderr.log',
               'guard-request.json','guard-result.json','bound-intent.json','run/run.json','run/runtime.json','run/summary.json',
               'run/local_ledger/ledger.json','run/local_ledger/campaign.lock','run/run_guard/campaign.lock','run/attempts/'+ident+'.json'}
        check('exact17 raw files and one attempt',set(raw_files)==names and len(raw_files)==17)
        successors=[p for p in (root/OUTPUT).iterdir() if p.name!='supervisor.lock']
        check('continuous local successor shard census',all(p.is_dir() and not p.is_symlink() for p in successors)
              and sorted(p.name for p in successors)==[f'order{n:03d}' for n in range(2,order+1)])
        previous=None
        predecessor_pins={}
        for n in range(1,order):
            prefix=FIRST_SHARD if n==1 else OUTPUT+f'/order{n:03d}'
            directory=root/prefix;inv=inventory(directory,{'git-preservation-receipt.json'})
            receipt=read(directory/'git-preservation-receipt.json');process=read(directory/'process.json')
            reservation=read(directory/'reservation.json');bound=read(directory/'bound-intent.json')
            identity=canonical(['local',MODEL,rows[n-1]['record_id']]);prior_id=FIRST_ID if n==1 else ID
            prior_names={p.replace(ident,identity) for p in names}
            check('predecessor raw inventory and receipt '+str(n),receipt['readback_verified'] is True and git_sha(receipt['commit'])
                  and receipt['files_sha256']==inv and receipt['process_sha256']==inv.get('process.json')
                  and set(inv)==prior_names and len(inv)==17
                  and integer(receipt.get('source_order'),n) and read(directory/'artifact-manifest.json')['files_sha256']=={k:v for k,v in inv.items() if k!='artifact-manifest.json'})
            prior_attempt=read(directory/'run/attempts'/(identity+'.json'));prior_ledger=read(directory/'run/local_ledger/ledger.json')
            prior_entry=prior_ledger['attempts'][identity]
            prior_meta={k:rows[n-1][k] for k in ['record_id','source_group_id','split','topic','genre','target_words']}
            prior_meta['prompt_sha256']=digest(rows[n-1]['prompt'].encode())
            check('predecessor single completed ledger/attempt '+str(n),set(prior_ledger['attempts'])=={identity}
                  and prior_attempt['identity']==prior_entry['identity']==identity
                  and prior_attempt['status']==prior_entry['status']=='completed'
                  and prior_attempt['result']==prior_entry['result'] and prior_attempt['record']==prior_entry['metadata']['record']==prior_meta
                  and prior_attempt['model']==prior_entry['metadata']['model']==MODEL
                  and prior_attempt['provider']==prior_entry['metadata']['provider']=='local'
                  and integer(prior_attempt['automatic_retries'],0) and integer(prior_entry['automatic_retries'],0))
            check('predecessor clean identity/outcome '+str(n),process['identity']==reservation['identity']==bound['identity']==identity
                  and integer(process['source_order'],n) and integer(reservation['source_order'],n) and integer(bound['source_order'],n)
                  and reservation['record_id']==bound['record_id']==rows[n-1]['record_id']
                  and reservation['experiment_id']==bound['experiment_id']==prior_id
                  and reservation['model_alias']==bound['model_alias']=='qwen05'
                  and process['status']=='completed_pending_remote_preservation' and integer(process['exit_code'],0)
                  and process['timed_out'] is False and not process.get('interrupted')
                  and reservation['intent_sha256']==digest(raw(directory/'bound-intent.json')))
            if n>1:
                check('predecessor receipt chain '+str(n),previous['commit']==reservation['predecessor_receipt_commit']==bound['predecessor_receipt_commit'])
                selected=inventory(directory) if n==order-1 else {'git-preservation-receipt.json':digest(raw(directory/'git-preservation-receipt.json'))}
                predecessor_pins.update({prefix+'/'+p:h for p,h in selected.items()})
            previous=receipt
        template=read(root/TEMPLATE);run=read(shard/'run/run.json');runtime=read(shard/'run/runtime.json')
        expected=copy.deepcopy(template);expected.update(model=MODEL,revision=REVISION,prompts_sha256=digest(expected_prompt),selected_records=[meta],selected_records_count=1)
        check('exact independent scientific run reconstruction',run==expected)
        check('frozen runtime',runtime==read(root/(PRE+'/runtime.json')) and digest(raw(shard/'run/runtime.json'))==RUNTIME_SHA)
        context=read(root/(PRE+'/all100-context.json'));config=read(root/(PRE+'/config.json'));contract=read(root/(PRE+'/model-contract.json'))
        check('context contract fixed model and runtime',context['model_id']==contract['model_id']==MODEL and context['revision']==contract['revision']==REVISION
              and context['source_sha256']==PROMPTS_SHA and context['model_config_sha256']==digest(raw(root/(PRE+'/config.json')))
              and context['model_context_limit']==config['max_position_embeddings']==32768 and contract['runtime_sha256']==RUNTIME_SHA
              and contract['settings']==template['settings'] and contract['dtype']=='float32' and len(context['records'])==100)
        for n,c in enumerate(context['records'],1):
            check('full100 context row '+str(n),integer(c['source_order'],n) and c['record_id']==rows[n-1]['record_id']
                  and c['prompt_sha256']==digest(rows[n-1]['prompt'].encode()) and integer(c['input_tokens']) and 0<c['input_tokens']<=32768-1024)
        fingerprint=canonical({'record':meta,'provider':'local','model':MODEL,'revision':REVISION,'dtype':'float32','settings':template['settings'],
            'pricing_sha256':None,'prompts_sha256':digest(expected_prompt),'runtime_sha256':canonical(runtime),
            'implementation':{k:template[k] for k in ['runner_sha256','ledger_code_sha256','transport_code_sha256']}})
        ledger=read(shard/'run/local_ledger/ledger.json');attempt=read(shard/'run/attempts'/(ident+'.json'));entry=ledger['attempts'][ident]
        check('single completed coherent ledger and attempt',set(ledger['attempts'])=={ident}
              and attempt['schema']=='synsteg-budget-generation-attempt-v1' and ledger['schema']=='synsteg-budget-ledger-recovery-v1'
              and attempt['identity']==entry['identity']==ident and attempt['status']==entry['status']=='completed'
              and attempt['result']==entry['result'] and attempt['record']==entry['metadata']['record']==meta
              and entry['metadata']=={'record':meta,'provider':'local','model':MODEL} and attempt['provider']=='local' and attempt['model']==MODEL
              and attempt['request_fingerprint']==entry['request_fingerprint']==fingerprint)
        check('zero retries and local zero monetary accounting',integer(attempt['automatic_retries'],0) and integer(entry['automatic_retries'],0)
              and integer(ledger['budget_nanodollars'],20_000_000_000) and not ledger.get('blocked_reason')
              and integer(entry['accounted_nanodollars'],0) and integer(entry['reserved_nanodollars'],0))
        generated=attempt['result'];usage=generated['usage'];tokens=usage['output_tokens']
        check('returned model and deterministic per-record seed',generated['returned_model']==MODEL and generated['revision']==REVISION
              and generated['dtype']=='float32' and integer(generated['seed'],int(canonical([20261024,MODEL,row['record_id']])[:8],16)))
        check('token usage and stop convention',integer(usage['input_tokens'],context['records'][order-1]['input_tokens'])
              and integer(tokens) and 1<=tokens<=1024 and generated['hit_output_limit'] is (tokens==1024)
              and generated['stop_reason']==('max_tokens' if tokens==1024 else 'end_turn'))
        check('nonempty cover',isinstance(generated['cover_text'],str) and bool(generated['cover_text'].strip()))
        summary=read(shard/'run/summary.json')
        check('single-request completed summary',summary['status']=='selected_records_completed'
              and all(integer(summary[k],v) for k,v in {'exit_code':0,'completed':1,'requests_attempted_this_invocation':1,'resumed_completed':0,
                    'selected_records_count':1,'max_new_records_this_invocation':1,'automatic_retries':0}.items())
              and summary['attempts']==[{'identity':ident,'record_id':row['record_id'],'status':'completed'}]
              and integer(summary['ledger']['attempt_count'],1) and integer(summary['ledger']['unknown_or_failed_attempts'],0))
        request={'schema':'synsteg-continuation-replay-request-v1','experiment_id':ID,'model_alias':'qwen05','source_order':order,
                 'intent_identity':ident,'key_sha256':None,'nonces':[]};request_bytes=canonical_bytes(request)+b'\n';guard_sha=digest(request_bytes)
        decision=read(shard/'guard-result.json');reservation=read(shard/'reservation.json');bound=read(shard/'bound-intent.json')
        activation=read(shard/'activation-receipt.json');process=read(shard/'process.json')
        check('exact guard request and deny-only result',rb(shard/'guard-request.json')==request_bytes
              and decision['request_sha256']==reservation['guard_request_sha256']==process['guard_request_sha256']==guard_sha
              and decision['status']=='not_denied_by_this_receipt' and decision['launch_authorized'] is False
              and decision['reasons']==[] and decision['guard_sha256']==digest(rb(root/DENIALS)))
        expected_intent={'schema':ID+'-intent','experiment_id':ID,'source_order':order,'model_alias':'qwen05','identity':ident,
            'record_id':row['record_id'],'status':'reserved_intent_no_execution_claim','protocol_sha256':digest(raw(protocol_path)),
            'helper_sha256':HELPER_SHA,'predecessor_receipt_commit':previous['commit'],'guard_request_sha256':guard_sha,
            'automatic_retries':0,'max_new_requests':1,'lost_execution_identity_policy':'Do not replay any saved intent after loss of execution identity.'}
        check('exact bound intent and protocol',canonical_bytes(bound)==canonical_bytes(expected_intent)
              and rb(shard/'bound-intent.json')==rb(intent_path))
        check('reservation coherence',reservation['experiment_id']==ID and reservation['scientific_execution_stratum']==FIRST_ID
              and reservation['model_alias']=='qwen05' and integer(reservation['source_order'],order) and reservation['identity']==ident
              and reservation['record_id']==row['record_id'] and reservation['status']=='reserved_outcome_unknown'
              and integer(reservation['automatic_retries'],0) and reservation['protocol_sha256']==digest(raw(protocol_path))
              and reservation['intent_sha256']==digest(raw(intent_path)) and reservation['predecessor_receipt_commit']==previous['commit'])
        activation_pins=allpins|predecessor_pins
        for p in [HELPER,FIRST_HELPER,prompt_path,protocol_rel,intent_rel]: activation_pins[p]=digest(raw(root/p))
        fields={'schema','readback_verified','commit','identity','source_order','files_sha256','remote_output_scope_complete',
                'remote_qwen_attempted_source_orders','remote_qwen_active_intents','remote_branches_and_prs_checked','predecessor_receipt_commit'}
        check('exact activation and census declaration',set(activation)==fields and activation['schema']==ID+'-activation-readback-v1'
              and activation['readback_verified'] is True and git_sha(activation['commit'])
              and activation['commit']==reservation['activation_commit'] and activation['identity']==ident and integer(activation['source_order'],order)
              and activation['files_sha256']==activation_pins and activation['remote_output_scope_complete'] is True
              and activation['remote_branches_and_prs_checked'] is True and activation['remote_qwen_attempted_source_orders']==list(range(1,order))
              and all(integer(n) for n in activation['remote_qwen_attempted_source_orders']) and activation['remote_qwen_active_intents']==[ident]
              and activation['predecessor_receipt_commit']==previous['commit'])
        provenance(root,activation_manifest,activation_pins,activation['commit'],check)
        first=read(root/FIRST);execution_root=Path(protocol['execution_root']);command=[str(execution_root/first['python_relative']),*first['argv']]
        for flag,value in {'--model':MODEL,'--revision':REVISION,'--prompts':str(execution_root/prompt_path),'--output':str(execution_root/OUTPUT/f'order{order:03d}'/'run'),'--max-records':'1','--max-new-records':'1'}.items():
            if command.count(flag)!=1: raise ValueError('Ambiguous command flag')
            command[command.index(flag)+1]=value
        elapsed=process['elapsed_seconds']
        check('exact bounded terminal command/process',process['command']==command and process['schema']==ID+'-process'
              and process['experiment_id']==ID and process['scientific_execution_stratum']==FIRST_ID and process['model_alias']=='qwen05'
              and integer(process['source_order'],order) and process['identity']==ident and integer(process['timeout_seconds'],1200)
              and integer(process['exit_code'],0) and integer(process['automatic_retries'],0) and process['timed_out'] is False
              and not process.get('interrupted') and process['status']=='completed_pending_remote_preservation'
              and process['observed_outcome']=={'status':'completed','details':[]} and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=1215)
        check('reservation precedes process start',type(reservation['time_unix']) in (int,float) and type(process['started_at_unix']) in (int,float)
              and math.isfinite(reservation['time_unix']) and math.isfinite(process['started_at_unix']) and reservation['time_unix']<=process['started_at_unix'])
        child=read(shard/'child.json');check('child record shape only',integer(child['pid']) and child['pid']>0 and integer(child['pgid'],child['pid']))
        check('artifact manifest matches all16 preceding raw files',read(shard/'artifact-manifest.json')['files_sha256']=={k:v for k,v in raw_files.items() if k!='artifact-manifest.json'})
        raw_provenance=decode(raw(raw_manifest));prefix=OUTPUT+f'/order{order:03d}/'
        provenance(root,raw_manifest,{prefix+p:h for p,h in raw_files.items()},raw_provenance.get('ref'),check)
        receipt_path=shard/'git-preservation-receipt.json'
        if require_git_receipt or receipt_path.exists():
            receipt=read(receipt_path)
            check('actual raw commit preservation receipt',receipt['readback_verified'] is True and git_sha(receipt['commit'])
                  and receipt['commit']==raw_provenance['ref'] and integer(receipt['source_order'],order)
                  and receipt['files_sha256']==raw_files and receipt['process_sha256']==digest(raw(shard/'process.json')))
        result.update(identity=ident,record_id=row['record_id'],usage=usage,seed=generated['seed'],request_fingerprint=fingerprint,
                      cover_sha256=digest(generated['cover_text'].encode()),raw_commit=raw_provenance['ref'],activation_commit=activation['commit'],
                      context_records_checked=100,predecessors_checked=order-1,raw_files_checked=len(raw_files))
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError) as exc:
        check('complete readable evidence',False);errors.append(type(exc).__name__+': '+str(exc))
    result.update(status='failed' if errors else 'passed',checks=len(checks),check_labels=checks,errors=errors,input_sha256=inputs)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-order',type=int,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--intent',type=Path,required=True)
    p.add_argument('--activation-manifest',type=Path,required=True);p.add_argument('--raw-readback-manifest',type=Path,required=True)
    p.add_argument('--require-git-receipt',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root,a.source_order,a.protocol,a.intent,a.activation_manifest,a.raw_readback_manifest,a.require_git_receipt)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream: stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','checks','errors']}));return 0 if result['status']=='passed' else 2


if __name__=='__main__': raise SystemExit(main())
