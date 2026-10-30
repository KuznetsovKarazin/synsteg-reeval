#!/usr/bin/env python3
"""Read-only first-Qwen saved-evidence audit; no model or tokenizer imports."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

ID = 'development-other-models-single-record-v1'
AUDIT_ID = 'qwen-first-postrun-independent-v1'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REV = '7ae557604adf67be50417f59c2c2f167def9a775'
ACTIVATION_COMMIT = '39fc60433972f4a359ed584f3151ebe07df3d1af'
PROMPTS = 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
PROTO = f'experiments/{ID}/qwen05-protocol.enabled.json'
PROMPT = f'experiments/{ID}/qwen05-source001.jsonl'
INTENT = f'experiments/{ID}/qwen05-intent-source001.json'
PRE = f'results/{ID}/qwen05-preflight'
REVIEW = f'results/{ID}/qwen05-source-prelaunch-independent/review.json'
HELPER = 'scripts/other_model_single_record_v1.py'
SHARD = f'runs/{ID}/qwen05/order001'
FIRST = 'experiments/large-campaign-development-v2/execution-manifest.json'
TEMPLATE = 'runs/baseline-single-record-recovery-v1/order090/run/run.json'
DENIALS = 'results/current-git-status-audit-20261028-v1/no-replay-guard.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()


def canonical(value):
    return digest(canonical_bytes(value))


def raw(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Missing regular file or symlink: ' + str(path))
    return path.read_bytes()


def decode(data):
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError('Duplicate JSON key')
            output[key] = value
        return output
    def bad(_):
        raise ValueError('Nonfinite JSON number')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def inventory(path, excluded=()):
    return {p.relative_to(path).as_posix():digest(raw(p)) for p in sorted(path.rglob('*'))
            if p.is_file() and p.relative_to(path).as_posix() not in excluded}


def audit(root, manifest, require_git_receipt=False):
    root, manifest = Path(root), Path(manifest)
    checks, errors, identities = [], [], {}
    def check(name, condition):
        checks.append(name)
        if not condition:
            errors.append(name)
    def read_bytes(path):
        data = raw(path); identities[str(path)] = digest(data); return data
    def read(path):
        return decode(read_bytes(path))
    result = {'schema': AUDIT_ID, 'model_alias':'qwen05','model':MODEL,'revision':REV,'source_order':1,
              'model_calls':0,'tokenizer_calls':0,'quality_assessed':False,'git_receipt_required':require_git_receipt,
              'git_network_access_by_this_script':False,'independent_source_git_readback_commit':ACTIVATION_COMMIT,
              'physical_process_identity_after_restore_assessed':False,
              'limitations':['Integrity of saved evidence and declared runtime; not cover factuality or semantic delivery.',
                             'This script checks source-manifest provenance and saved receipts; actual remote reads are separate evidence.',
                             'No physical PID continuity, model recomputation or bitwise reproducibility claim.']}
    try:
        provenance = read(manifest)
        check('independently retrieved activation source manifest', provenance['ref'] == ACTIVATION_COMMIT
              and provenance['actual_git_readback'] is True and len(provenance['files']) == 6)
        for item in provenance['files']:
            data = read_bytes(root/item['path'])
            check('actual Git source bytes: '+item['path'], digest(data) == item['sha256'] and len(data) == item['bytes']
                  and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == item['git_blob_sha1'])
        protocol = read(root/PROTO)
        check('fixed activated protocol SHA', digest(raw(root/PROTO)) == '71f4451b4b6ce65aa8e715f736da124372b492488c7b3df6a3a5b7d1f1d37b66')
        for path, expected in (protocol['core_pins'] | protocol['preflight_pins']).items():
            check('pinned input: '+path, digest(read_bytes(root/path)) == expected)
        check('exact frozen full100 source SHA', digest(raw(root/PROMPTS)) == '3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1')
        check('frozen scientific template SHA', digest(read_bytes(root/TEMPLATE)) == '8a6cd75473a9ec61d2f2c82da46e60b45f379361269b5e335923470c16f5ab16')
        rows = [decode(line) for line in raw(root/PROMPTS).splitlines() if line.strip()]
        check('all100 unique development inputs', len(rows) == len({r['record_id'] for r in rows}) == 100
              and all(r['split']=='development' for r in rows))
        row = rows[0]; single = read(root/PROMPT); job = protocol['job']
        ident = canonical(['local',MODEL,row['record_id']])
        meta = {k:row[k] for k in ('record_id','source_group_id','split','topic','genre','target_words')}
        meta['prompt_sha256'] = digest(row['prompt'].encode())
        check('source1 exact record and prompt bytes', single == row and job == {'identity':ident,'model_alias':'qwen05',
              'original_prompt_sha256':meta['prompt_sha256'],'prompt_file_sha256':digest(raw(root/PROMPT)),
              'prompt_path':PROMPT,'record_id':row['record_id'],'source_order':1})
        template = read(root/TEMPLATE)
        check('fixed model arm and scientific settings', protocol['model_id']==MODEL and protocol['revision']==REV
              and protocol['model_alias']=='qwen05' and protocol['dtype']=='float32' and protocol['settings']==template['settings']
              and protocol['source_orders']==[1] and protocol['execution_enabled'] is True and protocol['max_new_requests']==1
              and protocol['automatic_retries']==0 and protocol['full_denominator']==300 and protocol['per_model_denominator']==100
              and protocol['historical_outputs_imported']==0 and protocol['paid_calls_allowed'] is False and protocol['closed_test_allowed'] is False)
        review = read(root/REVIEW)
        check('reviewed helper source identity', review['status']=='passed_source_only' and review['wrapper_sha256']
              == protocol['helper_sha256'] == digest(read_bytes(root/HELPER)) == 'cfc5e284381964db30037931ff9a7091f4fc7e28618b1f0da3374e369eeadcdd')
        runtime_pre = read(root/(PRE+'/runtime.json'))
        contract = read(root/(PRE+'/model-contract.json'))
        config = read(root/(PRE+'/config.json'))
        context = read(root/(PRE+'/all100-context.json'))
        check('contract and declared context configuration', contract['model_id']==MODEL and contract['revision']==REV
              and contract['model_alias']=='qwen05' and contract['settings']==template['settings'] and contract['dtype']=='float32'
              and contract['runtime_sha256']==digest(raw(root/(PRE+'/runtime.json')))
              and context['model_id']==MODEL and context['revision']==REV and context['source_sha256']==digest(raw(root/PROMPTS))
              and context['model_config_sha256']==digest(raw(root/(PRE+'/config.json')))
              and context['model_context_limit']==config['max_position_embeddings']==32768)
        check('complete100 context rows', len(context['records'])==100)
        for n,item in enumerate(context['records'],1):
            check('context source row '+str(n), type(item['source_order']) is int and item['source_order']==n
                  and item['record_id']==rows[n-1]['record_id'] and item['prompt_sha256']==digest(rows[n-1]['prompt'].encode())
                  and type(item['input_tokens']) is int and 0 < item['input_tokens'] <= 32768-1024)
        shard = root/SHARD
        actual_run = read(shard/'run/run.json');runtime = read(shard/'run/runtime.json')
        expected_run = copy.deepcopy(template)
        expected_run.update(model=MODEL,revision=REV,prompts_sha256=digest(raw(root/PROMPT)),selected_records=[meta],selected_records_count=1)
        check('exact run plan except fixed model and source1', actual_run==expected_run)
        check('exact declared runtime and frozen science', runtime==runtime_pre and digest(raw(shard/'run/runtime.json'))
              == 'b9902f9a50d45933b0797788814ce9793347197c2c4b52bd517a0e0d8427f03a')
        ledger=read(shard/'run/local_ledger/ledger.json');attempt=read(shard/'run/attempts'/(ident+'.json'))
        summary=read(shard/'run/summary.json')
        check('single raw attempt and identity', set(ledger['attempts'])=={ident}
              and {p.name for p in (shard/'run/attempts').iterdir()}=={ident+'.json'})
        entry=ledger['attempts'][ident]
        fp=canonical({'record':meta,'provider':'local','model':MODEL,'revision':REV,'dtype':'float32',
            'settings':template['settings'],'pricing_sha256':None,'prompts_sha256':digest(raw(root/PROMPT)),
            'runtime_sha256':canonical(runtime),'implementation':{k:template[k] for k in
            ('runner_sha256','ledger_code_sha256','transport_code_sha256')}})
        check('entry and attempt schema identity metadata', entry['identity']==attempt['identity']==ident
              and entry['metadata']=={'record':meta,'provider':'local','model':MODEL} and attempt['record']==meta
              and attempt['provider']=='local' and attempt['model']==MODEL
              and attempt['schema']=='synsteg-budget-generation-attempt-v1')
        check('independently reconstructed fingerprints', entry['request_fingerprint']==attempt['request_fingerprint']==fp)
        check('both completed and zero retries', entry['status']==attempt['status']=='completed'
              and type(entry['automatic_retries']) is type(attempt['automatic_retries']) is int
              and entry['automatic_retries']==attempt['automatic_retries']==0)
        check('ledger schema cap zero accounting', ledger['schema']=='synsteg-budget-ledger-recovery-v1'
              and ledger['budget_nanodollars']==20_000_000_000 and not ledger.get('blocked_reason')
              and entry['accounted_nanodollars']==entry['reserved_nanodollars']==0)
        generated=entry['result'];usage=generated['usage'];tokens=usage['output_tokens']
        check('matching result objects', generated==attempt['result'])
        check('returned model revision dtype', generated['returned_model']==MODEL and generated['revision']==REV and generated['dtype']=='float32')
        check('original per-record seed', type(generated['seed']) is int and generated['seed']==int(canonical([20261024,MODEL,row['record_id']])[:8],16))
        check('input usage matches actual preflight source1', type(usage['input_tokens']) is int and usage['input_tokens']==context['records'][0]['input_tokens']==211)
        check('output token bound and frozen stop convention', type(tokens) is int and 1<=tokens<=1024
              and generated['hit_output_limit'] is (tokens==1024) and generated['stop_reason']==('max_tokens' if tokens==1024 else 'end_turn'))
        check('nonempty saved cover', isinstance(generated['cover_text'],str) and bool(generated['cover_text'].strip()))
        check('completed one-request summary', summary['status']=='selected_records_completed' and type(summary['exit_code']) is int
              and summary['exit_code']==0 and summary['selected_records_count']==summary['completed']==summary['requests_attempted_this_invocation']
              ==summary['max_new_records_this_invocation']==1 and summary['resumed_completed']==summary['automatic_retries']==0)
        check('summary attempt and ledger counters', summary['attempts']==[{'identity':ident,'record_id':row['record_id'],'status':'completed'}]
              and summary['ledger']['attempt_count']==1 and summary['ledger']['unknown_or_failed_attempts']==0)
        request={'schema':'synsteg-continuation-replay-request-v1','experiment_id':ID,'model_alias':'qwen05','source_order':1,
                 'intent_identity':ident,'key_sha256':None,'nonces':[]}
        request_bytes=canonical_bytes(request)+b'\n';request_sha=digest(request_bytes)
        decision=read(shard/'guard-result.json');reserve=read(shard/'reservation.json');intent=read(shard/'bound-intent.json')
        activation=read(shard/'activation-receipt.json');process=read(shard/'process.json')
        check('exact bound guard request and deny-only decision', read_bytes(shard/'guard-request.json')==request_bytes
              and decision['request_sha256']==reserve['guard_request_sha256']==process['guard_request_sha256']==request_sha
              and decision['status']=='not_denied_by_this_receipt' and decision['launch_authorized'] is False
              and decision['reasons']==[] and decision['guard_sha256']==digest(raw(root/DENIALS)))
        check('durable reservation and intent binding', reserve['experiment_id']==ID and reserve['model_alias']=='qwen05'
              and reserve['source_order']==1 and reserve['identity']==ident and reserve['record_id']==row['record_id']
              and reserve['status']=='reserved_outcome_unknown' and reserve['automatic_retries']==0
              and reserve['protocol_sha256']==digest(raw(root/PROTO)) and reserve['intent_sha256']==digest(read_bytes(root/INTENT))
              and intent==read(root/INTENT) and intent['identity']==ident and intent['protocol_sha256']==digest(raw(root/PROTO))
              and intent['helper_sha256']==protocol['helper_sha256'] and intent['max_new_requests']==1 and intent['automatic_retries']==0)
        pins=dict(protocol['core_pins'])|protocol['preflight_pins']
        pins.update({p:digest(raw(root/p)) for p in [HELPER,PROMPT,PROTO,INTENT]})
        check('activation exact independently fetched commit and pin map', activation['schema']==ID+'-activation-readback-v1'
              and activation['readback_verified'] is True and activation['commit']==reserve['activation_commit']==ACTIVATION_COMMIT
              and activation['identity']==ident and activation['files_sha256']==pins and activation['remote_output_scope_complete'] is True
              and activation['remote_branches_and_prs_checked'] is True and activation['remote_qwen_current100_attempts']==[]
              and activation['remote_qwen_active_intents']==[ident])
        first=read(root/FIRST);execution_root=Path(protocol['execution_root'])
        command=[str(execution_root/first['python_relative']),*first['argv']]
        for flag,value in {'--model':MODEL,'--revision':REV,'--prompts':str(execution_root/PROMPT),
                           '--output':str(execution_root/SHARD/'run'),'--max-records':'1','--max-new-records':'1'}.items():
            command[command.index(flag)+1]=value
        check('exact scientific command', process['command']==command)
        elapsed=process['elapsed_seconds']
        check('clean bounded terminal process', process['schema']==ID+'-process' and process['model_alias']=='qwen05'
              and process['source_order']==1 and process['identity']==ident and process['timeout_seconds']==1200
              and process['timed_out'] is False and not process.get('interrupted') and type(process['exit_code']) is int
              and process['exit_code']==0 and process['automatic_retries']==0 and process['status']=='completed_pending_remote_preservation'
              and process['observed_outcome']=={'status':'completed','details':[]} and type(elapsed) in (int,float)
              and math.isfinite(elapsed) and 0<=elapsed<=1215)
        check('reservation precedes declared process start', reserve['time_unix']<=process['started_at_unix'])
        child=read(shard/'child.json')
        check('child group record shape only', type(child['pid']) is int and child['pid']>0 and child['pgid']==child['pid'])
        read_bytes(shard/'stdout.log');read_bytes(shard/'stderr.log')
        artifact=read(shard/'artifact-manifest.json')
        check('raw artifact inventory', artifact['files_sha256']==inventory(shard,{'artifact-manifest.json','git-preservation-receipt.json'}))
        if require_git_receipt or (shard/'git-preservation-receipt.json').exists():
            receipt=read(shard/'git-preservation-receipt.json')
            check('postrun exact Git preservation receipt', receipt['readback_verified'] is True
                  and isinstance(receipt['commit'],str) and re.fullmatch('[0-9a-f]{40}',receipt['commit'])
                  and receipt['source_order']==1 and receipt['process_sha256']==digest(raw(shard/'process.json'))
                  and receipt['files_sha256']==inventory(shard,{'git-preservation-receipt.json'}))
        result.update(identity=ident,record_id=row['record_id'],usage=usage,seed=generated['seed'],
                      request_fingerprint=fp,cover_sha256=digest(generated['cover_text'].encode()),
                      context_records_checked=100,max_preflight_input_tokens=max(x['input_tokens'] for x in context['records']))
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError) as exc:
        check('complete readable evidence',False);errors.append(type(exc).__name__+': '+str(exc))
    result.update(status='failed' if errors else 'passed',checks=len(checks),check_labels=checks,errors=errors,input_sha256=identities)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--git-input-manifest',type=Path,required=True)
    p.add_argument('--require-git-receipt',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root,a.git_input_manifest,a.require_git_receipt)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks','errors')}))
    raise SystemExit(0 if result['status']=='passed' else 2)
