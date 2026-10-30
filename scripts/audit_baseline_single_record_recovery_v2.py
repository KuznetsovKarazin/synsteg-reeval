#!/usr/bin/env python3
"""Independent saved-shard checks. Stdlib only; never imports or runs a model.

The separate review manifest and saved Git receipts are inputs, not evidence of
network access by this program. Root must independently perform Git readback.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re

ID = 'baseline-single-record-recovery-v2'
SOURCE = '4050291345c43776e9c93528b6afe292c062ba7a'
MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
REVISION = 'fe8a4ea1ffedaf415f4da2f062534de366a451e6'
HELPER = 'scripts/baseline_single_record_recovery_v2.py'
PROPOSED = 'experiments/' + ID + '/protocol.proposed.json'
INPUTS = 'experiments/' + ID + '/inputs/git-source-inputs.json'
INPUTS_SHA = 'd222fd9e69ad2f7ce70731a5500847e9591b7fec1d8b7d8577fb9b36bd189b8d'
FIRST = 'experiments/large-campaign-development-v2/execution-manifest.json'
PROMPTS = 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
BASE = 'results/baseline-single-record-recovery-v1-root/full300-after090.json'
TEMPLATE = 'runs/baseline-single-record-recovery-v1/order090/run'
DENIALS = 'results/current-git-status-audit-20261028-v1/no-replay-guard.json'
PINS = {
    FIRST: '7b23df64b875799cc592640a9aadf7ea6bcd870b525335141169cf93a6003ff1',
    PROMPTS: '3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1',
    BASE: 'b25f56576ec2c47898b80cb95a59e1d70a15e99037a78b6b96a3f430ca4aaac7',
    TEMPLATE + '/run.json': '8a6cd75473a9ec61d2f2c82da46e60b45f379361269b5e335923470c16f5ab16',
    TEMPLATE + '/runtime.json': 'b9902f9a50d45933b0797788814ce9793347197c2c4b52bd517a0e0d8427f03a',
    DENIALS: 'ed7c17d68df5cc524ec8e441f3febb2a96e6d8108eda6f22f2ac154fae74db0c',
    'scripts/run_budget_generation.py': '4d5ab74c78c58214c1b5e11a346b4339eb0c6d08e713e8909d0d5fd9947c7e02',
    'scripts/baseline_single_record_recovery_v1.py': '12e7356d990242f7f09eaea88d045b16019f734956025ec86cfb5005bdf96294',
    'scripts/continuation_replay_guard_v1.py': 'fae8a6f191582dcf0d64a36ce6784cfe38495b3e3da38946723ea0fbebf8584c',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()


def canonical(value):
    return digest(compact(value))


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def bad(_):
        raise ValueError('nonfinite JSON number')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def reject_links(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink evidence refused')


def raw(path):
    path = Path(path)
    reject_links(path)
    if not path.is_file():
        raise ValueError('missing regular evidence file: ' + str(path))
    return path.read_bytes()


def inventory(directory, excluded=()):
    reject_links(directory)
    result = {}
    for path in sorted(directory.rglob('*')):
        reject_links(path)
        if path.is_file() and str(path.relative_to(directory)) not in excluded:
            result[str(path.relative_to(directory))] = digest(raw(path))
    return result


def safe_relative(value):
    return isinstance(value, str) and bool(value) and not Path(value).is_absolute() and '..' not in Path(value).parts


def commit(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{40}', value) is not None


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def audit(root, protocol_path, shard, order, review_manifest, require_git_receipt=False):
    root, protocol_path, shard, review_manifest = map(Path, (root, protocol_path, shard, review_manifest))
    checks, errors, input_hashes = [], [], {}

    def check(label, condition):
        checks.append(label)
        if not condition:
            errors.append(label)

    def read_bytes(path):
        data = raw(path)
        input_hashes[str(path)] = digest(data)
        return data

    def read(path):
        return strict_json(read_bytes(path))

    def finish():
        return {'schema': ID + '-independent-shard-audit-v1',
                'status': 'failed' if errors else 'passed', 'source_order': order,
                'checks': len(checks), 'check_labels': checks, 'errors': errors,
                'input_sha256': input_hashes, 'source_snapshot_commit': SOURCE,
                'model_calls': 0, 'tokenizer_calls': 0, 'git_remote_access_performed': False,
                'git_receipt_required': require_git_receipt, 'quality_assessed': False,
                'transport_assessed': False, 'live_process_identity_assessed': False,
                'limitation': 'Saved bytes, declared runtime and request/outcome integrity only. '
                    'Git readback and prelaunch source review are separate external evidence. '
                    'No claim of cover fidelity, delivery, same physical process after restore, or bitwise NN reproducibility.'}

    try:
        check('allowed unreserved source order92..100', type(order) is int and 92 <= order <= 100)
        if errors:
            return finish()
        check('shard relative path binds actual job', shard.relative_to(root).as_posix() == f'runs/{ID}/order{order:03d}')
        for path, expected in PINS.items():
            check('frozen source bytes: ' + path, digest(read_bytes(root/path)) == expected)
        if errors:
            return finish()
        source = read(root/INPUTS)
        check('pinned source manifest', digest(raw(root/INPUTS)) == INPUTS_SHA
              and source['source_commit'] == SOURCE and source['actual_git_readback'] is True)
        pinned = {f['path']: f['sha256'] for f in source['files']}
        check('57 unique safe frozen input paths', len(pinned) == len(source['files']) == 57 and all(map(safe_relative, pinned)))
        if errors:
            return finish()
        for item in source['files']:
            data = read_bytes(root/item['path'])
            check('source manifest file: ' + item['path'], digest(data) == item['sha256'] and len(data) == item['bytes']
                  and hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == item['git_blob_sha1'])
        review = read(review_manifest)
        proposal = read(root/PROPOSED)
        protocol = read(protocol_path)
        check('independent passed source review pins', review['schema'] == 'baseline92-successor-independent-review-v1'
              and review['status'] == 'passed' and review['wrapper_sha256'] == digest(read_bytes(root/HELPER))
              and review['proposed_protocol_sha256'] == digest(raw(root/PROPOSED)))
        expected_protocol = copy.deepcopy(proposal)
        expected_protocol['execution_enabled'] = True
        check('activation changes only execution_enabled', proposal['execution_enabled'] is False and protocol == expected_protocol)
        check('protocol scientific/scheduling boundaries', protocol['experiment_id'] == ID and protocol['source_commit'] == SOURCE
              and protocol['new_source_orders'] == list(range(92,101)) and protocol['full_denominator'] == 300
              and protocol['tinyllama_denominator'] == 100 and protocol['permanently_unknown_source_orders'] == [82]
              and protocol['blocked_intent_source_orders'] == [91] and protocol['automatic_retries'] == 0
              and protocol['max_new_requests_per_shard'] == 1 and protocol['timeout_seconds_per_shard_including_model_load'] == 1200
              and protocol['continue_after_new_failure'] is False and protocol['stop_after_each_shard_pending_remote_preservation'] is True
              and protocol['historical_ledger_rewritten'] is False and protocol['paid_calls_allowed'] is False
              and protocol['closed_test_allowed'] is False and protocol['helper_sha256'] == review['wrapper_sha256']
              and protocol['input_manifest_sha256'] == INPUTS_SHA and protocol['base_projection_sha256'] == PINS[BASE]
              and protocol['canonical_nn_lock'] == str(Path(protocol['execution_root'])/'results/project-nn.lock'))
        rows = [strict_json(line) for line in read_bytes(root/PROMPTS).splitlines() if line.strip()]
        check('all100 unique development sources', len(rows) == len({r['record_id'] for r in rows}) == 100
              and all(r['split'] == 'development' for r in rows))
        base = read(root/BASE)
        keyed = {(r['model_alias'],r['source_order']):r for r in base['records']}
        check('base full300 unique grid and full89/1/210 counts', len(base['records']) == len(keyed) == 300
              and set(keyed) == {(m,n) for m in ('tinyllama11','qwen05','falcon3_1b') for n in range(1,101)}
              and dict(Counter(r['status'] for r in base['records'])) == {'completed':89,'reserved_outcome_unknown':1,'not_attempted':210})
        check('historical82 unknown and91 outcome unchanged', keyed[('tinyllama11',82)]['status'] == 'reserved_outcome_unknown'
              and keyed[('tinyllama11',91)]['status'] == 'not_attempted'
              and all(r['record_id'] == rows[n-1]['record_id'] for (m,n),r in keyed.items()))
        denials = read(root/DENIALS)
        check('separate91 scheduling block and ending replay bans retained', denials['blocked_baseline_intents'][0]['source_order'] == 91
              and denials['blocked_baseline_intents'][0]['scheduling_status'] == 'blocked_reserved_intent_evidence_gap'
              and denials['deny_experiment_ids'] == ['framed-interval-public-ending-v2']
              and len(denials['conservatively_burned_nonce_pairs']) == 8)
        jobs = []
        for n in range(92,101):
            row = rows[n-1]
            payload = (json.dumps(row, ensure_ascii=False, separators=(',',':'))+'\n').encode()
            jobs.append({'source_order':n,'record_id':row['record_id'],'identity':canonical(['local',MODEL,row['record_id']]),
                         'prompt_path':f'experiments/{ID}/inputs/prompts/order{n:03d}.jsonl',
                         'prompt_file_sha256':digest(payload),'original_prompt_sha256':digest(row['prompt'].encode())})
            check('original source prompt ' + str(n), read_bytes(root/jobs[-1]['prompt_path']) == payload)
        check('all9 jobs bound to original source order', protocol['jobs'] == jobs)
        job, row = jobs[order-92], rows[order-1]
        ident = job['identity']
        meta = {k:row[k] for k in ('record_id','source_group_id','split','topic','genre','target_words')}
        meta['prompt_sha256'] = job['original_prompt_sha256']
        template = read(root/TEMPLATE/'run.json')
        expected_run = copy.deepcopy(template)
        expected_run.update(prompts_sha256=job['prompt_file_sha256'],selected_records=[meta],selected_records_count=1)
        actual_run, runtime = read(shard/'run/run.json'), read(shard/'run/runtime.json')
        check('run plan matches frozen science except exact shard metadata', actual_run == expected_run)
        check('declared runtime matches frozen runtime', runtime == read(root/TEMPLATE/'runtime.json'))
        ledger = read(shard/'run/local_ledger/ledger.json')
        attempt = read(shard/'run/attempts'/(ident+'.json'))
        summary = read(shard/'run/summary.json')
        check('one ledger identity and one raw attempt', set(ledger['attempts']) == {ident}
              and {p.name for p in (shard/'run/attempts').iterdir()} == {ident+'.json'})
        entry = ledger['attempts'][ident]
        fingerprint = canonical({'record':meta,'provider':'local','model':MODEL,'revision':REVISION,
            'dtype':template['dtype'],'settings':template['settings'],'pricing_sha256':None,
            'prompts_sha256':job['prompt_file_sha256'],'runtime_sha256':canonical(runtime),
            'implementation':{k:template[k] for k in ('runner_sha256','ledger_code_sha256','transport_code_sha256')}})
        check('ledger schema/cap', ledger['schema'] == 'synsteg-budget-ledger-recovery-v1'
              and ledger['budget_nanodollars'] == 20_000_000_000 and not ledger.get('blocked_reason'))
        check('ledger and attempt metadata', entry['metadata'] == {'record':meta,'provider':'local','model':MODEL}
              and attempt['record'] == meta and attempt['provider'] == 'local' and attempt['model'] == MODEL
              and attempt['schema'] == 'synsteg-budget-generation-attempt-v1')
        check('both request fingerprints', entry['request_fingerprint'] == attempt['request_fingerprint'] == fingerprint)
        check('both identities', entry['identity'] == attempt['identity'] == ident)
        check('both completed with zero retries', entry['status'] == attempt['status'] == 'completed'
              and type(entry['automatic_retries']) is type(attempt['automatic_retries']) is int
              and entry['automatic_retries'] == attempt['automatic_retries'] == 0)
        check('zero local accounted/reserved', entry['accounted_nanodollars'] == entry['reserved_nanodollars'] == 0)
        result = entry['result']
        check('matching raw result objects', isinstance(result,dict) and result == attempt['result'])
        check('result model revision dtype and seed', result['returned_model'] == MODEL and result['revision'] == REVISION
              and result['dtype'] == 'float32' and type(result['seed']) is int
              and result['seed'] == int(canonical([20261024,MODEL,row['record_id']])[:8],16))
        tokens = result['usage']['output_tokens']
        check('strict nonzero token bounds', type(tokens) is int and 1 <= tokens <= 1024
              and type(result['usage']['input_tokens']) is int and result['usage']['input_tokens'] > 0)
        check('frozen termination convention', result['stop_reason'] == ('max_tokens' if tokens == 1024 else 'end_turn')
              and result['hit_output_limit'] is (tokens == 1024))
        check('nonempty saved cover', isinstance(result['cover_text'],str) and bool(result['cover_text'].strip()))
        check('terminal summary', summary['status'] == 'selected_records_completed' and summary['exit_code'] == 0
              and summary['selected_records_count'] == summary['completed'] == summary['requests_attempted_this_invocation']
              == summary['max_new_records_this_invocation'] == 1 and summary['resumed_completed'] == summary['automatic_retries'] == 0)
        check('summary attempts and ledger count', summary['attempts'] == [{'identity':ident,'record_id':row['record_id'],'status':'completed'}]
              and summary['ledger']['attempt_count'] == 1 and summary['ledger']['unknown_or_failed_attempts'] == 0)
        reservation, process = read(shard/'reservation.json'), read(shard/'process.json')
        activation, intent, decision = read(shard/'activation-receipt.json'), read(shard/'bound-intent.json'), read(shard/'guard-result.json')
        request = {'schema':'synsteg-continuation-replay-request-v1','experiment_id':ID,'model_alias':'tinyllama11',
                   'source_order':order,'intent_identity':ident,'key_sha256':None,'nonces':[]}
        request_sha = digest(compact(request)+b'\n')
        check('raw guard request exact actual job', read_bytes(shard/'guard-request.json') == compact(request)+b'\n')
        check('guard decision bound to actual request and pinned denials', decision['schema'] == 'continuation-replay-guard-v1-decision'
              and decision['status'] == 'not_denied_by_this_receipt' and decision['launch_authorized'] is False
              and decision['reasons'] == [] and decision['guard_sha256'] == PINS[DENIALS]
              and decision['guard_source_commit'] == denials['source_commit'] and decision['request_sha256'] == request_sha
              and reservation['guard_request_sha256'] == intent['guard_request_sha256'] == process['guard_request_sha256'] == request_sha)
        check('reservation binds job protocol intent activation without retry', reservation['experiment_id'] == ID
              and reservation['source_order'] == order and reservation['record_id'] == row['record_id']
              and reservation['identity'] == ident and reservation['status'] == 'reserved_outcome_unknown'
              and reservation['protocol_sha256'] == digest(raw(protocol_path))
              and reservation['activation_commit'] == activation['commit'] and reservation['automatic_retries'] == 0)
        check('durable bound intent', intent['schema'] == ID+'-intent' and intent['experiment_id'] == ID
              and intent['status'] == 'reserved_intent_no_execution_claim' and intent['source_order'] == order
              and intent['record_id'] == row['record_id'] and intent['identity'] == ident
              and intent['source_snapshot_commit'] == SOURCE and intent['protocol_sha256'] == digest(raw(protocol_path))
              and intent['helper_sha256'] == review['wrapper_sha256'] and intent['max_new_requests'] == 1
              and intent['automatic_retries'] == 0)
        previous_path = root/('runs/baseline-single-record-recovery-v1/order090' if order == 92 else f'runs/{ID}/order{order-1:03d}')
        prior = read(previous_path/'git-preservation-receipt.json')
        prior_process = read(previous_path/'process.json')
        check('predecessor exact preserved clean terminal bytes', prior['readback_verified'] is True and commit(prior['commit'])
              and prior['source_order'] == (90 if order == 92 else order-1)
              and prior['files_sha256'] == inventory(previous_path, {'git-preservation-receipt.json'})
              and prior['process_sha256'] == digest(raw(previous_path/'process.json'))
              and prior_process['status'] == 'completed_pending_remote_preservation' and type(prior_process['exit_code']) is int
              and prior_process['exit_code'] == 0 and prior_process['timed_out'] is False and not prior_process.get('interrupted'))
        check('predecessor commit bound across activation and intent', activation['predecessor_receipt_commit']
              == intent['predecessor_receipt_commit'] == prior['commit'])
        expected_pins = dict(pinned)
        for path in [root/INPUTS,root/HELPER,protocol_path,*[root/j['prompt_path'] for j in jobs]]:
            expected_pins[path.relative_to(root).as_posix()] = digest(raw(path))
        extra_paths = set(activation['files_sha256']) - set(expected_pins)
        check('activation has exactly one safe original intent file', len(extra_paths) == 1 and all(map(safe_relative,extra_paths)))
        if len(extra_paths) == 1 and all(map(safe_relative,extra_paths)):
            original_intent = extra_paths.pop()
            expected_pins[original_intent] = digest(read_bytes(root/original_intent))
            check('original saved intent matches bound copy', read(root/original_intent) == intent
                  and expected_pins[original_intent] == reservation['intent_sha256'])
        check('exact activation readback pins and previous remote scope', activation['schema'] == ID+'-activation-readback-v1'
              and activation['readback_verified'] is True and commit(activation['commit'])
              and type(activation['source_order']) is int and activation['source_order'] == order and activation['identity'] == ident
              and activation['files_sha256'] == expected_pins and activation['remote_output_scope_complete'] is True
              and activation['remote_current_output_orders'] == list(range(92,order)))
        first = read(root/FIRST)
        execution_root = Path(protocol['execution_root'])
        command = [str(execution_root/first['python_relative']),*first['argv']]
        for flag,value in {'--prompts':str(execution_root/job['prompt_path']),
                           '--output':str(execution_root/f'runs/{ID}/order{order:03d}/run'),
                           '--max-records':'1','--max-new-records':'1'}.items():
            command[command.index(flag)+1] = value
        check('exact scientific child command', process['command'] == command)
        check('clean bounded terminal process', process['schema'] == ID+'-process' and process['source_order'] == order
              and process['identity'] == ident and process['timeout_seconds'] == 1200 and process['timed_out'] is False
              and process['automatic_retries'] == 0 and type(process['exit_code']) is int and process['exit_code'] == 0
              and not process.get('interrupted') and process['status'] == 'completed_pending_remote_preservation'
              and process['observed_outcome'] == {'status':'completed','details':[]}
              and finite_number(process['elapsed_seconds']) and 0 <= process['elapsed_seconds'] <= 1215)
        check('reservation precedes recorded process start', finite_number(reservation['time_unix'])
              and finite_number(process['started_at_unix']) and reservation['time_unix'] <= process['started_at_unix'])
        child = read(shard/'child.json')
        check('recorded child group shape only', type(child['pid']) is int and child['pid'] > 0 and child['pgid'] == child['pid'])
        read_bytes(shard/'stdout.log'); read_bytes(shard/'stderr.log')
        artifact = read(shard/'artifact-manifest.json')
        check('all raw shard artifacts preserved exactly', artifact['files_sha256']
              == inventory(shard, {'artifact-manifest.json','git-preservation-receipt.json'}))
        receipt_path = shard/'git-preservation-receipt.json'
        if require_git_receipt or receipt_path.exists():
            receipt = read(receipt_path)
            check('postrun Git receipt exact saved inventory', receipt['readback_verified'] is True and commit(receipt['commit'])
                  and type(receipt['source_order']) is int and receipt['source_order'] == order
                  and receipt['process_sha256'] == digest(raw(shard/'process.json'))
                  and receipt['files_sha256'] == inventory(shard, {'git-preservation-receipt.json'}))
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        check('complete readable well-formed evidence', False)
        errors.append(type(exc).__name__ + ': ' + str(exc))
    return finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--shard',type=Path,required=True)
    parser.add_argument('--source-order',type=int,required=True)
    parser.add_argument('--review-manifest',type=Path,required=True)
    parser.add_argument('--require-git-receipt',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.root,args.protocol,args.shard,args.source_order,args.review_manifest,args.require_git_receipt)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks','errors')}))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
