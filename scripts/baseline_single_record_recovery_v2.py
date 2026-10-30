#!/usr/bin/env python3
"""One-shot continuation92..100 after an explicitly blocked order91.

Frozen scientific runner/model/runtime are unchanged. This new outer launcher
integrates the deny-only guard and requires exact external Git readback before
each request. It never calls the old recovery execute/inspect functions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ID = 'baseline-single-record-recovery-v2'
SOURCE = '4050291345c43776e9c93528b6afe292c062ba7a'
ORDERS = list(range(92, 101))
INPUTS = 'experiments/' + ID + '/inputs/git-source-inputs.json'
INPUTS_SHA = 'd222fd9e69ad2f7ce70731a5500847e9591b7fec1d8b7d8577fb9b36bd189b8d'
HELPER = 'scripts/' + ID.replace('-', '_') + '.py'
OLD_HELPER = 'scripts/baseline_single_record_recovery_v1.py'
GUARD = 'scripts/continuation_replay_guard_v1.py'
DENIALS = 'results/current-git-status-audit-20261028-v1/no-replay-guard.json'
BASE = 'results/baseline-single-record-recovery-v1-root/full300-after090.json'
FIRST = 'experiments/large-campaign-development-v2/execution-manifest.json'
PROMPTS = 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
OLD_PROTOCOL = 'experiments/baseline-single-record-recovery-v1/inputs/protocol.enabled.json'
ORIGINAL = 'runs/large-campaign-development-v2/models/tinyllama11'
PREVIOUS = 'runs/baseline-single-record-recovery-v1/order090'
MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
ROOT_PIN = '/workspace/scratch/af7adce5db85/baseline92-continuation-v2/checkout'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')) + '\n').encode()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_bytes_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reject_links(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink evidence/path refused')


def raw(path):
    reject_links(path)
    require(Path(path).is_file(), 'Missing regular file: ' + str(path))
    return Path(path).read_bytes()


def sha(path):
    reject_links(path)
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    def pairs(values):
        output = {}
        for key, value in values:
            require(key not in output, 'Duplicate JSON key')
            output[key] = value
        return output
    def bad(_):
        raise ValueError('Nonfinite JSON refused')
    return json.loads(raw(path), object_pairs_hook=pairs, parse_constant=bad)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify_inputs(root):
    root = Path(root)
    require(sha(root / INPUTS) == INPUTS_SHA, 'Pinned source input manifest changed')
    manifest = load(root / INPUTS)
    require(manifest['source_commit'] == SOURCE and manifest['actual_git_readback'] is True,
            'Source provenance changed')
    names = [item['path'] for item in manifest['files']]
    require(len(names) == 57 and len(set(names)) == 57, 'Input manifest cardinality')
    for item in manifest['files']:
        name = Path(item['path'])
        require(not name.is_absolute() and '..' not in name.parts, 'Unsafe source path')
        data = raw(root / name)
        require(len(data) == item['bytes'] and digest_bytes(data) == item['sha256'], 'Source bytes changed: ' + str(name))
        require(hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == item['git_blob_sha1'], 'Git blob mismatch')
    return manifest


def original_rows(root):
    rows = [json.loads(line) for line in raw(root / PROMPTS).decode().splitlines() if line.strip()]
    require(len(rows) == 100 and len({r['record_id'] for r in rows}) == 100
            and all(r['split'] == 'development' for r in rows), 'Require unique development100')
    return rows


def validate_base(base, rows):
    records = base['records']
    require(len(records) == 300, 'Full300 denominator')
    keyed = {(r['model_alias'], r['source_order']): r for r in records}
    require(len(keyed) == 300 and set(keyed) == {(m, n) for m in ('tinyllama11', 'qwen05', 'falcon3_1b') for n in range(1, 101)}, 'Full300 unique grid')
    require(dict(Counter(r['status'] for r in records)) == {'completed': 89, 'reserved_outcome_unknown': 1, 'not_attempted': 210}, 'Base counts changed')
    for (model, order), record in keyed.items():
        wanted = ('completed' if order <= 90 and order != 82 else 'reserved_outcome_unknown' if order == 82 else 'not_attempted') if model == 'tinyllama11' else 'not_attempted'
        require(record['record_id'] == rows[order-1]['record_id'] and record['status'] == wanted, 'Base identity/outcome changed')
    return keyed


def job_for(root, order):
    require(type(order) is int and order in ORDERS, 'Only original orders92..100')
    rows = original_rows(root)
    old = module(root / OLD_HELPER, 'recovery_v1_pure_support')
    row = rows[order-1]
    payload = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    return {'source_order': order, 'record_id': row['record_id'],
            'identity': old.canonical(['local', MODEL, row['record_id']]),
            'prompt_path': f'experiments/{ID}/inputs/prompts/order{order:03d}.jsonl',
            'prompt_file_sha256': digest_bytes(payload), 'original_prompt_sha256': digest_bytes(row['prompt'].encode())}, payload


def expected_protocol(root, enabled=False):
    return {'schema': ID+'-protocol', 'experiment_id': ID, 'execution_enabled': enabled,
            'source_commit': SOURCE, 'execution_root': ROOT_PIN, 'new_source_orders': ORDERS,
            'full_denominator': 300, 'tinyllama_denominator': 100,
            'permanently_unknown_source_orders': [82], 'blocked_intent_source_orders': [91],
            'base_projection': BASE, 'base_projection_sha256': sha(root/BASE),
            'input_manifest_sha256': INPUTS_SHA, 'helper_sha256': sha(root/HELPER),
            'max_new_requests_per_shard': 1, 'timeout_seconds_per_shard_including_model_load': 1200,
            'automatic_retries': 0, 'continue_after_new_failure': False,
            'stop_after_each_shard_pending_remote_preservation': True,
            'canonical_nn_lock': str(root/'results/project-nn.lock'),
            'output_relative': 'runs/'+ID, 'historical_ledger_rewritten': False,
            'first_successor_predecessor': PREVIOUS, 'skip91_requires_new_id': True,
            'execution_stratum': 'same frozen scientific runner and runtime;1200seconds per fresh one-record process; no bitwise-equivalence claim',
            'paid_calls_allowed': False, 'closed_test_allowed': False,
            'jobs': [job_for(root, n)[0] for n in ORDERS]}


def prepare(root):
    root = Path(root)
    require(str(root) == ROOT_PIN, 'Root must equal fixed execution root')
    verify_inputs(root)
    validate_base(load(root/BASE), original_rows(root))
    old = module(root/OLD_HELPER, 'recovery_v1_prepare_support')
    for order in ORDERS:
        job, content = job_for(root, order)
        path = root/job['prompt_path']
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
    protocol = expected_protocol(root)
    old.write_new(root/f'experiments/{ID}/protocol.proposed.json', protocol)
    return protocol


def verify_protocol(root, path):
    require(str(root) == ROOT_PIN, 'Wrong execution root')
    verify_inputs(root)
    protocol = load(path)
    require(type(protocol.get('execution_enabled')) is bool, 'Execution flag must be bool')
    require(protocol == expected_protocol(root, protocol['execution_enabled']), 'Protocol differs from pinned design')
    validate_base(load(root/BASE), original_rows(root))
    for job in protocol['jobs']:
        require(sha(root/job['prompt_path']) == job['prompt_file_sha256'], 'Prompt bytes changed')
        require(load(root/job['prompt_path']) == original_rows(root)[job['source_order']-1], 'Prompt row changed')
    return protocol


def replay_request(job):
    return {'schema': 'synsteg-continuation-replay-request-v1', 'experiment_id': ID,
            'model_alias': 'tinyllama11', 'source_order': job['source_order'], 'intent_identity': job['identity'],
            'key_sha256': None, 'nonces': []}


def check_replay(root, job):
    guard = module(root/GUARD, 'continuation_replay_guard_bound_v2')
    decision = guard.evaluate_bytes(raw(root/DENIALS), canonical_bytes(replay_request(job)))
    require(decision['status'] == 'not_denied_by_this_receipt' and decision['launch_authorized'] is False,
            'Replay guard denied job')
    return decision


def inspect_shard(root, directory, job):
    old = module(root/OLD_HELPER, 'recovery_v1_shard_validator')
    if directory.exists():
        reject_links(directory)
        old.file_inventory(directory)
    return old.shard_status(directory, job, load(root/ORIGINAL/'run.json'), load(root/ORIGINAL/'runtime.json'))


def verify_preserved(root, directory, job):
    old = module(root/OLD_HELPER, 'recovery_v1_inventory_validator')
    require(directory.is_dir(), 'Missing predecessor')
    reject_links(directory)
    receipt = load(directory/'git-preservation-receipt.json')
    process = load(directory/'process.json')
    inventory = {k: v for k, v in old.file_inventory(directory).items() if k != 'git-preservation-receipt.json'}
    require(receipt.get('readback_verified') is True and type(receipt.get('commit')) is str
            and re.fullmatch('[0-9a-f]{40}', receipt['commit']) is not None
            and type(receipt.get('source_order')) is int and receipt['source_order'] == job['source_order']
            and receipt.get('files_sha256') == inventory
            and receipt.get('process_sha256') == sha(directory/'process.json'), 'Predecessor Git receipt differs')
    require(inspect_shard(root, directory, job) == {'status': 'completed', 'details': []}
            and process.get('status') == 'completed_pending_remote_preservation'
            and type(process.get('exit_code')) is int and process['exit_code'] == 0
            and process.get('timed_out') is False and not process.get('interrupted'), 'Previous shard not cleanly terminal')
    return receipt


def predecessor(root, order):
    output = root/'runs'/ID
    require(type(order) is int and order in ORDERS, 'Only original orders92..100')
    existing = []
    if output.exists():
        reject_links(output)
        for path in output.iterdir():
            require(path.name == 'supervisor.lock' or path.name in {f'order{n:03d}' for n in ORDERS}, 'Unexpected output or orphan shard')
            if path.name.startswith('order'):
                require(path.is_dir() and not path.is_symlink(), 'Invalid shard directory')
                existing.append(int(path.name[5:]))
    require(sorted(existing) == list(range(92, order)), 'Consumed/orphan/missing shard; no retry')
    old_protocol = load(root/OLD_PROTOCOL)
    receipt = verify_preserved(root, root/PREVIOUS, old_protocol['jobs'][90-83])
    for previous in range(92, order):
        receipt = verify_preserved(root, output/f'order{previous:03d}', job_for(root, previous)[0])
    return receipt


def intent_for(root, protocol_path, job, previous_commit):
    return {'schema': ID+'-intent', 'experiment_id': ID, 'status': 'reserved_intent_no_execution_claim',
            'source_order': job['source_order'], 'record_id': job['record_id'], 'identity': job['identity'],
            'source_snapshot_commit': SOURCE, 'protocol_sha256': sha(protocol_path),
            'helper_sha256': sha(root/HELPER), 'guard_request_sha256': digest_bytes(canonical_bytes(replay_request(job))),
            'predecessor_receipt_commit': previous_commit, 'max_new_requests': 1, 'automatic_retries': 0,
            'lost_execution_identity_policy': 'Do not replay this saved intent after loss of execution identity; preserve unknown and use a new recovery decision.'}


def readback_pins(root, protocol_path, intent_path, protocol):
    manifest = load(root/INPUTS)
    result = {item['path']: item['sha256'] for item in manifest['files']}
    for path in [root/INPUTS, root/HELPER, protocol_path, intent_path, *[root/j['prompt_path'] for j in protocol['jobs']]]:
        require(path.is_relative_to(root), 'Readback file outside root')
        result[str(path.relative_to(root))] = sha(path)
    return result


def validate_activation(root, protocol_path, intent_path, receipt_path, order):
    protocol = verify_protocol(root, protocol_path)
    require(protocol['execution_enabled'] is True, 'Source-only protocol is not an activation')
    job = job_for(root, order)[0]
    decision = check_replay(root, job)
    previous = predecessor(root, order)
    intent = load(intent_path)
    require(intent == intent_for(root, protocol_path, job, previous['commit']), 'Intent does not bind actual job')
    receipt = load(receipt_path)
    require(set(receipt) == {'schema', 'readback_verified', 'commit', 'source_order', 'identity', 'files_sha256',
                            'remote_output_scope_complete', 'remote_current_output_orders', 'predecessor_receipt_commit'}, 'Activation receipt fields')
    require(receipt['schema'] == ID+'-activation-readback-v1' and receipt['readback_verified'] is True
            and type(receipt['commit']) is str and re.fullmatch('[0-9a-f]{40}', receipt['commit']) is not None
            and type(receipt['source_order']) is int and receipt['source_order'] == order
            and receipt['identity'] == job['identity']
            and receipt['files_sha256'] == readback_pins(root, protocol_path, intent_path, protocol)
            and receipt['remote_output_scope_complete'] is True
            and receipt['remote_current_output_orders'] == list(range(92, order))
            and receipt['predecessor_receipt_commit'] == previous['commit'], 'Exact Git readback or remote reservation scope differs')
    return protocol, job, decision, receipt


def verify_runtime(root):
    first = load(root/FIRST)
    executable = root/first['python_relative']
    require(Path(sys.executable).absolute() == executable.absolute()
            and Path(sys.prefix).resolve() == executable.parent.parent.resolve(), 'Supervisor must use exact frozen venv Python')
    for item in first['model_files']:
        reject_links(root/item['path'])
    return module(root/OLD_HELPER, 'recovery_v1_runtime_validator').verify_environment(root)


def projection(root):
    verify_inputs(root)
    rows = original_rows(root)
    result = copy.deepcopy(load(root/BASE))
    keyed = validate_base(result, rows)
    for order in ORDERS:
        directory = root/'runs'/ID/f'order{order:03d}'
        if directory.exists():
            job = job_for(root, order)[0]
            observed = inspect_shard(root, directory, job)
            row = keyed[('tinyllama11', order)]
            row.update(status=observed['status'], details=observed['details'], source_stratum=ID,
                       evidence_path=str(directory.relative_to(root)))
    result.update(schema=ID+'-projection', base_projection=BASE,
                  counts=dict(Counter(r['status'] for r in result['records'])),
                  blocked_intents_separate_from_outcomes=[{'model_alias':'tinyllama11','source_order':91,
                    'scheduling_status':'blocked_reserved_intent_evidence_gap'}],
                  scope='TinyLlama successor only; other-model changes require the separately bound mixed300 merger',
                  model_calls_by_inspection=0)
    return result


def execute(root, protocol_path, intent_path, receipt_path, order):
    root = Path(root).absolute()
    require(str(root) == ROOT_PIN, 'Wrong execution root')
    verify_inputs(root)
    old = module(root/OLD_HELPER, 'recovery_v1_process_support')
    output = root/'runs'/ID
    reject_links(output)
    reject_links(root/'results/project-nn.lock')
    reject_links(output/'supervisor.lock')
    with old.exclusive(root/'results/project-nn.lock'), old.exclusive(output/'supervisor.lock'):
        protocol, job, decision, receipt = validate_activation(root, protocol_path, intent_path, receipt_path, order)
        first = verify_runtime(root)
        directory = output/f'order{order:03d}'
        directory.mkdir()
        old.write_new(directory/'reservation.json', {'experiment_id':ID,'source_order':order,'record_id':job['record_id'],
                      'identity':job['identity'],'status':'reserved_outcome_unknown','protocol_sha256':sha(protocol_path),
                      'activation_commit':receipt['commit'],'intent_sha256':sha(intent_path),'automatic_retries':0,
                      'guard_request_sha256':decision['request_sha256'],'time_unix':time.time()})
        old.write_new(directory/'activation-receipt.json', receipt)
        write_bytes_new(directory/'guard-request.json', canonical_bytes(replay_request(job)))
        old.write_new(directory/'guard-result.json', decision)
        old.write_new(directory/'bound-intent.json', load(intent_path))
        command = old.command_for(root, first, root/job['prompt_path'], directory/'run')
        environment = os.environ.copy()
        environment.update({k:v.replace('{repository_root}',str(root)) for k,v in first['environment'].items()})
        result = {'schema':ID+'-process','source_order':order,'identity':job['identity'],'command':command,
                  'timeout_seconds':1200,'timed_out':False,'automatic_retries':0,'status':'reserved_outcome_unknown',
                  'started_at_unix':time.time(),'guard_request_sha256':decision['request_sha256']}
        process = None
        started = time.monotonic()
        old_term = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Supervisor interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            with (directory/'stdout.log').open('xb') as stdout, (directory/'stderr.log').open('xb') as stderr:
                process = subprocess.Popen(command,cwd=root,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
                old.write_new(directory/'child.json',{'pid':process.pid,'pgid':process.pid})
                while not old.terminal_observed(process):
                    if time.monotonic()-started >= 1200:
                        result['timed_out'] = True
                        break
                    time.sleep(0.1)
        except BaseException as exc:
            result.update(error_type=type(exc).__name__,interrupted=True)
        finally:
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            old_int = signal.signal(signal.SIGINT,signal.SIG_IGN)
            if process is not None:
                result['exit_code'] = old.stop_and_reap(process)
            result['elapsed_seconds'] = time.monotonic()-started
            try:
                verify_inputs(root)
                observed = inspect_shard(root,directory,job)
                result['observed_outcome'] = observed
                result['status'] = ('completed_pending_remote_preservation'
                    if result.get('exit_code') == 0 and not result['timed_out'] and not result.get('interrupted')
                    and observed == {'status':'completed','details':[]} else 'stopped_requires_review_no_retry')
            except Exception as exc:
                result.update(status='integrity_review_required_no_retry',review_error_type=type(exc).__name__)
            old.write_new(directory/'process.json',result)
            old.write_new(directory/'artifact-manifest.json',{'files_sha256':old.file_inventory(directory)})
            signal.signal(signal.SIGTERM,old_term)
            signal.signal(signal.SIGINT,old_int)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','inspect','execute','intent'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--intent',type=Path)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--source-order',type=int)
    args = parser.parse_args()
    if args.operation == 'prepare':
        result = prepare(args.root)
    elif args.operation == 'inspect':
        result = projection(args.root)
    elif args.operation == 'intent':
        protocol = verify_protocol(args.root,args.protocol)
        require(protocol['execution_enabled'] is True,'Intent requires separately enabled protocol')
        job = job_for(args.root,args.source_order)[0]
        check_replay(args.root,job)
        prior = predecessor(args.root,args.source_order)
        result = intent_for(args.root,args.protocol,job,prior['commit'])
    else:
        result = execute(args.root,args.protocol,args.intent,args.receipt,args.source_order)
    print(json.dumps(result,ensure_ascii=False))
    return 0 if args.operation != 'execute' or result['status'] == 'completed_pending_remote_preservation' else 3


if __name__ == '__main__':
    raise SystemExit(main())
