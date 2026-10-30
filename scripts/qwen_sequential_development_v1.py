#!/usr/bin/env python3
"""Qwen original development orders2..100, one preserved request per invocation.

Only execute launches the unchanged scientific runner. All other operations are
stdlib inspection/preparation. No downloads, tokenizer calls, automatic retries,
Git writes, future source selection or changed model/runtime are implemented.
"""
from __future__ import annotations
import argparse
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

ID = 'development-qwen-sequential-v1'
HELPER = 'scripts/qwen_sequential_development_v1.py'
FIRST_HELPER = 'scripts/other_model_single_record_v1.py'
FIRST_HELPER_SHA = 'cfc5e284381964db30037931ff9a7091f4fc7e28618b1f0da3374e369eeadcdd'
ROOT_PIN = '/workspace/scratch/af7adce5db85/baseline92-continuation-v2/checkout'
OUTPUT = 'runs/'+ID
ORDERS = list(range(2, 101))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reject_links(path):
    require(not any(p.is_symlink() for p in (Path(path), *Path(path).parents)), 'Symlink path refused')


def sha(path):
    reject_links(path)
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def support(root):
    require(sha(root/FIRST_HELPER) == FIRST_HELPER_SHA, 'Frozen first adapter changed')
    spec = importlib.util.spec_from_file_location('qwen_first_frozen_support', root/FIRST_HELPER)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    value.verify_core(root)
    return value


def job_for(root, order):
    require(type(order) is int and order in ORDERS, 'Only original integer orders2..100')
    first = support(root); row = first.verify_core(root)[order-1]
    content = (json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n').encode()
    return {'model_alias': 'qwen05', 'source_order': order, 'record_id': row['record_id'],
            'identity': first.canonical(['local', first.MODEL, row['record_id']]),
            'prompt_path': f'experiments/{ID}/prompts/order{order:03d}.jsonl',
            'prompt_file_sha256': hashlib.sha256(content).hexdigest(),
            'original_prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest()}, content


def preview(root, order, directory):
    first = support(root); job = job_for(root, order)[0]
    require(sha(root/job['prompt_path']) == job['prompt_file_sha256'], 'Current prompt bytes differ')
    runner = first.module(root, first.RUNNER, 'qwen_successor_frozen_scientific_runner')
    code, value = runner.run(prompts=root/job['prompt_path'], output=directory/'run', provider='local',
        model=first.MODEL, revision=first.REVISION, dtype='float32', cpu_threads=6, seed=20261024,
        max_records=1, max_new_records=1, max_output_tokens=1024, split='development', execute=False)
    require(code == 0 and value['plan']['settings'] == first.SETTINGS, 'Scientific preview differs')
    return runner, value['plan']


def inspect_shard(root, directory, order):
    first = support(root); reject_links(directory)
    if order == 1:
        require(directory == root/first.SHARD, 'First shard path changed')
        return first.inspect_shard(root, directory)
    job = job_for(root, order)[0]
    if not directory.exists():
        return {'status': 'not_attempted', 'details': []}
    util = first.module(root, first.UTIL, 'qwen_successor_inventory')
    inventory = util.file_inventory(directory)
    identity = job['identity']; name = 'run/attempts/'+identity+'.json'
    require([p for p in inventory if p.startswith('run/attempts/')] in ([], [name]), 'Orphan/duplicate attempt')
    attempt_path = directory/name; ledger_path = directory/'run/local_ledger/ledger.json'
    if not (attempt_path.exists() and ledger_path.exists()):
        return {'status': 'reserved_outcome_unknown', 'details': ['incomplete_attempt_evidence']}
    attempt, ledger = first.load(attempt_path), first.load(ledger_path)
    require(set(ledger['attempts']) == {identity}, 'Unexpected ledger identity')
    entry = ledger['attempts'][identity]
    require(attempt['identity'] == entry['identity'] == identity
            and attempt['status'] == entry['status'] and attempt.get('result') == entry.get('result')
            and attempt['status'] in ['completed', 'failed', 'reserved_outcome_unknown'], 'Ledger/attempt mismatch')
    if entry['status'] != 'completed':
        return {'status': entry['status'], 'details': []}
    runner, expected = preview(root, order, directory)
    run, runtime = first.load(directory/'run/run.json'), first.load(directory/'run/runtime.json')
    require(first.canonical_bytes(run) == first.canonical_bytes(expected)
            and runtime == first.load(root/(first.PRE+'/runtime.json')), 'Scientific run/runtime differs')
    record = expected['selected_records'][0]
    fingerprint = first.canonical({'record': record, 'provider': 'local', 'model': first.MODEL,
        'revision': first.REVISION, 'dtype': 'float32', 'settings': first.SETTINGS, 'pricing_sha256': None,
        'prompts_sha256': run['prompts_sha256'], 'runtime_sha256': first.canonical(runtime),
        'implementation': {k: run[k] for k in ['runner_sha256', 'ledger_code_sha256', 'transport_code_sha256']}})
    result = entry['result']; usage = result.get('usage', {})
    require(attempt['request_fingerprint'] == entry['request_fingerprint'] == fingerprint
        and attempt['record'] == entry['metadata']['record'] == record
        and record['record_id'] == job['record_id'] and record['prompt_sha256'] == job['original_prompt_sha256']
        and attempt['provider'] == entry['metadata']['provider'] == 'local'
        and attempt['model'] == entry['metadata']['model'] == first.MODEL
        and type(attempt['automatic_retries']) is int and type(entry['automatic_retries']) is int
        and attempt['automatic_retries'] == entry['automatic_retries'] == 0
        and ledger['schema'] == 'synsteg-budget-ledger-recovery-v1' and not ledger.get('blocked_reason')
        and ledger['budget_nanodollars'] == 20000000000 and entry['accounted_nanodollars'] == entry['reserved_nanodollars'] == 0
        and result['returned_model'] == first.MODEL and result['revision'] == first.REVISION and result['dtype'] == 'float32'
        and type(result['seed']) is int and result['seed'] == int(first.canonical([20261024, first.MODEL, job['record_id']])[:8], 16)
        and isinstance(result.get('cover_text'), str) and bool(result['cover_text'].strip())
        and type(usage.get('input_tokens')) is int
        and usage['input_tokens'] == first.load(root/(first.PRE+'/all100-context.json'))['records'][order-1]['input_tokens']
        and type(usage.get('output_tokens')) is int and 1 <= usage['output_tokens'] <= 1024
        and result['hit_output_limit'] is (usage['output_tokens'] == 1024)
        and result['stop_reason'] == ('max_tokens' if usage['output_tokens'] == 1024 else 'end_turn'), 'Completed result differs')
    summary = first.load(directory/'run/summary.json')
    require(summary['status'] == 'selected_records_completed' and summary['completed'] == 1
        and summary['requests_attempted_this_invocation'] == 1 and summary['resumed_completed'] == 0
        and summary['exit_code'] == 0 and summary['selected_records_count'] == 1
        and summary['max_new_records_this_invocation'] == 1 and summary['automatic_retries'] == 0
        and summary['attempts'] == [{'identity': identity, 'record_id': job['record_id'], 'status': 'completed'}]
        and summary['ledger']['attempt_count'] == 1 and summary['ledger']['unknown_or_failed_attempts'] == 0, 'Summary differs')
    return {'status': 'completed', 'details': []}


def verify_preserved(root, directory, order):
    first = support(root); reject_links(directory)
    require(directory.is_dir(), 'Missing predecessor shard')
    util = first.module(root, first.UTIL, 'qwen_successor_receipt_inventory')
    raw = {p: value for p, value in util.file_inventory(directory).items() if p != 'git-preservation-receipt.json'}
    receipt = first.load(directory/'git-preservation-receipt.json')
    require(receipt.get('readback_verified') is True and type(receipt.get('commit')) is str
            and re.fullmatch('[0-9a-f]{40}', receipt['commit']) is not None
            and receipt.get('files_sha256') == raw and receipt.get('process_sha256') == raw.get('process.json')
            and ('source_order' not in receipt or type(receipt['source_order']) is int and receipt['source_order'] == order),
            'Predecessor Git receipt mismatch')
    process = first.load(directory/'process.json')
    require(process.get('status') == 'completed_pending_remote_preservation'
            and type(process.get('exit_code')) is int and process['exit_code'] == 0
            and process.get('timed_out') is False and not process.get('interrupted')
            and type(process.get('source_order')) is int and process['source_order'] == order
            and type(process.get('timeout_seconds')) is int and process['timeout_seconds'] == 1200
            and type(process.get('automatic_retries')) is int and process['automatic_retries'] == 0,
            'Predecessor process is not clean terminal')
    wanted_identity = first.job(root)[0]['identity'] if order == 1 else job_for(root, order)[0]['identity']
    wanted_job = first.job(root)[0] if order == 1 else job_for(root, order)[0]
    experiment = first.ID if order == 1 else ID
    require(process.get('identity') == wanted_identity, 'Predecessor process identity differs')
    required = {'activation-receipt.json', 'artifact-manifest.json', 'child.json', 'process.json',
                'reservation.json', 'stdout.log', 'stderr.log', 'guard-request.json', 'guard-result.json',
                'bound-intent.json', 'run/run.json', 'run/runtime.json', 'run/summary.json',
                'run/local_ledger/ledger.json', 'run/local_ledger/campaign.lock', 'run/run_guard/campaign.lock',
                'run/attempts/'+wanted_identity+'.json'}
    require(required <= set(raw), 'Predecessor raw evidence incomplete')
    reservation = first.load(directory/'reservation.json')
    require(reservation.get('identity') == wanted_identity and type(reservation.get('source_order')) is int
            and reservation['source_order'] == order and reservation.get('status') == 'reserved_outcome_unknown'
            and reservation.get('record_id') == wanted_job['record_id']
            and reservation.get('model_alias') == 'qwen05' and reservation.get('experiment_id') == experiment
            and type(reservation.get('automatic_retries')) is int and reservation['automatic_retries'] == 0,
            'Predecessor reservation identity differs')
    request, decision = first.guard_decision(root) if order == 1 else guard(root, order)
    require((directory/'guard-request.json').read_bytes() == first.canonical_bytes(request)
            and first.load(directory/'guard-result.json') == decision
            and reservation.get('guard_request_sha256') == process.get('guard_request_sha256') == decision['request_sha256'],
            'Predecessor guard evidence differs')
    bound = first.load(directory/'bound-intent.json')
    require(bound.get('schema') == experiment+'-intent' and bound.get('experiment_id') == experiment
            and bound.get('status') == 'reserved_intent_no_execution_claim'
            and bound.get('model_alias') == 'qwen05' and type(bound.get('source_order')) is int
            and bound['source_order'] == order and bound.get('identity') == wanted_identity
            and bound.get('record_id') == wanted_job['record_id']
            and type(bound.get('automatic_retries')) is int and bound['automatic_retries'] == 0
            and type(bound.get('max_new_requests')) is int and bound['max_new_requests'] == 1
            and bound.get('helper_sha256') == (FIRST_HELPER_SHA if order == 1 else sha(root/HELPER))
            and isinstance(bound.get('protocol_sha256'), str) and re.fullmatch('[0-9a-f]{64}', bound['protocol_sha256'])
            and bound['protocol_sha256'] == reservation.get('protocol_sha256')
            and sha(directory/'bound-intent.json') == reservation.get('intent_sha256'),
            'Predecessor bound intent identity/hash differs')
    activation = first.load(directory/'activation-receipt.json')
    require(activation.get('schema') == experiment+'-activation-readback-v1'
            and activation.get('readback_verified') is True and activation.get('identity') == wanted_identity
            and activation.get('remote_output_scope_complete') is True
            and activation.get('remote_branches_and_prs_checked') is True
            and activation.get('remote_qwen_active_intents') == [wanted_identity]
            and type(activation.get('commit')) is str and re.fullmatch('[0-9a-f]{40}', activation['commit'])
            and activation['commit'] == reservation.get('activation_commit')
            and isinstance(activation.get('files_sha256'), dict)
            and reservation['protocol_sha256'] in activation['files_sha256'].values()
            and reservation['intent_sha256'] in activation['files_sha256'].values(),
            'Predecessor activation identity/hash differs')
    for path, expected in dict(first.CORE_PINS, **first.preflight_pins(root)).items():
        require(activation['files_sha256'].get(path) == expected, 'Predecessor activation scientific pin differs')
    require(activation['files_sha256'].get(FIRST_HELPER) == FIRST_HELPER_SHA
            and activation['files_sha256'].get(wanted_job['prompt_path']) == wanted_job['prompt_file_sha256'],
            'Predecessor activation adapter/prompt pin differs')
    if order == 1:
        require(activation.get('remote_qwen_current100_attempts') == [] and bound.get('source_commit') == first.SOURCE,
                'First predecessor initial scope/source differs')
    if order > 1:
        previous_dir = root/first.SHARD if order == 2 else root/OUTPUT/f'order{order-1:03d}'
        previous_commit = first.load(previous_dir/'git-preservation-receipt.json')['commit']
        require(reservation.get('predecessor_receipt_commit') == previous_commit
                and bound.get('predecessor_receipt_commit') == previous_commit
                and bound.get('guard_request_sha256') == decision['request_sha256']
                and activation.get('predecessor_receipt_commit') == previous_commit
                and type(activation.get('source_order')) is int and activation['source_order'] == order
                and activation.get('remote_qwen_attempted_source_orders') == list(range(1, order))
                and all(type(n) is int for n in activation['remote_qwen_attempted_source_orders'])
                and activation['files_sha256'].get(HELPER) == sha(root/HELPER),
                'Predecessor receipt chain differs')
    require(first.load(directory/'artifact-manifest.json').get('files_sha256')
            == {p: v for p, v in raw.items() if p != 'artifact-manifest.json'}, 'Predecessor artifact inventory differs')
    require(inspect_shard(root, directory, order) == {'status': 'completed', 'details': []}, 'Prior result unresolved/failed')
    return receipt


def first_pins(root):
    first = support(root); directory = root/first.SHARD
    verify_preserved(root, directory, 1)
    util = first.module(root, first.UTIL, 'qwen_successor_first_inventory')
    return {first.SHARD+'/'+p: value for p, value in util.file_inventory(directory).items()}


def protocol(root, enabled=False):
    require(type(enabled) is bool, 'Execution flag must be bool')
    first = support(root)
    return {'schema': ID+'-protocol', 'experiment_id': ID, 'execution_enabled': enabled,
            'execution_root': ROOT_PIN, 'first_adapter_sha256': FIRST_HELPER_SHA,
            'helper_sha256': sha(root/HELPER), 'core_pins': first.CORE_PINS,
            'source_commit': first.SOURCE, 'model_alias': 'qwen05', 'model_id': first.MODEL,
            'revision': first.REVISION, 'dtype': 'float32', 'settings': first.SETTINGS,
            'full_denominator': 300, 'model_denominator': 100, 'source_orders': ORDERS,
            'jobs': [job_for(root, order)[0] for order in ORDERS],
            'first_preserved_shard_pins': first_pins(root), 'preflight_pins': first.preflight_pins(root),
            'scientific_execution_stratum': first.ID, 'operational_experiment_id': ID,
            'canonical_nn_lock': 'results/project-nn.lock', 'output': OUTPUT,
            'max_new_requests_per_invocation': 1, 'timeout_seconds_including_model_load': 1200,
            'automatic_retries': 0, 'stop_on_unknown_or_failure': True,
            'require_each_predecessor_remote_preservation': True, 'legacy_imported': 0,
            'paid_calls_allowed': False, 'closed_test_allowed': False}


def verify_protocol(root, path):
    first = support(root); value = first.load(path)
    require(first.canonical_bytes(value) == first.canonical_bytes(protocol(root, value.get('execution_enabled'))), 'Successor protocol changed')
    return value


def local_orders(root):
    directory = root/OUTPUT; reject_links(directory)
    if not directory.exists():
        return []
    result = []
    for path in directory.iterdir():
        reject_links(path)
        if path.name == 'supervisor.lock':
            require(path.is_file(), 'Invalid supervisor lock'); continue
        require(re.fullmatch('order[0-9]{3}', path.name) is not None and path.is_dir(), 'Orphan output')
        number = int(path.name[5:]); require(number in ORDERS, 'Invalid shard order'); result.append(number)
    return sorted(result)


def predecessor(root, order):
    require(type(order) is int and order in ORDERS, 'Only original integer orders2..100')
    first = support(root)
    require(local_orders(root) == list(range(2, order)), 'Consumed/missing/orphan shard; no retry')
    receipt = verify_preserved(root, root/first.SHARD, 1)
    for earlier in range(2, order):
        receipt = verify_preserved(root, root/OUTPUT/f'order{earlier:03d}', earlier)
    return receipt


def guard(root, order):
    first = support(root); job = job_for(root, order)[0]
    request = {'schema': 'synsteg-continuation-replay-request-v1', 'experiment_id': ID, 'model_alias': 'qwen05',
               'source_order': order, 'intent_identity': job['identity'], 'key_sha256': None, 'nonces': []}
    evaluator = first.module(root, first.GUARD, 'qwen_successor_replay_guard')
    result = evaluator.evaluate_bytes((root/first.DENIALS).read_bytes(), first.canonical_bytes(request))
    require(result['status'] == 'not_denied_by_this_receipt' and result['launch_authorized'] is False, 'Replay denied')
    return request, result


def intent(root, protocol_path, order):
    first = support(root); value = verify_protocol(root, protocol_path)
    require(value['execution_enabled'] is True, 'Protocol disabled')
    job = job_for(root, order)[0]; previous = predecessor(root, order)
    require(sha(root/job['prompt_path']) == job['prompt_file_sha256'], 'Current prompt not materialized exactly')
    _, decision = guard(root, order)
    return {'schema': ID+'-intent', 'experiment_id': ID, 'source_order': order, 'model_alias': 'qwen05',
            'identity': job['identity'], 'record_id': job['record_id'], 'status': 'reserved_intent_no_execution_claim',
            'protocol_sha256': sha(protocol_path), 'helper_sha256': sha(root/HELPER),
            'predecessor_receipt_commit': previous['commit'], 'guard_request_sha256': decision['request_sha256'],
            'automatic_retries': 0, 'max_new_requests': 1,
            'lost_execution_identity_policy': 'Do not replay any saved intent after loss of execution identity.'}


def activation_pins(root, protocol_path, intent_path, order, value):
    first = support(root); job = job_for(root, order)[0]
    pins = dict(value['core_pins']) | value['preflight_pins'] | value['first_preserved_shard_pins']
    for path in [root/HELPER, root/FIRST_HELPER, root/job['prompt_path'], protocol_path, intent_path]:
        require(path.is_relative_to(root), 'Activation evidence outside checkout')
        pins[str(path.relative_to(root))] = sha(path)
    util = first.module(root, first.UTIL, 'qwen_successor_activation_inventory')
    for earlier in range(2, order):
        directory = root/OUTPUT/f'order{earlier:03d}'
        # The immediate predecessor gets full raw readback; older immutable receipts bind the chain.
        inventory = util.file_inventory(directory) if earlier == order-1 else {'git-preservation-receipt.json': sha(directory/'git-preservation-receipt.json')}
        for path, digest in inventory.items():
            pins[str(directory.relative_to(root))+'/'+path] = digest
    return pins


def validate_activation(root, protocol_path, intent_path, receipt_path, order):
    first = support(root); value = verify_protocol(root, protocol_path)
    expected_intent = intent(root, protocol_path, order)
    require(first.canonical_bytes(first.load(intent_path)) == first.canonical_bytes(expected_intent), 'Intent differs')
    receipt = first.load(receipt_path)
    fields = {'schema', 'readback_verified', 'commit', 'identity', 'source_order', 'files_sha256',
              'remote_output_scope_complete', 'remote_qwen_attempted_source_orders', 'remote_qwen_active_intents',
              'remote_branches_and_prs_checked', 'predecessor_receipt_commit'}
    require(set(receipt) == fields and receipt['schema'] == ID+'-activation-readback-v1'
            and receipt['readback_verified'] is True and type(receipt['commit']) is str
            and re.fullmatch('[0-9a-f]{40}', receipt['commit']) is not None
            and type(receipt['source_order']) is int and receipt['source_order'] == order
            and receipt['identity'] == expected_intent['identity']
            and receipt['files_sha256'] == activation_pins(root, protocol_path, intent_path, order, value)
            and receipt['remote_output_scope_complete'] is True and receipt['remote_branches_and_prs_checked'] is True
            and type(receipt['remote_qwen_attempted_source_orders']) is list
            and all(type(n) is int for n in receipt['remote_qwen_attempted_source_orders'])
            and receipt['remote_qwen_attempted_source_orders'] == list(range(1, order))
            and receipt['remote_qwen_active_intents'] == [expected_intent['identity']]
            and receipt['predecessor_receipt_commit'] == expected_intent['predecessor_receipt_commit'],
            'Exact remote readback / full Qwen census differs')
    return value, receipt


def execute(root, protocol_path, intent_path, receipt_path, order):
    require(str(root) == ROOT_PIN, 'Wrong execution root')
    first = support(root); util = first.module(root, first.UTIL, 'qwen_successor_process_utils')
    output = root/OUTPUT
    for path in [output, root/'results/project-nn.lock', output/'supervisor.lock']:
        reject_links(path)
    with util.exclusive(root/'results/project-nn.lock'), util.exclusive(output/'supervisor.lock'):
        value, receipt = validate_activation(root, protocol_path, intent_path, receipt_path, order)
        manifest = first.verify_environment(root)
        require(first.resources(root) == first.load(root/(first.PRE+'/resources.json')), 'Resources changed')
        job = job_for(root, order)[0]; directory = output/f'order{order:03d}'
        runner, _ = preview(root, order, directory)
        require(runner.local_runtime() == first.load(root/(first.PRE+'/runtime.json')), 'Runtime changed')
        request, decision = guard(root, order)
        reject_links(directory); directory.mkdir()
        util.write_new(directory/'reservation.json', {'experiment_id': ID, 'model_alias': 'qwen05',
            'scientific_execution_stratum': first.ID, 'source_order': order, 'identity': job['identity'],
            'record_id': job['record_id'], 'status': 'reserved_outcome_unknown', 'automatic_retries': 0,
            'protocol_sha256': sha(protocol_path), 'intent_sha256': sha(intent_path),
            'activation_commit': receipt['commit'], 'predecessor_receipt_commit': receipt['predecessor_receipt_commit'],
            'guard_request_sha256': decision['request_sha256'], 'time_unix': time.time()})
        first.write_bytes(directory/'guard-request.json', first.canonical_bytes(request))
        # Preserve the exact remotely bound intent bytes, including its serialization.
        first.write_bytes(directory/'bound-intent.json', intent_path.read_bytes())
        for name, content in [('activation-receipt.json', receipt), ('guard-result.json', decision)]:
            util.write_new(directory/name, content)
        args = list(manifest['argv'])
        for flag, item in {'--model': first.MODEL, '--revision': first.REVISION,
            '--prompts': str(root/job['prompt_path']), '--output': str(directory/'run'),
            '--max-records': '1', '--max-new-records': '1'}.items():
            require(args.count(flag) == 1, 'Frozen argv layout changed'); args[args.index(flag)+1] = item
        command = [str(root/manifest['python_relative'])]+args
        environment = os.environ.copy(); environment.update(first.offline_environment(root, manifest))
        result = {'schema': ID+'-process', 'experiment_id': ID, 'scientific_execution_stratum': first.ID,
                  'model_alias': 'qwen05', 'source_order': order, 'identity': job['identity'], 'command': command,
                  'timeout_seconds': 1200, 'timed_out': False, 'automatic_retries': 0,
                  'status': 'reserved_outcome_unknown', 'started_at_unix': time.time(),
                  'guard_request_sha256': decision['request_sha256']}
        process = None; started = time.monotonic(); old_term = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Supervisor interrupted')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            with (directory/'stdout.log').open('xb') as stdout, (directory/'stderr.log').open('xb') as stderr:
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
                util.write_new(directory/'child.json', {'pid': process.pid, 'pgid': process.pid})
                while not util.terminal_observed(process):
                    if time.monotonic()-started >= 1200:
                        result['timed_out'] = True; break
                    time.sleep(0.1)
        except BaseException as exc:
            result.update(error_type=type(exc).__name__, interrupted=True)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN); old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            if process is not None:
                result['exit_code'] = util.stop_and_reap(process)
            result['elapsed_seconds'] = time.monotonic()-started
            try:
                support(root)
                observed = inspect_shard(root, directory, order); result['observed_outcome'] = observed
                result['status'] = ('completed_pending_remote_preservation' if result.get('exit_code') == 0
                    and not result['timed_out'] and not result.get('interrupted')
                    and observed == {'status': 'completed', 'details': []} else 'stopped_requires_review_no_retry')
            except Exception as exc:
                result.update(status='integrity_review_required_no_retry', review_error_type=type(exc).__name__)
            util.write_new(directory/'process.json', result)
            util.write_new(directory/'artifact-manifest.json', {'files_sha256': util.file_inventory(directory)})
            signal.signal(signal.SIGTERM, old_term); signal.signal(signal.SIGINT, old_int)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'prepare-order', 'next', 'intent', 'activation-pins', 'inspect', 'execute'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--intent', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--source-order', type=int)
    args = parser.parse_args(); root = args.root.absolute(); first = support(root)
    if args.operation == 'prepare':
        result = protocol(root)
        first.write_json(root/'experiments'/ID/'protocol.proposed.json', result)
    elif args.operation == 'prepare-order':
        verify_protocol(root, args.protocol); predecessor(root, args.source_order)
        result, content = job_for(root, args.source_order); guard(root, args.source_order)
        first.write_bytes(root/result['prompt_path'], content)
    elif args.operation == 'next':
        verify_protocol(root, args.protocol)
        orders = local_orders(root)
        require(orders == list(range(2, 2+len(orders))), 'Noncontiguous local successor sequence')
        next_order = len(orders)+2
        if next_order <= 100:
            predecessor(root, next_order); result = {'status': 'next_original_source', 'source_order': next_order}
        else:
            verify_preserved(root, root/first.SHARD, 1)
            for order in ORDERS:
                verify_preserved(root, root/OUTPUT/f'order{order:03d}', order)
            result = {'status': 'all100_preserved_completed', 'model_denominator': 100, 'full_denominator': 300}
    elif args.operation == 'intent':
        result = intent(root, args.protocol, args.source_order)
    elif args.operation == 'activation-pins':
        value = verify_protocol(root, args.protocol)
        result = activation_pins(root, args.protocol, args.intent, args.source_order, value)
    elif args.operation == 'inspect':
        result = inspect_shard(root, root/OUTPUT/f'order{args.source_order:03d}', args.source_order)
    else:
        result = execute(root, args.protocol, args.intent, args.receipt, args.source_order)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if args.operation != 'execute' or result['status'] == 'completed_pending_remote_preservation' else 3


if __name__ == '__main__':
    raise SystemExit(main())
