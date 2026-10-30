#!/usr/bin/env python3
"""Gated Qwen0.5 source1 adapter, same scientific runner, no download code.

Only explicit tokenize-preflight imports a tokenizer. Only execute launches the
frozen neural runner. Default operations are prepare/verify/intent; every actual
first request requires separately persisted enabled protocol, intent and exact
external Git readback. This version cannot select another model/source/order.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ID = 'development-other-models-single-record-v1'
MERGER = 'mixed-development-full300-merger-v2'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
ROOT_PIN = '/workspace/scratch/af7adce5db85/baseline92-continuation-v2/checkout'
SOURCE = '4050291345c43776e9c93528b6afe292c062ba7a'
HELPER = 'scripts/other_model_single_record_v1.py'
UTIL = 'scripts/baseline_single_record_recovery_v1.py'
RUNNER = 'scripts/run_budget_generation.py'
GUARD = 'scripts/continuation_replay_guard_v1.py'
DENIALS = 'results/current-git-status-audit-20261028-v1/no-replay-guard.json'
FIRST = 'experiments/large-campaign-development-v2/execution-manifest.json'
PROMPTS = 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
RESOURCE = 'experiments/' + ID + '/public-resource-manifest.json'
PRE = 'results/' + ID + '/qwen05-preflight'
PROMPT = 'experiments/' + ID + '/qwen05-source001.jsonl'
SHARD = 'runs/' + ID + '/qwen05/order001'
CORE_PINS = {
    FIRST: '7b23df64b875799cc592640a9aadf7ea6bcd870b525335141169cf93a6003ff1',
    PROMPTS: '3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1',
    RESOURCE: '741711fb21acd08db2a81fbcdb857f17bd1f0bdb53f819a9dd859a1bf220513f',
    RUNNER: '4d5ab74c78c58214c1b5e11a346b4339eb0c6d08e713e8909d0d5fd9947c7e02',
    UTIL: '12e7356d990242f7f09eaea88d045b16019f734956025ec86cfb5005bdf96294',
    GUARD: 'fae8a6f191582dcf0d64a36ce6784cfe38495b3e3da38946723ea0fbebf8584c',
    DENIALS: 'ed7c17d68df5cc524ec8e441f3febb2a96e6d8108eda6f22f2ac154fae74db0c',
    'src/synsteg_audit/budget_ledger.py': 'c04ca4e3141a9b9873054ab71c38545705353c8609c12ec0536eaabcd2e0b19b',
    'scripts/run_claude_smoke.py': '98411ed6af23a4fce6693c978e44617e3a6f6904cc5d7cd8547da2eb73bd6655',
    'scripts/check_claude_access.py': '1161cc279f126f2305fafc18cdd451d9a84eb1c9e925c768ecdec90bda6c0843',
}
# Explicitly retain the original scientific settings, including null API fields.
SETTINGS = {'api_version': None, 'cpu_threads': 6, 'max_output_tokens': 1024,
            'request_options': None, 'seed': 20261024,
            'system': 'Write clear English prose. Return only the requested text, without a heading or commentary.',
            'temperature': 0.8, 'top_k': 50, 'top_p': 0.95}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reject_links(path):
    require(not any(p.is_symlink() for p in (Path(path), *Path(path).parents)), 'Symlink path refused')


def sha(path):
    reject_links(path)
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'))+'\n').encode()


def load(path):
    reject_links(path)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def bad(_):
        raise ValueError('Nonfinite JSON')
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=bad)


def module(root, path, name):
    sys.path.insert(0, str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location(name, root / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write_bytes(path, content):
    reject_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, value):
    # Match frozen atomic_json serialization so runtime SHA binds actual output.
    write_bytes(path, (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + '\n').encode())


def verify_core(root):
    for path, expected in CORE_PINS.items():
        require(sha(root/path) == expected, 'Pinned source differs: ' + path)
    rows = [json.loads(line) for line in (root/PROMPTS).read_text().splitlines() if line.strip()]
    require(len(rows) == 100 and len({r['record_id'] for r in rows}) == 100
            and len({r['source_group_id'] for r in rows}) == 100
            and all(r['split'] == 'development' for r in rows), 'Require original development100')
    require(rows[0]['record_id'] == 'wikipedia-panel-recovery-v1-astronomy-popular_science-500-development-01'
            and hashlib.sha256(rows[0]['prompt'].encode()).hexdigest() == 'bcb0e1acea165b65cccc9e80077bceeb0037fcb46a525a1b8857d6879a0685e6', 'First source changed')
    return rows


def job(root):
    row = verify_core(root)[0]
    payload = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    return {'model_alias': 'qwen05', 'source_order': 1, 'record_id': row['record_id'],
            'identity': canonical(['local', MODEL, row['record_id']]), 'prompt_path': PROMPT,
            'prompt_file_sha256': hashlib.sha256(payload).hexdigest(),
            'original_prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest()}, payload


def verify_environment(root):
    require(str(root) == ROOT_PIN, 'Use the fixed shared execution checkout')
    first = load(root/FIRST)
    executable = root/first['python_relative']
    require(Path(sys.executable).absolute() == executable.absolute()
            and Path(sys.prefix).resolve() == executable.parent.parent.resolve(), 'Use exact frozen venv')
    require(sys.version == first['python'] and platform.platform() == first['platform'], 'Frozen Python/platform differs')
    for name, version in first['packages'].items():
        require(importlib.metadata.version(name) == version, 'Frozen package differs: ' + name)
    return first


def resources(root):
    manifest = load(root/RESOURCE)
    require(manifest['model_id'] == MODEL and manifest['revision'] == REVISION, 'Wrong resource identity')
    directory = root/manifest['expected_snapshot_relative']
    files = []
    for item in manifest['model_files']:
        path = directory/item['name']
        reject_links(path)
        require(path.is_file() and path.stat().st_size == item['bytes'], 'Resource missing/size differs: ' + item['name'])
        digest = sha(path)
        if item['known_file_sha256']:
            require(digest == item['known_file_sha256'], 'Resource SHA256 differs: ' + item['name'])
        if not item['upstream_lfs_sha256']:
            blob = hashlib.sha1(b'blob ' + str(item['bytes']).encode() + b'\0')
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    blob.update(block)
            require(blob.hexdigest() == item['upstream_git_blob_sha1'], 'Upstream Git blob differs: ' + item['name'])
        files.append({'path': str(path.relative_to(root)), 'bytes': item['bytes'], 'sha256': digest})
    require(load(directory/'config.json')['max_position_embeddings'] == 32768, 'Config context changed')
    return {'schema': ID+'-verified-resources', 'model_id': MODEL, 'revision': REVISION,
            'manifest_sha256': CORE_PINS[RESOURCE], 'files': files, 'actual_downloads_by_this_helper': 0}


def offline_environment(root, first):
    value = {k: v.replace('{repository_root}', str(root)) for k, v in first['environment'].items()}
    require(value['HF_HUB_OFFLINE'] == '1' and value['TRANSFORMERS_OFFLINE'] == '1'
            and value['HF_HUB_DISABLE_IMPLICIT_TOKEN'] == '1', 'Offline policy changed')
    return value


def tokenize_preflight(root):
    # This operation must be run only by the root execution owner, not preparation agents.
    rows = verify_core(root)
    first = verify_environment(root)
    util = module(root, UTIL, 'other_model_process_utils')
    directory = root/PRE
    reject_links(directory)
    reject_links(root/'results/project-nn.lock')
    with util.exclusive(root/'results/project-nn.lock'):
        require(not directory.exists(), 'Preflight already consumed; inspect it, never overwrite')
        resource_receipt = resources(root)
        write_json(directory/'started.json', {'schema': ID+'-preflight-start', 'time_unix': time.time(),
                   'model_id': MODEL, 'revision': REVISION, 'automatic_retries': 0,
                   'tokenizer_loads_planned': 1, 'tokenizer_encodes_planned': 100, 'model_loads_planned': 0})
        write_json(directory/'resources.json', resource_receipt)
        os.environ.update(offline_environment(root, first))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False)
        records = []
        for order, row in enumerate(rows, 1):
            rendered = tokenizer.apply_chat_template([{'role': 'system', 'content': SETTINGS['system']},
                        {'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True)
            count = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
            records.append({'source_order': order, 'record_id': row['record_id'],
                            'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(), 'input_tokens': count})
        runner = module(root, RUNNER, 'other_model_scientific_preview')
        runtime = runner.local_runtime()
        write_json(directory/'runtime.json', runtime)
        config_source = root/load(root/RESOURCE)['expected_snapshot_relative']/'config.json'
        write_bytes(directory/'config.json', config_source.read_bytes())
        context = {'schema': MERGER+'-context-report', 'model_id': MODEL, 'revision': REVISION,
                   'source_sha256': CORE_PINS[PROMPTS], 'model_config_sha256': sha(config_source),
                   'model_context_limit': 32768, 'records': records}
        write_json(directory/'all100-context.json', context)
        require(all(0 < r['input_tokens'] <= 32768-1024 for r in records), 'Context overflow; retain full100 evidence')
        require(resources(root) == resource_receipt, 'Resource changed during preflight')
        write_json(directory/'model-contract.json', expected_contract(root))
        return {'status': 'tokenizer_preflight_completed_no_model_load', 'records': 100,
                'model_calls': 0, 'tokenizer_loads': 1, 'tokenizer_encodes': 100,
                'max_input_tokens': max(r['input_tokens'] for r in records)}


def expected_contract(root):
    return {'schema': MERGER+'-model-contract', 'model_alias': 'qwen05', 'model_id': MODEL,
            'revision': REVISION, 'dtype': 'float32', 'settings': SETTINGS,
            'runtime_path': PRE+'/runtime.json', 'runtime_sha256': sha(root/(PRE+'/runtime.json')),
            'context_report_path': PRE+'/all100-context.json', 'model_config_path': PRE+'/config.json',
            'execution_stratum': ID, 'timeout_seconds_including_load': 1200}


def preflight_pins(root):
    paths = [PRE+'/'+n for n in ['started.json', 'resources.json', 'runtime.json', 'config.json', 'all100-context.json', 'model-contract.json']]
    pins = {path: sha(root/path) for path in paths}
    require(pins[PRE+'/config.json'] == load(root/RESOURCE)['metadata_config_sha256'], 'Pinned model config differs')
    require(load(root/(PRE+'/model-contract.json')) == expected_contract(root), 'Merger contract differs')
    rows = verify_core(root)
    context = load(root/(PRE+'/all100-context.json'))
    require(context['schema'] == MERGER+'-context-report' and context['model_id'] == MODEL
            and context['revision'] == REVISION and context['source_sha256'] == CORE_PINS[PROMPTS]
            and context['model_config_sha256'] == pins[PRE+'/config.json']
            and context['model_context_limit'] == 32768 and len(context['records']) == 100, 'Context identity differs')
    for i, row in enumerate(context['records']):
        require(set(row) == {'source_order', 'record_id', 'prompt_sha256', 'input_tokens'}
                and type(row['source_order']) is int and row['source_order'] == i+1 and row['record_id'] == rows[i]['record_id']
                and row['prompt_sha256'] == hashlib.sha256(rows[i]['prompt'].encode()).hexdigest()
                and type(row['input_tokens']) is int and 0 < row['input_tokens'] <= 32768-1024, 'Context source/count differs')
    return pins


def protocol(root, enabled=False):
    require(type(enabled) is bool, 'Execution flag must be bool')
    return {'schema': ID+'-protocol', 'experiment_id': ID, 'execution_enabled': enabled,
            'source_commit': SOURCE, 'execution_root': ROOT_PIN, 'model_alias': 'qwen05',
            'model_id': MODEL, 'revision': REVISION, 'dtype': 'float32', 'settings': SETTINGS,
            'full_denominator': 300, 'per_model_denominator': 100, 'source_orders': [1],
            'max_new_requests': 1, 'automatic_retries': 0, 'timeout_seconds_including_model_load': 1200,
            'output': SHARD, 'canonical_nn_lock': 'results/project-nn.lock',
            'helper_sha256': sha(root/HELPER), 'core_pins': CORE_PINS,
            'preflight_pins': preflight_pins(root), 'job': job(root)[0],
            'execution_stratum': ID, 'historical_outputs_imported': 0,
            'paid_calls_allowed': False, 'closed_test_allowed': False,
            'stop_after_request_pending_git_and_independent_audit': True}


def verify_protocol(root, path):
    value = load(path)
    require(value == protocol(root, value.get('execution_enabled')), 'Protocol differs from fixed design')
    require(sha(root/PROMPT) == value['job']['prompt_file_sha256'], 'Single prompt changed')
    return value


def guard_decision(root):
    guard = module(root, GUARD, 'other_model_replay_guard')
    request = {'schema': 'synsteg-continuation-replay-request-v1', 'experiment_id': ID, 'model_alias': 'qwen05',
               'source_order': 1, 'intent_identity': job(root)[0]['identity'], 'key_sha256': None, 'nonces': []}
    value = guard.evaluate_bytes((root/DENIALS).read_bytes(), canonical_bytes(request))
    require(value['status'] == 'not_denied_by_this_receipt' and value['launch_authorized'] is False, 'Replay denied')
    return request, value


def intent(root, protocol_path):
    value = verify_protocol(root, protocol_path)
    require(value['execution_enabled'] is True, 'Disabled protocol cannot create execution intent')
    guard_decision(root)
    require(not (root/SHARD).exists(), 'Consumed first request; never retry')
    output = root/'runs'/ID
    reject_links(output)
    if output.exists():
        require(set(p.name for p in output.iterdir()) <= {'supervisor.lock'}, 'Unexpected local model/shard reservation')
        for path in output.iterdir():
            reject_links(path)
            require(path.is_file(), 'Unexpected supervisor artifact')
    return {'schema': ID+'-intent', 'experiment_id': ID, 'model_alias': 'qwen05', 'source_order': 1,
            'record_id': value['job']['record_id'], 'identity': value['job']['identity'],
            'status': 'reserved_intent_no_execution_claim', 'source_commit': SOURCE,
            'protocol_sha256': sha(protocol_path), 'helper_sha256': sha(root/HELPER),
            'max_new_requests': 1, 'automatic_retries': 0,
            'lost_execution_identity_policy': 'Preserve unknown; do not replay this saved intent after scratch loss.'}


def activation_pins(root, protocol_path, intent_path, value):
    pins = dict(CORE_PINS) | value['preflight_pins']
    for path in [root/HELPER, root/PROMPT, protocol_path, intent_path]:
        require(path.is_relative_to(root), 'Activation evidence outside checkout')
        pins[str(path.relative_to(root))] = sha(path)
    return pins


def validate_activation(root, protocol_path, intent_path, receipt_path):
    value = verify_protocol(root, protocol_path)
    require(value['execution_enabled'] is True and load(intent_path) == intent(root, protocol_path), 'Intent differs')
    receipt = load(receipt_path)
    fields = {'schema', 'readback_verified', 'commit', 'identity', 'files_sha256', 'remote_output_scope_complete',
              'remote_qwen_current100_attempts', 'remote_qwen_active_intents', 'remote_branches_and_prs_checked'}
    require(set(receipt) == fields and receipt['schema'] == ID+'-activation-readback-v1'
            and receipt['readback_verified'] is True and type(receipt['commit']) is str
            and re.fullmatch('[0-9a-f]{40}', receipt['commit']) is not None
            and receipt['identity'] == value['job']['identity']
            and receipt['files_sha256'] == activation_pins(root, protocol_path, intent_path, value)
            and receipt['remote_output_scope_complete'] is True and receipt['remote_branches_and_prs_checked'] is True
            and receipt['remote_qwen_current100_attempts'] == []
            and receipt['remote_qwen_active_intents'] == [value['job']['identity']], 'Exact Git readback/census missing')
    return value, receipt


def preview(root, output):
    runner = module(root, RUNNER, 'other_model_scientific_preview')
    code, result = runner.run(prompts=root/PROMPT, output=output, provider='local', model=MODEL,
        revision=REVISION, max_records=1, max_new_records=1, max_output_tokens=1024,
        dtype='float32', cpu_threads=6, seed=20261024, split='development', execute=False)
    require(code == 0 and result['plan']['settings'] == SETTINGS, 'Scientific preview changed')
    return runner, result['plan']


def inspect_shard(root, directory):
    reject_links(directory)
    if not directory.exists():
        return {'status': 'not_attempted', 'details': []}
    util = module(root, UTIL, 'other_model_inventory')
    util.file_inventory(directory)
    runner, expected = preview(root, directory/'run')
    identity = job(root)[0]['identity']
    attempt_path = directory/'run/attempts'/(identity+'.json')
    ledger_path = directory/'run/local_ledger/ledger.json'
    if not (attempt_path.exists() and ledger_path.exists()):
        return {'status': 'reserved_outcome_unknown', 'details': ['incomplete_attempt_evidence']}
    attempt, ledger = load(attempt_path), load(ledger_path)
    require(set(ledger['attempts']) == {identity}, 'Unexpected ledger identity')
    entry = ledger['attempts'][identity]
    require(attempt['identity'] == entry['identity'] == identity and attempt['status'] == entry['status']
            and attempt.get('result') == entry.get('result'), 'Attempt/ledger differs')
    if entry['status'] != 'completed':
        return {'status': entry['status'], 'details': []}
    run, runtime = load(directory/'run/run.json'), load(directory/'run/runtime.json')
    require(run == expected and runtime == load(root/(PRE+'/runtime.json')), 'Scientific run/runtime differs')
    record = expected['selected_records'][0]
    fingerprint = canonical({'record': record, 'provider': 'local', 'model': MODEL, 'revision': REVISION,
        'dtype': 'float32', 'settings': SETTINGS, 'pricing_sha256': None, 'prompts_sha256': run['prompts_sha256'],
        'runtime_sha256': canonical(runtime), 'implementation': {k: run[k] for k in
        ['runner_sha256', 'ledger_code_sha256', 'transport_code_sha256']}})
    result = entry['result']; usage = result.get('usage', {})
    require(attempt['request_fingerprint'] == entry['request_fingerprint'] == fingerprint
        and attempt['record'] == entry['metadata']['record'] == record
        and attempt['provider'] == entry['metadata']['provider'] == 'local'
        and attempt['model'] == entry['metadata']['model'] == MODEL
        and attempt['automatic_retries'] == entry['automatic_retries'] == 0
        and ledger['schema'] == 'synsteg-budget-ledger-recovery-v1' and not ledger.get('blocked_reason')
        and ledger['budget_nanodollars'] == 20000000000 and entry['accounted_nanodollars'] == entry['reserved_nanodollars'] == 0
        and result['returned_model'] == MODEL and result['revision'] == REVISION and result['dtype'] == 'float32'
        and result['seed'] == int(canonical([20261024, MODEL, record['record_id']])[:8], 16)
        and isinstance(result.get('cover_text'), str) and bool(result['cover_text'].strip())
        and type(usage.get('input_tokens')) is int
        and usage['input_tokens'] == load(root/(PRE+'/all100-context.json'))['records'][0]['input_tokens']
        and type(usage.get('output_tokens')) is int and 1 <= usage['output_tokens'] <= 1024
        and result['hit_output_limit'] is (usage['output_tokens'] == 1024)
        and result['stop_reason'] == ('max_tokens' if usage['output_tokens'] == 1024 else 'end_turn'), 'Completed identity/result differs')
    summary = load(directory/'run/summary.json')
    require(summary['status'] == 'selected_records_completed' and summary['completed'] == 1
        and summary['requests_attempted_this_invocation'] == 1 and summary['resumed_completed'] == 0
        and summary['exit_code'] == 0 and summary['selected_records_count'] == 1
        and summary['max_new_records_this_invocation'] == 1 and summary['automatic_retries'] == 0
        and summary['attempts'] == [{'identity': identity, 'record_id': record['record_id'], 'status': 'completed'}]
        and summary['ledger']['attempt_count'] == 1 and summary['ledger']['unknown_or_failed_attempts'] == 0, 'Summary differs')
    return {'status': 'completed', 'details': []}


def execute(root, protocol_path, intent_path, receipt_path):
    require(str(root) == ROOT_PIN, 'Wrong execution root')
    verify_core(root)
    util = module(root, UTIL, 'other_model_process_utils')
    output = root/'runs'/ID
    for path in [output, root/'results/project-nn.lock', output/'supervisor.lock']:
        reject_links(path)
    with util.exclusive(root/'results/project-nn.lock'), util.exclusive(output/'supervisor.lock'):
        value, receipt = validate_activation(root, protocol_path, intent_path, receipt_path)
        first = verify_environment(root)
        require(resources(root) == load(root/(PRE+'/resources.json')), 'Resources differ from preflight')
        runner, _ = preview(root, root/SHARD/'run')
        require(runner.local_runtime() == load(root/(PRE+'/runtime.json')), 'Current runtime differs')
        request, decision = guard_decision(root)
        directory = root/SHARD
        reject_links(directory)
        directory.mkdir(parents=True)
        util.write_new(directory/'reservation.json', {'experiment_id': ID, 'model_alias': 'qwen05', 'source_order': 1,
            'record_id': value['job']['record_id'], 'identity': value['job']['identity'], 'status': 'reserved_outcome_unknown',
            'protocol_sha256': sha(protocol_path), 'activation_commit': receipt['commit'], 'intent_sha256': sha(intent_path),
            'automatic_retries': 0, 'guard_request_sha256': decision['request_sha256'], 'time_unix': time.time()})
        write_bytes(directory/'guard-request.json', canonical_bytes(request))
        for name, content in [('activation-receipt.json', receipt),
                              ('guard-result.json', decision), ('bound-intent.json', load(intent_path))]:
            util.write_new(directory/name, content)
        args = list(first['argv'])
        for flag, item in {'--model': MODEL, '--revision': REVISION, '--prompts': str(root/PROMPT),
                          '--output': str(directory/'run'), '--max-records': '1', '--max-new-records': '1'}.items():
            require(args.count(flag) == 1, 'Frozen argv layout changed')
            args[args.index(flag)+1] = item
        command = [str(root/first['python_relative'])] + args
        environment = os.environ.copy(); environment.update(offline_environment(root, first))
        result = {'schema': ID+'-process', 'model_alias': 'qwen05', 'source_order': 1, 'identity': value['job']['identity'],
                  'command': command, 'timeout_seconds': 1200, 'timed_out': False, 'automatic_retries': 0,
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
                        result['timed_out'] = True
                        break
                    time.sleep(0.1)
        except BaseException as exc:
            result.update(error_type=type(exc).__name__, interrupted=True)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN); old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            if process is not None:
                result['exit_code'] = util.stop_and_reap(process)
            result['elapsed_seconds'] = time.monotonic()-started
            try:
                verify_core(root)
                observed = inspect_shard(root, directory); result['observed_outcome'] = observed
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
    parser.add_argument('operation', choices=['verify-resources', 'tokenize-preflight', 'prepare', 'intent', 'activation-pins', 'execute', 'inspect'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--intent', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args(); root = args.root.absolute(); verify_core(root)
    if args.operation == 'verify-resources':
        result = resources(root)
    elif args.operation == 'tokenize-preflight':
        result = tokenize_preflight(root)
    elif args.operation == 'prepare':
        value = protocol(root)
        write_bytes(root/PROMPT, job(root)[1])
        target = root/'experiments'/ID/'qwen05-protocol.proposed.json'
        write_json(target, value); result = value
    elif args.operation == 'intent':
        result = intent(root, args.protocol)
    elif args.operation == 'activation-pins':
        value = verify_protocol(root, args.protocol)
        result = activation_pins(root, args.protocol, args.intent, value)
    elif args.operation == 'inspect':
        result = inspect_shard(root, root/SHARD)
    else:
        result = execute(root, args.protocol, args.intent, args.receipt)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if args.operation != 'execute' or result['status'] == 'completed_pending_remote_preservation' else 3


if __name__ == '__main__':
    raise SystemExit(main())
