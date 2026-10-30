#!/usr/bin/env python3
"""Additive, one-attempt recovery for never-attempted TinyLlama orders 83..100.

The frozen scientific runner is executed unmodified, with exact original rows
in separate one-record input files. The historical unknown remains immutable.
No model is imported by this supervisor. Root owns activation and Git receipts.
"""
from contextlib import contextmanager
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ID = 'baseline-single-record-recovery-v1'
MAIN = '5205e2a0e37611d6dd6326cebd45c2c78f0ce90b'
PRIMARY = '13d714588d3b0434d18791348158ff6859087464'
FIRST = 'experiments/large-campaign-development-v2/execution-manifest.json'
FIRST_SHA = '7b23df64b875799cc592640a9aadf7ea6bcd870b525335141169cf93a6003ff1'
RUN = 'runs/large-campaign-development-v2/models/tinyllama11'
PROMPTS = 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
PROMPTS_SHA = '3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1'
MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
REVISION = 'fe8a4ea1ffedaf415f4da2f062534de366a451e6'
HELPER = 'scripts/baseline_single_record_recovery_v1.py'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':')).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_inventory(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink evidence refused')
        if path.is_file():
            result[str(path.relative_to(root))] = sha(path)
    return result


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def source_rows(root):
    if sha(root / PROMPTS) != PROMPTS_SHA:
        raise ValueError('Original development prompts changed')
    rows = [json.loads(line) for line in (root / PROMPTS).read_text().splitlines() if line.strip()]
    if (len(rows) != 100 or any(row['split'] != 'development' for row in rows)
            or len({row['record_id'] for row in rows}) != 100):
        raise ValueError('Require full unique development 100')
    return rows


def verify_prior(prior_root, rows, expected_inventory=None):
    directory = prior_root / RUN
    inventory = file_inventory(directory)
    if expected_inventory is not None and inventory != expected_inventory:
        raise ValueError('Historical evidence changed')
    ledger = load(directory / 'local_ledger/ledger.json')
    run = load(directory / 'run.json')
    if run['model'] != MODEL or run['revision'] != REVISION or run['selected_records_count'] != 100:
        raise ValueError('Historical run identity changed')
    entries = ledger['attempts']
    expected_ids = [canonical(['local', MODEL, row['record_id']]) for row in rows]
    if set(entries) != set(expected_ids[:82]):
        raise ValueError('Prior attempted set must be exactly orders 1..82')
    statuses = []
    for order, (row, ident) in enumerate(zip(rows[:82], expected_ids), 1):
        entry = entries[ident]
        attempt = load(directory / 'attempts' / (ident + '.json'))
        wanted = 'completed' if order <= 81 else 'reserved_outcome_unknown'
        if (entry['status'] != wanted or attempt['status'] != wanted
                or entry['metadata']['record']['record_id'] != row['record_id']
                or attempt['record']['record_id'] != row['record_id']
                or entry.get('automatic_retries') != 0 or attempt.get('automatic_retries') != 0
                or entry.get('result') != attempt.get('result')):
            raise ValueError('Historical status or attempt mismatch')
        statuses.append(wanted)
    return inventory, statuses


def prepare(root, prior_root, target):
    root, prior_root, target = map(lambda p: Path(p).resolve(), (root, prior_root, target))
    if target.exists():
        raise ValueError('Preparation destination already exists')
    if sha(root / FIRST) != FIRST_SHA:
        raise ValueError('Frozen first manifest changed')
    rows = source_rows(root)
    inventory, _ = verify_prior(prior_root, rows)
    first = load(root / FIRST)
    target.mkdir(parents=True)
    jobs = []
    for order in range(83, 101):
        path = Path('prompts') / f'order{order:03d}.jsonl'
        # A JSONL row must be one line; serialize without changing row values.
        value = (json.dumps(rows[order - 1], ensure_ascii=False, separators=(',', ':')) + '\n').encode()
        (target / path).parent.mkdir(parents=True, exist_ok=True)
        with (target / path).open('xb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        jobs.append({'source_order': order, 'record_id': rows[order - 1]['record_id'],
                     'identity': canonical(['local', MODEL, rows[order - 1]['record_id']]),
                     'prompt_path': str(path), 'prompt_file_sha256': sha(target / path),
                     'original_prompt_sha256': hashlib.sha256(rows[order - 1]['prompt'].encode()).hexdigest()})
    protocol = {'schema': ID + '-protocol', 'experiment_id': ID, 'execution_enabled': False,
                'source_main': MAIN, 'prior_primary_commit': PRIMARY,
                'execution_root': str(root), 'output_relative': 'runs/' + ID,
                'canonical_nn_lock_relative': 'results/project-nn.lock',
                'full_denominator': 300, 'tinyllama_denominator': 100,
                'new_source_orders': list(range(83, 101)), 'permanently_unknown_source_orders': [82],
                'new_run_shards': 18, 'max_new_requests_per_shard': 1, 'automatic_retries': 0,
                'timeout_seconds_per_shard_including_model_load': 1200,
                'continue_after_new_failure': False,
                'stop_after_each_shard_pending_remote_preservation': True,
                'first_manifest_sha256': FIRST_SHA, 'original_prompts_sha256': PROMPTS_SHA,
                'scientific_runner': 'scripts/run_budget_generation.py',
                'scientific_runner_sha256': first['frozen_files_sha256']['scripts/run_budget_generation.py'],
                'helper_sha256': sha(Path(__file__)),
                'prior_run_files_sha256': inventory, 'jobs': jobs,
                'allowed_argv_changes': ['--prompts', '--output', '--max-records', '--max-new-records'],
                'execution_stratum': 'one record per fresh process; unchanged per-record seed formula and scientific settings; no bitwise NN-equivalence claim',
                'limit_comparability': '1200s per single request replaces prior1200s per pair; completion rates must be stratified',
                'old_summary_repaired': False, 'paid_calls_allowed': False, 'closed_test_allowed': False}
    write_new(target / 'protocol.proposed.json', protocol)
    return protocol


def verify_protocol(root, prior_root, protocol_path):
    protocol_path = Path(protocol_path)
    protocol = load(protocol_path)
    if (protocol.get('schema') != ID + '-protocol' or protocol.get('experiment_id') != ID
            or protocol.get('new_source_orders') != list(range(83, 101))
            or protocol.get('permanently_unknown_source_orders') != [82]
            or protocol.get('timeout_seconds_per_shard_including_model_load') != 1200
            or protocol.get('automatic_retries') != 0
            or protocol.get('continue_after_new_failure') is not False
            or protocol.get('stop_after_each_shard_pending_remote_preservation') is not True
            or protocol.get('max_new_requests_per_shard') != 1
            or protocol.get('new_run_shards') != 18
            or protocol.get('source_main') != MAIN or protocol.get('prior_primary_commit') != PRIMARY
            or protocol.get('first_manifest_sha256') != FIRST_SHA
            or protocol.get('original_prompts_sha256') != PROMPTS_SHA
            or protocol.get('scientific_runner') != 'scripts/run_budget_generation.py'
            or protocol.get('execution_root') != str(root.resolve())
            or protocol.get('output_relative') != 'runs/' + ID
            or protocol.get('canonical_nn_lock_relative') != 'results/project-nn.lock'
            or protocol.get('paid_calls_allowed') is not False or protocol.get('closed_test_allowed') is not False
            or protocol.get('full_denominator') != 300 or protocol.get('tinyllama_denominator') != 100
            or protocol.get('helper_sha256') != sha(Path(__file__))
            or sha(root / FIRST) != FIRST_SHA):
        raise ValueError('Recovery protocol changed')
    rows = source_rows(root)
    verify_prior(prior_root, rows, protocol['prior_run_files_sha256'])
    if len(protocol.get('jobs', [])) != 18:
        raise ValueError('Need all18 jobs')
    for order, job in zip(range(83, 101), protocol['jobs']):
        expected_path = f'prompts/order{order:03d}.jsonl'
        path = protocol_path.parent / expected_path
        parsed = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        row = rows[order - 1]
        if (job['source_order'] != order or job['prompt_path'] != expected_path
                or job['record_id'] != row['record_id'] or parsed != [row]
                or job['identity'] != canonical(['local', MODEL, row['record_id']])
                or job['prompt_file_sha256'] != sha(path)):
            raise ValueError('Shard differs from original fixed row')
    return protocol


def verify_environment(root):
    first = load(root / FIRST)
    if sys.version != first['python'] or platform.platform() != first['platform']:
        raise ValueError('Runtime Python/platform changed')
    for name, expected in first['packages'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Runtime package changed: ' + name)
    for name, expected in first['frozen_files_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError('Frozen source/input changed: ' + name)
    for item in first['model_files']:
        path = root / item['path']
        if path.is_symlink() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Frozen model resource changed')
    return first


def command_for(root, first, prompt_path, output):
    args = list(first['argv'])
    replacements = {'--prompts': str(prompt_path), '--output': str(output),
                    '--max-records': '1', '--max-new-records': '1'}
    for flag, value in replacements.items():
        if args.count(flag) != 1:
            raise ValueError('Frozen argument layout changed')
        args[args.index(flag) + 1] = value
    return [str(root / first['python_relative'])] + args


def terminal_observed(process):
    # Do not reap before the final group signal: avoids PID/PGID reuse.
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def stop_and_reap(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not terminal_observed(process):
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait()


def shard_status(directory, job, original_run=None, original_runtime=None):
    if not directory.exists():
        return {'status': 'not_attempted', 'details': []}
    # Reservation/shard directory is consumed even if child did not launch.
    status = 'reserved_outcome_unknown'
    details = []
    ledger_path = directory / 'run/local_ledger/ledger.json'
    attempt_path = directory / 'run/attempts' / (job['identity'] + '.json')
    if ledger_path.exists() and attempt_path.exists():
        ledger = load(ledger_path)
        entries = ledger['attempts']
        attempt = load(attempt_path)
        if set(entries) != {job['identity']}:
            raise ValueError('Unexpected shard identities')
        entry = entries[job['identity']]
        if (entry['status'] != attempt['status'] or entry.get('result') != attempt.get('result')
                or entry['metadata']['record']['record_id'] != job['record_id']
                or attempt['record']['record_id'] != job['record_id']):
            raise ValueError('Shard ledger/attempt mismatch')
        status = entry['status']
        if status == 'completed':
            if original_run is None or original_runtime is None:
                raise ValueError('Completed shard requires original scientific definition')
            run = load(directory / 'run/run.json')
            runtime = load(directory / 'run/runtime.json')
            expected_run = copy.deepcopy(original_run)
            record = original_run['selected_records'][job['source_order'] - 1]
            expected_run.update(prompts_sha256=job['prompt_file_sha256'],
                                selected_records=[record], selected_records_count=1)
            if run != expected_run or runtime != original_runtime:
                raise ValueError('Shard scientific run/runtime differs')
            fingerprint = canonical({'record': record, 'provider': 'local', 'model': MODEL,
                'revision': REVISION, 'dtype': run['dtype'], 'settings': run['settings'],
                'pricing_sha256': run['pricing_sha256'], 'prompts_sha256': run['prompts_sha256'],
                'runtime_sha256': canonical(runtime),
                'implementation': {key: run[key] for key in
                    ['runner_sha256', 'ledger_code_sha256', 'transport_code_sha256']}})
            result = entry['result']
            usage = result.get('usage', {})
            if (entry.get('identity') != job['identity'] or attempt.get('identity') != job['identity']
                    or ledger.get('schema') != 'synsteg-budget-ledger-recovery-v1'
                    or ledger.get('budget_nanodollars') != 20000000000 or ledger.get('blocked_reason')
                    or entry.get('request_fingerprint') != fingerprint or attempt.get('request_fingerprint') != fingerprint
                    or entry.get('automatic_retries') != 0 or attempt.get('automatic_retries') != 0
                    or entry.get('accounted_nanodollars') != 0 or entry.get('reserved_nanodollars') != 0
                    or entry['metadata'].get('provider') != 'local' or entry['metadata'].get('model') != MODEL
                    or attempt.get('provider') != 'local' or attempt.get('model') != MODEL
                    or entry['metadata']['record'] != record or attempt['record'] != record
                    or result.get('returned_model') != MODEL or result.get('revision') != REVISION
                    or result.get('dtype') != run['dtype']
                    or result.get('seed') != int(canonical([run['settings']['seed'], MODEL, job['record_id']])[:8], 16)
                    or not isinstance(result.get('cover_text'), str) or not result['cover_text'].strip()
                    or type(usage.get('input_tokens')) is not int or usage['input_tokens'] <= 0
                    or type(usage.get('output_tokens')) is not int or not 1 <= usage['output_tokens'] <= 1024
                    or result.get('stop_reason') != ('max_tokens' if usage['output_tokens'] == 1024 else 'end_turn')
                    or result.get('hit_output_limit') is not (usage['output_tokens'] == 1024)):
                raise ValueError('Completed shard identity/fingerprint/result mismatch')
            summary = load(directory / 'run/summary.json')
            if (summary.get('completed') != 1 or summary.get('status') != 'selected_records_completed'
                    or summary.get('requests_attempted_this_invocation') != 1 or summary.get('resumed_completed') != 0
                    or summary.get('exit_code') != 0 or summary.get('selected_records_count') != 1
                    or summary.get('max_new_records_this_invocation') != 1 or summary.get('automatic_retries') != 0
                    or summary.get('attempts') != [{'identity': job['identity'], 'record_id': job['record_id'], 'status': 'completed'}]
                    or summary.get('ledger', {}).get('attempt_count') != 1
                    or summary.get('ledger', {}).get('unknown_or_failed_attempts') != 0):
                details.append('completed_attempt_but_summary_inconsistent')
    elif ledger_path.exists() or attempt_path.exists():
        details.append('partial_ledger_or_attempt_evidence')
    return {'status': status, 'details': details}


def inspect(root, prior_root, protocol_path, output):
    protocol = verify_protocol(Path(root), Path(prior_root), protocol_path)
    rows = source_rows(Path(root))
    records = []
    for model in ['qwen05', 'tinyllama11', 'falcon3_1b']:
        for order, row in enumerate(rows, 1):
            status, details = 'not_attempted', []
            if model == 'tinyllama11':
                if order <= 81:
                    status = 'completed'
                elif order == 82:
                    status = 'reserved_outcome_unknown'
                else:
                    result = shard_status(Path(output) / f'order{order:03d}', protocol['jobs'][order - 83],
                        load(Path(prior_root) / RUN / 'run.json'), load(Path(prior_root) / RUN / 'runtime.json'))
                    status, details = result['status'], result['details']
            records.append({'model_alias': model, 'source_order': order,
                            'record_id': row['record_id'], 'status': status, 'details': details,
                            'source_stratum': ('historical_batch41' if order <= 82 else ID)
                                if model == 'tinyllama11' else 'unattempted_other_model',
                            'quality': None, 'transport': None})
    counts = {status: sum(row['status'] == status for row in records) for status in sorted({r['status'] for r in records})}
    return {'schema': ID + '-projection', 'records': records, 'counts': counts,
            'full_denominator': 300, 'tinyllama_denominator': 100,
            'historical_summary_repaired': False, 'successful_scientific_audit_claimed': False,
            'model_calls_by_inspection': 0}


def execute(root, prior_root, protocol_path, receipt_path, output, order, nn_lock):
    root, prior_root, protocol_path, receipt_path, output = map(lambda p: Path(p).resolve(),
                                                               (root, prior_root, protocol_path, receipt_path, output))
    if order not in range(83, 101):
        raise ValueError('Only never-attempted original orders83..100')
    if output.is_relative_to(prior_root) or output.is_relative_to(root / RUN):
        raise ValueError('Recovery output must not modify historical evidence')
    if output != root / 'runs' / ID or Path(nn_lock).resolve() != root / 'results/project-nn.lock':
        raise ValueError('Require frozen output namespace and canonical project NN lock')
    with exclusive(nn_lock), exclusive(output / 'supervisor.lock'):
        protocol = verify_protocol(root, prior_root, protocol_path)
        if protocol.get('execution_enabled') is not True:
            raise ValueError('Source-only protocol; root must freeze activation')
        receipt = load(receipt_path)
        if (receipt.get('readback_verified') is not True or receipt.get('source_order') != order
                or receipt.get('protocol_sha256') != sha(protocol_path)
                or receipt.get('helper_sha256') != sha(Path(__file__))
                or not re.fullmatch('[0-9a-f]{40}', receipt.get('commit', ''))):
            raise ValueError('Need fresh exact Git activation receipt')
        if order > 83:
            previous = output / f'order{order-1:03d}'
            prior_receipt = load(previous / 'git-preservation-receipt.json')
            old_job = protocol['jobs'][order - 84]
            old_process = load(previous / 'process.json')
            saved_inventory = {k: v for k, v in file_inventory(previous).items() if k != 'git-preservation-receipt.json'}
            if (shard_status(previous, old_job, load(prior_root / RUN / 'run.json'), load(prior_root / RUN / 'runtime.json')) != {'status': 'completed', 'details': []}
                    or old_process.get('status') != 'completed_pending_remote_preservation'
                    or prior_receipt.get('readback_verified') is not True
                    or prior_receipt.get('process_sha256') != sha(previous / 'process.json')
                    or prior_receipt.get('files_sha256') != saved_inventory
                    or not re.fullmatch('[0-9a-f]{40}', prior_receipt.get('commit', ''))):
                raise ValueError('Previous shard not cleanly completed and Git preserved')
        directory = output / f'order{order:03d}'
        if directory.exists() or directory.is_symlink():
            raise ValueError('One-shot shard already consumed; no retry')
        first = verify_environment(root)
        job = protocol['jobs'][order - 83]
        directory.mkdir(parents=True)
        write_new(directory / 'reservation.json', {'experiment_id': ID, 'source_order': order,
                  'record_id': job['record_id'], 'status': 'reserved_outcome_unknown',
                  'protocol_sha256': sha(protocol_path), 'activation_commit': receipt['commit'],
                  'automatic_retries': 0, 'time_unix': time.time()})
        write_new(directory / 'activation-receipt.json', receipt)
        command = command_for(root, first, protocol_path.parent / job['prompt_path'], directory / 'run')
        environment = os.environ.copy()
        environment.update({k: v.replace('{repository_root}', str(root)) for k, v in first['environment'].items()})
        result = {'schema': ID + '-process', 'source_order': order, 'command': command,
                  'timeout_seconds': 1200, 'timed_out': False, 'automatic_retries': 0,
                  'status': 'reserved_outcome_unknown', 'started_at_unix': time.time()}
        process = None
        started = time.monotonic()
        old_term = signal.getsignal(signal.SIGTERM)
        def interrupt(signum, frame):
            raise KeyboardInterrupt('Supervisor interrupted')
        signal.signal(signal.SIGTERM, interrupt)
        try:
            with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
                write_new(directory / 'child.json', {'pid': process.pid, 'pgid': process.pid})
                while not terminal_observed(process):
                    if time.monotonic() - started >= 1200:
                        result['timed_out'] = True
                        break
                    time.sleep(0.1)
        except BaseException as exc:
            result.update(error_type=type(exc).__name__, interrupted=True)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            if process is not None:
                result['exit_code'] = stop_and_reap(process)
            result['elapsed_seconds'] = time.monotonic() - started
            try:
                verify_prior(prior_root, source_rows(root), protocol['prior_run_files_sha256'])
                observed = shard_status(directory, job, load(prior_root / RUN / 'run.json'), load(prior_root / RUN / 'runtime.json'))
                result['observed_outcome'] = observed
                result['status'] = ('completed_pending_remote_preservation'
                    if result.get('exit_code') == 0 and not result['timed_out'] and not result.get('interrupted')
                    and observed == {'status': 'completed', 'details': []} else 'stopped_requires_review_no_retry')
            except Exception as exc:
                result.update(status='integrity_review_required_no_retry', review_error_type=type(exc).__name__)
            write_new(directory / 'process.json', result)
            write_new(directory / 'artifact-manifest.json', {'files_sha256': file_inventory(directory)})
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    for operation in ('prepare', 'inspect', 'execute'):
        child = sub.add_parser(operation)
        child.add_argument('--root', type=Path, required=True)
        child.add_argument('--prior-root', type=Path, required=True)
        child.add_argument('--output', type=Path, required=True)
        if operation != 'prepare':
            child.add_argument('--protocol', type=Path, required=True)
        if operation == 'execute':
            child.add_argument('--receipt', type=Path, required=True)
            child.add_argument('--source-order', type=int, required=True)
            child.add_argument('--nn-lock', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'prepare':
        value = prepare(args.root, args.prior_root, args.output)
    elif args.operation == 'inspect':
        value = inspect(args.root, args.prior_root, args.protocol, args.output)
    else:
        value = execute(args.root, args.prior_root, args.protocol, args.receipt, args.output, args.source_order, args.nn_lock)
    print(json.dumps(value, ensure_ascii=False))
    return 0 if args.operation != 'execute' or value['status'] == 'completed_pending_remote_preservation' else 3


if __name__ == '__main__':
    raise SystemExit(main())
