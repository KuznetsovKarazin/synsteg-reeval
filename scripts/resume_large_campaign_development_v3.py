#!/usr/bin/env python3
"""One two-request TinyLlama continuation after an externally verified Git checkpoint.

This is an operational supervisor, not a new generation variant. The frozen
runner, model, prompts and settings remain unchanged. A local Git receipt records
the caller's connector verification; it is not an independent remote Git proof.
"""
from contextlib import contextmanager
import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time

import checkpoint_large_campaign_development_v3 as audit

ROOT = Path(__file__).resolve().parents[1]
UTILITY = Path('experiments/large-campaign-resume-v3/manifest.json')
BATCHES = Path('results/large-campaign-development-v2/batches')
REPO = 'KuznetsovKarazin/synsteg-independent-reproduction'


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Another process owns the execution/evidence lock') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def evidence_locks(root):
    # Same order as the frozen runner. Released before launching its child;
    # the child repeats the lease check inside the runner's actual ledger lock.
    with exclusive(root / audit.RUN / 'local_ledger/campaign.lock'):
        with exclusive(root / audit.RUN / 'run_guard/campaign.lock'):
            yield


def verify_utility(root):
    obj = audit.load_json(root / UTILITY)
    expected_sources = {'scripts/resume_large_campaign_development_v3.py',
                        'scripts/checkpoint_large_campaign_development_v3.py',
                        'tests/test_large_campaign_resume_v3.py'}
    if (obj.get('schema') != 'large-campaign-resume-utility-v3'
            or obj.get('generation_settings_changed') is not False
            or obj.get('first_execution_manifest_sha256') != audit.FIRST_MANIFEST_SHA
            or set(obj.get('source_sha256', {})) != expected_sources):
        raise ValueError('Unexpected resume utility manifest')
    for name, expected in obj['source_sha256'].items():
        if audit.file_sha(root / name) != expected:
            raise ValueError('Resume utility source changed: ' + name)
    return obj


def verify_frozen_environment(root, *, resources=True):
    frozen = audit.first_manifest(root)
    for name, expected in frozen['frozen_files_sha256'].items():
        if audit.file_sha(root / name) != expected:
            raise ValueError('Original frozen input changed: ' + name)
    if sys.version != frozen['python'] or platform.platform() != frozen['platform']:
        raise ValueError('Python or platform differs from the original execution')
    for name, expected in frozen['packages'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Runtime package changed: ' + name)
    if resources:
        for item in frozen['model_files']:
            path = root / item['path']
            if path.is_symlink() or path.stat().st_size != item['bytes'] or audit.file_sha(path) != item['sha256']:
                raise ValueError('Pinned model/tokenizer resource changed')
    return frozen


def verify_lease(root, receipt, expected_completed):
    if type(expected_completed) is not int or not 2 <= expected_completed <= 98 or expected_completed % 2:
        raise ValueError('Require an even completed prefix from 2 through 98')
    if (receipt.get('schema') != 'large-campaign-verified-git-checkpoint-v3'
            or receipt.get('repository') != REPO
            or receipt.get('readback_verified') is not True
            or not re.fullmatch('[0-9a-f]{40}', receipt.get('commit', ''))
            or receipt.get('expected_completed') != expected_completed):
        raise ValueError('Require the caller verified Git checkpoint receipt')
    before = audit.inspect(root)
    if (before['status'] != 'verified' or before['completed_generations'] != expected_completed
            or before['attempted'] != expected_completed):
        raise ValueError('Stale count, inconsistent evidence or failed/unknown attempt; no retry')
    if receipt.get('run_files_sha256') != before['run_files_sha256']:
        raise ValueError('Local run differs from the verified Git checkpoint')
    return before


def capture(root, target):
    hashes = audit.run_file_hashes(root)
    for relative, expected in hashes.items():
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Run changed during evidence snapshot')
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    write_new(target / 'files.json', hashes)
    return hashes


def compare_old_evidence(before_root, root):
    before = audit.load_json(before_root / audit.RUN / 'local_ledger/ledger.json')['attempts']
    after = audit.load_json(root / audit.RUN / 'local_ledger/ledger.json')['attempts']
    issues = []
    for name in ['run.json', 'runtime.json'] + ['attempts/' + ident + '.json' for ident in before]:
        if audit.file_sha(before_root / audit.RUN / name) != audit.file_sha(root / audit.RUN / name):
            issues.append('Prior evidence bytes changed: ' + name)
    for ident, entry in before.items():
        if after.get(ident) != entry:
            issues.append('Prior ledger entry changed: ' + ident)
    return issues


def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # Also stop grandchildren if their parent exited after TERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def interrupt_on_term(signum, frame):
    raise KeyboardInterrupt('Supervisor terminated')


def install_locked_lease_guard(runner, root, receipt, expected_completed):
    """Recheck state after acquiring the original runner's real ledger lock."""
    original = runner.CampaignLedger

    class GuardedLedger(original):
        @contextmanager
        def locked(self):
            with super().locked():
                if self.directory.resolve() == (root / audit.RUN / 'local_ledger').resolve():
                    verify_lease(root, receipt, expected_completed)
                yield self

    runner.CampaignLedger = GuardedLedger


def child(job_path):
    job = audit.load_json(job_path)
    root = Path(job['repository_root'])
    if audit.file_sha(root / UTILITY) != job['utility_manifest_sha256']:
        raise ValueError('Child utility manifest changed')
    verify_utility(root)
    frozen = verify_frozen_environment(root, resources=False)
    receipt_path = job_path.parent / 'prior-git-receipt.json'
    if audit.file_sha(receipt_path) != job['prior_receipt_sha256']:
        raise ValueError('Child receipt changed')
    receipt = audit.load_json(receipt_path)
    sys.path.insert(0, str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location('frozen_budget_generator', root / 'scripts/run_budget_generation.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    install_locked_lease_guard(runner, root, receipt, job['expected_completed'])
    # All CLI arguments, sampling, generator and per-record seeds are the original ones.
    return runner.main(frozen['argv'][1:])


def execute(root, batch_id, expected_completed, receipt_path, utility_freeze_commit):
    root = Path(root).resolve()
    if not re.fullmatch('[0-9a-f]{40}', utility_freeze_commit):
        raise ValueError('Require the Git freeze SHA of this utility')
    if batch_id != f'batch{expected_completed // 2 + 1:02d}':
        raise ValueError('Batch ID must follow the fixed completed prefix')
    verify_utility(root)
    frozen = verify_frozen_environment(root)
    receipt_bytes = Path(receipt_path).read_bytes()
    receipt = json.loads(receipt_bytes)
    output = root / BATCHES / batch_id
    with exclusive(root / BATCHES / 'resume-supervisor.lock'):
        if output.exists():
            raise ValueError('Batch evidence already exists; do not rerun it')
        with evidence_locks(root):
            before = verify_lease(root, receipt, expected_completed)
            output.mkdir(parents=True, exist_ok=False)
            (output / 'prior-git-receipt.json').write_bytes(receipt_bytes)
            capture(root, output / 'before')
            write_new(output / 'before-checkpoint.json', before)
        job = {'schema': 'large-campaign-resume-job-v3', 'repository_root': str(root),
               'batch_id': batch_id, 'expected_completed': expected_completed,
               'utility_freeze_commit': utility_freeze_commit,
               'utility_manifest_sha256': audit.file_sha(root / UTILITY),
               'prior_git_commit': receipt['commit'],
               'prior_receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
               'max_new_requests': 2, 'automatic_retries': 0,
               'planned_next_record_ids': [row['record_id'] for row in before['records']
                    if row['model_alias'] == 'tinyllama11'][expected_completed:expected_completed + 2]}
        write_new(output / 'job.json', job)
        env = os.environ.copy()
        env.update({key: value.replace('{repository_root}', str(root)) for key, value in frozen['environment'].items()})
        command = [str(root / frozen['python_relative']), str(Path(__file__).resolve()),
                   '_child', str(output / 'job.json')]
        process = None
        started = time.monotonic()
        result = {**job, 'status': 'starting', 'started_at_unix': time.time(),
                  'command': command, 'timed_out': False}
        write_new(output / 'started.json', result)
        old_term = signal.signal(signal.SIGTERM, interrupt_on_term)
        old_int = signal.getsignal(signal.SIGINT)
        interrupted = None
        try:
            with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                try:
                    result['exit_code'] = process.wait(timeout=frozen['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    result['timed_out'] = True
                    stop_group(process)
                    result['exit_code'] = process.returncode
            result['status'] = 'process_finished'
        except BaseException as exc:
            interrupted = exc
            result.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'process_control_error',
                          error_type=type(exc).__name__)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            stop_group(process)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            result.update(elapsed_seconds=time.monotonic() - started, finished_at_unix=time.time(),
                          child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            try:
                with evidence_locks(root):
                    capture(root, output / 'after')
                    after = audit.inspect(root)
                    write_new(output / 'after-checkpoint.json', after)
                    preserved = compare_old_evidence(output / 'before', root)
                    delta = after['attempted'] - before['attempted']
                    result.update(prior_evidence_issues=preserved, new_attempts=delta,
                                  completed_total=after['completed_generations'],
                                  after_status=after['status'], full_denominator=after['planned_model_record_pairs'])
                    if delta < 0 or delta > 2 or preserved:
                        result['status'] = 'postrun_integrity_failure'
                    elif (result.get('exit_code') == 0 and not result['timed_out']
                          and after['status'] == 'verified' and delta == 2
                          and after['completed_generations'] == expected_completed + 2):
                        result['status'] = 'batch_completed_pending_remote_checkpoint'
                    else:
                        result['status'] = 'stopped_requires_review_no_retry'
            except Exception as exc:
                result.update(status='postrun_audit_error', audit_error_type=type(exc).__name__,
                              audit_error=str(exc))
            write_new(output / 'process.json', result)
            artifacts = {str(path.relative_to(root)): audit.file_sha(path)
                         for path in sorted(output.rglob('*')) if path.is_file()}
            write_new(output / 'artifact-manifest.json', {'schema': 'immutable-batch-artifacts-v3',
                                                         'files_sha256': artifacts})
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        if interrupted is not None:
            raise interrupted
        return result


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '_child':
        return child(Path(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--batch-id', required=True)
    parser.add_argument('--expected-completed', type=int, required=True)
    parser.add_argument('--verified-git-receipt', type=Path, required=True)
    parser.add_argument('--utility-freeze-commit', required=True)
    args = parser.parse_args()
    result = execute(args.root, args.batch_id, args.expected_completed,
                     args.verified_git_receipt, args.utility_freeze_commit)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'batch_completed_pending_remote_checkpoint' else 3


if __name__ == '__main__':
    raise SystemExit(main())
