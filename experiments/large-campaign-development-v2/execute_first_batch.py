#!/usr/bin/env python3
"""Execute the frozen first batch once, retain logs, and never retry it."""
import argparse
import hashlib
import importlib.metadata
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

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'experiments/large-campaign-development-v2/execution-manifest.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def interrupt_on_term(signum, frame):
    raise KeyboardInterrupt('Execution supervisor terminated')


def stop_child(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.freeze_commit):
        raise ValueError('Require the verified Git freeze commit')
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest['frozen_files_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen input changed: ' + name)
    for name, expected in manifest['packages'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Runtime package changed: ' + name)
    if sys.version != manifest['python'] or platform.platform() != manifest['platform']:
        raise ValueError('Runtime platform changed')
    if (ROOT / manifest['runner_output']).exists():
        raise ValueError('First-batch output already exists; execute is not a verification command')
    for item in manifest['model_files']:
        path = ROOT / item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Model asset changed')
    output = ROOT / manifest['process_output']
    output.mkdir(parents=True, exist_ok=False)
    started = {'freeze_commit': args.freeze_commit, 'execution_manifest_sha256': sha(MANIFEST),
               'started_at_unix': time.time(), 'max_new_requests': 2, 'automatic_retries': 0}
    (output / 'started.json').write_text(json.dumps(started, indent=2) + '\n')
    env = os.environ.copy()
    env.update({k: v.replace('{repository_root}', str(ROOT)) for k, v in manifest['environment'].items()})
    command = [str(ROOT / manifest['python_relative']), *manifest['argv']]
    timed_out, process = False, None
    result = {**started, 'command': command, 'status': 'starting'}
    before = time.monotonic()
    previous_term = signal.signal(signal.SIGTERM, interrupt_on_term)
    previous_int = signal.getsignal(signal.SIGINT)
    try:
        with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=manifest['timeout_seconds'])
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_child(process)
                code = process.returncode
        result.update(status='process_finished', exit_code=code, timed_out=timed_out)
    except BaseException as exc:
        result.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'process_control_error',
                      error_type=type(exc).__name__)
        # Allow one interrupted supervisor to clean up its entire child group.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_child(process)
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - before,
                      child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                      finished_at_unix=time.time())
        (output / 'process.json').write_text(json.dumps(result, indent=2) + '\n')
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    print(json.dumps(result))
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    sys.exit(main())
