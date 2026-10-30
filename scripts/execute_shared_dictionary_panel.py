#!/usr/bin/env python3
"""Own and wait for the registered local panel; never open experimental records."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, evidence = args.run.resolve(), args.evidence.resolve()
    config = json.loads((run / 'protocol.json').read_text())
    aliases = [m['alias'] for m in config['models']]
    if aliases != ['qwen05', 'tinyllama11', 'falcon3_1b']:
        raise ValueError('Controller requires the declared three-model panel')
    if (run / 'evaluation_opened.json').exists():
        raise ValueError('Refusing generation after opening')
    evidence.mkdir(parents=True, exist_ok=False)
    status = {'schema': 'shared-panel-owned-execution-v1', 'status': 'running',
              'controller_pid': os.getpid(), 'started_at_unix': time.time(),
              'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'run': str(run), 'schedule': [['qwen05', 'tinyllama11'], ['falcon3_1b']],
              'records_read_by_controller': False, 'automatic_retries': False, 'children': []}
    save(evidence / 'execution.json', status)

    def generate(alias):
        command = [sys.executable, str(root / 'scripts/run_shared_dictionary_pilot.py'),
                   'generate', '--run', str(run), '--model', alias]
        row = {'alias': alias, 'command': command, 'started_at_unix': time.time(),
               'status': 'running'}
        with (evidence / (alias + '.stdout')).open('wb') as stdout, (evidence / (alias + '.stderr')).open('wb') as stderr:
            child = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
            row['pid'] = child.pid
            save(evidence / (alias + '.json'), row)
            code = child.wait()
        row.update(actual_wait_returncode=code, finished_at_unix=time.time(),
                   status='completed' if code == 0 else 'failed')
        for suffix in ('stdout', 'stderr'):
            path = evidence / (alias + '.' + suffix)
            row[suffix + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        save(evidence / (alias + '.json'), row)
        print(json.dumps({'alias': alias, 'actual_wait_returncode': code}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(generate, alias) for alias in aliases[:2]]
        status['children'] = [job.result() for job in jobs]
    save(evidence / 'execution.json', status)
    status['children'].append(generate(aliases[2]))
    failed = any(row['actual_wait_returncode'] != 0 for row in status['children'])
    status.update(status='failed' if failed else 'completed', finished_at_unix=time.time(),
                  all_children_waited=True)
    save(evidence / 'execution.json', status)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
