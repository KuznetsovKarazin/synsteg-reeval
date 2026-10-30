#!/usr/bin/env python3
"""Run only the frozen free/offline Qwen long-summary pilot, once."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import prepare_long_summary_development_v3 as prep

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-commit', required=True)
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.frozen_commit):
        raise ValueError('An exact pre-generation Git commit is required')
    out = ROOT / prep.OUTPUT
    prep.verify(ROOT, out)
    readiness = prep.read(out / 'execution-readiness.json')
    for name, expected in readiness['file_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Readiness input changed: ' + name)
    check = prep.read(out / 'tokenizer_preflight.json')
    if check['status'] != 'passed' or len(check['records']) != 15:
        raise ValueError('All 15 preflight checks are required')
    command = [sys.executable, 'scripts/run_budget_generation.py', '--provider', 'local',
        '--model', prep.MODEL, '--revision', prep.REVISION, '--prompts', prep.OUTPUT + '/prompts.jsonl',
        '--output', prep.OUTPUT + '/run', '--split', 'development', '--max-records', '15',
        '--max-new-records', '15', '--max-output-tokens', '1536', '--cpu-threads', '6',
        '--dtype', 'float32', '--seed', '20261024', '--execute']
    if not args.execute:
        print(json.dumps({'status': 'verified_no_model_calls', 'command': command}))
        return 0
    if (out / 'run').exists() or (out / 'execution.json').exists():
        raise ValueError('Existing execution: no automatic retry or replacement')
    env = dict(os.environ)
    env.update(HF_HOME=str(args.hf_home.resolve()), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false')
    now = lambda: datetime.now(timezone.utc).isoformat()
    report = {'schema': 'synsteg-long-summary-v3-execution', 'status': 'running',
              'frozen_commit': args.frozen_commit, 'command': command,
              'planned_sources': 20, 'planned_generations': 15, 'source_abstentions': 5,
              'new_paid_calls': 0, 'automatic_retries': 0, 'started_at_utc': now(),
              'actual_returncode': None}
    path = out / 'execution.json'
    with path.open('x') as stream:
        json.dump(report, stream, indent=2)
    start = time.monotonic()
    with (out / 'generation.stdout').open('xb') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    report.update(status='completed' if result.returncode == 0 else 'failed_or_incomplete_no_retry',
                  actual_returncode=result.returncode, elapsed_seconds=time.monotonic()-start,
                  finished_at_utc=now())
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps(report))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
