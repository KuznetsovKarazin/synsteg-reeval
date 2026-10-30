#!/usr/bin/env python3
"""Replay frozen capacity-only code in a fresh temporary root; preserve originals."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'results/full-source-capacity-control-v1'
FREEZE = '3d97be120346f2e86a763953db91cefc04d68329'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replay(root):
    stage = root / STAGE
    protocol = json.loads((stage / 'protocol.json').read_bytes())
    manifest = {**protocol['source_files'], **protocol['code_files']}
    for name, digest in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path.read_bytes()) != digest:
            raise ValueError('Frozen dependency changed: ' + name)
    with tempfile.TemporaryDirectory(prefix='synsteg-full-source-replay-') as temporary:
        fresh = Path(temporary)
        for name in [*manifest, STAGE + '/protocol.json']:
            target = fresh / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        process = subprocess.run([sys.executable, str(fresh / 'scripts/audit_full_source_capacity_control_v1.py'),
            'execute', '--root', str(fresh), '--freeze-commit', FREEZE],
            cwd=fresh, capture_output=True, text=True, timeout=60)
        if process.returncode:
            raise RuntimeError('Fresh replay failed: ' + process.stderr)
        original = (stage / 'run/records.jsonl').read_bytes()
        rerun = (fresh / STAGE / 'run/records.jsonl').read_bytes()
        if original != rerun:
            raise ValueError('All20 records/60 values differ in fresh replay')
        old_summary = json.loads((stage / 'run/summary.json').read_bytes())
        new_summary = json.loads((fresh / STAGE / 'run/summary.json').read_bytes())
        old_finished = old_summary.pop('finished_at')
        new_finished = new_summary.pop('finished_at')
        if old_summary != new_summary:
            raise ValueError('Summary differs beyond wall-clock timestamp')
    return {'schema': 'full-source-control-fresh-replay-v1', 'freeze_commit': FREEZE,
        'records_sha256': sha(original), 'replay_records_sha256': sha(rerun),
        'all20_records_byte_identical': True, 'all60_capacities_equal': True,
        'summary_equal_except_finished_timestamp': True,
        'original_finished_at': old_finished, 'replay_finished_at': new_finished,
        'frozen_dependencies_copied': len(manifest), 'same_environment': True,
        'new_capacity_measurements': 60, 'encoding_calls': 0, 'decoding_calls': 0,
        'new_model_calls': 0, 'paid_calls': 0, 'original_outputs_overwritten': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / STAGE / 'replay-validation.json'
    if output.exists():
        raise FileExistsError('Choose a new replay evidence path')
    result = replay(root)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
