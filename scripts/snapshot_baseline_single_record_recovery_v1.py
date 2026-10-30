#!/usr/bin/env python3
"""Snapshot a clean Git-preserved TinyLlama recovery prefix; no model imports.

The frozen inspect function supplies the unchanged full300 projection. This
wrapper adds token/stop totals and provenance, writes only a fresh destination,
and requires all new prefix shards to be cleanly completed and Git receipted.
A failure/unknown in a new shard is refused here, never omitted: preserve it
and use the frozen inspect directly for an explicit failure checkpoint.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat

ID = 'baseline-single-record-recovery-v1'
HELPER = 'scripts/baseline_single_record_recovery_v1.py'
HELPER_SHA = '12e7356d990242f7f09eaea88d045b16019f734956025ec86cfb5005bdf96294'
PROTOCOL_SHA = '8f484dbdd25f91c214ef04c26804a85d38883bbfce6b823b5d7e15db490c0bf9'
PRIOR_RUN = 'runs/large-campaign-development-v2/models/tinyllama11'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


@contextmanager
def existing_lease(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Symlink lease refused')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('Regular existing canonical lease required')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def frozen_helper(root):
    path = root / HELPER
    if path.is_symlink() or sha(path) != HELPER_SHA:
        raise ValueError('Frozen helper identity differs')
    spec = importlib.util.spec_from_file_location('_frozen_recovery_snapshot_input', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary_from_inputs(projection, prior_attempts, new_attempts, orders):
    counts = dict(Counter(row['status'] for row in projection['records']))
    if (projection['schema'] != ID + '-projection' or len(projection['records']) != 300
            or projection['full_denominator'] != 300 or projection['tinyllama_denominator'] != 100
            or counts != projection['counts']):
        raise ValueError('Full300 projection/count mismatch')
    prior_values = list(prior_attempts.values())
    unknown = [e for e in prior_values if e['status'] != 'completed']
    old_completed = [e for e in prior_values if e['status'] == 'completed']
    if (len(prior_values) != 82 or len(old_completed) != 81 or len(unknown) != 1
            or unknown[0]['status'] != 'reserved_outcome_unknown' or unknown[0].get('result') is not None):
        raise ValueError('Historical81+unknown82 must remain unchanged')
    if (not orders or orders != list(range(83, orders[-1] + 1)) or orders[-1] > 100
            or len(new_attempts) != len(orders) or any(a['status'] != 'completed' for a in new_attempts)):
        raise ValueError('Require complete clean prefix; preserve new failures separately, never drop them')
    expected = {'completed': 81 + len(new_attempts), 'reserved_outcome_unknown': 1,
                'not_attempted': 218 - len(new_attempts)}
    if counts != expected:
        raise ValueError('Full denominator or another model arm changed')
    tiny = [r for r in projection['records'] if r['model_alias'] == 'tinyllama11']
    if len(tiny) != 100 or tiny[81]['source_order'] != 82 or tiny[81]['status'] != 'reserved_outcome_unknown':
        raise ValueError('Unknown82 row not retained')
    tokens, stops = 0, Counter()
    for entry in old_completed + new_attempts:
        result = entry['result']; value = result['usage']['output_tokens']
        if type(value) is not int or value < 1 or not isinstance(result['stop_reason'], str):
            raise ValueError('Completed output token/stop evidence invalid')
        tokens += value; stops[result['stop_reason']] += 1
    return {'counts': projection['counts'], 'completed_output_tokens': tokens,
            'stop_reasons': dict(sorted(stops.items())), 'old_unknown82_retained': True,
            'new_source_orders': orders, 'new_model_requests': len(new_attempts)}


def write_snapshot(destination, files):
    destination = Path(destination)
    encoded = {name: (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
               for name, value in files.items()}
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in encoded.items():
        with (destination / name).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {name: hashlib.sha256(raw).hexdigest() for name, raw in encoded.items()}


def snapshot(root, prior_root, protocol_path, through, destination):
    root, prior_root, protocol_path = map(lambda p: Path(p).resolve(), (root, prior_root, protocol_path))
    destination = Path(destination)
    if type(through) is not int or not 83 <= through <= 100:
        raise ValueError('Source order must be83..100')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Fresh snapshot destination required')
    if sha(protocol_path) != PROTOCOL_SHA:
        raise ValueError('Frozen enabled protocol differs')
    module = frozen_helper(root)
    output = root / 'runs' / ID
    if destination.resolve().is_relative_to(output) or destination.resolve().is_relative_to(prior_root):
        raise ValueError('Snapshot cannot alter source evidence trees')
    with existing_lease(root / 'results/project-nn.lock'):
        protocol = module.verify_protocol(root, prior_root, protocol_path)
        expected_dirs = {f'order{n:03d}' for n in range(83, through + 1)}
        observed_dirs = {p.name for p in output.glob('order*')}
        if observed_dirs != expected_dirs:
            raise ValueError('Observed order namespace differs; do not omit later reservations')
        model_base = root / 'runs/large-campaign-development-v2/models'
        for name in ('qwen05', 'falcon3_1b'):
            other = model_base / name
            if other.exists() and any(other.iterdir()):
                raise ValueError('Frozen inspect assumes other model arms unattempted')
        receipts, inventories, attempts = [], {}, []
        for order in range(83, through + 1):
            directory = output / f'order{order:03d}'
            receipt_path = directory / 'git-preservation-receipt.json'
            receipt, process = load(receipt_path), load(directory / 'process.json')
            inventory = module.file_inventory(directory)
            files = {k: v for k, v in inventory.items() if k != 'git-preservation-receipt.json'}
            if (process.get('status') != 'completed_pending_remote_preservation'
                    or process.get('exit_code') != 0 or process.get('timed_out') or process.get('interrupted')
                    or receipt.get('source_order') != order or receipt.get('readback_verified') is not True
                    or receipt.get('files_sha256') != files
                    or receipt.get('process_sha256') != sha(directory / 'process.json')
                    or re.fullmatch('[0-9a-f]{40}', receipt.get('commit', '')) is None):
                raise ValueError('Terminal clean full-inventory Git receipt required for every prefix shard')
            job = protocol['jobs'][order - 83]
            attempts.append(load(directory / 'run/attempts' / (job['identity'] + '.json')))
            inventories[str(directory)] = inventory
            receipts.append({'source_order': order, 'primary_commit': receipt['commit'],
                             'receipt_sha256': sha(receipt_path), 'full_files': len(files)})
        projection = module.inspect(root, prior_root, protocol_path, output)
        ledger_path = prior_root / PRIOR_RUN / 'local_ledger/ledger.json'
        summary = summary_from_inputs(projection, load(ledger_path)['attempts'], attempts, list(range(83, through + 1)))
        for directory, inventory in inventories.items():
            if module.file_inventory(directory) != inventory:
                raise ValueError('Input changed while reading snapshot')
        provenance = {'schema': ID + '-snapshot-provenance-v1', 'through_source_order': through,
                      'frozen_helper_sha256': HELPER_SHA, 'protocol_sha256': PROTOCOL_SHA,
                      'prior_ledger_sha256': sha(ledger_path), 'shards': receipts,
                      'full_input_inventories_sha256': inventories,
                      'cooperative_canonical_lease_held': True, 'model_calls': 0, 'tokenizer_calls': 0,
                      'scientific_shard_audit_reinvoked': False,
                      'scope': 'Clean terminal Git-receipted Tiny prefix only; unchanged frozen inspect; other model arms must remain unattempted.',
                      'receipt_scope': 'Caller has performed actual Git readback; this helper validates exact local receipt/inventory binding and does not access Git.'}
        files = {f'full300-after{through:03d}.json': projection,
                 f'summary-after{through:03d}.json': summary,
                 f'snapshot-provenance-after{through:03d}.json': provenance}
        return {'status': 'written', 'destination': str(destination), 'files_sha256': write_snapshot(destination, files),
                'counts': summary['counts'], 'model_calls': 0, 'tokenizer_calls': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prior-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--through-order', type=int, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(snapshot(a.root, a.prior_root, a.protocol, a.through_order, a.destination), sort_keys=True))
