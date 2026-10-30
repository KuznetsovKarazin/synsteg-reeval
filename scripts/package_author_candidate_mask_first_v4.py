#!/usr/bin/env python3
"""Preserve v4 raw results with unchanged, hash-pinned generic storage code.

No model/tokenizer/scientific-auditor imports. Archives every regular byte and
empty directory under this isolated checkout's results, including root process,
source readback, seals and reservations. A packaging receipt is not acceptance.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

STORAGE_SHA = '7a791b2685220f58c6b637b3642bf6e46c5f5d08269592e31739914d1193ffa6'
SCHEMA = 'author-candidate-mask-first-v4-raw-results-artifacts-v1'
BASE = 'results'
RAW = 'results/author-candidate-mask-first-v4/native-diagnostic-v4'
POST = 'results/author-candidate-mask-first-v4-postrun'
OUT = 'artifacts/author-candidate-mask-first-v4/native-diagnostic-v4'


def storage():
    path = Path(__file__).with_name('package_author_candidate_local_native_v3.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != STORAGE_SHA:
        raise ValueError('frozen generic storage helper differs')
    spec = importlib.util.spec_from_file_location('_v4_frozen_storage', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only the artifact identifier differs. The unchanged byte/archive/restore
    # algorithms run against explicit v4 paths; no native-v3 output is opened.
    module.SCHEMA = SCHEMA
    return module


def package(root, canonical_lease):
    s = storage(); root = Path(root).resolve(strict=True)
    allowed = {'author-candidate-mask-first-v4',
               'author-candidate-mask-first-v4-postrun', 'project-nn.lock'}
    if {p.name for p in (root / BASE).iterdir()} - allowed:
        raise ValueError('isolated v4 results tree required')
    required = [root / RAW / 'initial-denominator.json',
                root / POST / 'source-readback.json',
                root / POST / 'root-process/process.json',
                root / POST / 'root-process/stdout.log',
                root / POST / 'root-process/stderr.log']
    for path in required:
        s.regular_bytes(path)
    terminal = s.read_json(root / POST / 'root-process/process.json')
    if type(terminal.get('exit_code')) is not int:
        raise ValueError('root process exit must be known before packaging')
    lease_path = Path(canonical_lease)
    s.no_symlink_ancestors(lease_path)
    fd = os.open(lease_path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('canonical lease must be regular')
        if os.path.samefile(lease_path, root / 'results/project-nn.lock'):
            raise ValueError('outer canonical lease and inner checkout lease must be distinct here')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # BASE is the entire isolated results tree, not the nested campaign.
        # Thus no BASE/result.json is expected; this option is structural and
        # does not reinterpret the nested campaign outcome as incomplete.
        result = s.package(root, root / OUT, base=BASE, allow_incomplete=True)
        receipt = {'schema': 'author-candidate-mask-first-v4-packaging-receipt-v1',
            'storage_helper_sha256': STORAGE_SHA,
            'source_base': BASE, 'nested_campaign_result': RAW + '/result.json',
            'nested_campaign_result_present': (root / RAW / 'result.json').is_file(),
            'allow_incomplete_reason': 'results parent has no result.json; all known/unknown raw outcomes preserved without interpretation',
            'root_process_terminal': terminal, 'canonical_lease': str(lease_path),
            'canonical_lease_held_during_snapshot': True,
            'distinct_checkout_lease_held_by_storage': True,
            'preservation': result, 'scientific_acceptance_inferred': False,
            'actual_Git_readback_completed': False,
            'model_calls': 0, 'tokenizer_calls': 0, 'scientific_auditor_calls': 0}
        s.put(root / OUT / 'v4-packaging-receipt.json', s.json_bytes(receipt))
        return receipt
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('package')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--canonical-lease', type=Path, required=True)
    p = sub.add_parser('restore')
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = (package(args.root, args.canonical_lease) if args.command == 'package'
              else storage().restore(args.index, args.destination))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
