#!/usr/bin/env python3
"""Read-only preservation of already terminated v4 under its checkout lease.

Root explicitly permits this while a separate baseline owns the shared NN lease.
No baseline path is accessed. This is a new storage variant after a preserved
global-lock refusal; it does not retry any model, tokenizer or scientific audit.
"""
import argparse
import json
from pathlib import Path
from package_author_candidate_mask_first_v4 import storage, RAW, POST, OUT, STORAGE_SHA


def package(root):
    root = Path(root).resolve(strict=True); s = storage()
    allowed = {'author-candidate-mask-first-v4',
               'author-candidate-mask-first-v4-postrun', 'project-nn.lock'}
    if {p.name for p in (root / 'results').iterdir()} - allowed:
        raise ValueError('isolated v4 results tree required')
    process = s.read_json(root / POST / 'root-process/process.json')
    if type(process.get('exit_code')) is not int or process['exit_code'] != 0:
        raise ValueError('known normal outer termination required for this scoped variant')
    for name in ['source-readback.json', 'root-process/stdout.log', 'root-process/stderr.log']:
        s.regular_bytes(root / POST / name)
    result = s.read_json(root / RAW / 'result.json')
    if result.get('requested_chunks') != 2 or [x.get('chunk_order') for x in result.get('chunks', [])] != [0, 1]:
        raise ValueError('full fixed two-chunk result required')
    supervisors = []
    for number in (0, 1):
        proof = s.read_json(root / RAW / f'chunk{number}-attempt0/supervisor-result.json')
        if (proof.get('status') != 'completed_child' or proof.get('child_exit_code') != 0
                or proof.get('deadline_expired') is not False
                or proof.get('owned_group_present_after_cleanup') is not False
                or proof.get('group_shutdown_before_reap') is not True
                or proof.get('session_verified_before_worker') is not True
                or proof.get('shutdown_error') is not None):
            raise ValueError('known owned child quiescence required for scoped preservation')
        supervisors.append(proof)
    # Storage takes the exact checkout lease used by the v4 worker and checks
    # before/after snapshots. It creates no result or changes a raw input byte.
    archived = s.package(root, root / OUT, base='results', allow_incomplete=True)
    receipt = {'schema': 'author-candidate-mask-first-v4-scoped-packaging-v2',
        'status': 'raw_results_preserved_locally_pending_Git',
        'source_base': 'results', 'storage_helper_sha256': STORAGE_SHA,
        'lease_scope': 'v4 checkout only; no shared canonical lease and no baseline access',
        'root_authorized_completed_v4_read_only_during_separate_baseline': True,
        'root_process': process, 'supervisor_quiescence_observations': supervisors,
        'quiescence_scope': 'Recorded owned-supervisor and root terminal observations, not independent OS attestation',
        'allow_incomplete_reason': 'results parent has no result.json; nested two-chunk result is preserved exactly',
        'previous_packaging_global_lock_refusal_retained': True,
        'preservation': archived, 'scientific_acceptance_inferred': False,
        'model_calls': 0, 'tokenizer_calls': 0, 'scientific_auditor_calls': 0,
        'actual_Git_readback_completed': False}
    s.put(root / OUT / 'v4-scoped-packaging-receipt.json', s.json_bytes(receipt))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(package(args.root), sort_keys=True, ensure_ascii=False))


if __name__ == '__main__': main()
