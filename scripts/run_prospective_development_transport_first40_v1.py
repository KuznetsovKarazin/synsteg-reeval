#!/usr/bin/env python3
"""Add orders31–40 once; retain the complete frozen first30 outcomes verbatim.

No model code. Production requires a new enabled activation, intent and exact
external Git receipt. Existing runner, codecs, protocol and results are frozen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'prospective-development-transport-first40-v1'
ACTIVATION = 'experiments/' + VERSION + '/activation.json'
INTENT = 'results/' + VERSION + '/execution-intent.json'
PARENT_RUNNER = 'scripts/run_prospective_development_transport_v1.py'
PARENT_RUNNER_SHA = '2a927b07f0a375a77725b7b87a1a05fef7c87485b710e0bc3ffc44bf2b599f03'
PARENT_AUDITOR = 'scripts/audit_prospective_development_transport_v1.py'
PARENT_AUDITOR_SHA = 'f3919bfb86ff8063c722506c21e5fcc952a2c720c36fe1ee804f14e3b9020fab'
PRIOR = 'results/prospective-development-transport-v1/milestones/tinyllama-first30'
PRIOR_MANIFEST_SHA = 'db44f7bec7ff0dcd8cc9dea80162e58241a4663d6eb9749641acdcf2b0bbd5c9'
PRIOR_ARCHIVE_SHA = '3e5133a31283bc55953b4c11706a2eca236a9d963619e71890afa86e2ba4a43e'
IMPLEMENTATION_FILES = (
    'scripts/run_prospective_development_transport_first40_v1.py',
    'scripts/audit_prospective_development_transport_first40_v1.py',
    'tests/test_prospective_transport_first40_v1.py',
    'docs/prospective_development_transport_first40_v1_ru.md',
)


def parent(root):
    import hashlib
    path = root / PARENT_RUNNER
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_RUNNER_SHA:
        raise ValueError('Frozen parent runner changed')
    spec = importlib.util.spec_from_file_location('_first40_frozen_parent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prior_inputs(p, directory):
    """Identity checks only; never encode or decode historical covers."""
    p.require(p.sha((directory / 'output_manifest.json').read_bytes()) == PRIOR_MANIFEST_SHA,
              'Wrong frozen first30 manifest')
    manifest = p.read_json(directory / 'output_manifest.json')
    actual = {str(x.relative_to(directory)) for x in directory.rglob('*') if x.is_file()}
    p.require(actual == set(manifest) | {'output_manifest.json'}, 'Incomplete first30 archive')
    for path, digest in manifest.items():
        p.require(p.sha((directory / p.safe_relative(path)).read_bytes()) == digest, 'First30 file changed: ' + path)
    records = p.read_rows(directory / 'records.jsonl.gz')
    cells = p.read_rows(directory / 'cells.jsonl.gz')
    before = p.read_rows(directory / 'cells_before_receiver.jsonl.gz')
    jobs = p.read_rows(directory / 'recipient/receiver_jobs.jsonl.gz')
    outcomes = p.read_rows(directory / 'recipient/receiver_results.jsonl.gz')
    p.require(len(records) == 300 and len(cells) == len(before) == 15300, 'Prior denominator')
    p.require(sum(x['encoder_called'] is True for x in cells) == 720, 'Prior measured denominator')
    return records, cells, before, jobs, outcomes


def snapshot_records(p, root, config, activation):
    """Same exact snapshot validation as v1; the sole cohort change is40."""
    target = activation['target']
    cp_path = root / p.safe_relative(target['checkpoint'])
    p.require(cp_path.name == 'after-checkpoint.json' and cp_path.parent.name == 'batch20', 'Wrong immutable batch20 path')
    cp_raw = cp_path.read_bytes()
    cp = json.loads(cp_raw)
    p.require(cp['status'] == 'verified' and not cp['consistency_issues'] and cp['planned_model_record_pairs'] == 300, 'Checkpoint not verified/full300')
    p.require(cp['completed_generations'] == target['completed_covers'] == 40 and cp['counts'] == target['cover_counts'] == {'completed': 40, 'not_attempted': 260}, 'Target first40 counts differ')
    p.require(cp['model_counts'] == target['model_counts'] == {'qwen05': 0, 'tinyllama11': 40, 'falcon3_1b': 0} and not cp['closed_test_opened'] and cp['paid_calls'] == 0, 'Checkpoint scope differs')
    snapshot = cp_path.parent / 'after'
    for rel, digest in cp['run_files_sha256'].items():
        p.require(p.sha((snapshot / p.safe_relative(rel)).read_bytes()) == digest, 'Immutable snapshot SHA mismatch: ' + rel)
    expected = {(m['alias'], r['record_id']): (m, r, i) for m in config['local_models'] for i, r in enumerate(config['selected_record_metadata'], 1)}
    p.require(len(cp['records']) == 300 and len({(r['model_alias'], r['record_id']) for r in cp['records']}) == 300, 'Duplicate/omitted checkpoint records')
    records = []
    for r in cp['records']:
        key = (r['model_alias'], r['record_id'])
        p.require(key in expected, 'Unexpected checkpoint record')
        model, meta, order = expected[key]
        p.require(all(r[k] == v for k, v in meta.items()) and r['model'] == model['model_id'] and r['revision'] == model['revision'], 'Checkpoint metadata drift')
        p.require(r['status'] in ('completed', 'not_attempted'), 'Unexpected first40 cover state')
        row = dict(r, source_order=order, cover_text=None, attempt_sha256=None)
        if r['status'] == 'completed':
            rel = f"runs/large-campaign-development-v2/models/{r['model_alias']}/attempts/{r['identity']}.json"
            raw = (snapshot / rel).read_bytes()
            p.require(p.sha(raw) == cp['run_files_sha256'][rel], 'Unregistered attempt')
            attempt = json.loads(raw)
            text = attempt['result']['cover_text']
            p.require(attempt['status'] == 'completed' and attempt['identity'] == r['identity'] and attempt['record'] == meta, 'Attempt identity mismatch')
            p.require(p.sha(text.encode()) == r['cover_sha256'] and len(text.split()) == r['whitespace_words'], 'Cover content mismatch')
            row.update(cover_text=text, attempt_sha256=p.sha(raw))
        records.append(row)
    p.require([(r['model_alias'], r['source_order']) for r in records if r['status'] == 'completed'] == [('tinyllama11', n) for n in range(1, 41)], 'Exact first40 required')
    return records, cp, cp_raw, snapshot


def new_records_only(p, records, prior_records):
    """Gate delta membership before the inherited evaluator can be called."""
    key = lambda r: (r['model_alias'], r['record_id'])
    old = {key(r): r for r in prior_records}
    p.require(len(old) == len(prior_records) == len(records) == 300 and set(old) == {key(r) for r in records}, 'Changed full input membership')
    delta = []
    for row in records:
        if row['model_alias'] == 'tinyllama11' and 31 <= row['source_order'] <= 40:
            before = old[key(row)]
            p.require(before['status'] == 'not_attempted' and row['status'] == 'completed', 'Delta is not newly completed')
            for name in ('model_alias', 'record_id', 'source_group_id', 'source_order', 'model', 'revision', 'target_words', 'genre', 'topic', 'split', 'identity', 'prompt_sha256'):
                p.require(before[name] == row[name], 'Delta metadata changed: ' + name)
            delta.append(row)
        else:
            p.require(p.canonical(row) == p.canonical(old[key(row)]), 'Prior/unchanged input row changed')
    p.require([r['source_order'] for r in delta] == list(range(31, 41)), 'Exact new31–40 required')
    return delta


def merge_cells(p, old, new):
    """Replace only the510 formerly unavailable cells for orders31–40."""
    replacements = {r['row_id']: r for r in new}
    p.require(len(new) == len(replacements) == 510, 'Delta grid differs')
    p.require(all(r['model_alias'] == 'tinyllama11' and 31 <= r['source_order'] <= 40 for r in new), 'Delta includes old/noncohort cells')
    p.require(len(old) == len({r['row_id'] for r in old}) == 15300 and set(replacements) <= {r['row_id'] for r in old}, 'Prior grid differs')
    for row in old:
        if row['row_id'] in replacements:
            p.require(row['execution_status'] == 'not_run_cover_unavailable' and row['encoder_called'] is None, 'Previously attempted row cannot be replaced')
    return [replacements.get(r['row_id'], r) for r in old]


def preflight(root, frozen_root, receipt_path=None, intent_path=None):
    p = parent(root)
    p.require(p.sha((root / PARENT_AUDITOR).read_bytes()) == PARENT_AUDITOR_SHA, 'Frozen parent auditor changed')
    dependency = p.dependencies(root, frozen_root)
    activation = p.read_json(root / ACTIVATION)
    p.require(activation['experiment_id'] == VERSION and activation['feasibility_protocol_sha256'] == p.PLAN_SHA, 'Activation identity')
    p.require(activation['runtime'] == p.runtime_identity(), 'Runtime changed')
    p.require(set(activation['implementation_files_sha256']) == set(IMPLEMENTATION_FILES), 'Extension freeze incomplete')
    for relative, digest in activation['implementation_files_sha256'].items():
        p.require(p.sha((root / p.safe_relative(relative)).read_bytes()) == digest, 'Extension changed: ' + relative)
    p.require(activation['prior']['output_manifest_sha256'] == PRIOR_MANIFEST_SHA and activation['prior']['archive_sha256'] == PRIOR_ARCHIVE_SHA, 'Prior binding changed')
    prior = prior_inputs(p, root / PRIOR)
    if receipt_path is not None:
        p.require(activation['execution_enabled'] is True, 'New activation is disabled')
        p.require(intent_path is not None and intent_path.resolve() == (root / INTENT).resolve(), 'Explicit canonical intent required')
        intent, receipt = p.read_json(intent_path), p.read_json(receipt_path)
        p.require(intent['experiment_id'] == VERSION and intent['phase'] == 'prepared_before_encode' and intent['new_encoder_attempts'] == 240 and intent['reused_encoder_attempts'] == 720 and intent['outcomes_known'] is False, 'Intent scope differs')
        p.require(receipt['remote_readback_verified'] is True, 'External Git receipt missing')
        for k in ('implementation_freeze_commit', 'checkpoint_commit', 'intent_commit'):
            p.require(re.fullmatch('[0-9a-f]{40}', receipt.get(k, '')) is not None, 'Missing Git commit: ' + k)
        p.require(activation['observed_completed_covers_at_implementation_freeze'] == receipt['observed_completed_covers_at_implementation_freeze'] == 40, 'Known-output disclosure must be40')
        expected = {**activation['implementation_files_sha256'], ACTIVATION: p.sha((root / ACTIVATION).read_bytes()), INTENT: p.sha(intent_path.read_bytes()), p.PLAN: p.PLAN_SHA, dependency[0]['payloads']['map']: dependency[0]['payloads']['map_sha256'], PARENT_RUNNER: PARENT_RUNNER_SHA, PARENT_AUDITOR: PARENT_AUDITOR_SHA, PRIOR + '/output_manifest.json': PRIOR_MANIFEST_SHA, activation['target']['checkpoint']: p.sha((root / activation['target']['checkpoint']).read_bytes())}
        p.require(all(receipt.get('verified_files_sha256', {}).get(k) == v for k, v in expected.items()), 'Git receipt missing exact identities')
    return p, dependency, activation, prior


def run(root, frozen_root, output, receipt_path, intent_path):
    p, dep, activation, old = preflight(root, frozen_root, receipt_path, intent_path)
    p.require(not output.exists(), 'Output exists; no replay or overwrite')
    p.require(output == (root / activation['target']['output']).resolve(), 'Output differs from activation')
    plan, config, payload, shared, dictionaries, resources, resource_hashes, _ = dep
    records, cp, cp_raw, snapshot = snapshot_records(p, root, config, activation)
    prior_records, old_cells, old_before, old_jobs, old_outcomes = old
    delta = new_records_only(p, records, prior_records)
    receipt = p.read_json(receipt_path)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    p.write_json(output / 'status.json', {'status': 'running', 'experiment_id': VERSION})
    for origin, dest in ((root / plan['cover_cohort']['plan'], 'input_config.json'), (root / plan['payloads']['map'], 'payload-map.json'), (root / p.PLAN, 'feasibility-protocol.json'), (root / ACTIVATION, 'activation.json'), (receipt_path, 'git-readback-receipt.json'), (intent_path, 'execution-intent.json')):
        shutil.copyfile(origin, output / dest)
    (output / 'input_checkpoint.json').write_bytes(cp_raw)
    shutil.copytree(root / PRIOR, output / 'prior_milestone')
    for rel in cp['run_files_sha256']:
        dest = output / 'input_snapshot' / p.safe_relative(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / rel, dest)
    p.write_rows(output / 'records.jsonl.gz', records)
    # The sole encoder entry point: exactly10 new rows; never all40 or old30.
    new_cells, new_jobs, expected = p.evaluate(delta, payload, plan, shared, dictionaries)
    new_call_count = sum(r['encoder_called'] is True for r in new_cells)
    p.require(len([r for r in new_cells if r['arm'].startswith('public_')]) == 240 and new_call_count <= 240, 'New public condition count differs')
    p.write_rows(output / 'cells_before_receiver.jsonl.gz', merge_cells(p, old_before, new_cells))
    recipient = output / 'recipient'
    recipient.mkdir()
    shutil.copyfile(frozen_root / 'src/synsteg_audit/shared_dictionary.py', recipient / 'shared_dictionary.py')
    for version in ('v1', 'v2'):
        shutil.copyfile(frozen_root / f'data/shared_dictionary_recovery_{version}.json', recipient / f'dictionary_{version}.json')
    jobs = old_jobs + new_jobs
    p.require(len({r['row_id'] for r in jobs}) == len(jobs), 'Receiver jobs overlap')
    p.write_rows(recipient / 'receiver_jobs.jsonl.gz', jobs)
    process = subprocess.run([sys.executable, '-B', str((root / PARENT_RUNNER).resolve()), '--receiver', str(recipient)], cwd=recipient, capture_output=True, text=True, check=False)
    p.write_json(output / 'receiver_process.json', {'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr, 'job_count': len(jobs), 'payload_or_cover_argument': False, 'receiver_cwd': 'recipient', 'receiver_resource_files': sorted(x.name for x in recipient.iterdir()), 'replayed_prior_jobs': len(old_jobs), 'new_jobs': len(new_jobs)})
    p.require(process.returncode == 0, 'Receiver failed; preserve partial output')
    outcomes = p.read_rows(recipient / 'receiver_results.jsonl.gz')
    observed = {r['row_id']: r for r in outcomes}
    p.require(len(observed) == len(outcomes) == len(jobs) and set(observed) == {r['row_id'] for r in jobs}, 'Receiver membership differs')
    p.require(all(p.canonical(observed[r['row_id']]) == p.canonical(r) for r in old_outcomes), 'Historical receiver replay differs; never change prior rows')
    p.finalize(new_cells, expected, [observed[r['row_id']] for r in new_jobs])
    cells = merge_cells(p, old_cells, new_cells)
    old_public = [r for r in old_cells if r['encoder_called'] is True]
    current = {r['row_id']: r for r in cells}
    old_bytes = b''.join(p.canonical(r) + b'\n' for r in old_public)
    new_bytes = b''.join(p.canonical(current[r['row_id']]) + b'\n' for r in old_public)
    p.require(old_bytes == new_bytes, 'Old720 outcome bytes changed')
    p.write_rows(output / 'cells.jsonl.gz', cells)
    summary, accounting = p.summarize(cells, records, resources)
    summary.update(experiment_id=VERSION, new_encoder_calls=new_call_count, reused_encoder_calls=720, cumulative_encoder_calls=720 + new_call_count, cumulative_public_conditions=960)
    summary['limitations'].append('First30 transport outputs were known; this is a cumulative posthoc extension, not independent replication. Receiver decoding replays prior jobs; encoding never does.')
    p.write_json(output / 'summary.json', summary)
    p.write_json(output / 'resources.json', accounting)
    reuse = {'schema': 'prospective-first40-reuse-v1', 'prior_output_manifest_sha256': PRIOR_MANIFEST_SHA, 'prior_archive_sha256': PRIOR_ARCHIVE_SHA, 'prior_public_rows_sha256': p.sha(old_bytes), 'current_reused_public_rows_sha256': p.sha(new_bytes), 'reused_encoder_calls': 720, 'new_encoder_calls': new_call_count, 'cumulative_encoder_calls': 720 + new_call_count, 'cumulative_public_conditions': 960, 'new_public_conditions': 240, 'unchanged_record_rows': 290, 'reused_cell_rows': 14790, 'updated_cell_rows': 510, 'new_source_orders': list(range(31, 41)), 'prior_receiver_jobs_replayed': len(old_jobs), 'new_receiver_jobs': len(new_jobs)}
    p.write_json(output / 'reuse.json', reuse)
    execution = {'experiment_id': VERSION, 'activation_sha256': p.sha((root / ACTIVATION).read_bytes()), 'intent_sha256': p.sha(intent_path.read_bytes()), 'intent_commit': receipt['intent_commit'], 'feasibility_protocol_sha256': p.PLAN_SHA, 'payload_map_sha256': plan['payloads']['map_sha256'], 'config_sha256': plan['cover_cohort']['plan_sha256'], 'checkpoint_sha256': p.sha(cp_raw), 'source_checkpoint': activation['target']['checkpoint'], 'runtime': p.runtime_identity(), 'implementation_freeze_commit': receipt['implementation_freeze_commit'], 'checkpoint_commit': receipt['checkpoint_commit'], 'observed_completed_covers_at_implementation_freeze': 40, 'resource_files_sha256': resource_hashes, 'model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False, 'encoder_calls': 720 + new_call_count, 'public_conditions': 960, 'encoder_calls_scope': 'cumulative_including_exact_prior720', 'new_encoder_calls': new_call_count, 'reused_encoder_calls': 720, 'receiver_jobs': len(jobs), 'new_receiver_jobs': len(new_jobs), 'prior_receiver_jobs_replayed': len(old_jobs), 'wall_seconds': time.monotonic() - started, 'fresh_receiver_has_cover_or_payload': False, 'semantics_or_security_inferred': False}
    p.write_json(output / 'execution.json', execution)
    p.require((root / activation['target']['checkpoint']).read_bytes() == cp_raw, 'Source checkpoint changed')
    for rel, digest in cp['run_files_sha256'].items():
        p.require(p.sha((snapshot / rel).read_bytes()) == digest, 'Snapshot changed during evaluation')
    prior_inputs(p, root / PRIOR)
    status = {'status': 'completed_pending_independent_audit', 'experiment_id': VERSION, 'completed_covers': 40, 'full_denominator': 300, 'total_cells': 15300, 'unexpected_errors': sum(r['execution_status'] in ('decoder_mismatch', 'implementation_error') for r in cells), 'encoder_calls': 720 + new_call_count, 'public_conditions': 960, 'new_encoder_calls': new_call_count, 'reused_encoder_calls': 720, 'receiver_jobs': len(jobs), 'model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False}
    p.write_json(output / 'status.json', status)
    p.write_json(output / 'output_manifest.json', {str(f.relative_to(output)): p.sha(f.read_bytes()) for f in sorted(output.rglob('*')) if f.is_file()})
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--frozen-root', type=Path, default=ROOT / 'runtime/prospective-development-transport-v1/frozen')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--git-receipt', type=Path)
    parser.add_argument('--intent', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        p, _, activation, _ = preflight(args.root, args.frozen_root)
        print(json.dumps({'status': 'pass', 'execution_enabled': activation['execution_enabled'], 'new_encode_calls': 0, 'decode_calls': 0, 'model_calls': 0, 'prior720_preserved': True, 'runtime': p.runtime_identity()}))
    else:
        if not args.output or not args.git_receipt or not args.intent:
            parser.error('Production requires --output, --git-receipt and --intent')
        print(json.dumps(run(args.root.resolve(), args.frozen_root.resolve(), args.output.resolve(), args.git_receipt.resolve(), args.intent.resolve())))


if __name__ == '__main__':
    main()
