#!/usr/bin/env python3
"""Preserve a terminal unsuccessful/unknown batch without scientific replay.

No inference, tokenizer, codec, checkpoint projector, or scientific auditor is
imported. This records evidence availability, not scientific completion. An
external root receipt attests supervisor/process-group termination; local flock
checks are an additional guard, not an independent proof of process death.
"""
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import zipfile

RUN = Path('runs/large-campaign-development-v2/models/tinyllama11')
BATCHES = Path('results/large-campaign-development-v2/batches')
PACKAGER = Path('scripts/package_baseline81_100_v1.py')
PACKAGER_SHA = '8eb0ad811f7396b08ae9e5c62ca16490df15dcd145ca7388c14ae7e3cd4c9225'
MANIFEST = Path('experiments/baseline-failure-preservation-v1/manifest.json')
SOURCE_FILES = {'scripts/preserve_baseline_failure_v1.py',
                'tests/test_preserve_baseline_failure_v1.py',
                'docs/baseline_failure_preservation_v1_ru.md'}
SUCCESS = 'batch_completed_pending_remote_checkpoint'
PROOF_SCHEMA = 'baseline-failure-preservation-quiescence-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=True,
                          separators=(',', ':')).encode())


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def safe_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts
            and '..' not in relative.parts, 'Require repository-relative path')
    current = root
    for part in relative.parts:
        current /= part
        require(not current.is_symlink(), 'Refuse symlink: ' + str(relative))
    return current


@contextmanager
def existing_evidence_locks(root):
    """Use existing locks without creating or changing any evidence bytes."""
    locks = [BATCHES / 'resume-supervisor.lock',
             RUN / 'local_ledger/campaign.lock', RUN / 'run_guard/campaign.lock']
    with ExitStack() as stack:
        for relative in locks:
            path = safe_path(root, relative)
            require(path.is_file(), 'Missing evidence lock: ' + str(relative))
            stream = stack.enter_context(path.open('rb'))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError('Evidence writer may be active; refuse preservation') from exc
        yield


def snapshot(root, batch_number, proof_path):
    batch_id = f'batch{batch_number:02d}'
    directories = [RUN, BATCHES / batch_id]
    optional = [Path(f'results/large-campaign-continuation-v3/intents/{batch_id}.json'),
                Path(f'results/continuation-20261026-{batch_id}/preflight.json')]
    found, absent = {}, []
    for relative in directories:
        directory = safe_path(root, relative)
        require(directory.is_dir(), 'Missing source directory: ' + str(relative))
        for path in sorted(directory.rglob('*')):
            require(not path.is_symlink(), 'Refuse source symlink')
            require(path.is_dir() or path.is_file(), 'Refuse special source file')
            if path.is_file():
                name = path.relative_to(root).as_posix()
                found[name] = path.read_bytes()
    for relative in [*optional, proof_path]:
        path = safe_path(root, relative)
        if path.exists():
            require(path.is_file(), 'Require regular evidence file')
            found[relative.as_posix()] = path.read_bytes()
        else:
            absent.append(relative.as_posix())
    return dict(sorted(found.items())), absent


def parse_member(files, name, issues):
    raw = files.get(str(name))
    if raw is None:
        issues.append({'path': str(name), 'issue': 'missing'})
        return None
    try:
        result = json.loads(raw)
        require(isinstance(result, dict), 'Expected JSON object')
        return result
    except (ValueError, UnicodeError):
        issues.append({'path': str(name), 'issue': 'invalid_json_object'})
        return None


def availability(files, batch_number):
    """Conservative byte-availability inventory, deliberately not a projector."""
    issues = []
    run = parse_member(files, RUN / 'run.json', issues)
    ledger = parse_member(files, RUN / 'local_ledger/ledger.json', issues)
    selected = run.get('selected_records') if run else None
    valid_plan = (run is not None and run.get('provider') == 'local'
                  and isinstance(run.get('model'), str)
                  and run.get('selected_records_count') == 100
                  and isinstance(selected, list) and len(selected) == 100
                  and all(isinstance(row, dict) and row.get('split') == 'development'
                          and isinstance(row.get('record_id'), str) for row in selected)
                  and len({row['record_id'] for row in selected}) == 100)
    if not valid_plan:
        issues.append({'issue': 'unusable_100_development_run_plan'})
        return {'records': [], 'counts': {}, 'unclassified_unknown_records': 100,
                'issues': issues, 'classification_is_scientific_validation': False}
    model = run['model']
    identities = [canonical(['local', model, row['record_id']]) for row in selected]
    entries = ledger.get('attempts') if ledger else None
    ledger_valid = (ledger is not None
                    and ledger.get('schema') == 'synsteg-budget-ledger-recovery-v1'
                    and isinstance(entries, dict))
    if not ledger_valid:
        entries = {}; issues.append({'issue': 'ledger_unusable_absence_not_proven'})
    attempt_names = {name for name in files if name.startswith(str(RUN / 'attempts') + '/')}
    expected_names = {str(RUN / 'attempts' / (ident + '.json')) for ident in identities}
    orphan_names = sorted(attempt_names - expected_names)
    extra_entries = sorted(set(entries) - set(identities))
    fixed_prefix = set(entries) == set(identities[:len(entries)])
    metadata_valid = all(isinstance(entries.get(ident), dict)
                         and entries[ident].get('identity') == ident
                         and entries[ident].get('automatic_retries') == 0
                         and entries[ident].get('metadata') ==
                         {'record': meta, 'provider': 'local', 'model': model}
                         for ident, meta in zip(identities, selected) if ident in entries)
    absence_proven = ledger_valid and not orphan_names and not extra_entries and fixed_prefix and metadata_valid
    if not absence_proven:
        issues.append({'issue': 'untrusted_absence_or_identity', 'orphan_attempt_files': orphan_names,
                       'extra_ledger_identities': extra_entries, 'fixed_order_prefix': fixed_prefix})
    rows = []
    for order, (ident, meta) in enumerate(zip(identities, selected), 1):
        entry = entries.get(ident)
        attempt_path = RUN / 'attempts' / (ident + '.json')
        exists = str(attempt_path) in files
        attempt = parse_member(files, attempt_path, issues) if exists else None
        state, reason = 'unknown', 'missing_or_inconsistent_attempt_evidence'
        if entry is None and not exists and absence_proven:
            state, reason = 'not_attempted', 'no_reservation_or_attempt_in_quiescent_valid_ledger'
        elif isinstance(entry, dict) and attempt is not None:
            matching = (entry.get('identity') == ident
                        and entry.get('metadata') == {'record': meta, 'provider': 'local', 'model': model}
                        and entry.get('automatic_retries') == attempt.get('automatic_retries') == 0
                        and attempt.get('identity') == ident and attempt.get('record') == meta
                        and attempt.get('provider') == 'local' and attempt.get('model') == model
                        and isinstance(entry.get('request_fingerprint'), str)
                        and attempt.get('request_fingerprint') == entry['request_fingerprint']
                        and entry.get('result') == attempt.get('result'))
            result = entry.get('result')
            if (matching and entry.get('status') == attempt.get('status') == 'completed'
                    and isinstance(result, dict) and isinstance(result.get('cover_text'), str)
                    and result['cover_text'].strip()):
                state, reason = 'available', 'matching_retained_completed_text_without_quality_or_scientific_validation'
            else:
                reason = 'reserved_failed_unknown_or_disagreeing_evidence_no_retry'
        rows.append({'source_order': order, 'record_id': meta['record_id'], 'identity': ident,
                     'availability': state, 'reason': reason,
                     'ledger_status': entry.get('status') if isinstance(entry, dict) else None,
                     'attempt_status': attempt.get('status') if attempt else None,
                     'attempt_file_available': exists,
                     'current_fixed_pair': order in (2 * batch_number - 1, 2 * batch_number),
                     'automatic_retry_allowed': False})
    return {'records': rows, 'counts': dict(sorted(Counter(row['availability'] for row in rows).items())),
            'unclassified_unknown_records': 0, 'issues': issues,
            'classification_is_scientific_validation': False}


def preserve(root, batch_number, proof_path):
    root = Path(root).resolve(); proof_path = Path(proof_path)
    require(type(batch_number) is int and 41 <= batch_number <= 50, 'Only fixed batch41–50')
    batch_id = f'batch{batch_number:02d}'
    batch = BATCHES / batch_id
    out = Path(f'results/baseline-failure-preservation-v1/{batch_id}/artifacts')
    require(not safe_path(root, out).exists(), 'Preservation output exists; use explicit new recovery ID')
    source_manifest = json.loads(safe_path(root, MANIFEST).read_bytes())
    require(source_manifest.get('schema') == 'baseline-failure-preservation-v1-source-freeze', 'Unexpected source manifest')
    require(set(source_manifest.get('files_sha256', {})) == SOURCE_FILES, 'Incomplete source freeze')
    for name, expected in source_manifest['files_sha256'].items():
        require(sha(safe_path(root, name).read_bytes()) == expected, 'Source changed: ' + name)
    require(sha(safe_path(root, PACKAGER).read_bytes()) == PACKAGER_SHA, 'Frozen transport helper changed')
    spec = importlib.util.spec_from_file_location('frozen_baseline_transport_only', root / PACKAGER)
    transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
    transport.configure(root, batch_number); transport.OUT = out
    for relative, expected in [(transport.RESTORER, transport.RESTORER_SHA),
                               (transport.REASSEMBLER, transport.REASSEMBLER_SHA)]:
        require(sha(safe_path(root, relative).read_bytes()) == expected, 'Frozen restorer changed')
    with existing_evidence_locks(root):
        files, absent = snapshot(root, batch_number, proof_path)
        require(str(proof_path) in files, 'Missing external quiescence receipt')
        proof = json.loads(files[str(proof_path)])
        require(proof.get('schema') == PROOF_SCHEMA and proof.get('batch_id') == batch_id,
                'Invalid external quiescence receipt')
        for field in ['root_observed_supervisor_terminated', 'root_observed_child_group_terminated',
                      'exclusive_project_nn_lease_held_by_root']:
            require(proof.get(field) is True, 'Missing external termination/lease attestation: ' + field)
        require(isinstance(proof.get('observation_basis'), str) and bool(proof['observation_basis'].strip()),
                'Require concrete root terminal observation basis')
        for relative, field in [(batch / 'process.json', 'process_json_sha256'),
                                (batch / 'artifact-manifest.json', 'artifact_manifest_sha256')]:
            raw = files.get(str(relative))
            require(field in proof and proof[field] == (sha(raw) if raw is not None else None),
                    'Quiescence receipt source digest differs: ' + field)
        issues = []
        process = parse_member(files, batch / 'process.json', issues)
        require(not process or process.get('status') != SUCCESS,
                'Successful pair requires the successful packager; refuse failure-only helper')
        artifact = parse_member(files, batch / 'artifact-manifest.json', issues)
        if artifact and isinstance(artifact.get('files_sha256'), dict):
            actual = {name: sha(raw) for name, raw in files.items()
                      if name.startswith(str(batch) + '/') and name != str(batch / 'artifact-manifest.json')}
            if artifact['files_sha256'] != actual:
                issues.append({'issue': 'batch_manifest_disagreement_preserved_without_repair'})
        elif artifact is not None:
            issues.append({'issue': 'batch_manifest_unusable_preserved_without_repair'})
        inventory = availability(files, batch_number)
        (root / out).mkdir(parents=True, exist_ok=False)
        archive_path = root / out / 'full.zip'
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in files.items():
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 26, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED; info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, raw, compresslevel=9)
        archive_raw = archive_path.read_bytes()
        require(0 < len(archive_raw) <= transport.LOGICAL_ARCHIVE_MAX_BYTES, 'Archive exceeds frozen operational bound')
        native = {'schema': 'native-continuation-artifact-bundle-v1',
                  'artifact_id': 'failure-preservation-v1-' + batch_id,
                  'archive': {'path': str(out / 'full.zip'), 'bytes': len(archive_raw), 'sha256': sha(archive_raw)},
                  'files': {name: {'bytes': len(raw), 'sha256': sha(raw)} for name, raw in files.items()}}
        write_new(root / out / 'manifest.json', native)
        chunked = transport.chunk_archive(archive_raw, native)
        restored = transport.verify_chunk_restore(native, files)
        require(snapshot(root, batch_number, proof_path) == (files, absent), 'Input changed during preservation; no valid terminal package')
        write_new(root / out / 'fresh-restore.json', restored)
        write_new(root / out / 'availability.json', inventory)
        status = {'schema': 'baseline-failure-preservation-v1', 'batch_id': batch_id,
                  'status': 'preserved_locally_recovery_required_pending_git',
                  'supervisor_status_observed': process.get('status') if process else None,
                  'supervisor_exit_code_observed': process.get('exit_code') if process else None,
                  'supervisor_timed_out_observed': process.get('timed_out') if process else None,
                  'external_quiescence_receipt': str(proof_path), 'external_termination_independently_verified': False,
                  'primary_members': len(files), 'archive': native['archive'], 'fresh_restore': restored,
                  'availability_counts': inventory['counts'], 'unclassified_unknown_records': inventory['unclassified_unknown_records'],
                  'tinyllama_planned_records': 100, 'full_planned_model_record_pairs': 300,
                  'missing_optional_evidence': absent, 'evidence_issues': issues,
                  'generation_outcome_inferred_from_timeout': False, 'scientific_completed_count_asserted': False,
                  'next_batch_allowed': False, 'automatic_retry_allowed': False,
                  'successful_closure_allowed': False, 'recovery_requires_new_explicit_id': True,
                  'scientific_auditor_calls': 0, 'model_calls': 0, 'codec_calls': 0,
                  'remote_git_verified': False, 'old_evidence_modified': False}
        write_new(root / out / 'status.json', status)
        text_files = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
                      for p in sorted((root / out).iterdir()) if p.name.endswith('.json')]
        plan = {'schema': 'baseline-failure-preservation-save-plan-v1', 'batch_id': batch_id,
                'text_files': text_files, 'binary_parts': chunked['chunks'], 'include_this_plan': True,
                'logical_archive_direct_git_blob': False, 'source_run_paths_must_not_replace_prior_git_checkpoint': True,
                'required_after_save': 'Actual Git readback of every listed path and complete restore from fetched parts. No scientific replay or success closure. Durable handoff must block next batch until an explicit new recovery ID.',
                'remote_git_verified': False, 'model_calls': 0}
        write_new(root / out / 'primary-save-plan.json', plan)
        return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--batch-number', type=int, choices=range(41, 51), required=True)
    parser.add_argument('--quiescence-receipt', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preserve(args.root, args.batch_number, args.quiescence_receipt), ensure_ascii=False, indent=2))
