#!/usr/bin/env python3
"""Recompute the offline SynSteg audit from supplied report JSON and codec source.

This does not regenerate model outputs or recompute neural detector scores.
The default protocol performs 206 classifier fits, plus codec and split audits.
Every output directory is new: previously computed output is never reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time
import zipfile
import stat

PROJECT = Path(__file__).resolve().parents[1]
STEPS = Path(__file__).resolve().parent / 'audit_steps'
EXPECTED_ARCHIVE_SHA256 = '793c419e19b09a038af439fceb0c8727f2d809e2e855444d66839fd8132ef790'
EXPECTED_SOURCE_FILE_COUNT = 2406


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_manifest(directory: Path) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def verify_source_archive(source: Path, archive: Path) -> dict[str, str]:
    """Require the fixed submitted ZIP and byte-identical extracted members.

    Generated __pycache__ directories are excluded; all other extra files,
    missing files, changed contents and symlinks fail before any computation.
    """
    if not archive.is_file():
        raise ValueError('Original source archive is required and must exist')
    actual = sha256(archive)
    if actual != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f'Source archive hash mismatch: {actual}')
    expected = {}
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            relative = PurePosixPath(item.filename)
            if (relative.is_absolute() or '..' in relative.parts or chr(92) in item.filename
                    or not relative.parts or relative.parts[0] != 'synsteg_reproduction'
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError(f'Unsafe source archive member: {item.filename}')
            if item.is_dir():
                continue
            name = PurePosixPath(*relative.parts[1:]).as_posix()
            if name in expected:
                raise ValueError(f'Duplicate source archive member: {name}')
            expected[name] = hashlib.sha256(z.read(item)).hexdigest()
    if len(expected) != EXPECTED_SOURCE_FILE_COUNT:
        raise ValueError(f'Expected {EXPECTED_SOURCE_FILE_COUNT} source files, found {len(expected)}')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Source extraction must not contain symbolic links')
    extracted = file_manifest(source)
    missing = sorted(expected.keys() - extracted.keys())
    extra = sorted(extracted.keys() - expected.keys())
    changed = sorted(k for k in expected.keys() & extracted.keys() if expected[k] != extracted[k])
    if missing or extra or changed:
        raise ValueError(f'Source extraction differs from verified archive: '
                         f'missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}')
    return expected


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def validate_detection(output: Path) -> dict:
    from sklearn.metrics import confusion_matrix, roc_auc_score
    main = json.loads((output / 'clean_detection/results.json').read_text())
    extra = json.loads((output / 'clean_detection/additional_controls_results.json').read_text())
    runs = main['runs'] + extra['runs']
    # Each permutation contains a separate model fit. All other entries are one fit.
    fits = sum(r.get('n_permutations', 1) for r in runs)
    if fits != 206:
        raise ValueError(f'Expected full fixed protocol (206 fits), found {fits}')
    checked = 0
    primary_overlap = {}
    for r in runs:
        if 'predictions' not in r:
            continue  # Aggregate length and shuffled-label controls have no prediction manifest.
        checked += 1
        tr, va, te = (set(r.get(k, [])) for k in
                      ('train_uuids', 'validation_uuids', 'test_uuids'))
        if tr & va or tr & te or va & te:
            raise ValueError(f'UUID split intersection: {r["setting"]}')
        pp = r['predictions']
        keys = [(p['uuid'], p['arm']) for p in pp]
        if len(keys) != len(set(keys)):
            raise ValueError(f'Duplicate test prediction: {r["setting"]}')
        arms = Counter(u for u, _ in keys)
        if set(arms) != te or any(n != 2 for n in arms.values()):
            raise ValueError(f'Test set does not contain exactly both arms: {r["setting"]}')
        for p in pp:
            if p['arm'] not in ('cover', 'stego') or p['true'] != int(p['arm'] == 'stego'):
                raise ValueError(f'Incorrect stored label: {r["setting"]}')
            score = p['p_stego']
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'Invalid probability: {r["setting"]}')
            if p['pred'] != int(score >= .5):
                raise ValueError(f'Prediction threshold mismatch: {r["setting"]}')
        errors = sum(p['pred'] != p['true'] for p in pp)
        if (errors != r['errors'] or not math.isfinite(r['accuracy'])
                or not math.isclose(r['accuracy'], 1 - errors / len(pp), rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'Metric does not match predictions: {r["setting"]}')
        labels = [p['true'] for p in pp]
        scores = [p['p_stego'] for p in pp]
        predictions = [p['pred'] for p in pp]
        auc = float(roc_auc_score(labels, scores))
        if (not math.isfinite(r['AUC'])
                or not math.isclose(r['AUC'], auc, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'AUC does not match predictions: {r["setting"]}')
        matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
        if r['confusion_matrix_01'] != matrix:
            raise ValueError(f'Confusion matrix does not match predictions: {r["setting"]}')
        if r['setting'].endswith('_pair_disjoint'):
            n = r['exact_normalized_train_test_text_overlap']
            primary_overlap[f'{r["setting"]}:{r["seed"]}'] = n
            if n:
                raise ValueError('Primary split has an exact normalized text intersection')
    if checked != 96:
        raise ValueError(f'Expected 96 per-text prediction manifests, found {checked}')
    return {
        'classifier_fits': fits,
        'manifest_runs_checked': checked,
        'train_validation_test_uuid_overlap': 0,
        'test_pairs_have_both_arms': True,
        'stored_labels_match_arm': True,
        'error_counts_match_predictions': True,
        'probabilities_finite_and_in_unit_interval': True,
        'auc_matches_predictions': True,
        'confusion_matrices_match_predictions': True,
        'primary_normalized_train_test_overlap': primary_overlap,
        'limitations': [
            'Length-only and shuffled-label fits retain aggregate metrics, not per-text predictions.',
            'Repeated random test splits overlap; their SD is not a confidence interval.',
            'Themes are inferred from covers; exact historical generation prompt IDs are unavailable.',
            'A fresh recomputation of saved corpora is not a new generation experiment.',
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=PROJECT / 'data/source/synsteg_reproduction')
    parser.add_argument('--source-archive', type=Path, help='Original submitted ZIP (defaults to source.parent/synsteg_reproduction.zip); mandatory for verification')
    parser.add_argument('--output', type=Path, help='New output directory; must not already exist')
    args = parser.parse_args(argv)
    source = args.source.resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.output or PROJECT / 'results' / stamp).resolve()
    archive = (args.source_archive.resolve() if args.source_archive
               else source.parent / 'synsteg_reproduction.zip')
    if not (source / 'reports/gpt6_luna_missing').is_dir():
        parser.error('--source is not an extracted SynSteg source directory; run scripts/import_source.py first')
    if not archive.is_file():
        parser.error('Original ZIP required: --source-archive does not exist; run scripts/import_source.py first')
    if output == source or source in output.parents:
        parser.error('--output must be outside the source directory')
    if output.exists():
        parser.error('--output already exists; choose a fresh directory to avoid mixing runs')
    # Establish identity against the immutable ZIP before creating results or running source code.
    before = verify_source_archive(source, archive)
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    write_json(output / 'source_manifest.json', before)
    versions = {}
    for package in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    provenance = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': 'offline_saved_corpus_recomputation',
        'fresh_model_generation': False, 'fresh_neural_detector_inference': False,
        'python': sys.version, 'platform': platform.platform(), 'versions': versions,
        'git_commit': commit,
        'invocation': [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        'source_directory': str(source), 'source_file_count': len(before),
        'source_manifest_sha256': sha256(output / 'source_manifest.json'),
        'source_archive_sha256': sha256(archive),
        'source_verified_against_original_archive': True,
        'scripts_sha256': {p.name: sha256(p) for p in sorted(STEPS.glob('*.py'))},
        'reproduce_script_sha256': sha256(Path(__file__)),
        'thread_settings': {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'steps': [],
    }
    started = time.monotonic()
    write_json(output / 'provenance.json', provenance)
    plan = [
        ('codec_confirmation', 'codec_confirmation.py', 'codec_confirmation'),
        ('split_audit', 'split_audit.py', 'split_audit'),
        ('clean_detection', 'clean_detection.py', 'clean_detection'),
        ('assign_topics', 'assign_topics.py', 'clean_detection'),
        ('additional_controls', 'additional_controls.py', 'clean_detection'),
    ]
    env = os.environ.copy()
    env.update(provenance['thread_settings'])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONHASHSEED'] = '0'
    try:
        for step, script, target in plan:
            command = [sys.executable, str(STEPS / script), '--source', str(source),
                       '--output', str(output / target)]
            if archive:
                command += ['--source-archive', str(archive)]
            print(f'Running {step} ...', flush=True)
            step_start = time.monotonic()
            logfile = output / 'logs' / (step + '.log')
            with logfile.open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            provenance['steps'].append({'step': step, 'command': command,
                                        'returncode': result.returncode,
                                        'runtime_seconds': time.monotonic() - step_start,
                                        'log': logfile.relative_to(output).as_posix()})
            write_json(output / 'provenance.json', provenance)
            if result.returncode:
                raise RuntimeError(f'{step} failed (exit {result.returncode}); see {logfile}')
        validation = validate_detection(output)
        after = file_manifest(source)
        validation['source_unchanged'] = before == after
        validation['source_verified_against_original_archive'] = True
        if before != after:
            raise RuntimeError('Source files changed during execution')
        codec = json.loads((output / 'codec_confirmation/summary.json').read_text())
        validation['codec_reports_checked'] = sum(s['n'] for s in codec.values())
        validation['codec_reencoded_exact'] = sum(s['reencoded_exact'] for s in codec.values())
        if validation['codec_reports_checked'] != 2175 or validation['codec_reencoded_exact'] != 2175:
            raise ValueError('Source codec failed exact re-encoding of the expected 2175 saved texts')
        write_json(output / 'validation.json', validation)
        provenance['status'] = 'completed'
    except Exception as exc:
        provenance['status'] = 'failed'
        provenance['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        provenance['runtime_seconds'] = time.monotonic() - started
        provenance['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'provenance.json', provenance)
        write_json(output / 'output_manifest.json', file_manifest(output))
    print(f'Completed: 206 classifier fits; 2175 exact re-encodings. Results: {output}', flush=True)
    print('This run recomputed saved data. It did not generate new LLM outputs or neural detector scores.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
