"""Independent synthetic NLI contract tests; no model session or natural scoring."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_nli_diagnostic_v1 as prep
import run_nli_diagnostic_v1 as run


def synthetic_materials():
    rows = []
    for index in range(30):
        words = (['blue', 'green', 'red', 'white'] if index < 10 else
                 ['blue', 'red'] if index < 28 else ['blue', 'red', 'white'])
        text = f'Exact {index}:\t(blue),\nunchanged Ω tail.  '
        rows.append({'review_id': f'synthetic-{index}', 'text': text,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'token_index': 2, 'original_token': '(blue),', 'words': words,
            'cohort': 'source_excerpts' if index < 25 else 'historical_covers' if index < 27 else 'grounded_covers',
            'source_group_id': f'source-{index}'})
    return rows


def synthetic_pairs():
    pairs = prep.build_pairs(synthetic_materials())
    for pair in pairs:
        pair.update(pair_tokens=50, planned_status='score_once', tokenization_sha256='0' * 64)
    return pairs


def passing_results(pairs):
    return [run.score_pair(pair, lambda _: [-8.0, 8.0, -8.0]) for pair in pairs]


def test_complete_synthetic_plan_and_exact_span_preservation():
    pairs = synthetic_pairs()
    prep.validate_pair_plan(pairs)
    assert len(pairs) == 134
    assert sum(p['is_identity'] for p in pairs) == 30
    row = next(p for p in pairs if p['review_id'] == 'synthetic-0' and p['alternative_word'] == 'green'
               and p['direction'] == 'original_to_variant')
    assert row['premise'] == 'Exact 0:\t(blue),\nunchanged Ω tail.  '
    assert row['hypothesis'] == 'Exact 0:\t(green),\nunchanged Ω tail.  '


@pytest.mark.parametrize('index', [-1, 999, True, 2.0])
def test_invalid_focus_indices_fail_closed(index):
    rows = synthetic_materials()
    rows[0]['token_index'] = index
    with pytest.raises(ValueError):
        prep.build_pairs(rows)


@pytest.mark.parametrize('logits', [
    [float('nan'), 100.0, 0.0], [0.0, float('inf'), 0.0],
    [float('-inf'), 0.0, 0.0], [False, 8.0, -8.0],
    ['0', 8.0, -8.0], [0.0, 8.0], [0.0, 8.0, -8.0, 0.0], None,
])
def test_invalid_logits_are_retained_as_error(logits):
    result = run.score_pair(synthetic_pairs()[0], lambda _: logits)
    assert result['status'] == 'classifier_error'
    assert result['classifier_call_attempted'] is True
    assert result['passes_thresholds'] is False
    assert result['probabilities'] is None


def test_label_order_and_extreme_finite_logits():
    assert run.probabilities([-1e308, 1e308, 0.0]) == {
        'contradiction': 0.0, 'entailment': 1.0, 'neutral': 0.0}
    assert run.threshold_pass(run.probabilities([8.0, -8.0, -8.0])) is False


@pytest.mark.parametrize('entailment,contradiction,expected', [
    (.90, .05, True), (.899999999, .05, False), (.90, .050000001, False),
])
def test_fixed_inclusive_threshold_boundaries(entailment, contradiction, expected):
    value = {'entailment': entailment, 'contradiction': contradiction,
             'neutral': 1 - entailment - contradiction}
    assert run.threshold_pass(value) is expected


@pytest.mark.parametrize('tokens,planned,called,status', [
    (512, 'score_once', True, 'scored'),
    (513, 'abstain_pair_too_long', False, 'abstain_pair_too_long'),
])
def test_context_boundary_never_calls_predictor_for_oversize(tokens, planned, called, status):
    pair = synthetic_pairs()[0]
    pair.update(pair_tokens=tokens, planned_status=planned)
    observed = []
    def predictor(item):
        observed.append(item)
        return [-8.0, 8.0, -8.0]
    result = run.score_pair(pair, predictor)
    assert bool(observed) is called
    assert result['status'] == status


def test_missing_duplicate_and_unexpected_results_fail_closed():
    pairs = synthetic_pairs()
    good = passing_results(pairs)
    for results in [good[:-1], good[:-1] + [good[0]], good + [good[0]]]:
        with pytest.raises(ValueError):
            run.summarize(pairs, results)


def test_failed_identity_blocks_every_alternative_without_denominator_shrinkage():
    pairs = synthetic_pairs()
    results = passing_results(pairs)
    identity = next(p for p in pairs if p['review_id'] == 'synthetic-0' and p['is_identity'])
    results = [run.score_pair(identity, lambda _: [8.0, -8.0, -8.0]) if r['pair_id'] == identity['pair_id'] else r
               for r in results]
    summary = run.summarize(pairs, results)
    assert summary['planned_pairs'] == 134
    assert summary['planned_nonidentity_alternatives'] == 52
    assert summary['alternative_statuses'] == {
        'identity_control_rejected': 3, 'accepted_by_fixed_nli_criterion': 49}


def test_one_direction_error_blocks_alternative_without_denominator_shrinkage():
    pairs = synthetic_pairs()
    results = passing_results(pairs)
    failed = next(p for p in pairs if not p['is_identity'])
    def error(_):
        raise RuntimeError('synthetic failure')
    results = [run.score_pair(failed, error) if r['pair_id'] == failed['pair_id'] else r for r in results]
    summary = run.summarize(pairs, results)
    assert summary['planned_pairs'] == 134
    assert summary['planned_nonidentity_alternatives'] == 52
    assert summary['classifier_calls_attempted'] == 134
    assert summary['classifier_calls_completed'] == 133
    assert summary['alternative_statuses'] == {
        'directed_pair_unavailable': 1, 'accepted_by_fixed_nli_criterion': 51}


def test_summary_rejects_scored_outcome_for_frozen_oversize_pair():
    pairs = synthetic_pairs()
    results = passing_results(pairs)
    pair = next(p for p in pairs if not p['is_identity'])
    pair.update(pair_tokens=513, planned_status='abstain_pair_too_long')
    row = next(r for r in results if r['pair_id'] == pair['pair_id'])
    row.update(pair_tokens=513, planned_status='abstain_pair_too_long')
    with pytest.raises(ValueError):
        run.summarize(pairs, results)


def test_summary_rejects_unattempted_scored_outcome():
    pairs = synthetic_pairs()
    results = passing_results(pairs)
    results[0]['classifier_call_attempted'] = False
    with pytest.raises(ValueError):
        run.summarize(pairs, results)


def test_existing_output_refused_before_verification_or_model_loading(tmp_path, monkeypatch):
    existing = tmp_path / 'immutable-output'
    existing.mkdir()
    sentinel = existing / 'sentinel'
    sentinel.write_bytes(b'preserve')
    def forbidden(*args, **kwargs):
        raise AssertionError('verification/model must not be entered')
    monkeypatch.setattr(prep, 'verify_preparation', forbidden)
    monkeypatch.setattr(run, 'FixedOnnxPredictor', forbidden)
    with pytest.raises(FileExistsError):
        run.execute(tmp_path, tmp_path, tmp_path, existing, 'a' * 40)
    assert sentinel.read_bytes() == b'preserve'


def synthetic_preparation(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    prepared = tmp_path / 'prepared'
    resources = tmp_path / 'resources'
    root.mkdir(); prepared.mkdir(); resources.mkdir()
    rows = synthetic_materials()
    source_raw = b''.join(prep.canonical(r) + b'\n' for r in rows)
    source = root / prep.SOURCE
    source.parent.mkdir(parents=True)
    source.write_bytes(source_raw)
    (root / prep.SOURCE_PLAN).write_bytes(prep.formatted({
        'materials_sha256': prep.sha(source_raw), 'unique_positions': 30}))
    pairs = synthetic_pairs()
    (prepared / 'pairs.jsonl').write_bytes(b''.join(prep.canonical(p) + b'\n' for p in pairs))
    (prepared / 'resource-manifest.json').write_bytes(prep.formatted({}))
    for name in prep.CODE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'# synthetic frozen code\n')
    monkeypatch.setattr(prep, 'package_versions', lambda: prep.PACKAGES)
    monkeypatch.setattr(prep, 'verify_resources', lambda *args, **kwargs: {})
    protocol = {'schema': prep.SCHEMA, 'source_main_commit': prep.SOURCE_MAIN,
        'thresholds': prep.THRESHOLDS, 'max_pair_tokens': prep.MAX_TOKENS,
        'packages': prep.PACKAGES, 'labels': prep.LABELS,
        'model_revision': prep.REVISION, 'model_repository': prep.REPOSITORY, 'model_file': prep.MODEL_FILE,
        'source_files': {name: prep.sha((root / name).read_bytes()) for name in (prep.SOURCE, prep.SOURCE_PLAN)},
        'code_files': {name: prep.sha((root / name).read_bytes()) for name in prep.CODE_FILES},
        'frozen_files': {name: prep.sha((prepared / name).read_bytes()) for name in ('pairs.jsonl', 'resource-manifest.json')},
        'planned_status_counts': {'score_once': 134},
        'runtime_profile': deepcopy(prep.RUNTIME_PROFILE)}
    (prepared / 'protocol.json').write_bytes(prep.formatted(protocol))
    return root, prepared, resources, protocol


def test_verification_rejects_misreported_runtime_profile(tmp_path, monkeypatch):
    root, prepared, resources, protocol = synthetic_preparation(tmp_path, monkeypatch)
    protocol['runtime_profile']['intra_op_num_threads'] = 99
    (prepared / 'protocol.json').write_bytes(prep.formatted(protocol))
    with pytest.raises(ValueError):
        prep.verify_preparation(root, prepared, resources)


def test_model_load_error_preserves_all_outcomes_and_zero_call_counts(tmp_path, monkeypatch):
    root, prepared, resources, _ = synthetic_preparation(tmp_path, monkeypatch)
    attempts = []
    def broken_model(_):
        attempts.append(True)
        raise RuntimeError('synthetic constructor failure; no model loaded')
    monkeypatch.setattr(run, 'FixedOnnxPredictor', broken_model)
    output = tmp_path / 'model-error-output'
    summary = run.execute(root, prepared, resources, output, 'a' * 40)
    assert attempts == [True]
    assert summary['planned_pairs'] == 134
    assert summary['pair_statuses'] == {'model_load_error': 134}
    assert summary['classifier_calls_attempted'] == 0
    assert summary['classifier_calls_completed'] == 0
    assert summary['alternative_statuses'] == {'identity_control_unavailable': 52}
    records = [json.loads(line) for line in (output / 'pairs.jsonl').read_text().splitlines()]
    assert len(records) == 134
    execution = prep.read(output / 'execution.json')
    assert execution['status'] == 'completed_with_model_load_error'
    for name, digest in execution['output_sha256'].items():
        assert prep.sha((output / name).read_bytes()) == digest


def test_synthetic_execution_attempts_each_pair_once_and_never_retries_error(tmp_path, monkeypatch):
    root, prepared, resources, _ = synthetic_preparation(tmp_path, monkeypatch)
    calls = []
    class SyntheticPredictor:
        description = {'synthetic': True}
        def __init__(self, resources):
            pass
        def __call__(self, pair):
            calls.append(pair['pair_id'])
            if len(calls) == 2:
                raise ValueError('synthetic single error')
            return [-8.0, 8.0, -8.0]
    monkeypatch.setattr(run, 'FixedOnnxPredictor', SyntheticPredictor)
    summary = run.execute(root, prepared, resources, tmp_path / 'synthetic-output', 'a' * 40)
    assert len(calls) == len(set(calls)) == 134
    assert summary['classifier_calls_attempted'] == 134
    assert summary['classifier_calls_completed'] == 133
    assert summary['alternative_statuses'] == {
        'directed_pair_unavailable': 1, 'accepted_by_fixed_nli_criterion': 51}
