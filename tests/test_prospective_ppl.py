"""Secondary PPL checks use tiny synthetic records, never sealed test samples."""
import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p = load_module('prospective_ppl_checks', ROOT / 'scripts/prospective_ppl.py')
fixture_module = load_module('prospective_completed_fixture',
                             ROOT / 'tests/test_prospective_summary.py')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


@pytest.fixture
def completed(tmp_path):
    run = tmp_path / 'run'
    fixture_module.complete_fixture(run)
    return run


def score_fixture(corpus_dir, scores_dir):
    exported = json.loads((corpus_dir / 'manifest.json').read_text())
    corpus = [json.loads(line) for line in (corpus_dir / 'corpus.jsonl').read_text().splitlines()]
    identity = {'job': exported['expected_job'], 'runtime': {'fixture': True},
                'device': 'cpu', 'script_sha256': exported['runner_script_sha256']}
    fingerprint = p.runner.digest(identity)
    values = {'alpha omega': 2., 'aleph omega': 1., 'one two three four': 3.}
    rows = []
    for row in corpus:
        nll = values[row['text']]
        details = {'input_tokens': 3, 'scored_input_tokens': 3,
                   'predicted_tokens': 2, 'sum_nll': 2 * nll, 'mean_nll': nll,
                   'truncated': False, 'ppl_policy': 'sliding_window'}
        cache_key = p.runner.cache_key(fingerprint, row['text_sha256'])
        result = {'id': row['id'], 'text_sha256': row['text_sha256'],
                  'cache_key': cache_key, 'score': math.exp(nll), 'details': details}
        rows.append(result)
        write(scores_dir / 'checkpoints' / (cache_key + '.json'), {
            'status': 'complete', 'job_fingerprint': fingerprint,
            'text_sha256': row['text_sha256'], 'score': result['score'],
            'details': details, 'elapsed_seconds': 0})
    scores_dir.mkdir(parents=True, exist_ok=True)
    (scores_dir / 'scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest = {'status': 'complete', 'identity': identity,
                'job_fingerprint': fingerprint,
                'corpus_sha256': exported['corpus_sha256'],
                'n_records': len(corpus), 'n_unique_texts': len(values),
                'n_complete_unique_texts': len(values),
                'finished_utc': '2026-10-23T00:00:00+00:00',
                'scores_sha256': p.gate.file_hash(scores_dir / 'scores.jsonl')}
    write(scores_dir / 'manifest.json', manifest)
    return rows, manifest


def test_missing_evaluation_blocks_export_without_reading_json(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No JSON may be read before completion files exist')
    monkeypatch.setattr(p.gate, 'read_json', forbidden)
    with pytest.raises(ValueError, match='completed evaluation required'):
        p.export_corpus(tmp_path / 'run', tmp_path / 'export')
    assert not (tmp_path / 'export').exists()


@pytest.mark.parametrize('event_change', ['remove_completion', 'repeat_opening'])
def test_opening_completion_gate_precedes_test_record_access(completed, tmp_path,
                                                            monkeypatch, event_change):
    path = completed / 'events.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events = events[:1] if event_change == 'remove_completion' else [*events, events[0]]
    path.write_text(''.join(json.dumps(event) + '\n' for event in events))
    real_read = p.gate.read_json
    def guarded(path):
        assert 'records' not in Path(path).parts
        return real_read(path)
    monkeypatch.setattr(p.gate, 'read_json', guarded)
    with pytest.raises(ValueError, match='exactly one opening'):
        p.export_corpus(completed, tmp_path / 'export')


def test_changed_evaluated_records_are_rejected(completed, tmp_path):
    path = completed / 'records/test/success.json'
    record = json.loads(path.read_text())
    record['cover_text'] = 'Changed after the evaluated record hash was locked.'
    write(path, record)
    with pytest.raises(ValueError, match='records differ from evaluated'):
        p.export_corpus(completed, tmp_path / 'export')


def test_export_keeps_failed_cover_and_counts_duplicate_text_once(completed, tmp_path):
    output = tmp_path / 'export'
    manifest = p.export_corpus(completed, output)
    rows = [json.loads(line) for line in (output / 'corpus.jsonl').read_text().splitlines()]
    assert manifest['planned_attempts'] == 2
    assert manifest['common_success_samples'] == 1
    assert manifest['n_records'] == 4
    assert manifest['n_unique_texts'] == 3
    assert len(manifest['exclusions']) == 2
    assert {row['text'] for row in rows} == {'alpha omega', 'aleph omega', 'one two three four'}
    assert sum(row['text'] == 'alpha omega' for row in rows) == 2
    assert len({row['id'] for row in rows}) == len(rows)
    assert all(row['text_sha256'] == p.runner.text_hash(row['text']) for row in rows)
    assert {row['topic_id'] for row in rows} == {'a', 'b'}
    assert manifest['expected_job']['ppl_policy'] == 'sliding_window'
    assert manifest['expected_job']['dtype'] == 'float32'
    assert manifest['expected_job']['model_revision'] == '607a30d783dfa663caf39e06633721c8d4cfcd7e'


def test_pair_summary_uses_common_success_and_per_text_means(completed, tmp_path):
    corpus, scores = tmp_path / 'export', tmp_path / 'scores'
    p.export_corpus(completed, corpus)
    score_fixture(corpus, scores)
    result = p.summarize(completed, corpus, scores)
    assert result['common_success_samples'] == 1
    assert len(result['paired_rows']) == 1
    delta = result['paired_deltas']
    assert delta['quality_minus_canonical_mean_nll']['n'] == 1
    assert delta['quality_minus_canonical_mean_nll']['mean'] == -1.
    assert delta['quality_minus_canonical_mean_nll']['median'] == -1.
    assert delta['quality_minus_canonical_ppl']['mean'] == pytest.approx(math.e - math.exp(2))


@pytest.mark.parametrize('damage', ['incomplete', 'wrong_corpus', 'nonfinite_score', 'missing_checkpoint'])
def test_summary_rejects_incomplete_or_mismatched_inference(completed, tmp_path, damage):
    corpus, scores = tmp_path / 'export', tmp_path / 'scores'
    p.export_corpus(completed, corpus)
    rows, manifest = score_fixture(corpus, scores)
    if damage == 'incomplete':
        manifest['status'] = 'incomplete'
    elif damage == 'wrong_corpus':
        manifest['corpus_sha256'] = '0' * 64
    elif damage == 'nonfinite_score':
        rows[0]['score'] = float('nan')
        (scores / 'scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest['scores_sha256'] = p.gate.file_hash(scores / 'scores.jsonl')
    else:
        (scores / 'checkpoints' / (rows[0]['cache_key'] + '.json')).unlink()
    write(scores / 'manifest.json', manifest)
    with pytest.raises(ValueError):
        p.summarize(completed, corpus, scores)


def test_summary_revalidates_primary_gate_after_export(completed, tmp_path):
    corpus, scores = tmp_path / 'export', tmp_path / 'scores'
    p.export_corpus(completed, corpus)
    score_fixture(corpus, scores)
    (completed / 'evaluation.json').unlink()
    with pytest.raises(ValueError, match='completed evaluation required'):
        p.summarize(completed, corpus, scores)
