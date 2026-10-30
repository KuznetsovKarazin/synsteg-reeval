"""Failure retention, pairing and one-opening safeguards for transfer studies."""
import importlib.util
from pathlib import Path
import sys

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_crossmodel_runner', ROOT / 'scripts/crossmodel_transfer_pilot.py')
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)


@pytest.fixture
def design(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    texts = ['pleasant calm sunshine', 'quiet clear morning', 'ordinary green garden', 'plain bright sky',
             'secret hidden data', 'secret hidden signal', 'secret coded message', 'secret strange payload']
    model = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(random_state=0))])
    model.fit(texts, [0] * 4 + [1] * 4)
    attacks = {}
    for variant in pilot.VARIANTS:
        for feature in ('word', 'char'):
            name = variant + '__' + feature
            path = source / (name + '.joblib')
            joblib.dump(model, path)
            attacks[name] = {'variant': variant, 'features': feature,
                'score_direction': -1 if feature == 'char' else 1, 'decision_threshold': .5,
                'threshold': .6, 'artifact': path.name, 'artifact_sha256': pilot.file_hash(path)}
    lock = {'attacks': attacks, 'freeze_id': 'original-v3'}
    lock['lock_id'] = pilot.digest(lock)
    pilot.atomic_json(source / 'attack_lock.json', lock)
    config = {'schema': pilot.SCHEMA, 'execution_enabled': True, 'paid_api_allowed': False,
              'models': [{'alias': name, 'model_id': 'fixture/model', 'revision': 'a' * 40,
                          'cpu_threads': 1} for name in ('a', 'b')],
              'generation': {'max_new_tokens': 50, 'temperature': .8, 'top_p': .95, 'top_k': 50},
              'samples': [{'id': 's' + str(i), 'topic_id': 't' + str(i), 'genre_id': 'explanation',
                           'prompt': 'Fixture ' + str(i), 'payload_bits': 8 + 8 * i,
                           'generation_seed': 100 + i} for i in range(2)],
              'payload_seed': 81, 'method_config': {'variants': pilot.VARIANTS},
              'external_resource_files': [], 'source_run': str(source),
              'source_attack_lock_sha256': pilot.file_hash(source / 'attack_lock.json')}
    path = tmp_path / 'config.json'
    pilot.atomic_json(path, config)
    run = tmp_path / 'run'
    pilot.prepare(path, run)
    return run, config, source


class FakeAdapter:
    def __init__(self, fail_variant=None, fail_sample='s1'):
        self.calls = []
        self.fail_variant, self.fail_sample = fail_variant, fail_sample

    def generate_cover(self, sample, settings):
        self.calls.append(sample['id'])
        text = {'s0': 'pleasant calm sunshine crosses the ordinary green garden',
                's1': 'quiet clear morning reveals a plain bright sky'}[sample['id']]
        return {'text': text, 'metadata': {'fixture': True}}

    def prepare_methods(self, cover, sample, method):
        return {variant: {'fixture': sample['id']} for variant in pilot.VARIANTS}

    def encode_decode(self, variant, prepared, payload, cover, sample, method):
        if variant == self.fail_variant and sample['id'] == self.fail_sample:
            raise ValueError('fixture capacity failure')
        return {'stego_text': cover + ' secret hidden data', 'decoded_payload': payload,
                'embedded_bits': len(payload), 'key': {'fixture': sample['id']}, 'metadata': {}}


def test_same_payload_across_models_and_frozen_directions(design):
    run, config, _ = design
    for alias in ('a', 'b'):
        pilot.generate(run, alias, FakeAdapter())
    report = pilot.evaluate(run)
    a, b = [pilot.read_records(run, alias, complete=True) for alias in ('a', 'b')]
    assert [r['synthetic_payload'] for r in a] == [r['synthetic_payload'] for r in b]
    assert [r['sample'] for r in a] == config['samples']
    assert report['matched_success_sample_ids'] == ['s0', 's1']
    predictions = pilot.read_json(run / 'predictions/a/per_model_common_success/fixed_canonical__char.json')
    classifier = joblib.load(run / 'source_attacks/fixed_canonical__char.joblib')
    texts, _, _ = pilot.base.paired_rows(a, 'fixed_canonical')
    assert np.allclose([p['score'] for p in predictions], 1 - classifier.predict_proba(texts)[:, 1])
    assert report['models']['a']['cohorts']['per_model_common_success']['attacks']['fixed_canonical__char']['source_development_threshold'] == .6


def test_gate_requires_every_model_before_opening(design):
    run, _, _ = design
    pilot.generate(run, 'a', FakeAdapter())
    with pytest.raises(ValueError, match='All models must finish'):
        pilot.evaluate(run)
    assert not (run / 'evaluation_opened.json').exists()


def test_partial_generation_retains_failure_and_matched_cohort(design):
    run, _, _ = design
    pilot.generate(run, 'a', FakeAdapter(fail_variant='fixed_quality'))
    pilot.generate(run, 'b', FakeAdapter())
    report = pilot.evaluate(run)
    assert report['models']['a']['reliability']['fixed_quality']['attempted_samples'] == 2
    assert report['models']['a']['reliability']['fixed_quality']['full_success_samples'] == 1
    assert report['models']['a']['reliability']['fixed_quality']['attempted_payload_bits'] == 24
    assert report['models']['b']['cohorts']['per_model_common_success']['pairs'] == 2
    assert report['matched_success_sample_ids'] == ['s0']
    assert report['models']['b']['cohorts']['matched_all_models_success']['pairs'] == 1


def test_interrupted_attempt_never_regenerated(design):
    run, _, _ = design
    pilot.atomic_json(run / 'models/a/attempt_started/s0.json', {'time_unix': 1.0})
    adapter = FakeAdapter()
    pilot.generate(run, 'a', adapter)
    assert adapter.calls == ['s1']
    first = pilot.read_records(run, 'a', complete=True)[0]
    assert first['failure'] == 'interrupted_attempt_not_retried'
    assert first['cover_text'] is None


def test_no_regeneration_or_second_evaluation_after_opening(design):
    run, _, _ = design
    for alias in ('a', 'b'):
        pilot.generate(run, alias, FakeAdapter())
    pilot.evaluate(run)
    with pytest.raises(ValueError, match='already completed once'):
        pilot.evaluate(run)
    with pytest.raises(ValueError, match='generation is forbidden'):
        pilot.generate(run, 'a', FakeAdapter())
    events = [pilot.json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    assert sum(e['event'] == 'transfer_evaluation_opened_once' for e in events) == 1


def test_transferred_attack_copy_is_immutable(design):
    run, _, _ = design
    path = run / 'source_attacks/fixed_canonical__word.joblib'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='transferred attack changed'):
        pilot.generate(run, 'a', FakeAdapter())


def test_protocol_cannot_change_after_freeze(design):
    run, _, _ = design
    config = pilot.read_json(run / 'protocol.json')
    config['samples'][0]['prompt'] = 'A tuned prompt'
    pilot.atomic_json(run / 'protocol.json', config)
    with pytest.raises(ValueError, match='Frozen design'):
        pilot.generate(run, 'a', FakeAdapter())


def test_terminal_record_tampering_prevents_evaluation(design):
    run, _, _ = design
    for alias in ('a', 'b'):
        pilot.generate(run, alias, FakeAdapter())
    path = run / 'models/a/records/s0.json'
    row = pilot.read_json(path)
    row['cover_text'] = 'changed after completion'
    pilot.atomic_json(path, row)
    with pytest.raises(ValueError, match='Records changed'):
        pilot.evaluate(run)
    assert not (run / 'evaluation.json').exists()


def test_resume_duplicate_quarantine_matches_uninterrupted_run(design, tmp_path, monkeypatch):
    _, config, _ = design
    config['samples'].append({'id': 's2', 'topic_id': 't2', 'genre_id': 'explanation',
        'prompt': 'Fixture 2', 'payload_bits': 8, 'generation_seed': 102})
    path = tmp_path / 'three.json'
    pilot.atomic_json(path, config)
    monkeypatch.setattr(pilot.base, 'near_duplicate', lambda a, b, threshold: (a, b) in {('B', 'A'), ('C', 'B')})

    class DuplicateAdapter(FakeAdapter):
        def generate_cover(self, sample, settings):
            return {'text': {'s0': 'A', 's1': 'B', 's2': 'C'}[sample['id']]}

    resumed, uninterrupted = tmp_path / 'resumed', tmp_path / 'uninterrupted'
    for run in (resumed, uninterrupted):
        pilot.prepare(path, run)
    pilot.generate(resumed, 'a', DuplicateAdapter(), limit=2)
    pilot.generate(resumed, 'a', DuplicateAdapter())
    pilot.generate(uninterrupted, 'a', DuplicateAdapter())
    left, right = [pilot.read_records(run, 'a', complete=True) for run in (resumed, uninterrupted)]
    assert [r['failure'] for r in left] == [r['failure'] for r in right]
    assert left[1]['failure'] == 'ValueError: quarantined_duplicate_of:s0'
    assert left[2]['status'] == 'completed'


def test_empty_generation_keeps_provenance(design):
    run, _, _ = design

    class EmptyAdapter(FakeAdapter):
        def generate_cover(self, sample, settings):
            return {'text': '', 'metadata': {'output_token_ids': [2], 'chat_template_sha256': 'fixture'}}

    pilot.generate(run, 'a', EmptyAdapter())
    rows = pilot.read_records(run, 'a', complete=True)
    assert all('empty cover' in r['failure'] for r in rows)
    assert all(r['generation_metadata']['output_token_ids'] == [2] for r in rows)
