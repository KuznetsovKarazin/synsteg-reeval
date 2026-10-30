"""Closed-panel safeguards with real codecs/isolated decoders and synthetic covers."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_shared_pilot', ROOT / 'scripts/run_shared_dictionary_pilot.py')
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)


@pytest.fixture
def design(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    model = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(random_state=0))])
    model.fit(['calm clear morning', 'ordinary green garden', 'secret hidden data', 'coded strange payload'], [0, 0, 1, 1])
    attacks = {}
    for variant in ('fixed_canonical', 'fixed_quality'):
        for feature in ('word', 'char'):
            name = variant + '__' + feature
            path = source / (name + '.joblib')
            joblib.dump(model, path)
            attacks[name] = {'variant': variant, 'features': feature, 'score_direction': -1 if feature == 'char' else 1,
                'decision_threshold': .5, 'threshold': .6, 'artifact': path.name, 'artifact_sha256': pilot.file_hash(path)}
    lock = {'attacks': attacks, 'freeze_id': 'original-fixture'}
    lock['lock_id'] = pilot.digest(lock)
    pilot.atomic_json(source / 'attack_lock.json', lock)
    config = pilot.read_json(ROOT / 'configs/shared_dictionary_pilot_v1.json')
    config.update(models=[{'alias': alias, 'model_id': 'fixture/model', 'revision': 'a' * 40, 'cpu_threads': 1}
                          for alias in ('a', 'b')], source_run=str(source),
                  source_attack_lock_sha256=pilot.file_hash(source / 'attack_lock.json'),
                  dictionary_file='tests/fixtures/shared_dictionary_pilot.json', external_resource_files=[], additional_identity_files=[])
    config['samples'] = [{'id': f's{i}', 'topic_id': f't{i}', 'genre_id': 'explanation',
                          'prompt': f'Fixture {i}', 'generation_seed': 100 + i} for i in range(2)]
    path, run = tmp_path / 'config.json', tmp_path / 'run'
    pilot.atomic_json(path, config)
    pilot.prepare(path, run)
    dictionary = pilot.codec.PublicDictionary.from_dict(pilot.read_json(run / 'public_dictionary.json'))
    return run, config, dictionary


class FakeAdapter:
    def __init__(self, dictionary, empty=False):
        self.dictionary, self.calls, self.empty = dictionary, [], empty

    def generate_cover(self, sample, settings):
        self.calls.append(sample['id'])
        position = int(sample['id'][1:])
        word = self.dictionary.classes[position].words[0]
        return {'text': '' if self.empty else ' '.join([word] * (170 if position == 0 else 40)),
                'metadata': {'fixture': True, 'generated_tokens': 0 if self.empty else 170}}

    def prepare_shared(self, cover, dictionary, protocol):
        analysis = pilot.codec.analyze_text(cover, dictionary, protocol)
        classes = {c.class_id: c.words for c in dictionary.classes}
        return {'candidate_costs': {str(slot.index) + ':' + word: float(i)
                for slot in analysis.slots for i, word in enumerate(classes[slot.class_id])}, 'fixture': True}

    def prepare_baseline(self, cover, sample, config):
        return {'fixture': sample['id']}

    def encode_decode(self, variant, state, payload, cover, sample, config):
        return {'stego_text': cover, 'decoded_payload': payload, 'embedded_bits': len(payload),
                'key': {'fixture': sample['id']}, 'metadata': {'fixture': True}}


def populate(design):
    run, config, dictionary = design
    for alias in ('a', 'b'):
        pilot.generate(run, alias, FakeAdapter(dictionary))
    return run, config, dictionary


def test_paired_payloads_capacity_denominators_and_bundle(design):
    run, _, _ = populate(design)
    report = pilot.evaluate(run)
    a, b = [pilot.read_records(run, alias, complete=True) for alias in ('a', 'b')]
    assert [r['payloads_hex'] for r in a] == [r['payloads_hex'] for r in b]
    assert all(r['payloads_hex']['16'].startswith(r['payloads_hex']['8']) for r in a)
    stat = report['models']['a']['reliability']['aes_siv__canonical__8']
    assert stat['attempted_samples'] == 2
    assert stat['full_success_samples'] == 1
    assert stat['capacity_failures'] == 1
    assert stat['delivered_payload_bits'] == 8
    assert report['models']['a']['reliability']['plain_length16__quality__16']['full_success_samples'] == 2
    bundle = pilot.read_json(run / 'models/a/bundle.json')
    assert bundle['cover_text'] == '\n\n'.join(r['cover_text'] for r in a)
    assert bundle['sample_ids'] == ['s0', 's1']
    assert all(v['full_success'] for v in bundle['variants'].values())
    assert bundle['independent_container_count'] == 1
    assert report['models']['a']['cohorts']['aes_siv__8__per_model_success']['pairs'] == 1
    pilot.completed(run)
    manifest, rows, _, _ = pilot.make_export(run)
    assert len(manifest['exclusions']) == 8
    assert sum(r['arm'] == 'bundle_cover' for r in rows) == 2


def test_gate_requires_all_models(design):
    run, _, dictionary = design
    pilot.generate(run, 'a', FakeAdapter(dictionary))
    with pytest.raises(ValueError, match='All declared models'):
        pilot.evaluate(run)
    assert not (run / 'evaluation_opened.json').exists()


def test_interrupted_attempt_is_not_retried_and_bundle_fails(design):
    run, _, dictionary = design
    pilot.atomic_json(run / 'models/a/attempt_started/s0.json', {'time_unix': 1.0})
    adapter = FakeAdapter(dictionary)
    pilot.generate(run, 'a', adapter)
    assert adapter.calls == ['s1']
    assert pilot.read_records(run, 'a')[0]['failure'] == 'interrupted_attempt_not_retried'
    bundle = pilot.read_json(run / 'models/a/bundle.json')
    assert bundle['cover_text'] is None
    assert not any(v['full_success'] for v in bundle['variants'].values())


def test_empty_generation_preserves_metadata(design):
    run, _, dictionary = design
    pilot.generate(run, 'a', FakeAdapter(dictionary, empty=True))
    rows = pilot.read_records(run, 'a')
    assert all(r['generation_metadata']['generated_tokens'] == 0 for r in rows)
    assert all('Empty generated cover' in r['failure'] for r in rows)


@pytest.mark.parametrize('unusable', ['  \n\t', {'malformed': 'nonstring'}])
def test_unusable_cover_cannot_enter_bundle_or_secondary_corpus(design, unusable):
    run, _, dictionary = design
    class UnusableAdapter(FakeAdapter):
        def generate_cover(self, sample, settings):
            return {'text': unusable, 'metadata': {'fixture': True}}
    pilot.generate(run, 'a', UnusableAdapter(dictionary))
    pilot.generate(run, 'b', FakeAdapter(dictionary))
    bundle = pilot.read_json(run / 'models/a/bundle.json')
    assert bundle['cover_text'] is None
    assert all(not output['full_success'] for output in bundle['variants'].values())
    report = pilot.evaluate(run)
    assert report['models']['a']['generated_covers'] == 0
    manifest, rows, _, _ = pilot.make_export(run)
    assert all(row['text'].strip() and row['model_alias'] == 'b' for row in rows)
    assert any(row['model_alias'] == 'a' and row['arm'] == 'cover' for row in manifest['exclusions'])


def test_second_opening_and_postopening_generation_forbidden(design):
    run, _, dictionary = populate(design)
    pilot.evaluate(run)
    with pytest.raises(ValueError, match='already completed once'):
        pilot.evaluate(run)
    with pytest.raises(ValueError, match='generation is forbidden'):
        pilot.generate(run, 'a', FakeAdapter(dictionary))


@pytest.mark.parametrize('kind', ['dictionary', 'attack', 'protocol', 'bundle', 'record', 'opening'])
def test_frozen_evidence_changes_fail_closed(design, kind):
    run, _, dictionary = design
    if kind in {'bundle', 'record', 'opening'}:
        populate(design)
        pilot.evaluate(run)
    paths = {'dictionary': run / 'public_dictionary.json', 'attack': run / 'source_attacks/fixed_quality__char.joblib',
             'protocol': run / 'protocol.json', 'bundle': run / 'models/a/bundle.json',
             'record': run / 'models/a/records/s0.json', 'opening': run / 'evaluation_opened.json'}
    path = paths[kind]
    if kind == 'opening':
        value = pilot.read_json(path)
        value['source_lock_id'] = 'changed'
        pilot.atomic_json(path, value)
    else:
        path.write_bytes(path.read_bytes() + b' ')
        if kind == 'record':
            value = pilot.read_json(path)
            value['cover_text'] += ' changed'
            pilot.atomic_json(path, value)
    with pytest.raises(ValueError):
        if kind in {'bundle', 'record', 'opening'}:
            pilot.completed(run)
        else:
            pilot.generate(run, 'a', FakeAdapter(dictionary))


def test_isolated_receiver_authenticates_and_only_receives_public_contract(design):
    _, config, dictionary = design
    cover = ' '.join([dictionary.classes[0].words[0]] * 180)
    protocol = pilot.public_protocol(config, 'aes_siv')
    key = pilot.fixture_key(config)
    encoded = pilot.codec.encode(cover, b'hi', key, dictionary, protocol)
    decoded, proof = pilot.isolated_decode(encoded.stego_text, key, dictionary, protocol)
    assert decoded == b'hi'
    assert proof['input_keys'] == ['key_hex', 'public_dictionary', 'public_protocol', 'stego_text']
    assert all(proof['io_guard_self_checks'].values())
    with pytest.raises(ValueError, match='Isolated receiver rejected'):
        pilot.isolated_decode(encoded.stego_text, bytes(32), dictionary, protocol)


def test_quality_scores_original_cover_alternatives_once_without_slot_filtering(monkeypatch):
    # This test injects a fake full-text scorer and does not perform inference.
    # Isolate its thread-setting call from the optional PyTorch installation.
    thread_settings = []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=thread_settings.append))
    dictionary = pilot.codec.PublicDictionary.from_dict({'schema': 'synsteg-public-dictionary-v1',
        'version': 'fixture', 'classes': [{'id': 'bright', 'words': ['bright', 'clear', 'vivid']},
                                        {'id': 'calm', 'words': ['calm', 'still']}]})
    class FullScorer:
        def __init__(self):
            self.texts = []
        def score(self, text):
            self.texts.append(text)
            return math.e, {'input_tokens': 5, 'scored_input_tokens': 5, 'predicted_tokens': 4,
                            'sum_nll': 4., 'mean_nll': 1., 'truncated': False, 'ppl_policy': 'sliding_window'}
    adapter = pilot.SharedAdapter()
    adapter.full_scorer = FullScorer()
    prepared = adapter.prepare_shared('bright calm bright', dictionary,
        pilot.codec.Protocol(min_group_size=1, max_bits_per_slot=1, framing='plain_length16'))
    assert thread_settings == [pilot.read_json(pilot.JOBS['gpt2'])['cpu_threads']]
    assert len(prepared['analysis']['slots']) == 3
    assert prepared['unique_full_text_scoring_calls'] == prepared['model_forward_passes'] == 5
    assert set(adapter.full_scorer.texts) == {'bright calm bright', 'clear calm bright',
        'vivid calm bright', 'bright calm clear', 'bright calm vivid'}
    assert prepared['candidate_costs']['1:calm'] == prepared['candidate_costs']['1:still'] == 0
    assert all('not_a_model_score' in row['kind'] for row in prepared['candidate_score_evidence'] if row['slot'] == 1)


def test_evaluation_resume_is_explicit_and_does_not_reopen(design, monkeypatch):
    run, _, _ = populate(design)
    original = joblib.load
    monkeypatch.setattr(joblib, 'load', lambda _: (_ for _ in ()).throw(RuntimeError('interrupted analysis')))
    with pytest.raises(RuntimeError, match='interrupted analysis'):
        pilot.evaluate(run)
    monkeypatch.setattr(joblib, 'load', original)
    report = pilot.evaluate(run)
    assert report['evaluation_resume_count'] == 1
    pilot.completed(run)
