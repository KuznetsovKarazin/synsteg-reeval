"""Independent prediction audit: synthetic closed-panel summaries only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shared_prediction_audit_tests', ROOT / 'scripts/shared_dictionary_prediction_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def fixture(tmp_path):
    config = {'models': [{'alias': 'a'}, {'alias': 'b'}],
              'samples': [{'id': 's0'}, {'id': 's1'}], 'payload_bits': [8, 16],
              'framings': ['plain_length16', 'aes_siv'], 'selectors': ['canonical', 'quality']}
    lock = {'attacks': {f'fixed_{s}__{f}': {'variant': f'fixed_{s}', 'features': f,
        'score_direction': -1 if f == 'char' else 1, 'decision_threshold': .5, 'threshold': .6}
        for s in ('canonical', 'quality') for f in ('char', 'word')}}
    records = {alias: [{'sample': {'id': sid}, 'variants': {f'{framing}__{selector}__{bits}': {
        'full_success': framing == 'plain_length16' or (alias == 'a' and sid == 's0')}
        for framing in ('plain_length16', 'aes_siv') for selector in ('canonical', 'quality') for bits in (8, 16)}}
        for sid in ('s0', 's1')] for alias in ('a', 'b')}
    evaluation = {'models': {alias: {'cohorts': {}} for alias in ('a', 'b')}}
    for alias in ('a', 'b'):
        for framing in ('plain_length16', 'aes_siv'):
            for bits in (8, 16):
                for kind in ('per_model_success', 'matched_all_models_success'):
                    ids = ['s0', 's1'] if framing == 'plain_length16' else (['s0'] if alias == 'a' and kind == 'per_model_success' else [])
                    cid = f'{framing}__{bits}__{kind}'
                    cohort = {'pairs': len(ids), 'sample_ids': ids, 'attacks': {}}
                    for selector in ('canonical', 'quality'):
                        for feature in ('char', 'word'):
                            name = f'{framing}__{selector}__{bits}__{feature}'
                            if not ids:
                                cohort['attacks'][name] = {'status': 'no_common_success_pairs'}
                                continue
                            values = [{'sample_id': sid, 'label': label, 'score': .8 if label else .2,
                                       'prediction': label, 'prediction_at_source_dev_threshold': label}
                                      for sid in ids for label in (0, 1)]
                            sha = write(tmp_path / 'predictions' / alias / cid / (name + '.json'), values)
                            cohort['attacks'][name] = {'status': 'complete', 'source_attack': f'fixed_{selector}__{feature}',
                                'accuracy': 1., 'auc': 1., 'fpr_at_development_threshold': 0., 'tpr_at_development_threshold': 1.,
                                'source_development_threshold': .6, 'source_score_direction': -1 if feature == 'char' else 1,
                                'predictions_sha256': sha}
                    evaluation['models'][alias]['cohorts'][cid] = cohort
    return tmp_path, config, evaluation, records, lock


def test_exact_per_model_and_all_model_intersection_with_empty_cells(fixture):
    result = audit.audit_predictions(*fixture)
    assert result['cohorts_checked'] == 16
    assert result['complete_attack_cells'] == 40
    assert result['empty_attack_cells'] == 24
    assert result['prediction_rows_checked'] == 144
    assert result['unique_prediction_files'] == 40
    assert result['classifier_inference_recomputed'] is False


def test_pairwise_auc_ties_and_inclusive_threshold_are_exact():
    rows = [{'label': 0, 'score': .1}, {'label': 1, 'score': .1},
            {'label': 0, 'score': .9}, {'label': 1, 'score': .9}]
    assert audit.prediction_metrics(rows, .9) == {
        'accuracy': .5, 'auc': .5, 'fpr_at_development_threshold': .5, 'tpr_at_development_threshold': .5}
    assert audit.prediction_metrics(rows, 1.01)['tpr_at_development_threshold'] == 0


@pytest.mark.parametrize('mutation', ['cohort_removed', 'cohort_added', 'attack_removed', 'false_intersection',
                                    'direction', 'threshold', 'auc', 'empty_metrics'])
def test_changed_aggregate_contract_rejected(fixture, mutation):
    run, config, evaluation, records, lock = fixture
    cohorts = evaluation['models']['a']['cohorts']
    cell = cohorts['plain_length16__8__per_model_success']['attacks']['plain_length16__canonical__8__char']
    if mutation == 'cohort_removed':
        cohorts.pop('aes_siv__8__matched_all_models_success')
    elif mutation == 'cohort_added':
        cohorts['unexpected'] = copy.deepcopy(next(iter(cohorts.values())))
    elif mutation == 'attack_removed':
        cohorts['plain_length16__8__per_model_success']['attacks'].pop('plain_length16__canonical__8__char')
    elif mutation == 'false_intersection':
        cohorts['aes_siv__8__matched_all_models_success'].update(pairs=1, sample_ids=['s0'])
    elif mutation == 'direction':
        cell['source_score_direction'] = 1
    elif mutation == 'threshold':
        cell['source_development_threshold'] = .5
    elif mutation == 'auc':
        cell['auc'] = .99
    else:
        cohorts['aes_siv__8__matched_all_models_success']['attacks']['aes_siv__canonical__8__char']['auc'] = .5
    with pytest.raises(ValueError):
        audit.audit_predictions(run, config, evaluation, records, lock)


@pytest.mark.parametrize('mutation', ['hash', 'pair_order', 'prediction', 'nan', 'extra_file', 'repeated_score'])
def test_changed_prediction_evidence_rejected(fixture, mutation):
    run, config, evaluation, records, lock = fixture
    cid = 'plain_length16__8__per_model_success'
    name = 'plain_length16__canonical__8__char'
    path = run / 'predictions/a' / cid / (name + '.json')
    values = json.loads(path.read_text())
    if mutation == 'extra_file':
        write(run / 'predictions/unregistered.json', [])
    elif mutation == 'hash':
        path.write_bytes(path.read_bytes() + b' ')
    else:
        if mutation == 'pair_order':
            values.reverse()
        elif mutation == 'prediction':
            values[0]['prediction'] = 1
        elif mutation == 'nan':
            values[0]['score'] = float('nan')
        else:
            values[0]['score'] = .3  # Same aggregate metrics; intersection score still .2.
        evaluation['models']['a']['cohorts'][cid]['attacks'][name]['predictions_sha256'] = write(path, values)
    with pytest.raises(ValueError):
        audit.audit_predictions(run, config, evaluation, records, lock)


@pytest.mark.parametrize('rows', [[], [{'label': 0, 'score': .2}],
    [{'label': False, 'score': .2}, {'label': 1, 'score': .8}],
    [{'label': 0, 'score': -.1}, {'label': 1, 'score': .8}]])
def test_bad_metric_inputs_rejected(rows):
    with pytest.raises(ValueError):
        audit.prediction_metrics(rows, .6)
