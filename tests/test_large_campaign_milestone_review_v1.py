"""Synthetic bookkeeping; no real generation, transport or semantic labelling."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location('milestone_review_test', ROOT / 'scripts/check_large_campaign_milestone_review_v1.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def fixture():
    config = {'local_models': [{'alias': alias, 'model_id': alias, 'revision': 'frozen'}
        for alias in ('qwen05', 'tinyllama11', 'falcon3_1b')],
        'selected_record_metadata': [{'record_id': f'r{i:03d}', 'source_group_id': f's{i:03d}',
        'split': 'development', 'topic': 'fixture', 'genre': 'fixture', 'target_words': 250,
        'prompt_sha256': '0' * 64} for i in range(100)]}
    rows = [dict(meta, model_alias=model['alias'], model=model['model_id'], revision=model['revision'],
        identity=f"{model['alias']}-{meta['record_id']}", status='not_attempted')
        for model in config['local_models'] for meta in config['selected_record_metadata']]
    attempts = {}
    for index, words in zip(range(100, 103), (249, 250, 251)):
        row = rows[index]
        text = 'word ' * words
        row.update(status='completed', cover_sha256=m.sha(text.encode()), whitespace_words=words)
        attempts[row['identity']] = {'attempt_sha256': '1' * 64,
            'attempt': {'identity': row['identity'], 'status': 'completed', 'result': {'cover_text': text}}}
    rows[103]['status'] = 'failed'
    attempts[rows[103]['identity']] = {'attempt_sha256': '2' * 64,
        'attempt': {'identity': rows[103]['identity'], 'status': 'failed',
                    'result': {'error_type': 'SyntheticError', 'generation_outcome_known': False}}}
    rows[104]['status'] = 'unknown'
    return config, rows, attempts


def test_full300_preserves_failed_unknown_and_missing_separately():
    config, rows, attempts = fixture()
    result = m.compose_records(rows, config, attempts, {})
    assert len(result) == 300
    assert sum(r['status'] == 'completed' for r in result) == 3
    assert sum(r['status'] == 'failed' for r in result) == 1
    assert sum(r['status'] == 'unknown' for r in result) == 1
    assert sum(r['status'] == 'not_attempted' for r in result) == 295
    assert result[103]['retained_error'] == {'error_type': 'SyntheticError', 'generation_outcome_known': False}
    assert result[104]['retained_attempt_status'] is None
    for row in result:
        if row['status'] != 'completed':
            assert row['target_relation'] is None and row['target_word_ratio'] is None


def test_length_is_descriptive_and_never_posthoc_pass_failure():
    config, rows, attempts = fixture()
    result = m.compose_records(rows, config, attempts, {})
    assert [r['target_relation'] for r in result[100:103]] == ['under', 'equal', 'over']
    assert [r['target_word_delta'] for r in result[100:103]] == [-1, 0, 1]
    assert all(r['length_pass'] is None for r in result)
    assert all(r['length_pass_status'] == 'not_defined_no_frozen_numeric_tolerance' for r in result)


@pytest.mark.parametrize('mutation', ['missing_attempt', 'cover_hash', 'duplicate_record', 'changed_source'])
def test_inconsistent_primary_evidence_rejected(mutation):
    config, rows, attempts = fixture()
    if mutation == 'missing_attempt': del attempts[rows[100]['identity']]
    if mutation == 'cover_hash': rows[100]['cover_sha256'] = '0' * 64
    if mutation == 'duplicate_record': rows[0] = copy.deepcopy(rows[1])
    if mutation == 'changed_source': rows[0]['target_words'] = 500
    with pytest.raises(ValueError): m.compose_records(rows, config, attempts, {})


def test_reviews_apply_only_to_existing_tinyllama_identity():
    config, rows, attempts = fixture()
    review = {'r000': {'review_status': 'reviewed_existing_annotations'}}
    result = m.compose_records(rows, config, attempts, review)
    assert result[100]['text_review'] == review['r000']
    assert result[0]['text_review'] is None and result[200]['text_review'] is None


def test_nonfirst30_requires_explicit_provisional_before_any_read(tmp_path):
    with pytest.raises(ValueError, match='provisional'):
        m.inspect(tmp_path, through_batch=10, provisional=False)
