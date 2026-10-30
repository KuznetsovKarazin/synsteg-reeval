"""Synthetic controller/provenance tests; no ONNX model session or inference."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_nli_diagnostic_v1 as prep
import run_nli_diagnostic_v1 as run


def material(index=0, words=None):
    text = 'A quick sample.'
    return {'review_id': f'synthetic-{index}', 'text': text, 'text_sha256': prep.sha(text.encode()),
        'token_index': 1, 'original_token': 'quick', 'words': words or ['brisk', 'quick', 'rapid'],
        'cohort': 'source_excerpts' if index < 25 else 'historical_covers' if index < 27 else 'grounded_covers',
        'source_group_id': f'group-{index}'}


def full_pairs():
    materials = [material(i, ['brisk', 'quick', 'rapid'] if i < 22 else ['quick', 'rapid']) for i in range(30)]
    pairs = prep.build_pairs(materials)
    for pair in pairs:
        pair.update(pair_tokens=20, tokenization_sha256='synthetic-tokenization', planned_status='score_once')
    prep.validate_pair_plan(pairs)
    return pairs


def all_results(pairs):
    return [run.score_pair(p, lambda _: [-5.0, 5.0, -5.0]) for p in pairs]


def test_full_plan_has30identities_and104_directed_pairs():
    pairs = full_pairs()
    assert len(pairs) == 134
    assert sum(p['is_identity'] for p in pairs) == 30
    assert len({p['review_id'] for p in pairs}) == 30
    for pair in pairs:
        assert pair['premise'].startswith('A ') and pair['hypothesis'].endswith(' sample.')


def test_full_context_replacement_preserves_all_other_text():
    row = material()
    row.update(text='Long prefix. A (quick), sample! Tail stays.', token_index=3, original_token='(quick),')
    pairs = prep.build_pairs([row])
    selected = next(p for p in pairs if p['alternative_word'] == 'rapid' and p['direction'] == 'original_to_variant')
    assert selected['premise'] == row['text']
    assert selected['hypothesis'] == 'Long prefix. A (rapid), sample! Tail stays.'


@pytest.mark.parametrize('fault', ['focus_index', 'focus_literal', 'duplicate_words', 'uppercase_word', 'original_missing'])
def test_bad_focus_or_class_is_rejected(fault):
    row = material()
    if fault == 'focus_index': row['token_index'] = 999
    elif fault == 'focus_literal': row['original_token'] = 'other'
    elif fault == 'duplicate_words': row['words'] = ['quick', 'rapid', 'rapid']
    elif fault == 'uppercase_word': row['words'] = ['Quick', 'quick']
    else: row['words'] = ['brisk', 'rapid']
    with pytest.raises(ValueError): prep.build_pairs([row])


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'direction', 'text_hash', 'pair_hash', 'swapped_wrong'])
def test_plan_integrity_requires_every_exact_direction(fault):
    pairs = full_pairs()
    if fault == 'missing': pairs.pop()
    elif fault == 'duplicate': pairs[-1] = deepcopy(pairs[0])
    elif fault == 'direction': pairs[1]['direction'] = 'identity'
    elif fault == 'text_hash': pairs[1]['premise'] += '!'
    elif fault == 'pair_hash': pairs[1]['pair_id'] = 'bad'
    else:
        pairs[1]['premise'] = 'Altered whole text'
        pairs[1]['premise_sha256'] = prep.sha(pairs[1]['premise'].encode())
    with pytest.raises(ValueError): prep.validate_pair_plan(pairs)


def test_every_alternative_requires_identity_and_both_directions():
    pairs = full_pairs(); results = all_results(pairs)
    summary = run.summarize(pairs, results)
    assert summary['planned_nonidentity_alternatives'] == 52
    assert summary['alternative_statuses'] == {'accepted_by_fixed_nli_criterion': 52}
    assert summary['classifier_calls_completed'] == 134


@pytest.mark.parametrize('failed_part', ['identity', 'original_to_variant', 'variant_to_original'])
def test_threshold_failure_is_not_hidden_by_other_direction(failed_part):
    pairs = full_pairs()
    def predictor(pair):
        if pair['review_id'] == 'synthetic-0' and pair['direction'] == failed_part and (failed_part == 'identity' or pair['alternative_word'] == 'brisk'):
            return [-5.0, -5.0, 5.0]
        return [-5.0, 5.0, -5.0]
    summary = run.summarize(pairs, [run.score_pair(p, predictor) for p in pairs])
    assert sum(a['accepted'] for a in summary['alternatives']) == (50 if failed_part == 'identity' else 51)


def test_overlength_kept_and_never_calls_predictor():
    pair = full_pairs()[1]
    pair.update(pair_tokens=513, planned_status='abstain_pair_too_long')
    def forbidden(_): raise AssertionError('Overlength text must not reach inference')
    result = run.score_pair(pair, forbidden)
    assert result['status'] == 'abstain_pair_too_long'
    assert result['classifier_call_attempted'] is False
    assert result['passes_thresholds'] is False


def test_identity_overlength_blocks_all_its_alternatives():
    pairs = full_pairs(); pairs[0].update(pair_tokens=513, planned_status='abstain_pair_too_long')
    summary = run.summarize(pairs, all_results(pairs))
    assert summary['alternative_statuses']['identity_control_unavailable'] == 2
    assert summary['classifier_calls_attempted'] == 133


@pytest.mark.parametrize('logits', [[0.0, float('nan'), 0.0], [0.0, float('inf'), 0.0], [0.0, 1.0],
    [0.0, '1', 0.0], [False, 1.0, 0.0]])
def test_invalid_logits_fail_closed(logits):
    result = run.score_pair(full_pairs()[1], lambda _: logits)
    assert result['status'] == 'classifier_error' and result['passes_thresholds'] is False
    assert result['classifier_call_attempted'] is True


def test_predictor_exception_retained():
    def fail(_): raise RuntimeError('Synthetic failure')
    result = run.score_pair(full_pairs()[1], fail)
    assert result['status'] == 'classifier_error' and result['error_type'] == 'RuntimeError'


def test_fixed_threshold_boundaries_and_nonfinite_fail_closed():
    assert run.threshold_pass({'entailment': .90, 'contradiction': .05, 'neutral': .05})
    assert not run.threshold_pass({'entailment': .899999, 'contradiction': .05, 'neutral': .050001})
    assert not run.threshold_pass({'entailment': .90, 'contradiction': .050001, 'neutral': .049999})
    assert not run.threshold_pass({'entailment': float('nan'), 'contradiction': 0., 'neutral': 0.})


@pytest.mark.parametrize('fault', ['missing_result', 'duplicate_result', 'wrong_pair_provenance', 'changed_probability',
    'false_threshold', 'unknown_status', 'call_accounting', 'overlength_scored'])
def test_summarizer_rejects_altered_or_incomplete_outputs(fault):
    pairs = full_pairs(); results = all_results(pairs)
    if fault == 'missing_result': results.pop()
    elif fault == 'duplicate_result': results[-1] = deepcopy(results[0])
    elif fault == 'wrong_pair_provenance': results[0]['hypothesis_sha256'] = 'bad'
    elif fault == 'changed_probability': results[0]['probabilities']['entailment'] = .99
    elif fault == 'false_threshold': results[0]['passes_thresholds'] = False
    elif fault == 'unknown_status': results[0]['status'] = 'mystery'
    elif fault == 'call_accounting': results[0]['classifier_call_attempted'] = False
    else:
        pairs[0].update(pair_tokens=513, planned_status='abstain_pair_too_long')
        results[0].update(pair_tokens=513, planned_status='abstain_pair_too_long')
    with pytest.raises(ValueError): run.summarize(pairs, results)


def test_tokenizer_full_pair_is_not_truncated(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, '[CLS]': 1, '[SEP]': 2, 'word': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B [SEP]', special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    tokenizer.enable_truncation(max_length=10)
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    loaded = prep.load_tokenizer(tmp_path)
    value = prep.tokenization(loaded, ' '.join(['word'] * 300), ' '.join(['word'] * 300))
    assert len(value['input_ids']) == 603
    assert loaded.truncation is None and loaded.padding is None


def test_resource_content_mismatch_and_wrong_labels(tmp_path, monkeypatch):
    payloads = {'config.json': prep.formatted({'id2label': prep.LABELS, 'label2id': {v: int(k) for k, v in prep.LABELS.items()}, 'max_position_embeddings': 512}),
        'tokenizer_config.json': prep.formatted({'model_max_length': 512}), 'model.onnx': b'synthetic-not-loadable'}
    pins = {name: {'bytes': len(value), 'sha256': prep.sha(value)} for name, value in payloads.items()}
    monkeypatch.setattr(prep, 'FILE_PINS', pins)
    for name, value in payloads.items(): (tmp_path / name).write_bytes(value)
    assert prep.verify_resources(tmp_path)['labels'] == prep.LABELS
    (tmp_path / 'model.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError): prep.verify_resources(tmp_path)
    (tmp_path / 'model.onnx').write_bytes(payloads['model.onnx'])
    config = json.loads(payloads['config.json']); config['id2label'] = {'0': 'entailment', '1': 'contradiction', '2': 'neutral'}
    data = prep.formatted(config); (tmp_path / 'config.json').write_bytes(data)
    pins['config.json'] = {'bytes': len(data), 'sha256': prep.sha(data)}
    with pytest.raises(ValueError): prep.verify_resources(tmp_path)


def test_download_not_implied_by_verification(tmp_path, monkeypatch):
    def forbidden(*_a, **_k): raise AssertionError('Network must require explicit download flag')
    monkeypatch.setattr(prep.urllib.request, 'urlopen', forbidden)
    with pytest.raises(FileNotFoundError): prep.verify_resources(tmp_path)


def test_execute_refuses_overwrite_before_any_session(tmp_path, monkeypatch):
    def forbidden(*_a, **_k): raise AssertionError('No model load before checks')
    monkeypatch.setattr(run, 'FixedOnnxPredictor', forbidden)
    output = tmp_path / 'existing'; output.mkdir()
    with pytest.raises(FileExistsError): run.execute(tmp_path, tmp_path, tmp_path, output, 'a' * 40)


def test_execute_requires_exact_git_freeze_before_any_session(tmp_path, monkeypatch):
    def forbidden(*_a, **_k): raise AssertionError('No model load before checks')
    monkeypatch.setattr(run, 'FixedOnnxPredictor', forbidden)
    with pytest.raises(ValueError): run.execute(tmp_path, tmp_path, tmp_path, tmp_path / 'new', 'main')
