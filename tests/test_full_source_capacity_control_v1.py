"""Synthetic-only tests: no natural measurement, model, encoder or decoder."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_full_source_capacity_control_v1 as audit


def rows():
    out = []
    for i in range(1, 21):
        text = f'Synthetic unchanged source number {i}.'
        out.append({'record_id': f'fake-{i}', 'plan_index': i, 'text': text,
                    'source_sha256': audit.sha(text.encode()), 'source_words': len(text.split()),
                    'source_eligible': i not in audit.SOURCE_GATES, 'gguf_selected': i in audit.SELECTED,
                    'methods': {m: audit.capacity_result(0 if i % 2 else 32) for m in audit.METHODS}})
    return out


@pytest.mark.parametrize('capacity,plain8,plain16,aes8', [(0, False, False, False), (23, False, False, False),
    (24, True, False, False), (31, True, False, False), (32, True, True, False), (151, True, True, False), (152, True, True, True)])
def test_exact_arithmetic_thresholds(capacity, plain8, plain16, aes8):
    value = audit.capacity_result(capacity)
    assert value['frame_capacity_feasible'] == {'plain8': plain8, 'plain16': plain16, 'aes8': aes8}
    assert not value['encoding_called'] and not value['decoding_called'] and not value['delivery_established']


@pytest.mark.parametrize('capacity', [-1, True, 1.5, '8', None])
def test_invalid_measured_capacity(capacity):
    with pytest.raises(ValueError): audit.capacity_result(capacity)


def test_errors_are_null_not_zero_and_keep_all_denominators():
    records = rows()
    records[0]['methods']['base_v1'] = audit.capacity_result(None, 'SyntheticError')
    summary = audit.summarize(records)
    assert summary['measurement_denominator'] == 60
    assert {k: v['records'] for k, v in summary['groups'].items()} == {'all20': 20, 'source_eligible15': 15, 'gguf_selected8': 8}
    value = summary['groups']['all20']['methods']['base_v1']
    assert value['denominator'] == 20 and value['measured'] == 19 and value['analysis_errors'] == 1
    assert value['zero_capacity'] == 9
    assert records[0]['methods']['base_v1']['frame_capacity_feasible'] == dict.fromkeys(audit.THRESHOLDS)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'order', 'eligible', 'selected', 'methods', 'threshold', 'delivery'])
def test_invalid_summary_rejected(mutation):
    records = rows()
    if mutation == 'missing': records.pop()
    elif mutation == 'duplicate': records[1]['record_id'] = records[0]['record_id']
    elif mutation == 'order': records[0], records[1] = records[1], records[0]
    elif mutation == 'eligible': records[1]['source_eligible'] = False
    elif mutation == 'selected': records[1]['gguf_selected'] = False
    elif mutation == 'methods': records[0]['methods'].pop('base_v2')
    elif mutation == 'threshold': records[0]['methods']['base_v1']['frame_capacity_feasible']['plain8'] = True
    else: records[0]['methods']['base_v1']['delivery_established'] = True
    with pytest.raises(ValueError): audit.summarize(records)


def test_each_analyzer_receives_whole_ineligible_text_once_and_failure_retained():
    row = rows()[18]
    assert not row['source_eligible']
    calls = []
    def okay(text):
        calls.append(text); return 0
    def failed(text):
        calls.append(text); raise RuntimeError('synthetic refusal')
    result = audit.analyze_record(row, {'base_v1': okay, 'base_v2': failed, 'universal_morph_v3': okay})
    assert calls == [row['text']] * 3 and result['text'] == row['text']
    assert result['methods']['base_v2']['status'] == 'analysis_error'
    assert result['methods']['base_v1']['capacity_bits'] == 0


def test_prepare_never_invokes_natural_analyzers(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Natural analyzer invoked during preparation')
    monkeypatch.setattr(audit.base, 'analyze_text', forbidden)
    monkeypatch.setattr(audit.universal, 'analyze_text', forbidden)
    protocol = audit.make_protocol(ROOT)
    assert protocol['analysis_calls_during_preparation'] == 0 and protocol['measurement_count'] == 60


def test_analyze_refuses_changed_source_before_any_call():
    row = rows()[0]
    row['text'] += ' new invented sentence'
    calls = []
    with pytest.raises(ValueError):
        audit.analyze_record(row, {m: lambda text: calls.append(text) for m in audit.METHODS})
    assert not calls
