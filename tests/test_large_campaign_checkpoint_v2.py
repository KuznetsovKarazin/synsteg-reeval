import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('checkpoint', Path(__file__).resolve().parents[1] /
                                             'scripts/checkpoint_large_campaign_development_v2.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fixture():
    selected = [{'record_id': str(i), 'target_words': 500} for i in range(3)]
    models = [{'alias': 'tiny', 'model_id': 'model', 'revision': 'revision'},
              {'alias': 'other', 'model_id': 'other', 'revision': 'revision'}]
    ident = M.identity('model', '0')
    entry = {'status': 'reserved_outcome_unknown', 'metadata':
             {'record': selected[0], 'provider': 'local', 'model': 'model'},
             'request_fingerprint': 'fingerprint', 'reserved_nanodollars': 0,
             'accounted_nanodollars': 0}
    snapshots = {'tiny': {'run': {'selected_records': selected},
                 'ledger': {'attempts': {ident: entry}}, 'attempts': {}}}
    return selected, models, snapshots, ident


def test_entire_plan_without_any_generation_remains_present():
    selected, models, _, _ = fixture()
    value = M.project(selected, models, {})
    assert value['counts'] == {'not_attempted': 6}
    assert value['attempted'] == 0


def test_reserved_without_attempt_file_is_unknown_not_missing_from_denominator():
    selected, models, snapshots, ident = fixture()
    value = M.project(selected, models, snapshots)
    assert value['counts'] == {'not_attempted': 5, 'reserved_outcome_unknown': 1}
    assert value['consistency_issues'] == [{'identity': ident, 'issue': 'attempt_json_missing'}]
    assert value['records'][0]['automatic_retry_allowed'] is False


def test_cherry_picked_nonprefix_attempt_is_rejected():
    selected, models, snapshots, ident = fixture()
    entry = snapshots['tiny']['ledger']['attempts'].pop(ident)
    snapshots['tiny']['ledger']['attempts'][M.identity('model', '2')] = entry
    with pytest.raises(ValueError, match='fixed-order prefix'):
        M.project(selected, models, snapshots)


def test_removed_planned_record_is_rejected():
    selected, models, snapshots, _ = fixture()
    snapshots['tiny']['run']['selected_records'] = copy.deepcopy(selected[:2])
    with pytest.raises(ValueError, match='denominator changed'):
        M.project(selected, models, snapshots)


def test_failed_attempt_stays_failed_and_is_never_a_completion():
    selected, models, snapshots, ident = fixture()
    entry = snapshots['tiny']['ledger']['attempts'][ident]
    entry['status'] = 'failed_or_unknown_no_retry'
    entry['result'] = {'error_type': 'RuntimeError', 'generation_outcome_known': False}
    snapshots['tiny']['attempts'][ident] = {'identity': ident, **entry['metadata'],
        'automatic_retries': 0, 'request_fingerprint': 'fingerprint',
        'status': entry['status'], 'result': entry['result']}
    value = M.project(selected, models, snapshots)
    assert value['counts'] == {'failed_or_unknown_no_retry': 1, 'not_attempted': 5}
    assert value['completed_generations'] == 0
    assert value['consistency_issues'] == []
