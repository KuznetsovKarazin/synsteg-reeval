"""Reject misleading or incomplete review evidence without judging semantics."""
import copy
import importlib.util
from pathlib import Path

import pytest
from synsteg_audit import shared_dictionary as base

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('acceptance_review', ROOT / 'scripts/audit_universal_context_acceptance_reviews_v1.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture():
    dictionary = base.PublicDictionary.from_mapping({'pace': ['quick', 'rapid']}, 'evidence-fixture')
    material = {'review_id': 'last-token', 'text': 'quick X quick.', 'token_index': 2,
                'class_id': 'pace', 'words': ['quick', 'rapid']}
    flags = {f: False for f in module.FLAGS}
    alternatives = [{'word': word, 'unchanged': word == 'quick', **flags,
        'quoted_original': 'quick.', 'substituted_phrase': word + '.', 'explanation': 'Synthetic evidence fixture.'}
        for word in material['words']]
    data = {'schema': 'universal-accepted-slots-review-v1', 'reviewer': 'A',
        'review_type': 'independent_assistant_not_human', 'materials_sha256': 'materials', 'plan_sha256': 'plan',
        'records': [{'review_id': 'last-token', **flags, 'alternatives': alternatives}]}
    return data, [material], dictionary


def verify(data, materials, dictionary):
    return module.verify_review(data, 'A', materials, 'plan', 'materials', dictionary)


def test_exact_focus_and_complete_identity_alternative():
    data, materials, dictionary = fixture()
    rows, count = verify(data, materials, dictionary)
    assert count == 2 and set(rows) == {'last-token'}


@pytest.mark.parametrize('fault', ['other_occurrence', 'extra_change', 'missing_member', 'duplicate_member',
    'identity_damage', 'nonboolean', 'wrong_or', 'stale_plan', 'missing_position'])
def test_rejects_misleading_or_incomplete_evidence(fault):
    data, materials, dictionary = fixture()
    alternative = data['records'][0]['alternatives'][1]
    if fault == 'other_occurrence':
        alternative.update(quoted_original='quick X', substituted_phrase='rapid X')
    elif fault == 'extra_change':
        alternative['substituted_phrase'] = 'rapid!'
    elif fault == 'missing_member':
        data['records'][0]['alternatives'].pop()
    elif fault == 'duplicate_member':
        data['records'][0]['alternatives'][1] = copy.deepcopy(data['records'][0]['alternatives'][0])
    elif fault == 'identity_damage':
        data['records'][0]['alternatives'][0]['definite_meaning_change'] = True
    elif fault == 'nonboolean':
        alternative['definite_meaning_change'] = 1
    elif fault == 'wrong_or':
        alternative['definite_meaning_change'] = True
    elif fault == 'stale_plan':
        data['plan_sha256'] = 'wrong'
    else:
        data['records'] = []
    with pytest.raises(ValueError):
        verify(data, materials, dictionary)
