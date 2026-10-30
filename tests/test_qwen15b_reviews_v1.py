"""Synthetic-only evidence validation; never open production generations or test data."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_qwen15b_reviews_v1 as audit

FLAGS = ('unsupported_claim', 'contradiction', 'language_or_entity_error', 'ambiguous_or_borderline')
SELECTED = {2, 5, 6, 7, 8, 9, 10, 11}
GATES = {1, 3, 4, 16, 19}


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def materials():
    records = []
    for index in range(1, 21):
        selected = index in SELECTED
        available = selected and index != 11
        action = ('generate_once' if selected else 'abstain_source_gate'
                  if index in GATES else 'not_selected_fixed_eight_budget')
        source = f'Synthetic source {index} states that the bridge opened in 1998.'
        output = 'The bridge opened in 1999.' if available else None
        records.append({'record_id': f'synthetic-{index:02d}', 'plan_index': index,
                        'planned_action': action,
                        'status': 'generated' if available else 'request_error' if selected else action,
                        'available': available, 'source_text': source, 'source_sha256': sha(source),
                        'generation_schema_issues': [] if available else None,
                        'generation_truncated': False if available else None,
                        'output_text': output, 'output_sha256': sha(output) if output is not None else None})
    return {'schema': 'qwen15b-review-materials-v1', 'records': records}


def review(material=None, reviewer_id='A'):
    material = material or materials()
    return {'schema': 'qwen15b-source-bound-review-v1', 'reviewer_id': reviewer_id,
            'review_type': 'source_bound_assistant_diagnostic',
            'records': [{'record_id': row['record_id'], 'source_sha256': row['source_sha256'],
                         'output_sha256': row['output_sha256'], 'whole_source_read': row['available'],
                         'whole_output_read': row['available'],
                         'flags': {flag: False if row['available'] else None for flag in FLAGS},
                         'findings': []} for row in material['records']]}


def add_finding(record, category='unsupported_claim', source_quote=None):
    record['flags'][category] = True
    record['findings'].append({'category': category, 'output_quote': 'opened in 1999',
                               'source_quote': source_quote,
                               'reason': 'The supplied synthetic reference gives a different year.'})


def test_full_denominator_and_unavailable_nulls_are_valid():
    material = materials()
    audit.validate_materials(material)
    summary = audit.validate_review(review(material), material)
    assert isinstance(summary, dict)


def test_available_output_cannot_be_hidden_with_false_availability():
    material = materials()
    material['records'][1]['available'] = False
    with pytest.raises(ValueError):
        audit.validate_materials(material)


@pytest.mark.parametrize('field,value', [('generation_schema_issues', ['unknown_stop_metadata']),
                                        ('generation_truncated', True), ('generation_truncated', None)])
def test_uncertain_or_truncated_generation_cannot_be_marked_available(field, value):
    material = materials()
    material['records'][1][field] = value
    with pytest.raises(ValueError):
        audit.validate_materials(material)


@pytest.mark.parametrize('change', ['missing', 'duplicate_id', 'duplicate_index', 'source_hash',
                                    'output_hash', 'available_without_text', 'wrong_action_counts'])
def test_material_identity_and_availability_tampering_rejected(change):
    material = materials()
    if change == 'missing': material['records'].pop()
    elif change == 'duplicate_id': material['records'][1]['record_id'] = material['records'][0]['record_id']
    elif change == 'duplicate_index': material['records'][1]['plan_index'] = 1
    elif change == 'source_hash': material['records'][1]['source_sha256'] = '0' * 64
    elif change == 'output_hash': material['records'][1]['output_sha256'] = '0' * 64
    elif change == 'available_without_text': material['records'][0]['available'] = True
    else: material['records'][0]['planned_action'] = 'generate_once'
    with pytest.raises(ValueError):
        audit.validate_materials(material)


@pytest.mark.parametrize('change', ['missing_record', 'duplicate_record', 'source_hash', 'output_hash',
                                    'whole_source_false', 'whole_output_false', 'missing_flag',
                                    'string_flag', 'integer_flag', 'true_without_finding'])
def test_review_coverage_hashes_and_explicit_read_attestation(change):
    material = materials()
    value = review(material)
    row = value['records'][1]
    if change == 'missing_record': value['records'].pop()
    elif change == 'duplicate_record': value['records'][1] = deepcopy(value['records'][0])
    elif change == 'source_hash': row['source_sha256'] = 'f' * 64
    elif change == 'output_hash': row['output_sha256'] = 'f' * 64
    elif change == 'whole_source_false': row['whole_source_read'] = False
    elif change == 'whole_output_false': row['whole_output_read'] = False
    elif change == 'missing_flag': del row['flags']['unsupported_claim']
    elif change == 'string_flag': row['flags']['unsupported_claim'] = 'false'
    elif change == 'integer_flag': row['flags']['unsupported_claim'] = 0
    else: row['flags']['unsupported_claim'] = True
    with pytest.raises(ValueError):
        audit.validate_review(value, material)


@pytest.mark.parametrize('change', ['unknown_category', 'quote_not_in_output', 'blank_output_quote',
                                    'quote_not_in_source', 'blank_source_quote', 'blank_reason',
                                    'finding_without_true_flag', 'contradiction_without_source_quote'])
def test_findings_require_literal_evidence_and_coherent_flags(change):
    material = materials()
    value = review(material)
    row = value['records'][1]
    add_finding(row)
    finding = row['findings'][0]
    if change == 'unknown_category': finding['category'] = 'human_gold_pass'
    elif change == 'quote_not_in_output': finding['output_quote'] = 'opened in 2000'
    elif change == 'blank_output_quote': finding['output_quote'] = ' '
    elif change == 'quote_not_in_source': finding['source_quote'] = 'opened in 2000'
    elif change == 'blank_source_quote': finding['source_quote'] = ''
    elif change == 'blank_reason': finding['reason'] = '\n '
    elif change == 'finding_without_true_flag': row['flags']['unsupported_claim'] = False
    else:
        row['flags']['unsupported_claim'] = False
        row['flags']['contradiction'] = True
        finding['category'] = 'contradiction'
    with pytest.raises(ValueError):
        audit.validate_review(value, material)


@pytest.mark.parametrize('change', ['false_flag', 'true_flag', 'read_source', 'read_output', 'finding'])
def test_unavailable_records_cannot_receive_quality_ratings(change):
    material = materials()
    value = review(material)
    row = value['records'][0]
    if change == 'false_flag': row['flags']['unsupported_claim'] = False
    elif change == 'true_flag': row['flags']['unsupported_claim'] = True
    elif change == 'read_source': row['whole_source_read'] = True
    elif change == 'read_output': row['whole_output_read'] = True
    else: add_finding(row)
    with pytest.raises(ValueError):
        audit.validate_review(value, material)


def test_exact_source_quote_required_for_contradiction_and_null_allowed_for_absence():
    material = materials()
    value = review(material)
    add_finding(value['records'][1], 'contradiction', 'opened in 1998')
    add_finding(value['records'][4], 'unsupported_claim', None)
    audit.validate_review(value, material)


def test_malformed_response_raw_text_is_preserved_but_unavailable():
    material = materials()
    row = material['records'][10]
    row.update(status='response_schema_uncertain', output_text='Raw text retained for inspection.',
               output_sha256=sha('Raw text retained for inspection.'))
    audit.validate_materials(material)
    value = review(material)
    audit.validate_review(value, material)
    assert value['records'][10]['output_sha256'] == row['output_sha256']
    assert all(flag is None for flag in value['records'][10]['flags'].values())


def test_aggregation_preserves_denominators_and_disagreement_without_adjudication():
    material = materials()
    a, b = review(material, 'A'), review(material, 'B')
    add_finding(a['records'][1], 'unsupported_claim')
    add_finding(b['records'][1], 'unsupported_claim')
    add_finding(a['records'][4], 'unsupported_claim')
    add_finding(b['records'][5], 'contradiction', 'opened in 1998')
    result = audit.aggregate_reviews(a, b, material)
    assert len(result['records']) == 20
    assert result['denominators'] == {'all20': 20, 'planned8': 8, 'available': 7}
    assert result['counts']['unsupported_claim'] == {'union': 2, 'intersection': 1, 'disagreements': 1}
    assert result['counts']['contradiction'] == {'union': 1, 'intersection': 0, 'disagreements': 1}
    assert result['counts']['language_or_entity_error'] == {'union': 0, 'intersection': 0, 'disagreements': 0}


def test_aggregation_requires_two_distinct_reviewer_ids():
    material = materials()
    with pytest.raises(ValueError):
        audit.aggregate_reviews(review(material, 'A'), review(material, 'A'), material)


@pytest.mark.parametrize('field,value', [('schema', 'unbound-review'),
                                        ('review_type', 'external_truth_human_gold'),
                                        ('reviewer_id', 'anonymous')])
def test_review_scope_and_identity_are_not_silently_relabelled(field, value):
    material = materials()
    candidate = review(material)
    candidate[field] = value
    with pytest.raises(ValueError):
        audit.validate_review(candidate, material)
