#!/usr/bin/env python3
"""Independently check the saved assessment without classifier/generator calls."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import verify_morph_nli_outputs_independent_v1 as independent


ROOT = Path(__file__).resolve().parents[1]
STAGE = 'results/nli-diagnostic-v1'
PRIOR = 'results/universal-context-development-v1'


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(flags):
    definite = [v['meaning'] or v['grammar'] for v in flags.values()]
    if all(definite):
        return 'both_assistants_definite'
    if any(definite):
        return 'one_assistant_definite'
    if any(v['ambiguous'] for v in flags.values()):
        return 'ambiguity_or_register_only'
    return 'no_flag_by_either_not_certified'


def review(root=ROOT):
    assessment = read(root / STAGE / 'assessment.json')
    plan = read(root / STAGE / 'assessment-plan.json')
    # Recalculate from raw outputs using our separate mathematical implementation.
    # This calls neither the production summarizer nor the assessment code.
    nli = independent.validate_nli(root)
    morph = independent.validate_morphology(root)
    natural = read(root / 'results/morphology-nli-independent-v1/natural-results-validation.json')
    assert natural['morphology'] == morph
    # The frozen independent validator enumerates alternative sets; compare
    # identities, not its incidental process-dependent set iteration order.
    assert {k: v for k, v in natural['nli'].items() if k != 'alternative_evidence'} == {k: v for k, v in nli.items() if k != 'alternative_evidence'}
    assert {(r['review_id'], r['alternative_word']): r for r in natural['nli']['alternative_evidence']} == {(r['review_id'], r['alternative_word']): r for r in nli['alternative_evidence']}
    assert assessment['planned_alternatives'] == 52
    assert assessment['planned_positions'] == 30
    assert len(assessment['records']) == 52 and len(assessment['positions']) == 30
    assert len({(r['review_id'], r['alternative_word']) for r in assessment['records']}) == 52
    assert len({r['review_id'] for r in assessment['positions']}) == 30
    materials = {r['review_id']: r for r in rows(root / PRIOR / 'accepted-slots.jsonl')}
    expected = {(r['review_id'], r['alternative_word']): r for r in nli['alternative_evidence']}
    planned = rows(root / STAGE / 'pairs.jsonl')
    identity = {p['review_id']: p['pair_id'] for p in planned if p['is_identity']}
    pair_groups = defaultdict(list)
    for pair in planned:
        if not pair['is_identity']:
            pair_groups[(pair['review_id'], pair['alternative_word'])].append(pair['pair_id'])
    assert set(expected) == set(pair_groups)
    aggregate = {name: {'planned_alternatives': 0, 'accepted': 0, 'outcomes': Counter()} for name in plan['categories']}
    groups = defaultdict(list)
    for saved in assessment['records']:
        key = saved['review_id'], saved['alternative_word']
        check = expected[key]
        flags = check['prior_assistant_flags']
        assert saved['accepted'] is check['accepted']
        assert saved['status'] == check['nli_status']
        assert saved['reference_category'] == category(flags)
        assert saved['identity_pair_id'] == identity[key[0]]
        assert saved['directed_pair_ids'] == pair_groups[key]
        assert saved['cohort'] == materials[key[0]]['cohort']
        assert saved['source_group_id'] == materials[key[0]]['source_group_id']
        for label, names in [('A', 'a'), ('B', 'b')]:
            assert saved['reference_flags'][label] == {
                'definite_meaning_change': flags[names]['meaning'],
                'definite_grammar_damage': flags[names]['grammar'],
                'ambiguous_or_register_risk': flags[names]['ambiguous'],
            }
        item = aggregate[category(flags)]
        item['planned_alternatives'] += 1
        item['accepted'] += int(check['accepted'])
        item['outcomes'][check['nli_status']] += 1
        groups[key[0]].append(check)
    for item in aggregate.values():
        item['outcomes'] = dict(item['outcomes'])
    assert assessment['reference_categories'] == aggregate
    assert set(groups) == set(materials)
    for position in assessment['positions']:
        rid = position['review_id']
        assert position['cohort'] == materials[rid]['cohort']
        assert position['all_observed_original_comparisons_accepted'] is all(r['accepted'] for r in groups[rid])
        assert position['has_both_definite_reference'] is any(category(r['prior_assistant_flags']) == 'both_assistants_definite' for r in groups[rid])
    assert assessment['position_all_member_passes'] == 28
    assert assessment['position_all_member_passes_with_definite_reference'] == 8
    assert assessment['accepted_both_definite_counterexamples'] == nli['accepted_despite_both_assistants_definite'] == 8
    assert assessment['score_summary_recomputed_exactly'] is True
    for name in ('human_validation', 'semantic_safety_certified', 'receiver_invariant_codec_gate', 'threshold_tuning_performed'):
        assert assessment[name] is False
    for name, sha in assessment['inputs_sha256'].items():
        assert digest(root / name) == sha
    for name, sha in plan['reference_files'].items():
        assert digest(root / name) == sha
    assert digest(root / 'scripts/assess_nli_diagnostic_v1.py') == plan['analysis_code_sha256']
    summary = read(root / 'results/morphology-development-v3/audit/summary.json')
    excerpt = summary['cohorts']['source_excerpts']
    exposure = {
        'all_positions': len(materials),
        'distinct_source_groups': len({r['source_group_id'] for r in materials.values()}),
        'cohort_positions': dict(Counter(r['cohort'] for r in materials.values())),
    }
    evidence = [
        'scripts/review_morph_nli_actual_v1.py',
        'scripts/verify_morph_nli_outputs_independent_v1.py',
        'results/morphology-nli-independent-v1/natural-results-validation.json',
        'results/morphology-development-v3/protocol.json',
        'results/morphology-development-v3/audit/records.jsonl',
        'results/morphology-development-v3/audit/summary.json',
        'results/morphology-development-v3/audit/execution.json',
        STAGE + '/protocol.json', STAGE + '/pairs.jsonl',
        STAGE + '/scores/pairs.jsonl', STAGE + '/scores/summary.json',
        STAGE + '/scores/execution.json', STAGE + '/assessment-plan.json', STAGE + '/assessment.json',
        *plan['reference_files'],
    ]
    return {
        'schema': 'synsteg-independent-actual-review-v1',
        'status': 'all_saved_aggregates_match_independent_calculation',
        'frozen_commit': nli['freeze_commit'],
        'evidence_sha256': {name: digest(root / name) for name in evidence},
        'morphology': {k: v for k, v in morph.items()},
        'all100_excerpt_capacity_comparison': {
            'unit': 'gross_slot_bits_before_length_or_authentication_framing',
            'v2_base_mean': excerpt['reference_capacity']['base_v2']['mean'],
            'v2_guarded_mean': excerpt['reference_capacity']['universal_v2']['mean'],
            'v3_base_mean': excerpt['reference_capacity']['base_morph_v3']['mean'],
            'v3_guarded_mean': excerpt['new_capacity']['mean'],
            'relative_loss_from_v2_base': 1 - 17 / 28,
            'relative_loss_from_v2_guarded': 1 - 17 / 25,
            'zero_capacity_denominator100': 85,
        },
        'nli': {k: v for k, v in nli.items() if k != 'alternative_evidence'},
        'assessment': {
            'all_alternative_records_independently_matched': 52,
            'all_position_records_independently_matched': 30,
            'reference_categories': aggregate,
            'position_all_member_passes': 28,
            'position_all_member_passes_with_definite_reference': 8,
            'exposure': exposure,
        },
        'interpretation': [
            'Morphology capacity loss is measured on every 100 development excerpts, including zeros; 44 reused covers are descriptive dependent cohorts, not 44 independent new sources.',
            'The 0.17 bits per excerpt are gross eligible slot bits, not successfully delivered payload; all 432 framed arms abstained or lacked capacity.',
            '323 saved alias comparisons (45 individual, 278 simultaneous) match; this review checks their evidence and does not rerun them. This does not establish semantic preservation.',
            'Matching complete static inflection tables and POS spectra prevents specified morphology ambiguities, not contextual meaning change, countability errors or all grammar failures.',
            'The fixed NLI criterion accepted all 8 alternatives with a definite problem in both prior assistant reviews. These are diagnostic counterexamples to relying on this criterion as a safety screen; labels are not human gold.',
            'The 3 nonaccepted alternatives are length-related identity-control abstentions, not successful detection of semantic problems.',
            'The 52 alternatives share 30 positions and source groups; these are not independent samples. This selected diagnostic cannot estimate general semantic error rates.',
            'Observed-original bidirectional comparisons with identity controls are not all ordered class-member comparisons or full joint sentence-orbit comparisons. No receiver-invariant NLI codec is demonstrated.',
            'The two prior assistant reviews may have correlated errors. Unflagged alternatives are not certified safe; current conclusions need human validation before deployment claims.',
            'Model/tokenizer/runtime hashes identify the local CPU execution. Cross-hardware exact reproducibility is not claimed; no model replay is performed by this review.',
        ],
        'new_classifier_calls': 0, 'new_generator_calls': 0,
        'paid_calls': 0, 'closed_test_opened': False,
        'threshold_tuning': False, 'new_human_or_assistant_semantic_judgments': False,
        'semantic_safety_certified': False,
    }


if __name__ == '__main__':
    output = ROOT / 'results/morphology-nli-independent-v1/actual-review.json'
    if output.exists():
        raise FileExistsError('Independent review evidence is never overwritten')
    report = review()
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'sha256': digest(output), 'assessment': report['assessment']}, indent=2))
