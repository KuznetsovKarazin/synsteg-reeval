#!/usr/bin/env python3
"""Crosswalk saved morphology decisions to prior reviews; never run a model/codec.

This post-hoc descriptive comparison transfers unchanged historical assistant
ratings only after an exact source/position/whole-class match. It is not a new
review, semantic certification, or an estimate of generalization performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = 'bbe77c1fe81b7bf6242edd49e505830e615b167f'
OLD = 'results/universal-context-development-v1'
MORPH = 'results/morphology-development-v3'
BUILD = 'results/morphology-recovery-v3'
OUTPUT = 'results/morphology-nli-independent-v1/review-crosswalk.json'
FLAGS = ('definite_meaning_change', 'definite_grammar_damage', 'ambiguous_or_register_risk')
FILES = (
    f'{OLD}/inputs.jsonl', f'{OLD}/accepted-slots.jsonl', f'{OLD}/acceptance-review-plan.json',
    f'{OLD}/audit/records.jsonl', f'{OLD}/reviews/reviewer_a.json', f'{OLD}/reviews/reviewer_b.json',
    f'{OLD}/reviews/comparison.json', f'{MORPH}/protocol.json', f'{MORPH}/audit/records.jsonl',
    f'{MORPH}/audit/execution.json', f'{BUILD}/parent-ledger.json', f'{BUILD}/trials.jsonl',
    f'{BUILD}/annotations.json', 'data/shared_dictionary_morphology_v3.json',
    'data/shared_dictionary_recovery_v2.json',
)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def formatted(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def unique(rows, key, description):
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate ' + description)
    return result


def position(row):
    return row['audit_record_id'], row['token_index']


def focus(item):
    tokens = list(re.finditer(r'\S+', item['text']))
    index = item['token_index']
    if type(index) is not int or not 0 <= index < len(tokens):
        raise ValueError('Invalid original focus index')
    token = tokens[index]
    if token.group() != item['original_token']:
        raise ValueError('Original focus literal changed')
    match = re.fullmatch(r'([^A-Za-z]*)([a-z]+)([^A-Za-z]*)', token.group())
    if match is None or match.group(2) not in item['words']:
        raise ValueError('Original lowercase class member not found')
    return match.group(2), token.start() + match.start(2), token.start() + match.end(2)


def verify_review(review, label, materials, materials_hash, plan_hash):
    if (review.get('schema') != 'universal-accepted-slots-review-v1'
            or review.get('review_type') != 'independent_assistant_not_human'
            or review.get('reviewer') != label or review.get('materials_sha256') != materials_hash
            or review.get('plan_sha256') != plan_hash):
        raise ValueError('Historical review provenance changed: ' + label)
    by_id = unique(review['records'], lambda r: r['review_id'], 'review identity')
    if set(by_id) != {m['review_id'] for m in materials}:
        raise ValueError('Historical review must include all30 positions')
    for item in materials:
        row = by_id[item['review_id']]
        original, left, right = focus(item)
        alternatives = unique(row['alternatives'], lambda a: a['word'], 'review alternative')
        if set(alternatives) != set(item['words']):
            raise ValueError('Historical review omitted a complete class member')
        for word, alternative in alternatives.items():
            if any(type(alternative.get(flag)) is not bool for flag in FLAGS):
                raise ValueError('Historical alternative flags must be boolean')
            if type(alternative.get('unchanged')) is not bool or alternative['unchanged'] != (word == original):
                raise ValueError('Historical identity alternative changed')
            if alternative['unchanged'] and any(alternative[flag] for flag in FLAGS):
                raise ValueError('Historical identity must not introduce damage')
            quote, replacement = alternative['quoted_original'], alternative['substituted_phrase']
            if not isinstance(quote, str) or not quote or not isinstance(replacement, str):
                raise ValueError('Nonempty exact original and replacement required')
            candidates = [m.start() for m in re.finditer(re.escape(quote), item['text'])]
            expected = [quote[:left - start] + word + quote[right - start:] for start in candidates
                        if start <= left and right <= start + len(quote)]
            if replacement not in expected:
                raise ValueError('Historical quote/replacement does not match the exact focus')
        if any(type(row.get(flag)) is not bool or row[flag] != any(a[flag] for a in alternatives.values()) for flag in FLAGS):
            raise ValueError('Historical position flags are not the OR of alternatives')
    return by_id


def build(root):
    root = Path(root).resolve()
    raw = {name: (root / name).read_bytes() for name in FILES}
    hashes = {name: sha(value) for name, value in raw.items()}
    read = lambda name: json.loads(raw[name])
    lines = lambda name: [json.loads(line) for line in raw[name].splitlines() if line.strip()]
    protocol = read(f'{MORPH}/protocol.json')
    execution = read(f'{MORPH}/audit/execution.json')
    if (execution['frozen_commit'] != FROZEN_COMMIT
            or execution['protocol_sha256'] != hashes[f'{MORPH}/protocol.json']
            or execution['output_sha256']['records.jsonl'] != hashes[f'{MORPH}/audit/records.jsonl']):
        raise ValueError('Saved morphology execution/freeze binding differs')
    for name, digest in protocol['source_files'].items():
        if name in hashes and hashes[name] != digest:
            raise ValueError('Frozen morphology source changed: ' + name)
    inputs = unique(lines(f'{OLD}/inputs.jsonl'), lambda r: r['audit_record_id'], 'source input')
    records = unique(lines(f'{MORPH}/audit/records.jsonl'), lambda r: r['audit_record_id'], 'morphology record')
    if len(records) != 144 or set(records) != set(inputs):
        raise ValueError('Complete144-record saved morphology denominator required')
    materials = lines(f'{OLD}/accepted-slots.jsonl')
    old_by_position = unique(materials, position, 'old accepted position')
    plan = read(f'{OLD}/acceptance-review-plan.json')
    comparison = read(f'{OLD}/reviews/comparison.json')
    if (len(materials) != 30 or plan['unique_positions'] != 30
            or plan['materials_sha256'] != hashes[f'{OLD}/accepted-slots.jsonl']
            or plan['records_sha256'] != hashes[f'{OLD}/audit/records.jsonl']
            or comparison['materials_sha256'] != hashes[f'{OLD}/accepted-slots.jsonl']
            or comparison['plan_sha256'] != hashes[f'{OLD}/acceptance-review-plan.json']):
        raise ValueError('Complete30-position prior review provenance required')
    old_classes = unique(read('data/shared_dictionary_recovery_v2.json')['classes'], lambda c: c['id'], 'old class')
    classes = unique(read('data/shared_dictionary_morphology_v3.json')['classes'], lambda c: c['id'], 'morphology class')
    annotation_file = read(f'{BUILD}/annotations.json')
    if annotation_file['dictionary_sha256'] != hashes['data/shared_dictionary_morphology_v3.json']:
        raise ValueError('Morphology annotation/dictionary mismatch')
    annotations = unique(annotation_file['classes'], lambda c: c['class_id'], 'morphology annotation')
    ledgers = unique(read(f'{BUILD}/parent-ledger.json'), lambda c: c['parent_id'], 'parent ledger')
    trials = unique(lines(f'{BUILD}/trials.jsonl'), lambda c: c['trial_id'], 'morphology trial')
    reviews = {}
    for label in ('A', 'B'):
        name = f'{OLD}/reviews/reviewer_{label.lower()}.json'
        if comparison['review_sha256'][label] != hashes[name]:
            raise ValueError('Historical review bytes changed')
        reviews[label] = verify_review(read(name), label, materials,
            hashes[f'{OLD}/accepted-slots.jsonl'], hashes[f'{OLD}/acceptance-review-plan.json'])
    for old in materials:
        source, record = inputs[old['audit_record_id']], records[old['audit_record_id']]
        if (old['words'] != old_classes[old['class_id']]['words']
                or old['text'] != source['text'] or sha(old['text'].encode()) != old['text_sha256']
                or any(old[field] != source[field] or old[field] != record[field]
                       for field in ('text_sha256', 'cohort', 'source_group_id'))):
            raise ValueError('Prior complete class or exact source context changed')
        focus(old)
    accepted = []
    accepted_positions = set()
    for record_id, record in records.items():
        analysis = record['methods']['universal_morph_v3']['analysis']
        for decision in (analysis or {}).get('slot_decisions', []):
            if type(decision['accepted']) is not bool:
                raise ValueError('Saved acceptance must be boolean')
            if not decision['accepted']:
                continue
            key = record_id, decision['token_index']
            if key in accepted_positions or key not in old_by_position:
                raise ValueError('Duplicate or unmatched new accepted position; cannot transfer reviews')
            accepted_positions.add(key)
            old = old_by_position[key]
            words, annotation = classes[decision['class_id']]['words'], annotations[decision['class_id']]
            if (words != old['words'] or annotation['parent_id'] != old['class_id']
                    or record['text_sha256'] != old['text_sha256']
                    or inputs[record_id]['availability'] != 'available'):
                raise ValueError('New complete surface class/parent/source differs; cannot transfer reviews')
            original, _, _ = focus(old)
            accepted.append({'audit_record_id': record_id, 'token_index': key[1], 'cohort': old['cohort'],
                'source_group_id': old['source_group_id'], 'text': old['text'], 'text_sha256': old['text_sha256'],
                'old_review_id': old['review_id'], 'old_class_id': old['class_id'],
                'new_class_id': decision['class_id'], 'complete_surface_words': words,
                'original_token': old['original_token'], 'original_word': original,
                'saved_morphology_decision': decision,
                'exact_match': {'record_and_token': True, 'source_text_and_hash': True,
                    'complete_surface_word_class': True, 'annotated_parent_class': True},
                'review_transfer': 'unchanged_historical_assistant_flags_after_exact_match_not_new_review',
                'prior_reviews': {label: reviews[label][old['review_id']] for label in ('A', 'B')}})
    if len(accepted) != 21:
        raise ValueError('All21 accepted morphology positions required')
    excluded = []
    for old in materials:
        if position(old) in accepted_positions:
            continue
        ledger = ledgers[old['class_id']]
        decision_rows = (records[old['audit_record_id']]['methods']['universal_morph_v3']['analysis'] or {}).get('slot_decisions', [])
        at_position = [d for d in decision_rows if d['token_index'] == old['token_index']]
        if (ledger['status'] != 'no_retained_profile' or ledger['output_class_ids'] or at_position
                or any(c['words'] == old['words'] for c in classes.values())
                or any(a['parent_id'] == old['class_id'] for a in annotations.values())):
            raise ValueError('Unexplained removed old position; update descriptive accounting explicitly')
        evidence_trials = [trials[trial_id] for trial_id in ledger['trial_ids']]
        if any(t['parent_id'] != old['class_id'] or t['status'] != 'rejected' for t in evidence_trials):
            raise ValueError('Removed-class trial provenance mismatch')
        excluded.append({'audit_record_id': old['audit_record_id'], 'token_index': old['token_index'],
            'old_review_id': old['review_id'], 'old_class_id': old['class_id'], 'cohort': old['cohort'],
            'source_group_id': old['source_group_id'], 'text_sha256': old['text_sha256'],
            'original_token': old['original_token'], 'complete_old_surface_words': old['words'],
            'status': 'old_position_absent_from_new_dictionary_candidates',
            'cause': 'entire_parent_class_has_no_retained_static_morphology_profile',
            'contextual_rejection': False, 'saved_decisions_at_old_token': at_position,
            'parent_ledger_evidence': ledger, 'rejected_profile_trials': evidence_trials,
            'historical_flags_only_not_transferred': {label: {flag: reviews[label][old['review_id']][flag] for flag in FLAGS}
                                                    for label in ('A', 'B')}})
    if len(excluded) != 9 or len(accepted) + len(excluded) != len(materials):
        raise ValueError('Full30-position old denominator was not partitioned')
    counts = {label: {**{flag: sum(r['prior_reviews'][label][flag] for r in accepted) for flag in FLAGS},
        'any_definite_issue': sum(any(r['prior_reviews'][label][f] for f in FLAGS[:2]) for r in accepted),
        'no_flags_identified_not_certified': sum(not any(r['prior_reviews'][label][f] for f in FLAGS) for r in accepted)}
        for label in ('A', 'B')}
    meaning_cases = [{'audit_record_id': r['audit_record_id'], 'token_index': r['token_index'],
        'old_review_id': r['old_review_id'], 'new_class_id': r['new_class_id'],
        'original_token': r['original_token'], 'complete_surface_words': r['complete_surface_words'],
        'source_text_sha256': r['text_sha256'],
        'exact_historical_evidence': {label: [a for a in r['prior_reviews'][label]['alternatives'] if a['definite_meaning_change']]
                                      for label in ('A', 'B')}}
        for r in accepted if any(r['prior_reviews'][label]['definite_meaning_change'] for label in ('A', 'B'))]
    result = {'schema': 'synsteg-morphology-review-crosswalk-v1', 'frozen_production_commit': FROZEN_COMMIT,
        'scope': 'post_hoc_descriptive_crosswalk_of_saved_development_results_and_prior_assistant_reviews',
        'matching_rule': 'audit_record_id+token_index, exact full source text/hash, complete sorted surface word class and annotated parent class',
        'source_files_sha256': hashes, 'script_sha256': sha(Path(__file__).read_bytes()),
        'denominators': {'saved_morphology_records': 144, 'prior_reviewed_positions': 30,
            'all_new_accepted_positions': 21, 'exact_matches_with_transferred_reviews': 21,
            'new_accepted_positions_without_exact_prior_match': 0, 'old_positions_not_newly_accepted': 9,
            'new_accepted_nonidentity_alternatives': sum(len(r['complete_surface_words']) - 1 for r in accepted)},
        'accepted_cohort_counts': dict(Counter(r['cohort'] for r in accepted)),
        'historical_flag_counts_over_all21_accepted_positions': counts,
        'removed_old_position_counts_by_parent_class': dict(Counter(r['old_class_id'] for r in excluded)),
        'remaining_prior_definite_meaning_cases': meaning_cases,
        'remaining_prior_definite_meaning_case_count': len(meaning_cases),
        'accepted_positions': accepted, 'old_positions_not_newly_accepted': excluded,
        'limitations': ['Ratings are the unchanged prior A/B assistant judgments, not new or human review.',
            'No-flag and no-definite-issue outcomes are not semantic or grammatical certification.',
            'The comparison is post-hoc and feedback-informed; no held-out/generalization or causal improvement estimate is made.',
            'Removed km positions reflect missing static morphology, not a validated contextual number-agreement rule.',
            'Exact quote checks verify provenance and substitution identity, not judgment correctness.'],
        'new_review_performed': False, 'human_validation': False, 'semantic_safety_certified': False,
        'generalization_metrics_reported': False, 'new_model_inference_calls': 0, 'new_capacity_audit_runs': 0,
        'closed_test_opened': False, 'paid_calls': 0}
    if any(sha((root / name).read_bytes()) != digest for name, digest in hashes.items()):
        raise ValueError('Input changed during crosswalk construction')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', action='store_true', help='Recompute and compare existing JSON without changing it')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (args.output or root / OUTPUT).resolve()
    if output.exists() and not args.verify:
        raise FileExistsError('Crosswalk never overwrites saved evidence')
    result = build(root)
    encoded = formatted(result)
    if args.verify:
        if output.read_bytes() != encoded:
            raise ValueError('Saved crosswalk is not byte-identical to source reconstruction')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({'verified' if args.verify else 'created': str(output),
        'sha256': sha(encoded), 'denominators': result['denominators'],
        'historical_flag_counts': result['historical_flag_counts_over_all21_accepted_positions'],
        'remaining_prior_definite_meaning_case_count': result['remaining_prior_definite_meaning_case_count']}, indent=2))


if __name__ == '__main__':
    main()
