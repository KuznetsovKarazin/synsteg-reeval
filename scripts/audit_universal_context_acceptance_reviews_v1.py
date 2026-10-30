#!/usr/bin/env python3
"""Verify all accepted-slot materials and compare two unaltered assistant reviews.

Exact evidence checks establish provenance/coverage, not judgment correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synsteg_audit import shared_dictionary as base

STAGE = 'results/universal-context-development-v1'
FLAGS = ('definite_meaning_change', 'definite_grammar_damage', 'ambiguous_or_register_risk')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(s) for s in path.read_bytes().splitlines() if s.strip()]


def rebuild_materials(root):
    stage = root / STAGE
    inputs = {r['audit_record_id']: r for r in lines(stage / 'inputs.jsonl')}
    dictionaries = {v: {c['id']: c['words'] for c in read(root / f'data/shared_dictionary_recovery_{v}.json')['classes']} for v in ('v1', 'v2')}
    accepted = []
    for row in lines(stage / 'audit/records.jsonl'):
        source = inputs[row['audit_record_id']]
        groups = {}
        for method, result in row['methods'].items():
            if not result['analysis']:
                continue
            for decision in result['analysis']['slot_decisions']:
                if not decision['accepted']:
                    continue
                key = (decision['token_index'], decision['class_id'])
                version = method.split('_')[-1]
                words = dictionaries[version][decision['class_id']]
                group = groups.setdefault(key, {'token_index': key[0], 'class_id': key[1], 'words': words, 'methods': []})
                if group['words'] != words:
                    raise ValueError('Methods disagree about full class members')
                group['methods'].append(method)
        for (index, class_id), group in sorted(groups.items()):
            tokens = list(re.finditer(r'\S+', source['text']))
            accepted.append({'review_id': f"{row['audit_record_id']}@{index}:{class_id}",
                'audit_record_id': row['audit_record_id'], 'cohort': row['cohort'],
                'source_group_id': row['source_group_id'], 'text': source['text'],
                'text_sha256': source['text_sha256'], 'original_token': tokens[index].group(), **group})
    return accepted


def verify_review(data, label, materials, plan_hash, materials_hash, dictionary):
    if (data.get('schema') != 'universal-accepted-slots-review-v1' or data.get('reviewer') != label
            or data.get('review_type') != 'independent_assistant_not_human'
            or data.get('materials_sha256') != materials_hash or data.get('plan_sha256') != plan_hash):
        raise ValueError('Review provenance mismatch: ' + label)
    rows = data.get('records')
    if not isinstance(rows, list) or len(rows) != len(materials):
        raise ValueError('Review omitted planned positions')
    by_id = {r['review_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {m['review_id'] for m in materials}:
        raise ValueError('Missing/duplicate/unexpected review identity')
    quote_checks = 0
    for item in materials:
        row = by_id[item['review_id']]
        raw = base.analyze_text(item['text'], dictionary, base.Protocol(1, 1, 'plain_length16'))
        slot = next(s for s in raw.slots if s.token_index == item['token_index'])
        if slot.class_id != item['class_id'] or slot.surface.lower() not in item['words']:
            raise ValueError('Focus slot changed')
        alternatives = row.get('alternatives')
        if (not isinstance(alternatives, list) or len(alternatives) != len(item['words'])
                or {a['word'] for a in alternatives} != set(item['words'])):
            raise ValueError('Every complete class member must be rated')
        for alternative in alternatives:
            word = alternative['word']
            if any(type(alternative.get(flag)) is not bool for flag in FLAGS):
                raise ValueError('Alternative flags must be booleans')
            unchanged = word == slot.surface
            if type(alternative.get('unchanged')) is not bool or alternative['unchanged'] != unchanged:
                raise ValueError('Identity alternative incorrectly labeled')
            if unchanged and any(alternative[f] for f in FLAGS):
                raise ValueError('Identity alternative cannot introduce substitution damage')
            quote = alternative.get('quoted_original')
            substitute = alternative.get('substituted_phrase')
            if not isinstance(quote, str) or not quote or not isinstance(substitute, str):
                raise ValueError('Nonempty exact quote and substituted phrase required')
            candidates = [m.start() for m in re.finditer(re.escape(quote), item['text'])]
            expected = [quote[:slot.start - start] + word + quote[slot.end - start:]
                        for start in candidates if start <= slot.start and slot.end <= start + len(quote)]
            if substitute not in expected:
                raise ValueError('Quote or focus replacement mismatch: ' + item['review_id'] + ' ' + word)
            if not isinstance(alternative.get('explanation'), str) or not alternative['explanation'].strip():
                raise ValueError('Missing alternative explanation')
            quote_checks += 1
        if any(type(row.get(flag)) is not bool or row[flag] != any(a[flag] for a in alternatives) for flag in FLAGS):
            raise ValueError('Position flags must equal OR of alternative judgments')
    return by_id, quote_checks


def compare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Review comparisons never overwrite existing evidence')
    stage = root / STAGE
    plan_bytes = (stage / 'acceptance-review-plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    materials_bytes = (stage / 'accepted-slots.jsonl').read_bytes()
    rebuilt = rebuild_materials(root)
    rebuilt_bytes = ''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n' for x in rebuilt).encode()
    if (rebuilt_bytes != materials_bytes or sha(materials_bytes) != plan['materials_sha256']
            or sha((stage / 'audit/records.jsonl').read_bytes()) != plan['records_sha256']
            or len(rebuilt) != plan['unique_positions']
            or sum(len(r['methods']) for r in rebuilt) != plan['method_position_occurrences']):
        raise ValueError('Accepted-position materials do not exactly reconstruct from complete saved audit')
    dictionary = base.PublicDictionary.from_dict(read(root / 'data/shared_dictionary_recovery_v2.json'))
    reviews, checked, hashes = {}, {}, {}
    for label in ('A', 'B'):
        p = stage / f'reviews/reviewer_{label.lower()}.json'
        data = p.read_bytes()
        hashes[label] = sha(data)
        reviews[label], checked[label] = verify_review(json.loads(data), label, rebuilt, sha(plan_bytes), sha(materials_bytes), dictionary)
    records = []
    for item in rebuilt:
        rid = item['review_id']
        ratings = {label: {f: reviews[label][rid][f] for f in FLAGS} for label in ('A', 'B')}
        any_definite = {label: ratings[label]['definite_meaning_change'] or ratings[label]['definite_grammar_damage'] for label in ('A', 'B')}
        records.append({'review_id': rid, 'cohort': item['cohort'], 'methods': item['methods'],
            'ratings': ratings, 'any_definite_issue': any_definite,
            'both_definite': all(any_definite.values()), 'either_definite': any(any_definite.values()),
            'disagreed_flags': [f for f in FLAGS if ratings['A'][f] != ratings['B'][f]]})
    summary = {'schema': 'universal-accepted-slot-review-comparison-v1', 'unique_positions': len(records),
        'method_position_occurrences': sum(len(r['methods']) for r in records),
        'distinct_source_groups': len({r['source_group_id'] for r in rebuilt}),
        'materials_rebuilt_byte_identical': True, 'materials_sha256': sha(materials_bytes),
        'plan_sha256': sha(plan_bytes), 'review_sha256': hashes, 'verified_quote_and_replacement_pairs': checked,
        'reviewer_counts': {label: {**{f: sum(r['ratings'][label][f] for r in records) for f in FLAGS},
            'any_definite_issue': sum(r['any_definite_issue'][label] for r in records)} for label in ('A', 'B')},
        'both_definite': sum(r['both_definite'] for r in records), 'either_definite': sum(r['either_definite'] for r in records),
        'positions_with_any_disagreement': sum(bool(r['disagreed_flags']) for r in records),
        'cohorts': {cohort: {'positions': sum(r['cohort'] == cohort for r in records),
            'both_definite': sum(r['cohort'] == cohort and r['both_definite'] for r in records),
            'either_definite': sum(r['cohort'] == cohort and r['either_definite'] for r in records)} for cohort in sorted({r['cohort'] for r in records})},
        'human_validation': False, 'external_truth_check': False, 'semantic_safety_certified': False,
        'independent_population_observations': False, 'judgment_correctness_validated_by_quote_check': False,
        'records': records}
    with output.open('xb') as stream:
        stream.write((json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode())
    return {k: v for k, v in summary.items() if k != 'records'}


def detail_summary(root, output):
    """Retain alternative-level disagreement hidden by position-level OR."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Details never overwrite existing evidence')
    stage = root / STAGE
    materials = lines(stage / 'accepted-slots.jsonl')
    comparison = read(stage / 'reviews/comparison.json')
    reviews = {label: read(stage / f'reviews/reviewer_{label.lower()}.json') for label in ('A', 'B')}
    if any(sha((stage / f'reviews/reviewer_{label.lower()}.json').read_bytes()) != comparison['review_sha256'][label] for label in reviews):
        raise ValueError('Reviews changed after exact evidence verification')
    if sha((stage / 'accepted-slots.jsonl').read_bytes()) != comparison['materials_sha256']:
        raise ValueError('Materials changed after verification')
    by_label = {label: {r['review_id']: r for r in value['records']} for label, value in reviews.items()}
    differences = []
    for material in materials:
        rid = material['review_id']
        alternatives = {label: {a['word']: a for a in rows[rid]['alternatives']} for label, rows in by_label.items()}
        for word in material['words']:
            flags = [f for f in FLAGS if alternatives['A'][word][f] != alternatives['B'][word][f]]
            if flags:
                differences.append({'review_id': rid, 'word': word, 'disagreed_flags': flags,
                    'ratings': {label: {f: alternatives[label][word][f] for f in FLAGS} for label in alternatives}})
    details = {'schema': 'universal-accepted-slot-review-details-v1', 'comparison_sha256': sha((stage / 'reviews/comparison.json').read_bytes()),
        'alternative_flag_counts': {label: {f: sum(a[f] for r in rows.values() for a in r['alternatives']) for f in FLAGS} for label, rows in by_label.items()},
        'alternative_disagreements': differences, 'alternative_disagreement_count': len(differences),
        'methods': {method: {'accepted_positions': sum(method in m['methods'] for m in materials),
            'both_definite_positions': sum(method in r['methods'] and r['both_definite'] for r in comparison['records'])}
            for method in ('universal_v1', 'universal_v2')},
        'human_validation': False, 'judgment_correctness_verified': False}
    with output.open('xb') as stream:
        stream.write((json.dumps(details, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode())
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--details-output', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.root, args.output or args.root / STAGE / 'reviews/comparison.json'), ensure_ascii=False, indent=2))
    if args.details_output:
        print(json.dumps(detail_summary(args.root, args.details_output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
