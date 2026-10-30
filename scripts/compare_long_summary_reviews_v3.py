#!/usr/bin/env python3
"""Compare retained independent assistant diagnostics; never certify quality."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/long-summary-development-v3'
FLAGS = ('unsupported_claim', 'contradiction', 'language_or_entity_error', 'ambiguous_or_borderline')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    material = OUT / 'mechanical-audit/review-materials.jsonl'
    materials = [json.loads(line) for line in material.read_text().splitlines()]
    assert len(materials) == 20 and sum(r['availability'] == 'available' for r in materials) == 15
    reviews = {}
    for label in ('a', 'b'):
        path = OUT / f'reviews/reviewer_{label}.json'
        review = json.loads(path.read_text())
        assert review['materials_sha256'] == sha(material)
        assert review['rubric_sha256'] == sha(OUT / 'review_rubric.json')
        assert len(review['records']) == 20
        for m, row in zip(materials, review['records']):
            assert (m['plan_index'], m['record_id'], m['availability'], m['cover_sha256']) == (
                row['plan_index'], row['record_id'], row['availability'], row['cover_sha256'])
            if m['availability'] != 'available':
                assert row['flags'] is None and not row['issues']
            else:
                assert set(row['flags']) == set(FLAGS)
                assert all(type(value) is bool for value in row['flags'].values())
        reviews[label] = review
    flagged = lambda review, flag: {r['plan_index'] for r in review['records'] if r['flags'] and r['flags'][flag]}
    comparisons = {}
    for flag in FLAGS:
        a, b = (flagged(reviews[label], flag) for label in ('a', 'b'))
        comparisons[flag] = {'reviewer_a': sorted(a), 'reviewer_b': sorted(b),
            'both': sorted(a & b), 'a_only': sorted(a - b), 'b_only': sorted(b - a)}
    a, b = (flagged(reviews[label], 'unsupported_claim') | flagged(reviews[label], 'contradiction') for label in ('a', 'b'))
    report = {'schema': 'synsteg-long-summary-v3-review-comparison', 'planned_sources': 20,
        'planned_generations': 15, 'available_outputs_reviewed': 15, 'source_abstentions_not_rated': 5,
        'materials_sha256': sha(material), 'review_hashes': {label: sha(OUT / f'reviews/reviewer_{label}.json') for label in ('a', 'b')},
        'by_flag': comparisons, 'both_flagged_unsupported_or_contradiction': sorted(a & b),
        'either_flagged_unsupported_or_contradiction': sorted(a | b),
        'review_type': 'Two independent source-bound assistant diagnostics; not human validation or external truth checking.',
        'disagreements_preserved': True, 'no_flag_is_quality_certification': False,
        'population_inference': False, 'causal_comparison_to_v2': False}
    path = OUT / 'reviews/comparison.json'
    with path.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
