#!/usr/bin/env python3
"""Verify retained evidence identities/quotes; no semantic truth certification."""
from pathlib import Path
import hashlib
import json
import tempfile
import audit_long_summary_development_v3 as audit
import prepare_long_summary_development_v3 as prep

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / prep.OUTPUT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    prep.verify(ROOT, OUT)
    for row in read(OUT / 'generation-completion-manifest.json')['files']:
        path = OUT / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    with tempfile.TemporaryDirectory() as directory:
        replay = Path(directory) / 'audit'
        audit.execute(ROOT, OUT, replay)
        for name in ('records.jsonl', 'review-materials.jsonl', 'summary.json'):
            assert (replay / name).read_bytes() == (OUT / 'mechanical-audit' / name).read_bytes()
    records = {r['record_id']: r for r in audit.jsonlines(OUT / 'mechanical-audit/records.jsonl')}
    materials = audit.jsonlines(OUT / 'mechanical-audit/review-materials.jsonl')
    normalized = lambda text: ' '.join(text.split())
    quote_fields = 0
    for label in ('a', 'b'):
        review = read(OUT / f'reviews/reviewer_{label}.json')
        assert review['materials_sha256'] == sha(OUT / 'mechanical-audit/review-materials.jsonl')
        assert review['rubric_sha256'] == sha(OUT / 'review_rubric.json')
        assert len(review['records']) == len(materials) == 20
        for row, item in zip(review['records'], materials):
            for key in ('plan_index', 'record_id', 'cover_sha256', 'availability'):
                assert row[key] == item[key]
            for issue in row['issues']:
                for field, text in (('cover_quote', item['cover_text'] or ''), ('source_quote', item['source_window'])):
                    quote = issue.get(field, '')
                    if quote:
                        quote_fields += 1
                        assert normalized(quote) in normalized(text), (label, row['plan_index'], field)
    stego = read(OUT / 'stego-review.json')
    for name, value in stego['input_sha256'].items():
        assert sha(ROOT / name) == value
    assert len(stego['arms']) == 180
    seen, changed = set(), 0
    for arm in stego['arms']:
        key = (arm['record_id'], arm['method'], arm['arm'])
        assert key not in seen
        seen.add(key)
        row = records[arm['record_id']]
        original = row['methods'][arm['method']]['arms'][arm['arm']]
        assert arm['mechanical_status'] == original['status']
        assert arm['encoding_called'] == original['encoding_called']
        if not arm['encoding_called']:
            assert arm['definite_new_substitution_damage'] is None
            continue
        cover, text = row['cover_text'], original['stego_text']
        assert hashlib.sha256(text.encode()).hexdigest() == arm['stego_sha256']
        before, after = cover.split(), text.split()
        assert len(before) == len(after)
        actual = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert actual == [p['token_index_zero_based'] for p in arm['changed_token_pairs']]
        for pair in arm['changed_token_pairs']:
            i = pair['token_index_zero_based']; changed += 1
            assert pair['cover_token'] == before[i] and pair['stego_token'] == after[i]
            assert normalized(pair['cover_quote']) in normalized(cover)
            assert normalized(pair['stego_quote']) in normalized(text)
    print(json.dumps({'schema': 'synsteg-long-summary-v3-evidence-verification', 'status': 'passed',
        'mechanical_records_replayed': 20, 'mechanical_arms_replayed': 180,
        'review_records_checked': 40, 'factual_review_nonempty_quote_fields': quote_fields,
        'stego_changed_token_occurrences_verified': changed, 'new_model_calls': 0,
        'semantic_judgments_or_external_truth_certified': False}, indent=2))


if __name__ == '__main__':
    main()
