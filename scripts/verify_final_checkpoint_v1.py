#!/usr/bin/env python3
"""Offline preservation/replay check; no model generation or network calls.

Run from a complete repository checkout with cryptography installed.
Prints fresh verification JSON; does not overwrite historical evidence.
"""
from pathlib import Path
import hashlib
import json
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from synsteg_audit.shared_dictionary import PublicDictionary, Protocol, analyze_text
import audit_invariant_context_development as audit
import restore_recovery_cloud_state as restore


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    captured = read(root / 'results/final-checkpoint-20261024-0816/capture-manifest.json')
    for row in captured['files']:
        path = root / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    out = root / 'results/grounded-development-v2'
    for name, row in read(out / 'generation_snapshot_manifest.json')['files'].items():
        assert sha(out / name) == row['sha256'], name
    plan = read(out / 'run/run.json')['selected_records']
    prompts = {r['record_id']: r for r in map(json.loads, (out / 'prompts.jsonl').read_text().splitlines())}
    attempts = {}
    for path in (out / 'run/attempts').glob('*.json'):
        item = read(path)
        rid = item['record']['record_id']
        assert rid not in attempts
        attempts[rid] = (item, path)
    assert len(plan) == len(prompts) == len(attempts) == 20
    assert {r['record_id'] for r in plan} == set(attempts) == set(prompts)
    summary = read(out / 'mechanical_summary.json')
    dictionaries = {v: PublicDictionary.from_dict(read(root / f'data/shared_dictionary_recovery_{v}.json')) for v in ('v1', 'v2')}
    proto = Protocol(min_group_size=1, max_bits_per_slot=1, framing='plain_length16')
    caps = {v: [] for v in dictionaries}
    words, band = [], 0
    assert len(summary['records']) == 20
    for planned, row in zip(plan, summary['records']):
        rid = planned['record_id']; attempt, path = attempts[rid]
        assert planned['split'] == 'development' and attempt['record'] == planned
        assert attempt['status'] == 'completed' and row['record_id'] == rid
        assert row['attempt_sha256'] == sha(path)
        text = attempt['result']['cover_text']; prompt = prompts[rid]
        assert row['cover_sha256'] == hashlib.sha256(text.encode()).hexdigest()
        count = len(text.split()); words.append(count)
        assert row['cover_words'] == count and row['source_words'] == len(prompt['excerpt'].split())
        inside = prompt['preferred_word_band'][0] <= count <= prompt['preferred_word_band'][1]
        assert row['within_soft_word_band'] == inside
        band += inside
        for v, dictionary in dictionaries.items():
            value = analyze_text(text, dictionary, proto).capacity_bits
            assert row['capacity_bits'][v] == value
            caps[v].append(value)
    assert band == summary['within_soft_band'] == 11
    assert sum(words) / 20 == summary['mean_cover_words']
    for v, values in caps.items():
        assert sum(values) / 20 == summary['dictionary_capacity'][v]['mean_bits_available']
        assert max(values) < 24
    comparison = read(out / 'review_comparison.json')
    for name, expected in comparison['review_hashes'].items():
        assert sha(out / name) == expected
    # These checks preserve earlier judgments; they are not a new factual review.
    assert len(comparison['both_flagged_unsupported_or_contradiction']) == 12
    assert len(comparison['by_record_flag']['contradiction']['both_flagged']) == 7
    context = root / 'results/invariant-context-development-v1'
    audit.verify_freeze(root, context)
    inputs = [json.loads(line) for line in (context / 'inputs.jsonl').read_text().splitlines()]
    collected, _ = audit.collect_inputs(root)
    assert inputs == collected
    expected = [json.loads(line) for line in (context / 'evaluation/records.jsonl').read_text().splitlines()]
    assert len(inputs) == len(expected) == 124
    dictionary = PublicDictionary.from_dict(read(context / 'public_dictionary.json'))
    policies = {m: audit.guarded.ProtectedPhrasePolicy.from_dict(read(context / file), dictionary)
                for m, file in [('wrapper_guard', 'guard_policy.json'), ('wrapper_empty', 'empty_policy.json')]}
    replay = [audit.evaluate_record(root, row, dictionary, policies) for row in inputs]
    assert replay == expected, 'Context replay differs from saved complete records'
    saved_summary = read(context / 'evaluation/summary.json')
    for name, value in audit.summarize(replay).items():
        assert saved_summary[name] == value, name
    assert saved_summary['records_sha256'] == sha(context / 'evaluation/records.jsonl')
    assert saved_summary['protocol_sha256'] == sha(context / 'protocol.json')
    arms = [a for r in replay for m in r['methods'].values() for a in m['arms'].values()]
    statuses = dict(Counter(a['status'] for a in arms))
    restorations = {}
    for name in ('grounded', 'length-probe', 'baseline'):
        manifest = root / f'results/cloud-continuation-2026-10-24/{name}-restore-manifest.json'
        restorations[name] = restore.operate(manifest, repo=root, restore=False)
    return {'schema': 'synsteg-final-checkpoint-verification-v1', 'status': 'passed',
            'captured_files_verified': len(captured['files']), 'grounded_attempts_verified': 20,
            'grounded_capacity_comparisons': 40, 'within_soft_word_band': band,
            'context_records_replayed_exactly': len(replay), 'context_arms': len(arms),
            'context_statuses': statuses, 'restore_verification': restorations,
            'new_model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False,
            'factual_reviews': 'preserved assistant judgments; not re-rated, human validated or population estimates'}


if __name__ == '__main__':
    print(json.dumps(verify(ROOT), ensure_ascii=False, indent=2))
