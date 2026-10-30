#!/usr/bin/env python3
"""Independently aggregate saved morphology and NLI outputs; no model calls.

This intentionally does not call either new evaluator or NLI summarizer.
It shares only the frozen base tokenizer for static dictionary positions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synsteg_audit import shared_dictionary as base


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(s) for s in path.read_bytes().splitlines() if s.strip()]


def literal(token):
    match = re.fullmatch(r'([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)', token)
    if match is None or any(c == '_' or not unicodedata.category(c).startswith('P') for c in match[1] + match[3]):
        return None
    return match[2].lower()


def direct_gate(text, dictionary):
    raw = base.analyze_text(text, dictionary, base.Protocol(1, 1, 'aes_siv'))
    tokens = text.split()
    slots = {s.token_index: s for s in raw.slots}
    classes = {c.class_id: c.words for c in dictionary.classes}
    decisions = []
    for slot in raw.slots:
        i = slot.token_index
        start, stop = max(0, i - 1), min(len(tokens), i + 2)
        size = math.prod(len(classes[slots[j].class_id]) if j in slots else 1 for j in range(start, stop))
        previous = classes[slots[i - 1].class_id] if i - 1 in slots else (tokens[i - 1],) if i else ()
        if size > 4096:
            reason = 'orbit_cap_exceeded'
        elif slot.case_style != 'lower':
            reason = 'case_not_lowercase'
        elif any(literal(w) in ('a', 'an') for w in previous):
            reason = 'adjacent_indefinite_article'
        else:
            reason = 'accepted'
        decisions.append((i, slot.class_id, size, reason == 'accepted', reason))
    return raw.capacity_bits, sum(d[3] for d in decisions), decisions


def validate_morphology(root):
    stage = root / 'results/morphology-development-v3'
    frozen = read(stage / 'protocol.json')
    inputs = lines(root / 'results/universal-context-development-v1/inputs.jsonl')
    output = lines(stage / 'audit/records.jsonl')
    summary = read(stage / 'audit/summary.json')
    execution = read(stage / 'audit/execution.json')
    assert len(inputs) == len(output) == 144
    assert Counter(x['cohort'] for x in inputs) == {'source_excerpts': 100, 'historical_covers': 4, 'grounded_covers': 20, 'long_summary_covers': 20}
    dictionary = base.PublicDictionary.from_dict(read(root / 'data/shared_dictionary_morphology_v3.json'))
    reference = base.PublicDictionary.from_dict(read(root / 'data/shared_dictionary_recovery_v2.json'))
    results = {r['audit_record_id']: r for r in output}
    assert len(results) == 144 and set(results) == {r['audit_record_id'] for r in inputs}
    outcomes, capacities, checks = Counter(), {}, 0
    alias_individual = alias_joint = 0
    for source in inputs:
        result = results[source['audit_record_id']]
        assert result['cohort'] == source['cohort'] and result['availability'] == source['availability']
        assert result['text_sha256'] == source['text_sha256']
        method = result['methods']['universal_morph_v3']
        assert set(result['methods']) == {'universal_morph_v3'}
        assert len(method['arms']) == 3
        if source['availability'] == 'available':
            raw, capacity, decisions = direct_gate(source['text'], dictionary)
            oldraw, oldguarded, _ = direct_gate(source['text'], reference)
            assert method['capacity_bits'] == capacity
            assert result['capacity_only_references']['base_morph_v3']['capacity_bits'] == raw
            assert result['capacity_only_references']['base_v2']['capacity_bits'] == oldraw
            assert result['capacity_only_references']['universal_v2']['capacity_bits'] == oldguarded
            actual = [(d['token_index'], d['class_id'], d['orbit_size'], d['accepted'], d['reason']) for d in method['analysis']['slot_decisions']]
            assert actual == decisions
            checks += len(decisions)
            capacities.setdefault(source['cohort'], []).append(capacity)
            alias = method['alias_audit']
            alias_individual += len(alias['individual_checks'])
            alias_joint += len(alias['simultaneous_checks'])
            assert alias['all_equal'] and all(c['equal'] for k in ('individual_checks', 'simultaneous_checks') for c in alias[k])
            for arm in method['arms'].values():
                if capacity < arm['required_frame_bits']:
                    assert arm['status'] == 'insufficient_capacity' and not arm['encoding_called']
                else:
                    assert arm['encoding_called']
        else:
            assert method['capacity_bits'] is None
            assert all(a['status'] == source['availability'] and not a['encoding_called'] for a in method['arms'].values())
        outcomes.update(a['status'] for a in method['arms'].values())
    assert sum(outcomes.values()) == 432 and summary['all_outcomes'] == dict(outcomes)
    for cohort, values in capacities.items():
        actual = summary['cohorts'][cohort]['new_capacity']
        assert actual == {'count': len(values), 'min': min(values), 'max': max(values), 'mean': statistics.mean(values), 'zero_capacity': values.count(0)}
    for name, digest in execution['output_sha256'].items():
        assert sha((stage / 'audit' / name).read_bytes()) == digest
    return {'records': 144, 'available_texts': sum(r['availability'] == 'available' for r in inputs),
        'arms': 432, 'outcomes': dict(outcomes), 'slot_decisions_checked': checks,
        'new_capacity_pairs_checked': sum(len(v) for v in capacities.values()),
        'saved_alias_individual_evidence_checked': alias_individual, 'saved_alias_joint_evidence_checked': alias_joint,
        'alias_checks_rerun': False, 'independent_capacity': {k: {'count': len(v), 'mean': statistics.mean(v), 'max': max(v), 'zero': v.count(0)} for k, v in capacities.items()},
        'freeze_commit': execution['frozen_commit'], 'protocol_sha256': sha((stage / 'protocol.json').read_bytes())}


def softmax(values):
    assert isinstance(values, list) and len(values) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in values)
    weights = [math.exp(v - max(values)) for v in values]
    denom = sum(weights)
    return dict(zip(('contradiction', 'entailment', 'neutral'), (w / denom for w in weights)))


def validate_nli(root):
    stage = root / 'results/nli-diagnostic-v1'
    plan = lines(stage / 'pairs.jsonl')
    results = lines(stage / 'scores/pairs.jsonl')
    summary = read(stage / 'scores/summary.json')
    execution = read(stage / 'scores/execution.json')
    protocol = read(stage / 'protocol.json')
    assert len(plan) == len(results) == 134
    assert protocol['thresholds'] == {'entailment_min': .90, 'contradiction_max': .05}
    actual = {r['pair_id']: r for r in results}
    assert len(actual) == 134 and set(actual) == {p['pair_id'] for p in plan}
    scores = {}
    for p in plan:
        r = actual[p['pair_id']]
        for key in ('review_id', 'cohort', 'source_group_id', 'alternative_word', 'original_word', 'direction', 'is_identity', 'premise_sha256', 'hypothesis_sha256', 'source_text_sha256', 'pair_tokens', 'tokenization_sha256', 'planned_status'):
            assert r[key] == p[key]
        assert p['premise_sha256'] == sha(p['premise'].encode()) and p['hypothesis_sha256'] == sha(p['hypothesis'].encode())
        if p['pair_tokens'] > 512:
            assert r['status'] == 'abstain_pair_too_long' and p['planned_status'] == 'abstain_pair_too_long'
        if r['status'] == 'scored':
            assert p['pair_tokens'] <= 512 and p['planned_status'] == 'score_once'
            prob = softmax(r['logits'])
            assert prob == r['probabilities']
            accepted = prob['entailment'] >= .90 and prob['contradiction'] <= .05
            assert r['passes_thresholds'] == accepted and r['classifier_call_attempted'] is True
            assert r['predicted_label'] == ('contradiction', 'entailment', 'neutral')[max(range(3), key=lambda i: r['logits'][i])]
            scores[p['pair_id']] = accepted
        else:
            assert r['status'] in ('abstain_pair_too_long', 'classifier_error', 'model_load_error')
            assert r['logits'] is None and r['probabilities'] is None and r['passes_thresholds'] is False
            assert r['classifier_call_attempted'] is (r['status'] == 'classifier_error')
            scores[p['pair_id']] = False
    by_position = {}
    for p in plan:
        by_position.setdefault(p['review_id'], []).append(p)
    alternatives = {}
    for rid, pairs in by_position.items():
        identities = [p for p in pairs if p['is_identity']]
        assert len(identities) == 1
        identity = actual[identities[0]['pair_id']]
        for word in {p['alternative_word'] for p in pairs if not p['is_identity']}:
            directed = [actual[p['pair_id']] for p in pairs if not p['is_identity'] and p['alternative_word'] == word]
            assert len(directed) == 2
            if identity['status'] != 'scored': state = 'identity_control_unavailable'
            elif not identity['passes_thresholds']: state = 'identity_control_rejected'
            elif any(r['status'] != 'scored' for r in directed): state = 'directed_pair_unavailable'
            elif not all(r['passes_thresholds'] for r in directed): state = 'directed_pair_rejected'
            else: state = 'accepted_by_fixed_nli_criterion'
            alternatives[(rid, word)] = state
    assert len(by_position) == 30 and len(alternatives) == 52
    saved = {(a['review_id'], a['alternative_word']): a for a in summary['alternatives']}
    assert set(saved) == set(alternatives)
    for key, state in alternatives.items():
        assert saved[key]['status'] == state and saved[key]['accepted'] is (state == 'accepted_by_fixed_nli_criterion')
    statuses = Counter(r['status'] for r in results)
    assert summary['pair_statuses'] == dict(statuses) and summary['alternative_statuses'] == dict(Counter(alternatives.values()))
    for name, digest in execution['output_sha256'].items():
        assert sha((stage / 'scores' / name).read_bytes()) == digest
    prior = root / 'results/universal-context-development-v1/reviews'
    ratings = {label: {(r['review_id'], a['word']): a for r in read(prior / f'reviewer_{label}.json')['records'] for a in r['alternatives'] if not a['unchanged']} for label in ('a', 'b')}
    assert all(set(rating) == set(alternatives) for rating in ratings.values())
    evidence = []
    for key, state in alternatives.items():
        flags = {label: {'meaning': ratings[label][key]['definite_meaning_change'], 'grammar': ratings[label][key]['definite_grammar_damage'], 'ambiguous': ratings[label][key]['ambiguous_or_register_risk']} for label in ratings}
        evidence.append({'review_id': key[0], 'alternative_word': key[1], 'nli_status': state,
            'accepted': state == 'accepted_by_fixed_nli_criterion', 'prior_assistant_flags': flags})
    return {'planned_pairs': 134, 'planned_positions': 30, 'planned_nonidentity_alternatives': 52,
        'pair_statuses': dict(statuses), 'alternative_statuses': dict(Counter(alternatives.values())),
        'accepted_alternatives': sum(e['accepted'] for e in evidence),
        'both_assistants_definite_alternatives': sum(all(v['meaning'] or v['grammar'] for v in e['prior_assistant_flags'].values()) for e in evidence),
        'accepted_despite_both_assistants_definite': sum(e['accepted'] and all(v['meaning'] or v['grammar'] for v in e['prior_assistant_flags'].values()) for e in evidence),
        'accepted_despite_both_assistants_meaning': sum(e['accepted'] and all(v['meaning'] for v in e['prior_assistant_flags'].values()) for e in evidence),
        'accepted_despite_both_assistants_grammar': sum(e['accepted'] and all(v['grammar'] for v in e['prior_assistant_flags'].values()) for e in evidence),
        'classifier_inference_rerun_by_this_script': False, 'human_gold_labels': False,
        'freeze_commit': execution['frozen_commit'], 'alternative_evidence': evidence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / 'results/morphology-nli-independent-v1/natural-results-validation.json'
    if output.exists():
        raise FileExistsError('Independent evidence never overwrites old output')
    report = {'schema': 'morphology-nli-independent-results-v1', 'status': 'matched',
        'morphology': validate_morphology(root), 'nli': validate_nli(root),
        'new_classifier_or_generator_calls': 0, 'paid_calls': 0, 'closed_test_opened': False,
        'semantic_safety_certified': False, 'cohorts_independent': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps({k: v if k != 'nli' else {n: x for n, x in v.items() if n != 'alternative_evidence'} for k, v in report.items()}, indent=2))


if __name__ == '__main__':
    main()
