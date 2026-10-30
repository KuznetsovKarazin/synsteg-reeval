#!/usr/bin/env python3
"""Predeclare/measure gross capacity of all20 unchanged source windows only.

No encoding, decoding, generation, NLI, text replacement or semantic rating.
Prepare and verify never call a natural-text capacity analyzer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synsteg_audit import shared_dictionary as base
from synsteg_audit import universal_context as universal

ID = 'full-source-capacity-control-v1'
OUT = 'results/' + ID
WINDOWS = 'data/development-full-revisions-v3/source-windows.jsonl'
OLD_PLAN = 'results/long-summary-development-v3/plan.json'
GGUF_PLAN = 'results/long-summary-development-qwen15b-gguf-v1/plan.json'
METHODS = ('base_v1', 'base_v2', 'universal_morph_v3')
THRESHOLDS = {'plain8': 24, 'plain16': 32, 'aes8': 152}
DICTIONARIES = {'base_v1': 'data/shared_dictionary_recovery_v1.json',
                'base_v2': 'data/shared_dictionary_recovery_v2.json',
                'universal_morph_v3': 'data/shared_dictionary_morphology_v3.json'}
POLICY = 'results/morphology-development-v3/policy.json'
INPUTS = (WINDOWS, OLD_PLAN, GGUF_PLAN, *DICTIONARIES.values(), POLICY)
CODE = ('scripts/audit_full_source_capacity_control_v1.py', 'tests/test_full_source_capacity_control_v1.py',
        'docs/full_source_capacity_control_v1_plan_ru.md', 'src/synsteg_audit/shared_dictionary.py',
        'src/synsteg_audit/universal_context.py', 'src/synsteg_audit/invariant_context.py', 'src/synsteg_audit/__init__.py')
SELECTED = [2, 5, 6, 7, 8, 9, 10, 11]
SOURCE_GATES = [1, 3, 4, 16, 19]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def formatted(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def exclusive(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def source_rows(root):
    windows = [json.loads(x) for x in (root / WINDOWS).read_text(encoding='utf-8').splitlines() if x.strip()]
    old, pilot = (read(root / p)['records'] for p in (OLD_PLAN, GGUF_PLAN))
    if len(windows) != 20 or len(old) != 20 or len(pilot) != 20:
        raise ValueError('All20 full saved source windows and both complete plans required')
    records = []
    for index, (window, prior, selected) in enumerate(zip(windows, old, pilot), 1):
        if any(r['split'] != 'development' for r in (window, prior, selected)):
            raise ValueError('Development-only control')
        if prior['plan_index'] != index or selected['plan_index'] != index:
            raise ValueError('Frozen source order differs')
        for key in ('record_id', 'source_group_id', 'source_window_sha256'):
            if not window[key] == prior[key] == selected[key]:
                raise ValueError('Frozen source identity differs: ' + key)
        text = window['source_window']
        if (not isinstance(text, str) or sha(text.encode()) != window['source_window_sha256']
                or len(text.split()) != window['selected_window_words']
                or prior['source_status'] != window['status']):
            raise ValueError('Whole source text/hash/length/status differs')
        records.append({'record_id': window['record_id'], 'source_group_id': window['source_group_id'],
            'plan_index': index, 'split': 'development', 'text': text,
            'source_sha256': window['source_window_sha256'], 'source_words': len(text.split()),
            'source_characters': len(text), 'source_utf8_bytes': len(text.encode()),
            'source_gate_status': window['status'], 'source_eligible': window['status'] == 'word_heuristic_met',
            'gguf_selected': selected['planned_action'] == 'generate_once', 'attribution': prior['attribution'],
            'source_gate_is_text_unavailable': False, 'generated_cover': False})
    if (len({r['record_id'] for r in records}) != 20 or len({r['source_group_id'] for r in records}) != 20
            or [r['plan_index'] for r in records if not r['source_eligible']] != SOURCE_GATES
            or [r['plan_index'] for r in records if r['gguf_selected']] != SELECTED
            or any(r['gguf_selected'] and not r['source_eligible'] for r in records)):
        raise ValueError('Frozen all20/eligible15/selected8 cohorts differ')
    return records


def resources(root):
    dictionaries = {name: base.PublicDictionary.from_dict(read(root / path)) for name, path in DICTIONARIES.items()}
    dictionary = dictionaries['universal_morph_v3']
    policy = universal.UniversalPolicy.from_dict(read(root / POLICY), dictionary)
    expected = universal.UniversalPolicy.from_dictionaries(dictionary, dictionary,
        version='morphology-complete-tag-spectrum-v3-reference-v1', radius=1, orbit_cap=4096)
    if policy.to_dict() != expected.to_dict():
        raise ValueError('Exact frozen complete-morphology policy required')
    if {'plain8': 8 + base.Protocol(1, 1, 'plain_length16').frame_overhead_bits,
        'plain16': 16 + base.Protocol(1, 1, 'plain_length16').frame_overhead_bits,
        'aes8': 8 + base.Protocol(1, 1, 'aes_siv').frame_overhead_bits} != THRESHOLDS:
        raise ValueError('Frame arithmetic differs from frozen base codec')
    return dictionaries, policy


def make_protocol(root):
    rows = source_rows(root)
    dictionaries, policy = resources(root)
    return {'schema': ID, 'status': 'prepared_not_measured',
        'parent_head_at_preparation': 'd02d7a895e2ef74367881f20f918e63d64d34b8c',
        'scope': 'Exploratory control planned after earlier generation results; no independent population inference',
        'question': 'Do complete saved source windows provide enough gross capacity for whole-frame thresholds, even without short generated covers?',
        'source_files': {p: sha((root / p).read_bytes()) for p in INPUTS},
        'code_files': {p: sha((root / p).read_bytes()) for p in CODE},
        'records': [{k: v for k, v in r.items() if k != 'text'} for r in rows],
        'methods': list(METHODS), 'measurement_count': 60, 'physical_text_denominator': 20,
        'groups': {'all20': 20, 'source_eligible15': 15, 'gguf_selected8': 8}, 'overlapping_groups': True,
        'min_group_size': 1, 'max_bits_per_slot': 1,
        'dictionary_hashes': {name: d.sha256 for name, d in dictionaries.items()}, 'policy_sha256': policy.sha256,
        'frame_threshold_bits': THRESHOLDS, 'feasibility_rule': 'capacity_bits >= required_frame_bits; arithmetic only, no payload delivery claim',
        'zero_and_error_policy': 'Keep measured zero as0; analysis error capacity/thresholdflags=null; retain all60 records and group denominators',
        'text_policy': 'Every complete saved source_window byte string exactly unchanged; no compression, generation, copying-as-cover or text edits',
        'source_gate_policy': 'Five old generation source-gate refusals retained as metadata; all20 physical windows remain available for capacity-only analysis',
        'analysis_calls_during_preparation': 0, 'encoding_calls': 0, 'decoding_calls': 0,
        'llm_calls': 0, 'nli_calls': 0, 'paid_calls': 0, 'closed_test_used': False,
        'claims_not_made': ['delivery/roundtrip', 'factual or semantic quality', 'safe substitutions', 'steganographic secrecy',
                           'external source truth', 'representativeness', 'source copies as generated covers'],
        'required_before_measurement': 'Root review and durable Git freeze of protocol, script, tests and plan; full verified commit passed explicitly'}


def verify(root):
    p = read(root / OUT / 'protocol.json')
    if p != make_protocol(root):
        raise ValueError('Frozen full-source control input/code/protocol changed')
    return p


def capacity_result(capacity, error=None):
    if error is not None:
        if capacity is not None or not isinstance(error, str) or not error:
            raise ValueError('Error requires null capacity and a reason')
        status = 'analysis_error'
    else:
        if type(capacity) is not int or capacity < 0:
            raise ValueError('Measured gross capacity must be a nonnegative integer')
        status = 'measured'
    return {'status': status, 'capacity_bits': capacity, 'error': error,
            'frame_capacity_feasible': {k: capacity >= bits if capacity is not None else None for k, bits in THRESHOLDS.items()},
            'encoding_called': False, 'decoding_called': False, 'delivery_established': False, 'semantic_quality_assessed': False}


def analyze_record(row, analyzers):
    if set(analyzers) != set(METHODS):
        raise ValueError('Exactly three fixed analyzers required')
    text = row['text']
    if sha(text.encode()) != row['source_sha256'] or len(text.split()) != row['source_words']:
        raise ValueError('Source text changed before analysis')
    result = {**row, 'methods': {}}
    for name in METHODS:
        try:
            result['methods'][name] = capacity_result(analyzers[name](text))
        except Exception as error:
            result['methods'][name] = capacity_result(None, type(error).__name__ + ': ' + str(error))
    return result


def summarize(records):
    if (len(records) != 20 or [r['plan_index'] for r in records] != list(range(1, 21))
            or len({r['record_id'] for r in records}) != 20
            or any(type(r['source_eligible']) is not bool or type(r['gguf_selected']) is not bool for r in records)
            or [r['plan_index'] for r in records if not r['source_eligible']] != SOURCE_GATES
            or [r['plan_index'] for r in records if r['gguf_selected']] != SELECTED
            or any(r['gguf_selected'] and not r['source_eligible'] for r in records)):
        raise ValueError('Complete unique ordered20 and overlapping15/8 cohorts required')
    for row in records:
        if set(row['methods']) != set(METHODS) or type(row['source_words']) is not int or row['source_words'] < 0:
            raise ValueError('Three measurements and source length required per row')
        for value in row['methods'].values():
            if value != capacity_result(value.get('capacity_bits'), value.get('error')):
                raise ValueError('Capacity status/threshold/null or no-delivery contract changed')
    groups = {}
    for group, cohort in (('all20', records), ('source_eligible15', [r for r in records if r['source_eligible']]),
                          ('gguf_selected8', [r for r in records if r['gguf_selected']])):
        methods = {}
        for name in METHODS:
            measured = [r for r in cohort if r['methods'][name]['status'] == 'measured']
            values = [r['methods'][name]['capacity_bits'] for r in measured]
            total = sum(values)
            methods[name] = {'denominator': len(cohort), 'measured': len(measured), 'analysis_errors': len(cohort) - len(measured),
                'zero_capacity': values.count(0), 'capacity_sum': total,
                'mean_capacity_over_measured': total / len(values) if values else None,
                'measured_source_words': sum(r['source_words'] for r in measured),
                'frame_capacity_feasible': {k: sum(r['methods'][name]['frame_capacity_feasible'][k] is True for r in measured) for k in THRESHOLDS}}
        groups[group] = {'records': len(cohort), 'source_words': [r['source_words'] for r in cohort], 'methods': methods}
    return {'schema': ID + '-summary', 'measurement_denominator': 60, 'groups': groups,
            'framing_tests_are_capacity_inequalities_only': True, 'encoding_calls': 0, 'decoding_calls': 0,
            'generation_calls': 0, 'nli_calls': 0, 'semantic_quality_assessed': False, 'population_inference': False}


def execute(root, freeze_commit):
    if not re.fullmatch('[0-9a-f]{40}', freeze_commit):
        raise ValueError('Explicit full verified Git freeze commit required')
    verify(root)
    rows = source_rows(root)
    dictionaries, policy = resources(root)
    output = root / OUT / 'run'
    output.mkdir(parents=True, exist_ok=False)
    exclusive(output / 'execution.json', formatted({'freeze_commit': freeze_commit,
        'freeze_binding': 'Caller verifies remote Git commit; this script verifies all frozen local hashes',
        'started_at': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha((root / OUT / 'protocol.json').read_bytes())}))
    analyzers = {name: (lambda text, dictionary=dictionaries[name]: base.analyze_text(text, dictionary, base.Protocol(1, 1, 'plain_length16')).capacity_bits)
                 for name in ('base_v1', 'base_v2')}
    analyzers['universal_morph_v3'] = lambda text: universal.analyze_text(text, dictionaries['universal_morph_v3'], policy,
        universal.UniversalProtocol(1, 1, 'plain_length16')).capacity_bits
    results = []
    with (output / 'records.jsonl').open('xb') as stream:
        for row in rows:
            result = analyze_record(row, analyzers)
            results.append(result)
            stream.write((json.dumps(result, ensure_ascii=False, sort_keys=True) + '\n').encode()); stream.flush()
    verify(root)
    summary = summarize(results)
    exclusive(output / 'summary.json', formatted({**summary, 'post_run_hashes_verified': True,
        'finished_at': datetime.now(timezone.utc).isoformat()}))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'verify', 'execute'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--freeze-commit')
    args = p.parse_args(argv); root = args.root.resolve()
    if args.stage == 'prepare':
        exclusive(root / OUT / 'protocol.json', formatted(make_protocol(root)))
        result = {'status': 'prepared_not_measured', 'measurements_planned': 60, 'capacity_analyzer_calls': 0}
    elif args.stage == 'verify':
        verify(root); result = {'status': 'verified_not_measured', 'capacity_analyzer_calls': 0}
    else:
        if not args.freeze_commit: p.error('execute requires --freeze-commit; no measurement during prepare/verify')
        result = execute(root, args.freeze_commit)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
