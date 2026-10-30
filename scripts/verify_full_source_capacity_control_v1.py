#!/usr/bin/env python3
"""Independent token-level oracle for the full saved-window capacity control.

Shares only the frozen base token pattern and public dictionary representation.
Does not import/call the new evaluator or any encode/decode/universal analyzer.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synsteg_audit import shared_dictionary as base

STAGE = 'results/full-source-capacity-control-v1'
METHODS = ('base_v1', 'base_v2', 'universal_morph_v3')
THRESHOLDS = {'plain8': 24, 'plain16': 32, 'aes8': 152}
SELECTED = (2, 5, 6, 7, 8, 9, 10, 11)
GATES = (1, 3, 4, 16, 19)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def check(value, label):
    if not value:
        raise ValueError(label)


def core(token):
    match = base._TOKEN.fullmatch(token)
    if match is None or any(c == '_' or not unicodedata.category(c).startswith('P')
                            for c in match.group('prefix') + match.group('suffix')):
        return None
    return match


def raw_slots(text, dictionary):
    lookup = {word: entry for entry in dictionary.classes for word in entry.words}
    slots = []
    for index, word in enumerate(text.split()):
        match = core(word)
        if match is None:
            continue
        surface = match.group('word')
        entry = lookup.get(surface.lower())
        if entry is None or len(entry.words) < 2:
            continue
        style = ('lower' if surface.islower() else 'upper' if surface.isupper()
                 else 'title' if surface.istitle() else None)
        if style is None:
            continue
        slots.append({'token_index': index, 'class_id': entry.class_id, 'surface': surface,
                      'style': style, 'prefix': match.group('prefix'), 'suffix': match.group('suffix')})
    return slots


def morph_slots(text, dictionary, strict_dictionary, radius=1, cap=4096):
    check(radius == 1, 'This oracle is for the frozen radius1 control')
    check(not any(c in text for c in ('\ue000', '\ue001')), 'Reserved external characters')
    slots = raw_slots(text, dictionary)
    tokens = text.split()
    classes = {entry.class_id: entry.words for entry in dictionary.classes}
    reference = {entry.class_id: entry.words for entry in strict_dictionary.classes}
    options = [(token,) for token in tokens]
    for slot in slots:
        def styled(word):
            return word.upper() if slot['style'] == 'upper' else word.capitalize() if slot['style'] == 'title' else word
        options[slot['token_index']] = tuple(sorted(set(slot['prefix'] + styled(w) + slot['suffix'] for w in classes[slot['class_id']])))
    decisions = []
    for slot in slots:
        i = slot['token_index']
        orbit_size = math.prod(len(values) for values in options[max(0, i - 1):min(len(tokens), i + 2)])
        if orbit_size > cap:
            reason = 'orbit_cap_exceeded'
        elif reference.get(slot['class_id']) != classes[slot['class_id']]:
            reason = 'class_not_exact_reference'
        elif slot['style'] != 'lower':
            reason = 'case_not_lowercase'
        elif i > 0 and any((match := core(value)) is not None and match.group('word').lower() in ('a', 'an') for value in options[i - 1]):
            reason = 'adjacent_indefinite_article'
        else:
            reason = 'accepted'
        decisions.append({**slot, 'orbit_size': orbit_size, 'accepted': reason == 'accepted', 'reason': reason})
    return decisions


def self_test():
    dictionary = base.PublicDictionary.from_mapping({'speed': ['rapid', 'swift'], 'article': ['an', 'on']})
    checks = 0
    for text, expected in [('rapid Swift RAPID rApId', 3), ('"rapid," _rapid rapid-like (swift)', 2),
                           ('a rapid rapid', 2), ('on rapid', 2)]:
        check(len(raw_slots(text, dictionary)) == expected, 'Synthetic raw token oracle'); checks += 1
    for text, expected in [('rapid Swift RAPID rApId', 1), ('a rapid rapid', 1), ('on rapid', 1), ('"an," rapid', 1)]:
        check(sum(x['accepted'] for x in morph_slots(text, dictionary, dictionary)) == expected, 'Synthetic direct morphology gate'); checks += 1
    check(not any(x['accepted'] for x in morph_slots('rapid swift', dictionary, dictionary, cap=1)), 'Synthetic cap fail-closed'); checks += 1
    other = base.PublicDictionary.from_mapping({'different': ['rapid', 'swift'], 'article': ['an', 'on']})
    check(not morph_slots('rapid', dictionary, other)[0]['accepted'], 'Synthetic class identity mismatch'); checks += 1
    try:
        morph_slots('rapid\ue000', dictionary, dictionary)
    except ValueError:
        checks += 1
    else:
        raise ValueError('Synthetic reserved character rejection')
    return checks


def audit(root, expected_freeze):
    stage = root / STAGE
    summary = read(stage / 'run/summary.json')
    check(summary.get('post_run_hashes_verified') is True and summary.get('finished_at'), 'Completed control required')
    execution = read(stage / 'run/execution.json')
    check(execution['freeze_commit'] == expected_freeze and re.fullmatch('[0-9a-f]{40}', expected_freeze), 'Explicit common freeze')
    protocol = read(stage / 'protocol.json')
    check(execution['protocol_sha256'] == sha((stage / 'protocol.json').read_bytes()), 'Protocol identity')
    manifests = {**protocol['source_files'], **protocol['code_files']}
    evidence = {str(p.relative_to(root)): sha(p.read_bytes()) for p in (stage / 'run').iterdir() if p.is_file()}
    evidence[STAGE + '/protocol.json'] = sha((stage / 'protocol.json').read_bytes())
    for name, expected in manifests.items():
        check(sha((root / name).read_bytes()) == expected, 'Frozen file ' + name)
    windows = lines(root / 'data/development-full-revisions-v3/source-windows.jsonl')
    records = lines(stage / 'run/records.jsonl')
    check(len(windows) == len(records) == len(protocol['records']) == 20, 'Full20 windows')
    check(protocol['groups'] == {'all20':20, 'source_eligible15':15, 'gguf_selected8':8} and protocol['measurement_count'] == 60, 'Predefined groups/60 measurements')
    check(protocol['frame_threshold_bits'] == THRESHOLDS and protocol['min_group_size'] == protocol['max_bits_per_slot'] == 1, 'Frozen frame arithmetic')
    dictionaries = {name: base.PublicDictionary.from_dict(read(root / path)) for name, path in {
        'base_v1':'data/shared_dictionary_recovery_v1.json', 'base_v2':'data/shared_dictionary_recovery_v2.json',
        'universal_morph_v3':'data/shared_dictionary_morphology_v3.json'}.items()}
    policy = read(root / 'results/morphology-development-v3/policy.json')
    strict = base.PublicDictionary.from_dict(policy['strict_dictionary'])
    check(policy['radius'] == 1 and policy['orbit_cap'] == 4096 and strict.sha256 == dictionaries['universal_morph_v3'].sha256, 'Exact morphology reference')
    check({name:d.sha256 for name,d in dictionaries.items()} == protocol['dictionary_hashes'], 'Canonical dictionary hashes')
    oracle = []
    for index, (window, row, planned) in enumerate(zip(windows, records, protocol['records']), 1):
        text = window['source_window']
        check(row['plan_index'] == index and row['text'] == text and {k:v for k,v in row.items() if k not in ('text','methods')} == planned, 'Unchanged row and metadata')
        check(row['record_id'] == window['record_id'] and row['source_group_id'] == window['source_group_id'] and row['split'] == 'development', 'Development identities')
        check(sha(text.encode()) == row['source_sha256'] == window['source_window_sha256'] and len(text.split()) == row['source_words'], 'Source bytes/hash/words')
        check(row['source_characters'] == len(text) and row['source_utf8_bytes'] == len(text.encode()), 'Source lengths')
        check(row['source_eligible'] == (index not in GATES) and row['gguf_selected'] == (index in SELECTED) and row['generated_cover'] is False and row['source_gate_is_text_unavailable'] is False, 'Group and availability metadata')
        check(set(row['methods']) == set(METHODS), 'All three methods')
        result = {'plan_index':index, 'record_id':row['record_id'], 'source_sha256':row['source_sha256'], 'source_words':row['source_words'],
                  'source_eligible':row['source_eligible'], 'gguf_selected':row['gguf_selected'], 'methods':{}}
        for method in METHODS:
            slots = (morph_slots(text, dictionaries[method], strict, policy['radius'], policy['orbit_cap']) if method == 'universal_morph_v3'
                     else raw_slots(text, dictionaries[method]))
            capacity = sum(slot['accepted'] for slot in slots) if method == 'universal_morph_v3' else len(slots)
            measured = row['methods'][method]
            expected = {'status':'measured', 'capacity_bits':capacity, 'error':None,
                'frame_capacity_feasible':{name:capacity >= required for name,required in THRESHOLDS.items()},
                'encoding_called':False, 'decoding_called':False, 'delivery_established':False, 'semantic_quality_assessed':False}
            check(measured == expected, 'Independent token capacity/flags mismatch')
            result['methods'][method] = {'capacity_bits':capacity, 'token_oracle':slots}
        oracle.append(result)
    groups = {}
    for name, rows in (('all20',oracle), ('source_eligible15',[r for r in oracle if r['source_eligible']]), ('gguf_selected8',[r for r in oracle if r['gguf_selected']])):
        values = {'records':len(rows), 'source_words':[r['source_words'] for r in rows], 'methods':{}}
        for method in METHODS:
            capacities = [r['methods'][method]['capacity_bits'] for r in rows]
            values['methods'][method] = {'denominator':len(rows), 'measured':len(rows), 'analysis_errors':0,
                'zero_capacity':capacities.count(0), 'capacity_sum':sum(capacities), 'mean_capacity_over_measured':sum(capacities)/len(rows),
                'measured_source_words':sum(r['source_words'] for r in rows),
                'frame_capacity_feasible':{label:sum(c >= threshold for c in capacities) for label,threshold in THRESHOLDS.items()}}
        groups[name] = values
    check(summary['groups'] == groups and summary['measurement_denominator'] == 60, 'Independently recomputed group summaries')
    check(summary['encoding_calls'] == summary['decoding_calls'] == summary['generation_calls'] == summary['nli_calls'] == 0 and summary['framing_tests_are_capacity_inequalities_only'] is True and summary['semantic_quality_assessed'] is False and summary['population_inference'] is False, 'No extra claims')
    for name, expected in {**evidence, **manifests}.items():
        check(sha((root / name).read_bytes()) == expected, 'Evidence changed during audit')
    return {'schema':'full-source-capacity-control-v1-independent-oracle', 'status':'all60_token_capacities_match',
        'completed_utc':datetime.now(timezone.utc).isoformat(), 'freeze_commit':expected_freeze,
        'verifier_sha256':sha(Path(__file__).read_bytes()), 'synthetic_oracle_checks_passed':self_test(),
        'analysis_implementation':'Shared frozen base token regex and dictionary parser only; independent token iteration, width1 counting, direct universal morphology predicate and finite orbit-size accounting.',
        'new_evaluator_called':False, 'base_analyze_text_called':False, 'universal_analyze_text_called':False,
        'production_files_unchanged':True, 'natural_capacity_recomputations':60,
        'groups':groups, 'records':oracle, 'input_sha256':{**manifests, **evidence},
        'limitations':['All20 are complete saved source windows, not necessarily whole articles.',
            'This is a capacity-only post-observation exploratory control, not copying-as-generation or factual/semantic validation.',
            'Threshold feasibility means only integer capacity inequalities; no encoding, decoding, payload delivery, or secrecy claim.',
            'The oracle shares the frozen tokenizer regex and dictionary parser; it independently evaluates the slot and morphology rules.',
            'Groups20/15/8 overlap and cannot be treated as independent samples.'],
        'encoding_calls':0, 'decoding_calls':0, 'model_calls':0, 'paid_calls':0, 'closed_test_used':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--freeze-commit')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps({'synthetic_checks_passed':self_test(),'natural_measurements':0}));return
    if not args.freeze_commit:
        parser.error('--freeze-commit is required')
    root=args.root.resolve(); output=root/STAGE/'independent-postrun-review.json'
    if output.exists():
        raise FileExistsError('Audit evidence never overwritten')
    value=audit(root,args.freeze_commit)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({'status':value['status'],'sha256':sha(output.read_bytes()),'groups':value['groups']},indent=2))


if __name__=='__main__':
    main()
