#!/usr/bin/env python3
"""Freeze and audit a universal local-orbit gate on saved development texts.

This is an exploratory mechanical/lexical ablation, not semantic validation.
No generation, paid calls, closed split, retries or outcome-based selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synsteg_audit import shared_dictionary as base
from synsteg_audit import universal_context as universal

SCHEMA = 'synsteg-universal-context-development-v1'
OUTPUT = 'results/universal-context-development-v1'
PARENT = '7b4cb325021dd0acfb6e617a7e1dcb7653029d77'
OLD_INPUTS = 'results/invariant-context-development-v1/inputs.jsonl'
OLD_PROTOCOL = 'results/invariant-context-development-v1/protocol.json'
PROMPTS = 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
LONG_INPUTS = 'results/long-summary-development-v3/mechanical-audit/records.jsonl'
LONG_EXECUTION = 'results/long-summary-development-v3/mechanical-audit/execution.json'
DICTIONARIES = {v: f'data/shared_dictionary_recovery_{v}.json' for v in ('v1', 'v2')}
SOURCES = (OLD_INPUTS, OLD_PROTOCOL, PROMPTS, LONG_INPUTS, LONG_EXECUTION, *DICTIONARIES.values())
CODE = ('scripts/audit_universal_context_development_v1.py', 'src/synsteg_audit/universal_context.py',
        'src/synsteg_audit/invariant_context.py', 'src/synsteg_audit/shared_dictionary.py',
        'src/synsteg_audit/__init__.py')
COUNTS = {'source_excerpts': 100, 'historical_covers': 4, 'grounded_covers': 20, 'long_summary_covers': 20}
ARMS = (('plain_length16', 8), ('plain_length16', 16), ('aes_siv', 8))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def formatted(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def read(path):
    return json.loads(Path(path).read_bytes())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def exclusive(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def fixture(label, group, size=32):
    return hashlib.sha256(('PUBLIC-UNIVERSAL-CONTEXT-' + label + '-v1\0' + group).encode()).digest()[:size]


def collect_inputs(root):
    root = Path(root)
    inventory = {p: sha((root / p).read_bytes()) for p in SOURCES}
    if inventory[OLD_INPUTS] != read(root / OLD_PROTOCOL)['frozen_files']['inputs.jsonl']:
        raise ValueError('Historical frozen input hash mismatch')
    if inventory[LONG_INPUTS] != read(root / LONG_EXECUTION)['output_sha256']['records.jsonl']:
        raise ValueError('Long-summary saved output hash mismatch')
    prompts = jsonl(root / PROMPTS)
    if len(prompts) != 100 or any(p.get('split') != 'development' for p in prompts):
        raise ValueError('Exactly all100 development prompts required')
    by_id = {p['record_id']: p for p in prompts}
    if len(by_id) != 100 or len({p['source_group_id'] for p in prompts}) != 100:
        raise ValueError('Duplicate development record or group')
    old = jsonl(root / OLD_INPUTS)
    if Counter(r['cohort'] for r in old) != Counter({k: v for k, v in COUNTS.items() if k != 'long_summary_covers'}):
        raise ValueError('Historical denominator changed')
    rows = []
    for r in old:
        p = by_id.get(r['record_id'])
        if p is None or p['source_group_id'] != r['source_group_id'] or r.get('split') != 'development':
            raise ValueError('Unexpected historical source identity')
        text = r['text']
        if not isinstance(text, str) or sha(text.encode()) != r['text_sha256']:
            raise ValueError('Historical text/hash unavailable')
        if r['cohort'] == 'source_excerpts' and (text != p['excerpt'] or r['text_sha256'] != p['excerpt_sha256']):
            raise ValueError('Development excerpt changed')
        state = 'available' if text.strip() and r.get('generation_stop_reason') != 'refusal' else 'unavailable_input'
        rows.append({**r, 'availability': state})
    long = jsonl(root / LONG_INPUTS)
    if len(long) != 20 or [r['plan_index'] for r in long] != list(range(1, 21)):
        raise ValueError('Long-summary full plan required')
    if Counter(r['planned_action'] for r in long) != Counter({'generate_once': 15, 'abstain_source_gate': 5}):
        raise ValueError('Long-summary source gate changed')
    for i, r in enumerate(long):
        p = by_id.get(r['record_id'])
        if p is None or p['source_group_id'] != r['source_group_id'] or r.get('split') != 'development':
            raise ValueError('Long-summary identity outside development')
        text = r['cover_text']
        if (text is not None and (not isinstance(text, str) or sha(text.encode()) != r['cover_sha256'])):
            raise ValueError('Long-summary cover/hash mismatch')
        if r['availability'] == 'available' and (not text or not text.strip()):
            raise ValueError('Available cover must be nonempty')
        if r['planned_action'] == 'abstain_source_gate' and (text is not None or r['availability'] != 'source_gate_abstention'):
            raise ValueError('Source gate must remain an unrated abstention')
        rows.append({'cohort': 'long_summary_covers', 'variant': 'long_summary_v3', 'cohort_index': i,
            'audit_record_id': 'long_summary_v3:' + r['record_id'], 'record_id': r['record_id'],
            'source_group_id': r['source_group_id'], 'split': 'development', 'text': text,
            'text_sha256': r['cover_sha256'], 'source_file': LONG_INPUTS,
            'source_provenance': r['attribution'], 'input_status': r['availability'],
            'availability': r['availability'], 'planned_action': r['planned_action']})
    validate_rows(rows)
    for p, h in inventory.items():
        if sha((root / p).read_bytes()) != h:
            raise ValueError('Input changed during collection')
    return rows, inventory


def validate_rows(rows):
    if Counter(r['cohort'] for r in rows) != Counter(COUNTS) or len({r['audit_record_id'] for r in rows}) != sum(COUNTS.values()):
        raise ValueError('Full unique planned denominator required')
    if any(r['split'] != 'development' for r in rows):
        raise ValueError('Only development allowed')
    for cohort, count in COUNTS.items():
        selected = [r for r in rows if r['cohort'] == cohort]
        if sorted(r['cohort_index'] for r in selected) != list(range(count)):
            raise ValueError('Cohort index omitted or duplicated')
        groups = len({r['source_group_id'] for r in selected})
        if groups != (2 if cohort == 'historical_covers' else count):
            raise ValueError('Source-group denominator changed')


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preparation never overwrites an existing namespace')
    rows, inventory = collect_inputs(root)
    dictionaries = {v: base.PublicDictionary.from_dict(read(root / p)) for v, p in DICTIONARIES.items()}
    frozen = {'inputs.jsonl': b''.join(canonical(r) + b'\n' for r in rows)}
    policies = {}
    for v, dictionary in dictionaries.items():
        policy = universal.UniversalPolicy.from_dictionaries(dictionary, dictionaries['v2'], radius=1, orbit_cap=4096)
        policies[v] = policy.sha256
        frozen[f'policy_{v}.json'] = formatted(policy.to_dict())
    protocol = {'schema': SCHEMA, 'parent_main_commit': PARENT,
        'scope': 'exploratory_post_observation_mechanical_and_lexical_ablation',
        'reference_gate': 'exact full class ID+word-list match to strict_v2; lowercase only; reject immediately after a/an',
        'reference_gate_is_semantic_certificate': False, 'radius': 1, 'orbit_cap': 4096,
        'all_neighbor_class_members_included': True, 'orbit_cap_is_invariant_and_fail_closed': True,
        'cohorts': COUNTS, 'cohorts_are_independent': False,
        'selection': 'All100 development excerpts, all4 historical covers, all20 short-grounded plans, all20 long-summary plans. No review/capacity filtering.',
        'methods': ['universal_v1', 'universal_v2'], 'capacity_only_references': ['base_v1', 'base_v2'],
        'arms': [{'framing': f, 'payload_bits': b} for f, b in ARMS],
        'base_min_group_size': 1, 'max_bits_per_slot': 1, 'policy_hashes': policies,
        'source_files': inventory, 'code_files': {p: sha((root / p).read_bytes()) for p in CODE},
        'frozen_files': {p: sha(b) for p, b in frozen.items()},
        'natural_invariance_checks': 'Every individual alias of every candidate including denied slots, plus simultaneous first/last representatives. Does not exhaust global Cartesian product.',
        'invariance_signature_excludes': 'virtual normalization hash because denied original spellings are intentionally authenticated and change under diagnostic aliases',
        'fixture_key_rule': 'SHA256 PUBLIC-UNIVERSAL-CONTEXT-KEY-v1 + NUL + source_group_id; public not secret',
        'fixture_payload_rule': 'SHA256 PUBLIC-UNIVERSAL-CONTEXT-PAYLOAD-v1 + NUL + source_group_id, first payload bytes',
        'receiver': 'Fresh process for each deliverable frame, key+stego+public only; Python I/O guard, not OS sandbox',
        'quality_or_population_claim': False, 'new_model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in frozen.items():
        exclusive(output / name, data)
    exclusive(output / 'protocol.json', formatted(protocol))
    return {'status': 'prepared_not_executed', 'protocol_sha256': sha(formatted(protocol)), 'cohorts': COUNTS}


def verify_freeze(root, prepared):
    protocol = read(prepared / 'protocol.json')
    if protocol.get('schema') != SCHEMA or protocol.get('cohorts') != COUNTS or protocol.get('parent_main_commit') != PARENT:
        raise ValueError('Unsupported frozen protocol')
    if (set(protocol.get('source_files', {})) != set(SOURCES)
            or set(protocol.get('code_files', {})) != set(CODE)
            or set(protocol.get('frozen_files', {})) != {'inputs.jsonl', 'policy_v1.json', 'policy_v2.json'}
            or protocol.get('methods') != ['universal_v1', 'universal_v2']
            or protocol.get('radius') != 1 or protocol.get('orbit_cap') != 4096
            or protocol.get('arms') != [{'framing': f, 'payload_bits': b} for f, b in ARMS]):
        raise ValueError('Incomplete or changed frozen protocol')
    for location, manifest in ((root, protocol['source_files']), (root, protocol['code_files']), (prepared, protocol['frozen_files'])):
        for name, digest in manifest.items():
            path = (location / name).resolve()
            if not path.is_relative_to(location.resolve()) or sha(path.read_bytes()) != digest:
                raise ValueError('Hash-bound file changed: ' + name)
    rows = jsonl(prepared / 'inputs.jsonl')
    validate_rows(rows)
    return protocol, rows


def signature(analysis):
    result = asdict(analysis)
    result.pop('virtual_normalization_sha256')
    return sha(canonical(result))


def replace_words(text, replacements):
    pieces, start = [], 0
    for left, right, value in sorted(replacements):
        pieces.extend((text[start:left], value))
        start = right
    return ''.join(pieces) + text[start:]


def styled(word, style):
    return word.upper() if style == 'upper' else word.capitalize() if style == 'title' else word


def audit_aliases(text, dictionary, policy, original):
    raw = base.analyze_text(text, dictionary, base.Protocol(min_group_size=1, max_bits_per_slot=1))
    classes = {c.class_id: c.words for c in dictionary.classes}
    expected = signature(original)
    checks = []
    for slot in raw.slots:
        for word in classes[slot.class_id]:
            value = styled(word, slot.case_style)
            if value == slot.surface:
                continue
            alias = replace_words(text, [(slot.start, slot.end, value)])
            observed = signature(universal.analyze_text(alias, dictionary, policy))
            checks.append({'token_index': slot.token_index, 'replacement': value, 'signature': observed, 'equal': observed == expected})
    joint = []
    for index in (0, -1):
        alias = replace_words(text, [(s.start, s.end, styled(classes[s.class_id][index], s.case_style)) for s in raw.slots])
        observed = signature(universal.analyze_text(alias, dictionary, policy))
        joint.append({'representative': 'first' if index == 0 else 'last', 'text_sha256': sha(alias.encode()),
                      'signature': observed, 'equal': observed == expected})
    return {'reference_signature': expected, 'individual_checks': checks, 'simultaneous_checks': joint,
            'all_equal': all(c['equal'] for c in checks + joint), 'global_product_exhausted': False}


def isolated_receive(root, text, key, dictionary, policy, framing):
    protocol = universal.UniversalProtocol(framing=framing)
    request = {'key_hex': key.hex(), 'stego_text': text, 'public_dictionary': dictionary.to_dict(),
               'public_policy': policy.to_dict(), 'public_protocol': protocol.to_dict()}
    env = {n: os.environ[n] for n in ('SYSTEMROOT', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP') if n in os.environ}
    env.update(PATH=os.defpath, PYTHONIOENCODING='utf-8')
    with tempfile.TemporaryDirectory(prefix='universal-receiver-') as directory:
        child = subprocess.run([sys.executable, '-I', str(root / 'src/synsteg_audit/universal_context.py'), 'receiver'],
            input=canonical(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=directory, env=env, timeout=60, check=False)
    response = json.loads(child.stdout)
    expected = {'dictionary_sha256': dictionary.sha256, 'policy_sha256': policy.sha256, 'protocol_sha256': protocol.sha256}
    if (child.returncode != 0 or response.get('status') != 'decoded' or response.get('input_keys') != sorted(request)
            or response.get('io_guard_self_checks') != {'file': True, 'socket': True, 'subprocess': True}
            or any(response.get(k) != v for k, v in expected.items())):
        raise ValueError('Fresh receiver failed or resource/isolation evidence mismatched')
    return bytes.fromhex(response['payload_hex']), {'returncode': child.returncode,
        'input_keys': response['input_keys'], 'io_guard_self_checks': response['io_guard_self_checks'],
        'stdout_sha256': sha(child.stdout), 'stderr_sha256': sha(child.stderr)}


def evaluate(root, row, dictionaries, policies):
    text, state = row['text'], row['availability']
    result = {'audit_record_id': row['audit_record_id'], 'cohort': row['cohort'],
        'source_group_id': row['source_group_id'], 'text_sha256': row['text_sha256'],
        'availability': state, 'capacity_only_references': {}, 'methods': {}}
    key = fixture('KEY', row['source_group_id'])
    for v, dictionary in dictionaries.items():
        baseline = None
        if state == 'available':
            baseline = base.analyze_text(text, dictionary, base.Protocol(min_group_size=1, max_bits_per_slot=1)).capacity_bits
        result['capacity_only_references']['base_' + v] = {'capacity_bits': baseline, 'encoding_called': False}
        policy = policies[v]
        entry = {'input_status': state, 'capacity_bits': None, 'analysis': None, 'alias_audit': None, 'arms': {}}
        if state == 'available':
            try:
                analysis = universal.analyze_text(text, dictionary, policy)
                entry.update(capacity_bits=analysis.capacity_bits, analysis=asdict(analysis),
                    alias_audit=audit_aliases(text, dictionary, policy, analysis))
                if not entry['alias_audit']['all_equal']:
                    entry['input_status'] = 'invariance_failure'
            except base.CodecError as error:
                entry.update(input_status='invalid_input', error_type=type(error).__name__)
        for framing, bits in ARMS:
            payload = fixture('PAYLOAD', row['source_group_id'], bits // 8)
            required = bits + (16 if framing == 'plain_length16' else 144)
            arm = {'payload_bits': bits, 'framing': framing, 'required_frame_bits': required,
                'encoding_called': False, 'mechanical_success': False}
            if entry['input_status'] != 'available':
                arm['status'] = entry['input_status']
            elif entry['capacity_bits'] < required:
                arm['status'] = 'insufficient_capacity'
            else:
                arm['encoding_called'] = True
                try:
                    encoded = universal.encode(text, payload, key, dictionary, policy, universal.UniversalProtocol(framing=framing))
                    decoded, evidence = isolated_receive(root, encoded.stego_text, key, dictionary, policy, framing)
                    before, after = text.split(), encoded.stego_text.split()
                    preserved = len(before) == len(after) and all(before[i] == after[i] for i in analysis.protected_token_indices)
                    okay = decoded == payload and preserved
                    arm.update(status='decoded' if okay else 'payload_or_protection_mismatch', mechanical_success=okay,
                        payload_hex=payload.hex(), decoded_payload_hex=decoded.hex(), stego_text=encoded.stego_text,
                        changed_slots=encoded.changed_slots, protected_spellings_preserved=preserved, receiver=evidence)
                except (base.CodecError, ValueError, OSError, subprocess.SubprocessError) as error:
                    arm.update(status='codec_or_receiver_failure', error_type=type(error).__name__)
            entry['arms'][f'{framing}__{bits}'] = arm
        result['methods']['universal_' + v] = entry
    return result


def stats(values):
    return {'count': len(values), 'min': min(values), 'max': max(values), 'mean': statistics.mean(values),
            'zero_capacity': sum(v == 0 for v in values)} if values else {'count': 0}


def summarize(records):
    if Counter(r['cohort'] for r in records) != Counter(COUNTS):
        raise ValueError('Incomplete result denominator')
    summary = {'schema': SCHEMA + '-summary', 'planned_records': sum(COUNTS.values()), 'cohorts': {},
        'cohorts_independent': False, 'new_model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False,
        'semantic_safety_validated': False, 'population_inference': False,
        'all_outcome_counts': dict(Counter(a['status'] for r in records for m in r['methods'].values() for a in m['arms'].values()))}
    for cohort, count in COUNTS.items():
        selected = [r for r in records if r['cohort'] == cohort]
        entry = {'planned_records': count, 'availability_counts': dict(Counter(r['availability'] for r in selected)),
                 'distinct_source_groups': len({r['source_group_id'] for r in selected}), 'methods': {}, 'capacity_only_references': {}}
        for v in ('v1', 'v2'):
            reference = [r['capacity_only_references']['base_' + v]['capacity_bits'] for r in selected]
            entry['capacity_only_references']['base_' + v] = stats([x for x in reference if x is not None])
            methods = [r['methods']['universal_' + v] for r in selected]
            values = [m['capacity_bits'] for m in methods if m['capacity_bits'] is not None]
            reasons = Counter(d['reason'] for m in methods if m['analysis'] for d in m['analysis']['slot_decisions'])
            individual = [c for m in methods if m['alias_audit'] for c in m['alias_audit']['individual_checks']]
            joint = [c for m in methods if m['alias_audit'] for c in m['alias_audit']['simultaneous_checks']]
            entry['methods']['universal_' + v] = {'capacity_bits': stats(values), 'slot_reasons': dict(reasons),
                'individual_alias_checks': len(individual), 'simultaneous_alias_checks': len(joint),
                'invariance_mismatches': sum(not c['equal'] for c in individual + joint),
                'arms': {f'{f}__{b}': {'statuses': dict(Counter(m['arms'][f'{f}__{b}']['status'] for m in methods)),
                    'planned_denominator': count, 'available_denominator': sum(r['availability'] == 'available' for r in selected),
                    'mechanical_successes': sum(m['arms'][f'{f}__{b}']['mechanical_success'] for m in methods)} for f, b in ARMS}}
        summary['cohorts'][cohort] = entry
    return summary


def execute(root, prepared, output, frozen_commit):
    root, prepared, output = Path(root).resolve(), Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Execution never overwrites prior results')
    if not re.fullmatch(r'[a-f0-9]{40}', frozen_commit):
        raise ValueError('Exact externally verified Git freeze commit required')
    protocol, rows = verify_freeze(root, prepared)
    dictionaries = {v: base.PublicDictionary.from_dict(read(root / p)) for v, p in DICTIONARIES.items()}
    policies = {v: universal.UniversalPolicy.from_dict(read(prepared / f'policy_{v}.json'), d) for v, d in dictionaries.items()}
    if any(policies[v].sha256 != protocol['policy_hashes'][v] for v in policies):
        raise ValueError('Policy identity changed')
    started = datetime.now(timezone.utc).isoformat()
    records = [evaluate(root, row, dictionaries, policies) for row in rows]
    verify_freeze(root, prepared)
    summary = summarize(records)
    content = {'records.jsonl': b''.join(canonical(r) + b'\n' for r in records), 'summary.json': formatted(summary)}
    content['execution.json'] = formatted({'schema': SCHEMA + '-execution', 'frozen_commit': frozen_commit,
        'frozen_commit_verification': 'performed by repository operator before invocation, not remotely checked by this offline script',
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha((prepared / 'protocol.json').read_bytes()), 'records': len(records),
        'arms': len(records) * 2 * len(ARMS), 'output_sha256': {p: sha(b) for p, b in content.items()},
        'runtime': {'python': sys.version, 'cryptography': importlib.metadata.version('cryptography')},
        'new_model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False})
    output.mkdir(parents=True, exist_ok=False)
    for p, b in content.items():
        exclusive(output / p, b)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'execute', 'verify'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frozen-commit')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    prepared = (args.prepared or root / OUTPUT).resolve()
    if args.command == 'prepare':
        result = prepare(root, prepared)
    elif args.command == 'verify':
        protocol, rows = verify_freeze(root, prepared)
        result = {'status': 'verified', 'records': len(rows), 'schema': protocol['schema']}
    else:
        result = execute(root, prepared, args.output or prepared / 'audit', args.frozen_commit or '')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
