#!/usr/bin/env python3
"""Predeclare and audit the new morphology dictionary on all saved development.

No corpus-dependent dictionary construction, model generation or closed split.
The old universal codec is reused without changing its public predicate.
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
sys.path.insert(0, str(ROOT / 'scripts'))
from synsteg_audit import shared_dictionary as base
from synsteg_audit import universal_context as universal
import audit_universal_context_development_v1 as previous

SCHEMA = 'synsteg-morphology-development-v3'
STAGE = 'results/morphology-development-v3'
DICTIONARY = 'data/shared_dictionary_morphology_v3.json'
PARENT = '026b5b94494d65461fb0f440f12ddfc99c8ee2fb'
INPUTS = 'results/universal-context-development-v1/inputs.jsonl'
REFERENCE_DICTIONARY = 'data/shared_dictionary_recovery_v2.json'
REFERENCE_POLICY = 'results/universal-context-development-v1/policy_v2.json'
CODE = ('scripts/audit_morphology_development_v3.py', 'scripts/build_morphology_dictionary_v3.py',
        'scripts/audit_universal_context_development_v1.py', 'src/synsteg_audit/shared_dictionary.py',
        'src/synsteg_audit/universal_context.py', 'src/synsteg_audit/invariant_context.py',
        'src/synsteg_audit/__init__.py')
SOURCE = (INPUTS, DICTIONARY, REFERENCE_DICTIONARY, REFERENCE_POLICY)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def build_evidence_paths(root):
    build = sorted(p for p in (root / 'results/morphology-recovery-v3').rglob('*') if p.is_file())
    source = sorted(p for p in (root / 'data/morphology-recovery-v3/source').rglob('*') if p.is_file())
    if not build or not source:
        raise ValueError('Complete morphology build evidence/source bundle required')
    return [p.relative_to(root).as_posix() for p in build + source]


def validate_inputs(path):
    rows = [json.loads(s) for s in path.read_bytes().splitlines() if s.strip()]
    previous.validate_rows(rows)
    for row in rows:
        text = row['text']
        if text is None and row['text_sha256'] is not None:
            raise ValueError('Absent text must have an absent hash')
        if (text is not None and (not isinstance(text, str) or sha(text.encode()) != row['text_sha256'])):
            raise ValueError('Input text hash mismatch')
        if row['availability'] == 'available' and (not isinstance(text, str) or not text.strip()):
            raise ValueError('Available input must have text')
        if row.get('planned_action') == 'abstain_source_gate' and (text is not None or row['availability'] != 'source_gate_abstention'):
            raise ValueError('Preserve source gate and absence of output')
    return rows


def prepare(root, prepared):
    root, prepared = Path(root).resolve(), Path(prepared).resolve()
    if prepared.exists():
        raise FileExistsError('Use a fresh preparation namespace')
    rows = validate_inputs(root / INPUTS)
    dictionary = base.PublicDictionary.from_dict(read(root / DICTIONARY))
    policy = universal.UniversalPolicy.from_dictionaries(dictionary, dictionary,
        version='morphology-complete-tag-spectrum-v3-reference-v1', radius=1, orbit_cap=4096)
    policy_bytes = previous.formatted(policy.to_dict())
    # Bind all source-only build evidence, without depending on filenames of
    # the independent builder's individual annotation/ledger outputs.
    extra = build_evidence_paths(root)
    protocol = {'schema': SCHEMA, 'parent_main': PARENT,
        'scope': 'feedback-informed exploratory development, not held-out validation',
        'selection': 'All144 previously saved planned records, no capacity/rating filtering; all source gates retained',
        'cohorts': previous.COUNTS, 'cohorts_independent': False,
        'methods': ['universal_morph_v3'], 'capacity_only_references': ['base_morph_v3', 'base_v2', 'universal_v2'],
        'arms': [{'framing': f, 'payload_bits': b} for f, b in previous.ARMS],
        'policy_sha256': policy.sha256, 'policy_file_sha256': sha(policy_bytes),
        'public_dictionary_sha256': dictionary.sha256,
        'source_files': {p: sha((root / p).read_bytes()) for p in (*SOURCE, *extra)},
        'code_files': {p: sha((root / p).read_bytes()) for p in CODE},
        'metadata_rule': 'WordNet sense/POS and source inflection tag-spectrum are provenance, not contextual certification',
        'alias_rule': 'Every individual member alias and simultaneous first/last representatives using old audited helper',
        'new_generator_calls': 0, 'paid_calls': 0, 'closed_test_opened': False}
    prepared.mkdir(parents=True, exist_ok=False)
    previous.exclusive(prepared / 'policy.json', policy_bytes)
    previous.exclusive(prepared / 'protocol.json', previous.formatted(protocol))
    return {'status': 'prepared_not_executed', 'records': len(rows), 'protocol_sha256': sha(previous.formatted(protocol))}


def verify(root, prepared):
    root, prepared = Path(root).resolve(), Path(prepared).resolve()
    protocol = read(prepared / 'protocol.json')
    if (protocol.get('schema') != SCHEMA or protocol.get('parent_main') != PARENT
            or protocol.get('cohorts') != previous.COUNTS or protocol.get('methods') != ['universal_morph_v3']
            or protocol.get('arms') != [{'framing': f, 'payload_bits': b} for f, b in previous.ARMS]
            or set(protocol.get('code_files', {})) != set(CODE)
            or set(protocol.get('source_files', {})) != set((*SOURCE, *build_evidence_paths(root)))):
        raise ValueError('Changed or incomplete protocol')
    for manifest in (protocol['source_files'], protocol['code_files']):
        for name, expected in manifest.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or sha(path.read_bytes()) != expected:
                raise ValueError('Hash-bound input changed: ' + name)
    if sha((prepared / 'policy.json').read_bytes()) != protocol['policy_file_sha256']:
        raise ValueError('Policy file changed')
    dictionary = base.PublicDictionary.from_dict(read(root / DICTIONARY))
    policy = universal.UniversalPolicy.from_dict(read(prepared / 'policy.json'), dictionary)
    if policy.sha256 != protocol['policy_sha256'] or dictionary.sha256 != protocol['public_dictionary_sha256']:
        raise ValueError('Public resource identity mismatch')
    return protocol, validate_inputs(root / INPUTS), dictionary, policy


def summarize(records):
    if Counter(r['cohort'] for r in records) != Counter(previous.COUNTS):
        raise ValueError('Incomplete planned denominator')
    result = {'schema': SCHEMA + '-summary', 'planned_records': len(records),
        'arms': len(records) * len(previous.ARMS), 'cohorts': {},
        'all_outcomes': dict(Counter(a['status'] for r in records for a in r['methods']['universal_morph_v3']['arms'].values())),
        'new_generator_calls': 0, 'paid_calls': 0, 'closed_test_opened': False,
        'cohorts_independent': False, 'semantic_safety_validated': False}
    for cohort, count in previous.COUNTS.items():
        selected = [r for r in records if r['cohort'] == cohort]
        methods = [r['methods']['universal_morph_v3'] for r in selected]
        aliases = [c for m in methods if m['alias_audit'] for key in ('individual_checks', 'simultaneous_checks') for c in m['alias_audit'][key]]
        result['cohorts'][cohort] = {'planned': count,
            'availability': dict(Counter(r['availability'] for r in selected)),
            'new_capacity': previous.stats([m['capacity_bits'] for m in methods if m['capacity_bits'] is not None]),
            'reference_capacity': {name: previous.stats([r['capacity_only_references'][name]['capacity_bits'] for r in selected
                if r['capacity_only_references'][name]['capacity_bits'] is not None]) for name in ('base_morph_v3', 'base_v2', 'universal_v2')},
            'slot_reasons': dict(Counter(d['reason'] for m in methods if m['analysis'] for d in m['analysis']['slot_decisions'])),
            'alias_checks': len(aliases), 'invariance_mismatches': sum(not c['equal'] for c in aliases),
            'arms': {f'{f}__{b}': {'planned_denominator': count,
                'available_denominator': sum(r['availability'] == 'available' for r in selected),
                'outcomes': dict(Counter(m['arms'][f'{f}__{b}']['status'] for m in methods))} for f, b in previous.ARMS}}
    return result


def execute(root, prepared, output, frozen_commit):
    root, prepared, output = Path(root).resolve(), Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refuse to overwrite audit results')
    if not isinstance(frozen_commit, str) or not re.fullmatch('[a-f0-9]{40}', frozen_commit):
        raise ValueError('Externally verified exact Git freeze commit required')
    protocol, rows, dictionary, policy = verify(root, prepared)
    reference = base.PublicDictionary.from_dict(read(root / REFERENCE_DICTIONARY))
    reference_policy = universal.UniversalPolicy.from_dict(read(root / REFERENCE_POLICY), reference)
    started = datetime.now(timezone.utc).isoformat()
    results = []
    for row in rows:
        result = previous.evaluate(root, row, {'morph_v3': dictionary}, {'morph_v3': policy})
        raw = guarded = None
        if row['availability'] == 'available':
            raw = base.analyze_text(row['text'], reference, base.Protocol(1, 1, 'aes_siv')).capacity_bits
            guarded = universal.analyze_text(row['text'], reference, reference_policy).capacity_bits
        for name, value in (('base_v2', raw), ('universal_v2', guarded)):
            result['capacity_only_references'][name] = {'capacity_bits': value, 'encoding_called': False}
        results.append(result)
    verify(root, prepared)
    summary = summarize(results)
    content = {'records.jsonl': b''.join(previous.canonical(r) + b'\n' for r in results), 'summary.json': previous.formatted(summary)}
    content['execution.json'] = previous.formatted({'schema': SCHEMA + '-execution', 'frozen_commit': frozen_commit,
        'freeze_verification_scope': 'operator verified Git before invocation; this script is offline',
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha((prepared / 'protocol.json').read_bytes()),
        'output_sha256': {n: sha(b) for n, b in content.items()}, 'records': len(results), 'arms': len(results) * 3,
        'new_generator_calls': 0, 'paid_calls': 0, 'closed_test_opened': False})
    output.mkdir(parents=True, exist_ok=False)
    for name, data in content.items():
        previous.exclusive(output / name, data)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'verify', 'execute'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--frozen-commit')
    a = p.parse_args(argv)
    root = a.root.resolve(); prepared = a.prepared or root / STAGE
    if a.command == 'prepare': result = prepare(root, prepared)
    elif a.command == 'verify': result = {'status': 'verified', 'records': len(verify(root, prepared)[1])}
    else: result = execute(root, prepared, a.output or Path(prepared) / 'audit', a.frozen_commit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
