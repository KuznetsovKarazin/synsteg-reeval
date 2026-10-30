#!/usr/bin/env python3
"""Preregistered mechanical evaluation of all20 fixed GGUF pilot ledger rows.

No model imports, downloads or generation. Mechanical decoding is never a
semantic success, particularly for the known-unsafe raw v1 reference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from synsteg_audit import shared_dictionary as base
from synsteg_audit import universal_context as universal
import audit_universal_context_development_v1 as previous
import prepare_qwen15b_gguf_v1 as genprep
import run_qwen15b_gguf_v1 as genrun

SCHEMA = 'synsteg-qwen15b-capacity-v1'
STAGE = 'results/long-summary-development-qwen15b-gguf-v1'
PARENT = '0dd89f821e639ce66ed5c0708ae3a9db41644299'
METHODS = ('base_v1', 'base_v2', 'universal_morph_v3')
ARMS = (('plain_length16', 8), ('plain_length16', 16), ('aes_siv', 8))
DICTIONARIES = {'base_v1': 'data/shared_dictionary_recovery_v1.json',
                'base_v2': 'data/shared_dictionary_recovery_v2.json',
                'universal_morph_v3': 'data/shared_dictionary_morphology_v3.json'}
POLICY = 'results/morphology-development-v3/policy.json'
CODE = ('scripts/audit_qwen15b_capacity_v1.py', 'scripts/audit_universal_context_development_v1.py',
        'scripts/prepare_qwen15b_gguf_v1.py', 'scripts/run_qwen15b_gguf_v1.py',
        'src/synsteg_audit/shared_dictionary.py', 'src/synsteg_audit/universal_context.py',
        'src/synsteg_audit/invariant_context.py', 'src/synsteg_audit/__init__.py')
SOURCE = (*DICTIONARIES.values(), POLICY, *(STAGE + '/' + n for n in ('protocol.json', 'plan.json', 'prompts.jsonl', 'resources.json')))
TERMINAL = frozenset(('generated', 'empty_output', 'response_schema_uncertain', 'protocol_violation_truncated',
                     'context_budget_reject', 'timeout', 'request_error', 'server_error'))
PLAN_COUNTS = {'generate_once': 8, 'abstain_source_gate': 5, 'not_selected_fixed_eight_budget': 7}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def validate_plan(plan):
    rows = plan['records']
    if (len(rows) != 20 or [r['plan_index'] for r in rows] != list(range(1, 21))
            or len({r['record_id'] for r in rows}) != 20 or len({r['source_group_id'] for r in rows}) != 20
            or Counter(r['planned_action'] for r in rows) != Counter(PLAN_COUNTS)
            or any(r['split'] != 'development' for r in rows)
            or [r['plan_index'] for r in rows if r['planned_action'] == 'generate_once'] != genprep.SELECTED_INDICES):
        raise ValueError('Exactly the fixed20 development rows and fixed8 selection required')
    return rows


def resources(root):
    dictionaries = {name: base.PublicDictionary.from_dict(read(root / path)) for name, path in DICTIONARIES.items()}
    policy = universal.UniversalPolicy.from_dict(read(root / POLICY), dictionaries['universal_morph_v3'])
    expected = universal.UniversalPolicy.from_dictionaries(dictionaries['universal_morph_v3'], dictionaries['universal_morph_v3'],
        version='morphology-complete-tag-spectrum-v3-reference-v1', radius=1, orbit_cap=4096)
    if policy.to_dict() != expected.to_dict():
        raise ValueError('The unchanged complete morphology policy is required')
    return dictionaries, policy


def prepare(root, stage):
    root, stage = Path(root).resolve(), Path(stage).resolve()
    output = stage / 'capacity-protocol.json'
    if output.exists() or (stage / 'run').exists():
        raise FileExistsError('Capacity protocol must be new and prepared before generation')
    genprep.verify(root)
    validate_plan(read(stage / 'plan.json'))
    dictionaries, policy = resources(root)
    protocol = {'schema': SCHEMA, 'implementation_base_commit': PARENT,
        'generation_protocol_sha256': sha((stage / 'protocol.json').read_bytes()),
        'methods': list(METHODS), 'arms': [{'framing': f, 'payload_bits': b} for f, b in ARMS],
        'plan_counts': PLAN_COUNTS, 'planned_records': 20, 'selected_records': 8,
        'all_arm_rows': 180, 'selected_arm_rows': 72, 'source_gate_arm_rows': 45, 'not_selected_arm_rows': 63,
        'base_protocol': {'min_group_size': 1, 'max_bits_per_slot': 1},
        'public_dictionary_sha256': {name: d.sha256 for name, d in dictionaries.items()},
        'policy_sha256': policy.sha256,
        'source_files': {p: sha((root / p).read_bytes()) for p in SOURCE},
        'code_files': {p: sha((root / p).read_bytes()) for p in CODE},
        'available_rule': 'Only nonempty status=generated with valid raw response, no schema issues and truncated=False; length limit alone stays available',
        'unavailable_rule': 'Preserve source gates, not-selected, selected errors, empty/schema/truncated outputs separately; capacity=null, no encoding',
        'fixture_key_rule': 'Existing PUBLIC-UNIVERSAL-CONTEXT-KEY-v1 SHA256 rule by source_group_id; public test fixture, not a secret',
        'fixture_payload_rule': 'Existing PUBLIC-UNIVERSAL-CONTEXT-PAYLOAD-v1 SHA256 rule; identical payloads across methods for each source group',
        'receiver': 'Fresh process, only stego+public fixture key+public dictionary/protocol, and public policy for universal; Python I/O guard, not OS sandbox',
        'changes': 'Preserve exact generated text, all saved stego texts and every changed whitespace token for later quality review',
        'mechanical_success_is_semantic_success': False, 'raw_v1_known_semantic_failures': True,
        'quality_review_performed_by_this_script': False, 'outcome_or_quality_filtering': False,
        'generator_runner_unchanged': True, 'new_model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False}
    previous.exclusive(output, previous.formatted(protocol))
    return protocol


def verify(root, stage):
    root, stage = Path(root).resolve(), Path(stage).resolve()
    p = read(stage / 'capacity-protocol.json')
    fixed = {'schema': SCHEMA, 'implementation_base_commit': PARENT, 'methods': list(METHODS),
        'arms': [{'framing': f, 'payload_bits': b} for f, b in ARMS], 'plan_counts': PLAN_COUNTS,
        'planned_records': 20, 'selected_records': 8, 'all_arm_rows': 180, 'selected_arm_rows': 72,
        'source_gate_arm_rows': 45, 'not_selected_arm_rows': 63, 'base_protocol': {'min_group_size': 1, 'max_bits_per_slot': 1}}
    if (any(p.get(k) != v for k, v in fixed.items()) or set(p.get('source_files', {})) != set(SOURCE)
            or set(p.get('code_files', {})) != set(CODE)):
        raise ValueError('Incomplete or changed capacity protocol')
    for manifest in (p['source_files'], p['code_files']):
        for name, expected in manifest.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or sha(path.read_bytes()) != expected:
                raise ValueError('Changed frozen capacity input: ' + name)
    if sha((stage / 'protocol.json').read_bytes()) != p['generation_protocol_sha256']:
        raise ValueError('Generator protocol changed')
    genprep.verify(root)
    plan = validate_plan(read(stage / 'plan.json'))
    dictionaries, policy = resources(root)
    if p['public_dictionary_sha256'] != {name: d.sha256 for name, d in dictionaries.items()} or p['policy_sha256'] != policy.sha256:
        raise ValueError('Public codec resource identity changed')
    return p, plan, dictionaries, policy


def load_ledger(root, stage, frozen_commit):
    root, stage = Path(root).resolve(), Path(stage).resolve()
    plan = validate_plan(read(stage / 'plan.json'))
    run = stage / 'run'
    ledger, execution, summary = (read(run / name) for name in ('ledger.json', 'execution.json', 'summary.json'))
    runtime = read(stage / 'preflight/runtime-preflight.json')['runtime']
    servers = [f['sha256'] for f in runtime['files'] if f['path'] == runtime['server_path']]
    if (execution.get('freeze_commit') != frozen_commit or execution.get('protocol_sha256') != sha((stage / 'protocol.json').read_bytes())
            or execution.get('model_sha256') != read(stage / 'resources.json')['model']['sha256']
            or len(servers) != 1 or execution.get('server_sha256') != servers[0]
            or summary.get('post_run_hashes_verified') is not True or not isinstance(summary.get('finished_at'), str)):
        raise ValueError('Completed generation and matching immutable Git/protocol evidence required')
    if not isinstance(ledger, list) or len(ledger) != 20:
        raise ValueError('Complete20-row generation ledger required')
    evidence = {str((run / n).relative_to(root)): sha((run / n).read_bytes()) for n in ('ledger.json', 'execution.json', 'summary.json')}
    rows = []
    for expected, item in zip(plan, ledger):
        if any(item.get(k) != value for k, value in expected.items()):
            raise ValueError('Ledger altered frozen plan fields/order')
        attempted = item.get('generation_request_attempted')
        if type(attempted) is not bool:
            raise ValueError('Explicit request-attempt state required')
        action, status = expected['planned_action'], item.get('status')
        text, text_hash = item.get('text'), item.get('text_sha256')
        if text is None:
            if text_hash is not None:
                raise ValueError('Missing text cannot have a hash')
        elif not isinstance(text, str) or sha(text.encode()) != text_hash:
            raise ValueError('Ledger text/hash mismatch')
        if action != 'generate_once':
            if status != action or attempted or text is not None:
                raise ValueError('Source gates and not-selected rows must remain ungenerated')
            availability = 'source_gate_abstention' if action == 'abstain_source_gate' else action
        else:
            if status not in TERMINAL:
                raise ValueError('Selected generation is incomplete or has an unknown status')
            directory = run / 'attempts' / ('%02d' % item['plan_index'])
            attempt = read(directory / 'attempt.json')
            if any(attempt.get(k) != item.get(k) for k in ('record_id', 'plan_index', 'status', 'generation_request_attempted', 'text', 'text_sha256')):
                raise ValueError('Attempt/ledger evidence differs')
            evidence[str((directory / 'attempt.json').relative_to(root))] = sha((directory / 'attempt.json').read_bytes())
            if text is not None:
                cover = directory / 'cover.txt'
                if cover.exists():
                    if cover.read_bytes() != text.encode():
                        raise ValueError('Saved cover and exact ledger text differ')
                    evidence[str(cover.relative_to(root))] = sha(cover.read_bytes())
                elif status in ('generated', 'empty_output', 'response_schema_uncertain', 'protocol_violation_truncated'):
                    raise ValueError('Completed response must preserve its cover file')
            if status in ('generated', 'empty_output', 'response_schema_uncertain', 'protocol_violation_truncated'):
                response_path = directory / 'completion.response.json'
                response = read(response_path)
                recomputed = genrun.completion_summary(response, item)
                response_status = 'protocol_violation_truncated' if response.get('truncated') is True else recomputed['status']
                if not attempted or status != response_status or any(item.get(k) != v for k, v in recomputed.items() if k != 'status'):
                    raise ValueError('Saved output status/metadata do not follow raw completion response')
                evidence[str(response_path.relative_to(root))] = sha(response_path.read_bytes())
            if status == 'generated':
                if not text or not text.strip() or item.get('truncated') is not False or item.get('schema_issues') != []:
                    raise ValueError('Generated availability requires nonempty untruncated valid response')
                availability = 'available'
            else:
                availability = 'generation_' + status
        rows.append({'audit_record_id': 'qwen15b_gguf_v1:' + item['record_id'],
            'record_id': item['record_id'], 'plan_index': item['plan_index'], 'source_group_id': item['source_group_id'],
            'split': 'development', 'cohort': 'qwen15b_gguf_v1', 'planned_action': action,
            'generation_status': status, 'generation_request_attempted': attempted,
            'text': text, 'text_sha256': text_hash, 'availability': availability,
            'generation_metadata': {k: v for k, v in item.items() if k not in expected and k not in ('text', 'text_sha256')},
            'attribution': expected.get('attribution')})
    if any(summary.get(k) != v for k, v in genrun.summarize(ledger).items()):
        raise ValueError('Generation summary does not match the complete20-row ledger')
    return rows, evidence


def raw_receiver_main():
    """Same inspected-program I/O guard as the frozen universal receiver."""
    import socket
    from cryptography.hazmat.primitives.ciphers.aead import AESSIV
    warm = AESSIV(bytes(64))
    warm.decrypt(warm.encrypt(b'', [b'public-warmup']), [b'public-warmup'])
    def guard(event, args):
        if event in {'open', 'os.listdir', 'os.scandir', 'os.system', 'os.posix_spawn'} or event.startswith(('socket.', 'subprocess.')):
            raise PermissionError('Receiver external I/O forbidden')
    sys.addaudithook(guard)
    checks = {}
    for name, operation in (('file', lambda: open('/receiver-forbidden')), ('socket', socket.socket),
                            ('subprocess', lambda: subprocess.Popen(['receiver-forbidden']))):
        try:
            operation()
        except PermissionError:
            checks[name] = True
        else:
            raise RuntimeError('Receiver I/O guard self-check failed')
    request = json.loads(sys.stdin.buffer.read())
    if not isinstance(request, dict) or set(request) != {'key_hex', 'stego_text', 'public_dictionary', 'public_protocol'}:
        raise ValueError('Receiver accepts only key, stego and public resources')
    p = request['public_protocol']
    protocol = base.Protocol(1, 1, p['framing'])
    if p != protocol.to_dict():
        raise ValueError('Unexpected raw protocol')
    dictionary = base.PublicDictionary.from_dict(request['public_dictionary'])
    decoded = base.decode(request['stego_text'], bytes.fromhex(request['key_hex']), dictionary, protocol)
    print(json.dumps({'status': 'decoded', 'payload_hex': decoded.hex(), 'input_keys': sorted(request),
        'dictionary_sha256': dictionary.sha256, 'protocol_sha256': protocol.sha256,
        'io_guard_self_checks': checks, 'isolation_scope': 'Python audit guard, not OS sandbox'}, sort_keys=True))


def raw_isolated_receive(root, text, key, dictionary, framing):
    protocol = base.Protocol(1, 1, framing)
    request = {'key_hex': key.hex(), 'stego_text': text, 'public_dictionary': dictionary.to_dict(), 'public_protocol': protocol.to_dict()}
    env = {n: os.environ[n] for n in ('SYSTEMROOT', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP') if n in os.environ}
    env.update(PATH=os.defpath, PYTHONIOENCODING='utf-8')
    with tempfile.TemporaryDirectory(prefix='qwen-raw-receiver-') as directory:
        child = subprocess.run([sys.executable, '-I', str(root / 'scripts/audit_qwen15b_capacity_v1.py'), 'raw-receiver'],
            input=previous.canonical(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=directory, env=env, timeout=60, check=False)
    response = json.loads(child.stdout)
    if (child.returncode != 0 or response.get('status') != 'decoded' or response.get('input_keys') != sorted(request)
            or response.get('dictionary_sha256') != dictionary.sha256 or response.get('protocol_sha256') != protocol.sha256
            or response.get('io_guard_self_checks') != {'file': True, 'socket': True, 'subprocess': True}):
        raise ValueError('Raw fresh-process receiver evidence mismatched')
    return bytes.fromhex(response['payload_hex']), {'returncode': child.returncode, 'input_keys': response['input_keys'],
        'io_guard_self_checks': response['io_guard_self_checks'], 'stdout_sha256': sha(child.stdout), 'stderr_sha256': sha(child.stderr)}


def changes(before, after):
    a, b = before.split(), after.split()
    if len(a) != len(b):
        raise ValueError('Codec changed whitespace-token count')
    return [{'token_index': i, 'original_token': x, 'stego_token': y} for i, (x, y) in enumerate(zip(a, b)) if x != y]


def evaluate_raw(root, row, dictionary):
    text, state = row['text'], row['availability']
    entry = {'input_status': state, 'capacity_bits': None, 'analysis': None, 'arms': {}}
    analysis = None
    if state == 'available':
        try:
            analysis = base.analyze_text(text, dictionary, base.Protocol(1, 1))
            entry.update(capacity_bits=analysis.capacity_bits, analysis=asdict(analysis))
        except base.CodecError as error:
            entry.update(input_status='invalid_input', error_type=type(error).__name__)
    key = previous.fixture('KEY', row['source_group_id'])
    for framing, bits in ARMS:
        payload = previous.fixture('PAYLOAD', row['source_group_id'], bits // 8)
        protocol = base.Protocol(1, 1, framing)
        required = bits + protocol.frame_overhead_bits
        arm = {'payload_bits': bits, 'framing': framing, 'required_frame_bits': required,
            'encoding_called': False, 'mechanical_success': False}
        if entry['input_status'] != 'available':
            arm['status'] = entry['input_status']
        elif analysis.capacity_bits < required:
            arm['status'] = 'insufficient_capacity'
        else:
            arm['encoding_called'] = True
            try:
                encoded = base.encode(text, payload, key, dictionary, protocol)
                arm.update(stego_text=encoded.stego_text, stego_sha256=sha(encoded.stego_text.encode()),
                    payload_hex=payload.hex(), changed_slots=encoded.changed_slots, token_changes=changes(text, encoded.stego_text))
                decoded, evidence = raw_isolated_receive(root, encoded.stego_text, key, dictionary, framing)
                before, after = text.split(), encoded.stego_text.split()
                mutable = {s.token_index for s in analysis.slots}
                protected = len(before) == len(after) and all(before[i] == after[i] for i in range(len(before)) if i not in mutable)
                invariant = base.analyze_text(encoded.stego_text, dictionary, protocol).normalization_sha256 == analysis.normalization_sha256
                okay = decoded == payload and protected and invariant
                arm.update(status='decoded' if okay else 'payload_or_protection_mismatch', mechanical_success=okay,
                    decoded_payload_hex=decoded.hex(), protected_spellings_preserved=protected,
                    normalization_preserved=invariant, receiver=evidence)
            except (base.CodecError, ValueError, OSError, subprocess.SubprocessError) as error:
                arm.update(status='codec_or_receiver_failure', error_type=type(error).__name__)
        entry['arms'][f'{framing}__{bits}'] = arm
    return entry


def evaluate_record(root, row, dictionaries, policy):
    result = {**row, 'methods': {name: evaluate_raw(root, row, dictionaries[name]) for name in ('base_v1', 'base_v2')}}
    audited = previous.evaluate(root, row, {'morph_v3': dictionaries['universal_morph_v3']}, {'morph_v3': policy})
    result['methods']['universal_morph_v3'] = audited['methods']['universal_morph_v3']
    for name, entry in result['methods'].items():
        entry['semantic_safety_certified'] = False
        entry['known_unsafe_reference'] = name == 'base_v1'
        for arm in entry['arms'].values():
            arm['semantic_success'] = None
            if 'stego_text' in arm:
                arm['stego_sha256'] = sha(arm['stego_text'].encode())
                arm['token_changes'] = changes(row['text'], arm['stego_text'])
                if len(arm['token_changes']) != arm['changed_slots']:
                    raise ValueError('Saved changed-slot count differs from exact token differences')
    return result


def flatten(records):
    return [{**{k: row[k] for k in ('audit_record_id', 'record_id', 'plan_index', 'source_group_id', 'planned_action', 'generation_status', 'availability', 'text_sha256')},
             'method': method, 'arm_id': arm_id, 'capacity_bits': entry['capacity_bits'], **arm}
            for row in records for method, entry in row['methods'].items() for arm_id, arm in entry['arms'].items()]


def summarize(records):
    if (len(records) != 20 or [r['plan_index'] for r in records] != list(range(1, 21))
            or len({r['audit_record_id'] for r in records}) != 20
            or Counter(r['planned_action'] for r in records) != Counter(PLAN_COUNTS)
            or any(set(r['methods']) != set(METHODS) for r in records)
            or any(set(m['arms']) != {f'{f}__{b}' for f, b in ARMS} for r in records for m in r['methods'].values())):
        raise ValueError('Full20-record/three-method/three-arm denominator required')
    rows = flatten(records)
    selected = [r for r in rows if r['planned_action'] == 'generate_once']
    for row in rows:
        if (type(row['encoding_called']) is not bool or type(row['mechanical_success']) is not bool
                or row['mechanical_success'] != (row['status'] == 'decoded')
                or row.get('semantic_success') is not None
                or (row['capacity_bits'] is not None and (type(row['capacity_bits']) is not int or row['capacity_bits'] < 0))):
            raise ValueError('Inconsistent mechanical outcome fields')
        if row['availability'] != 'available' and (row['capacity_bits'] is not None or row['encoding_called'] or row['mechanical_success']):
            raise ValueError('Unavailable output must not become measured zero capacity or successful encoding')
        if row['availability'] != 'available' and row['status'] != row['availability']:
            raise ValueError('Unavailable output status must remain explicit in all arms')
        if row['status'] == 'decoded' and (not row['encoding_called'] or row['capacity_bits'] is None
                or row['capacity_bits'] < row['required_frame_bits'] or not isinstance(row.get('stego_text'), str)
                or row.get('decoded_payload_hex') != row.get('payload_hex')):
            raise ValueError('Decoded claim lacks complete-frame/payload evidence')
        if row['status'] == 'insufficient_capacity' and (row['encoding_called'] or row['capacity_bits'] is None
                or row['capacity_bits'] >= row['required_frame_bits']):
            raise ValueError('Incorrect insufficient-capacity claim')
    return {'schema': SCHEMA + '-summary', 'planned_records': 20, 'selected_records': 8, 'all_arm_rows': len(rows),
        'selected_arm_rows': len(selected), 'source_gate_arm_rows': sum(r['planned_action'] == 'abstain_source_gate' for r in rows),
        'not_selected_arm_rows': sum(r['planned_action'] == 'not_selected_fixed_eight_budget' for r in rows),
        'availability_all20': dict(Counter(r['availability'] for r in records)),
        'all180_outcomes': dict(Counter(r['status'] for r in rows)), 'selected72_outcomes': dict(Counter(r['status'] for r in selected)),
        'methods': {name: {'selected_denominator': 8, 'planned_denominator': 20,
            'available_denominator': sum(r['availability'] == 'available' for r in records),
            'gross_capacity_bits': previous.stats([r['methods'][name]['capacity_bits'] for r in records if r['methods'][name]['capacity_bits'] is not None]),
            'arms': {f'{f}__{b}': {'all20_outcomes': dict(Counter(r['methods'][name]['arms'][f'{f}__{b}']['status'] for r in records)),
                'selected8_outcomes': dict(Counter(r['methods'][name]['arms'][f'{f}__{b}']['status'] for r in records if r['planned_action'] == 'generate_once')),
                'mechanical_successes': sum(r['methods'][name]['arms'][f'{f}__{b}']['mechanical_success'] for r in records)} for f, b in ARMS}}
            for name in METHODS},
        'mechanical_success_is_semantic_success': False, 'semantic_safety_certified': False,
        'quality_review_performed': False, 'new_model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False}


def execute(root, stage, output, frozen_commit):
    root, stage, output = Path(root).resolve(), Path(stage).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Never overwrite mechanical evidence')
    if not isinstance(frozen_commit, str) or not re.fullmatch('[a-f0-9]{40}', frozen_commit):
        raise ValueError('Exact externally verified Git freeze commit required')
    protocol, _, dictionaries, policy = verify(root, stage)
    rows, evidence = load_ledger(root, stage, frozen_commit)
    started = datetime.now(timezone.utc).isoformat()
    result = [evaluate_record(root, row, dictionaries, policy) for row in rows]
    summary = summarize(result)
    verify(root, stage)
    if any(sha((root / name).read_bytes()) != digest for name, digest in evidence.items()):
        raise ValueError('Saved generation input changed during evaluation')
    content = {'records.jsonl': b''.join(previous.canonical(r) + b'\n' for r in result),
        'arms.jsonl': b''.join(previous.canonical(r) + b'\n' for r in flatten(result)), 'summary.json': previous.formatted(summary)}
    content['execution.json'] = previous.formatted({'schema': SCHEMA + '-execution', 'frozen_commit': frozen_commit,
        'freeze_verification_scope': 'Operator verifies Git; script verifies frozen hashes without network',
        'capacity_protocol_sha256': sha((stage / 'capacity-protocol.json').read_bytes()),
        'generation_protocol_sha256': protocol['generation_protocol_sha256'], 'generation_input_sha256': evidence,
        'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'output_sha256': {name: sha(data) for name, data in content.items()}, 'records': 20, 'arms': 180,
        'new_model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False})
    output.mkdir(parents=True, exist_ok=False)
    for name, data in content.items():
        previous.exclusive(output / name, data)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'verify', 'execute', 'raw-receiver'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--stage', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--frozen-commit')
    a = p.parse_args(argv)
    if a.command == 'raw-receiver':
        return raw_receiver_main()
    root = a.root.resolve(); stage = a.stage or root / STAGE
    if a.command == 'prepare':
        result = prepare(root, stage)
    elif a.command == 'verify':
        verify(root, stage); result = {'status': 'verified', 'records': 20, 'selected': 8, 'arms': 180}
    else:
        result = execute(root, stage, a.output or stage / 'capacity-audit', a.frozen_commit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
