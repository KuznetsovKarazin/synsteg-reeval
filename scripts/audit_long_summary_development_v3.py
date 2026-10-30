#!/usr/bin/env python3
"""Offline mechanical audit of all twenty frozen long-summary source groups.

No model/API calls, quality scores, outcome filtering or causal comparisons.
Use --output for a new audit directory; existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from synsteg_audit import shared_dictionary as base
from synsteg_audit import invariant_context as context
from synsteg_audit.budget_ledger import canonical_hash
import audit_invariant_context_development as previous

SCHEMA = 'synsteg-long-summary-development-v3-mechanical-audit'
PREPARED = 'results/long-summary-development-v3'
SOURCE_WINDOWS = 'data/development-full-revisions-v3/source-windows.jsonl'
RESOURCE_FILES = {
    'v1': 'data/shared_dictionary_recovery_v1.json',
    'v2': 'data/shared_dictionary_recovery_v2.json',
    'guard': 'results/invariant-context-development-v1/guard_policy.json',
}
CODE_FILES = ('scripts/audit_long_summary_development_v3.py',
              'scripts/audit_invariant_context_development.py',
              'scripts/run_shared_dictionary_pilot.py',
              'src/synsteg_audit/shared_dictionary.py',
              'src/synsteg_audit/invariant_context.py',
              'src/synsteg_audit/__init__.py')
METHODS = ('base_v1', 'base_v2', 'context_guard_v1')
ARMS = (('plain_length16', 8), ('plain_length16', 16), ('aes_siv', 8))
META_KEYS = ('record_id', 'source_group_id', 'split', 'topic', 'genre', 'target_words', 'prompt_sha256')
ATTEMPT_STATES = {'completed', 'reserved_outcome_unknown',
                  'failed_or_unknown_no_retry', 'accounting_bound_violation_stop'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def formatted(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def checked(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe bound path')
    return path


def verify_preparation(root, prepared):
    """Validate every frozen input and the complete gate; no extraction rebuild."""
    protocol = read(prepared / 'protocol.prepared.json')
    if (protocol.get('schema') != 'synsteg-long-summary-development-v3'
            or protocol.get('planned_source_groups') != 20
            or protocol.get('planned_generations') != 15
            or protocol.get('paid_calls_allowed') is not False
            or protocol.get('test_data_used') is not False):
        raise ValueError('Unknown or incomplete development preparation')
    bindings = dict(protocol['input_hashes'])
    bindings.update(protocol.get('mechanical_analysis', {}).get('input_hashes', {}))
    for name in (*CODE_FILES, *RESOURCE_FILES.values(), SOURCE_WINDOWS):
        if name not in bindings:
            raise ValueError('Missing frozen audit dependency: ' + name)
    inventory = {}
    for name, expected in bindings.items():
        path = checked(root, name)
        actual = sha(path.read_bytes())
        if actual != expected:
            raise ValueError('Frozen input changed: ' + name)
        inventory[str(path)] = actual
    for name, expected in protocol['files'].items():
        path = checked(prepared, name)
        actual = sha(path.read_bytes())
        if actual != expected:
            raise ValueError('Prepared file changed: ' + name)
        inventory[str(path)] = actual
    for name in ('plan.json', 'prompts.jsonl', 'review_rubric.json'):
        if name not in protocol['files']:
            raise ValueError('Missing frozen preparation file')
    inventory[str(prepared / 'protocol.prepared.json')] = sha((prepared / 'protocol.prepared.json').read_bytes())
    plan = read(prepared / 'plan.json')['records']
    windows = jsonlines(root / SOURCE_WINDOWS)
    prompts = jsonlines(prepared / 'prompts.jsonl')
    if len(plan) != 20 or len(windows) != 20 or len(prompts) != 15:
        raise ValueError('All twenty source groups and fifteen prompts required')
    for rows in (plan, windows, prompts):
        if any(r.get('split') != 'development' for r in rows):
            raise ValueError('Closed/nondevelopment split refused')
        if any(len({r[k] for r in rows}) != len(rows) for k in ('record_id', 'source_group_id')):
            raise ValueError('Duplicate planned identity')
    eligible = []
    for index, (item, window) in enumerate(zip(plan, windows), 1):
        if item['plan_index'] != index or any(item[k] != window[k] for k in ('record_id', 'source_group_id', 'source_window_sha256')):
            raise ValueError('Source window/plan identity mismatch')
        text = window['source_window']
        if (not isinstance(text, str) or sha(text.encode()) != item['source_window_sha256']
                or len(text.split()) != item['source_words']
                or item['source_status'] != window['status']
                or item['target_words'] != window['original_long_target_words']):
            raise ValueError('Source window content/gate mismatch')
        allowed = item['source_status'] == 'word_heuristic_met'
        if item['planned_action'] != ('generate_once' if allowed else 'abstain_source_gate'):
            raise ValueError('Changed source gate')
        if allowed:
            eligible.append(item)
        elif item.get('prompt_sha256') is not None:
            raise ValueError('Abstained group has a generation prompt')
    if len(eligible) != 15:
        raise ValueError('Source gate changed the planned denominator')
    by_window = {w['record_id']: w for w in windows}
    for item, prompt in zip(eligible, prompts):
        if any(item[k] != prompt[k] for k in ('record_id', 'source_group_id', 'target_words', 'prompt_sha256')):
            raise ValueError('Ordered eligible prompt mismatch')
        if (sha(prompt['prompt'].encode()) != prompt['prompt_sha256']
                or prompt['excerpt'] != by_window[item['record_id']]['source_window']
                or prompt['excerpt_sha256'] != item['source_window_sha256']):
            raise ValueError('Prompt/source bytes changed')
    return protocol, plan, windows, prompts, inventory


def load_run(root, prepared, protocol, prompts, inventory):
    run_dir = prepared / 'run'
    path = run_dir / 'run.json'
    run = read(path)
    generator = protocol['generator']
    expected_meta = [{k: p[k] for k in META_KEYS} for p in prompts]
    if (run.get('schema') != 'synsteg-budget-generation-run-v1'
            or run.get('split') != 'development' or run.get('provider') != 'local'
            or run.get('selected_records') != expected_meta
            or run.get('selected_records_count') != 15
            or run.get('prompts_sha256') != sha((prepared / 'prompts.jsonl').read_bytes())
            or run.get('automatic_retries') != 0):
        raise ValueError('Runner plan differs from all fifteen frozen prompts')
    for name in ('provider', 'model', 'revision', 'dtype'):
        if run.get(name) != generator[name]:
            raise ValueError('Runner generator mismatch: ' + name)
    for actual, expected in [('max_output_tokens', 'max_output_tokens'), ('system', 'system'),
                             ('seed', 'master_seed'), ('cpu_threads', 'cpu_threads'),
                             ('temperature', 'temperature'), ('top_p', 'top_p'), ('top_k', 'top_k')]:
        if run['settings'].get(actual) != generator[expected]:
            raise ValueError('Runner setting mismatch: ' + actual)
    for key, name in [('runner_sha256', 'scripts/run_budget_generation.py'),
                      ('ledger_code_sha256', 'src/synsteg_audit/budget_ledger.py')]:
        if run.get(key) != sha((root / name).read_bytes()):
            raise ValueError('Runner implementation mismatch')
    inventory[str(path)] = sha(path.read_bytes())
    runtime_path = run_dir / 'runtime.json'
    runtime_sha = canonical_hash(read(runtime_path))
    inventory[str(runtime_path)] = sha(runtime_path.read_bytes())
    by_id = {r['record_id']: r for r in expected_meta}
    attempts = {}
    attempt_files = sorted((run_dir / 'attempts').glob('*.json'))
    for path in attempt_files:
        item = read(path); meta = item.get('record', {}); rid = meta.get('record_id')
        identity = canonical_hash([run['provider'], run['model'], rid])
        if (rid not in by_id or rid in attempts or meta != by_id[rid]
                or item.get('schema') != 'synsteg-budget-generation-attempt-v1'
                or item.get('provider') != run['provider'] or item.get('model') != run['model']
                or item.get('identity') != identity or path.name != identity + '.json'
                or item.get('automatic_retries') != 0 or item.get('status') not in ATTEMPT_STATES):
            raise ValueError('Unexpected, duplicate or altered generation attempt')
        fingerprint = canonical_hash({'record': meta, 'provider': run['provider'], 'model': run['model'],
            'revision': run['revision'], 'dtype': run['dtype'], 'settings': run['settings'],
            'pricing_sha256': run['pricing_sha256'], 'prompts_sha256': run['prompts_sha256'],
            'runtime_sha256': runtime_sha, 'implementation': {k: run[k] for k in
            ('runner_sha256', 'ledger_code_sha256', 'transport_code_sha256')}})
        if item.get('request_fingerprint') != fingerprint:
            raise ValueError('Attempt request fingerprint differs from frozen run')
        if item['status'] == 'completed':
            result = item.get('result')
            if (not isinstance(result, dict) or not isinstance(result.get('cover_text'), str)
                    or result.get('returned_model') != run['model']
                    or result.get('revision') != run['revision']
                    or type(result.get('hit_output_limit')) is not bool
                    or result.get('stop_reason') not in ('end_turn', 'max_tokens', 'refusal')):
                raise ValueError('Completed result model/shape mismatch')
        attempts[rid] = item
        inventory[str(path)] = sha(path.read_bytes())
    return run, attempts, attempt_files


def availability(item, attempt):
    if item['planned_action'] == 'abstain_source_gate':
        return 'source_gate_abstention', None
    if attempt is None:
        return 'not_attempted', None
    if attempt['status'] != 'completed':
        return attempt['status'], None
    result = attempt.get('result')
    if not isinstance(result, dict) or not isinstance(result.get('cover_text'), str):
        raise ValueError('Completed attempt has no string cover')
    if result.get('stop_reason') == 'refusal':
        return 'model_refusal', result['cover_text']
    if not result['cover_text'].strip():
        return 'empty_output', result['cover_text']
    return 'available', result['cover_text']


def analyze(text, method, dictionaries, policy):
    dictionary = dictionaries['v2' if method == 'base_v2' else 'v1']
    if method == 'context_guard_v1':
        value = context.analyze_text(text, dictionary, policy)
        tokens = text.split()
        protected = [{'token_index': i, 'token': tokens[i]} for i in value.protected_token_indices]
        return value.capacity_bits, protected
    return base.analyze_text(text, dictionary, base.Protocol(min_group_size=1, max_bits_per_slot=1)).capacity_bits, []


def source_capacities(text, dictionaries, policy):
    result = {}
    for method in METHODS:
        try:
            capacity, protected = analyze(text, method, dictionaries, policy)
            result[method] = {'status': 'measured', 'capacity_bits': capacity,
                              'protected_tokens': protected, 'encoding_called': False}
        except base.CodecError as error:
            result[method] = {'status': 'invalid_input', 'capacity_bits': None,
                              'error_type': type(error).__name__, 'encoding_called': False}
    return result


def measure(root, text, state, group, dictionaries, policy):
    """All nine arms, including missing/refused/zero-capacity cases."""
    key = previous.fixture('KEY', group)
    methods = {}
    for method in METHODS:
        dictionary = dictionaries['v2' if method == 'base_v2' else 'v1']
        entry = {'input_status': state, 'capacity_bits': None, 'protected_tokens': [], 'arms': {}}
        if state == 'available':
            try:
                entry['capacity_bits'], entry['protected_tokens'] = analyze(text, method, dictionaries, policy)
            except base.CodecError as error:
                entry.update(input_status='invalid_input', error_type=type(error).__name__)
        for framing, bits in ARMS:
            payload = previous.fixture('PAYLOAD', group, bits // 8)
            needed = bits + (16 if framing == 'plain_length16' else 144)
            arm = {'payload_bits': bits, 'payload_hex': payload.hex(), 'framing': framing,
                   'required_frame_bits': needed, 'full_success': False, 'encoding_called': False}
            if entry['input_status'] != 'available':
                arm['status'] = entry['input_status']
            elif entry['capacity_bits'] < needed:
                arm['status'] = 'insufficient_capacity'
            else:
                arm['encoding_called'] = True
                try:
                    if method == 'context_guard_v1':
                        encoded = context.encode(text, payload, key, dictionary, policy, context.ContextProtocol(framing=framing))
                    else:
                        encoded = base.encode(text, payload, key, dictionary,
                            base.Protocol(min_group_size=1, max_bits_per_slot=1, framing=framing))
                    decoded, receiver = previous.isolated_receive(root,
                        'wrapper_guard' if method == 'context_guard_v1' else 'base_v1',
                        encoded.stego_text, key, dictionary, policy if method == 'context_guard_v1' else None, framing)
                    tokens = encoded.stego_text.split()
                    preserved = all(tokens[t['token_index']] == t['token'] for t in entry['protected_tokens'])
                    okay = decoded == payload and preserved
                    arm.update(status='decoded' if okay else 'payload_or_protection_mismatch', full_success=okay,
                        decoded_payload_hex=decoded.hex(), stego_text=encoded.stego_text,
                        stego_text_sha256=sha(encoded.stego_text.encode()), changed_slots=encoded.changed_slots,
                        protected_spellings_preserved=preserved, receiver=receiver)
                except (base.CodecError, ValueError, OSError, previous.subprocess.SubprocessError) as error:
                    arm.update(status='codec_or_receiver_failure', error_type=type(error).__name__)
            entry['arms'][f'{framing}__{bits}'] = arm
        methods[method] = entry
    return methods


def summarize(records):
    if len(records) != 20 or sum(r['planned_action'] == 'generate_once' for r in records) != 15:
        raise ValueError('Complete planned denominator required')
    available = [r for r in records if r['availability'] == 'available']
    summary = {'schema': SCHEMA + '-summary', 'planned_source_groups': 20, 'planned_generations': 15,
        'planned_source_abstentions': 5, 'availability_counts': dict(Counter(r['availability'] for r in records)),
        'available_covers': len(available), 'within_soft_word_band': sum(r['within_soft_word_band'] for r in available),
        'exact_source_copies': sum(r['exact_source_copy'] for r in available),
        'whitespace_normalized_source_copies': sum(r['whitespace_normalized_source_copy'] for r in available),
        'output_limit_hits': sum(r['hit_output_limit'] for r in available), 'methods': {},
        'factual_quality': 'not_assessed', 'population_inference': False, 'causal_comparison_to_v2': False,
        'fixture_keys_are_public_not_secrets': True, 'new_model_calls': 0, 'paid_calls': 0,
        'test_split_opened': False, 'source_reference_is_capacity_only_not_generated_cover': True}
    for method in METHODS:
        entries = [r['methods'][method] for r in records]
        values = [r['capacity_bits'] for r in entries if r['capacity_bits'] is not None]
        arms = {}
        for framing, bits in ARMS:
            name = f'{framing}__{bits}'
            counts = dict(Counter(e['arms'][name]['status'] for e in entries))
            successes = sum(e['arms'][name]['full_success'] for e in entries)
            arms[name] = {'statuses_all20': counts, 'full_success': successes,
                          'fraction_all20': successes / 20, 'fraction_planned_generation15': successes / 15,
                          'denominator_all_sources': 20, 'denominator_planned_generations': 15}
        summary['methods'][method] = {'available_capacity_records': len(values),
            'capacity_bits_available': {'min': min(values), 'max': max(values), 'mean': statistics.mean(values)} if values else None,
            'arms': arms}
    return summary


def execute(root, prepared, output):
    root, prepared, output = Path(root).resolve(), Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refuse to overwrite an existing audit')
    protocol, plan, windows, prompts, inventory = verify_preparation(root, prepared)
    run, attempts, attempt_files = load_run(root, prepared, protocol, prompts, inventory)
    dictionaries = {v: base.PublicDictionary.from_dict(read(root / RESOURCE_FILES[v])) for v in ('v1', 'v2')}
    policy = context.ProtectedPhrasePolicy.from_dict(read(root / RESOURCE_FILES['guard']), dictionaries['v1'])
    records, materials = [], []
    started = datetime.now(timezone.utc).isoformat()
    for item, window in zip(plan, windows):
        rid = item['record_id']; attempt = attempts.get(rid)
        state, text = availability(item, attempt)
        result = attempt.get('result') if attempt else None
        if attempt is not None and attempt['status'] == 'completed' and result.get('seed') != item['derived_sampling_seed']:
            raise ValueError('Returned sampling seed mismatch')
        record = {**item, 'availability': state, 'generation_status': attempt['status'] if attempt else 'not_attempted',
                  'cover_text': text, 'cover_sha256': sha(text.encode()) if text is not None else None,
                  'public_fixture_key_hex': previous.fixture('KEY', item['source_group_id']).hex(),
                  'source_reference_capacity': source_capacities(window['source_window'], dictionaries, policy),
                  'methods': measure(root, text, state, item['source_group_id'], dictionaries, policy)}
        if state == 'available':
            words = len(text.split()); low, high = item['preferred_word_band']
            record.update(cover_words=words, within_soft_word_band=low <= words <= high,
                exact_source_copy=text == window['source_window'],
                whitespace_normalized_source_copy=' '.join(text.split()) == ' '.join(window['source_window'].split()),
                hit_output_limit=result.get('hit_output_limit', False), stop_reason=result.get('stop_reason'),
                usage=result.get('usage'), elapsed_seconds=result.get('elapsed_seconds'), sampling_seed=result['seed'])
        records.append(record)
        materials.append({'plan_index': item['plan_index'], 'record_id': rid, 'source_group_id': item['source_group_id'],
            'attribution': item['attribution'], 'source_window': window['source_window'],
            'source_window_sha256': window['source_window_sha256'], 'target_words': item['target_words'],
            'availability': state, 'cover_text': text, 'cover_sha256': record['cover_sha256'],
            'generated_result_if_failed_or_refused': result if state != 'available' else None,
            'stego_by_method_and_arm': {m: {a: v['stego_text'] for a, v in e['arms'].items() if 'stego_text' in v}
                                      for m, e in record['methods'].items()},
            'factual_ratings': None})
    for name, digest in inventory.items():
        if sha(Path(name).read_bytes()) != digest:
            raise ValueError('Input changed during audit: ' + name)
    if sorted((prepared / 'run/attempts').glob('*.json')) != attempt_files:
        raise ValueError('Generation attempts changed during audit')
    summary = summarize(records)
    output.mkdir(parents=True, exist_ok=False)
    content = {'records.jsonl': b''.join(previous.canonical(r) + b'\n' for r in records),
               'review-materials.jsonl': b''.join(previous.canonical(r) + b'\n' for r in materials),
               'summary.json': formatted(summary)}
    content['execution.json'] = formatted({'schema': SCHEMA + '-execution', 'status': 'completed',
        'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_hashes': {str(Path(p).relative_to(root)): h for p, h in inventory.items()},
        'output_sha256': {n: sha(b) for n, b in content.items()}, 'records': 20,
        'arms': 180, 'source_capacity_reference_records': 20,
        'encoding_calls': sum(a['encoding_called'] for r in records for m in r['methods'].values() for a in m['arms'].values()),
        'new_model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False})
    for name, data in content.items():
        with (output / name).open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve(); prepared = (args.prepared or root / PREPARED).resolve()
    result = execute(root, prepared, args.output or prepared / 'mechanical-audit')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
