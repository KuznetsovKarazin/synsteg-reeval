#!/usr/bin/env python3
"""Offline full300 merger; no inference, tokenizer, network or Git writes.

The exact historical base is inherited, not a fresh raw audit of its 89 covers.
New outcomes require complete supplied shard bytes and preservation receipts.
Global completeness still requires the caller's fresh remote reservation census.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re

ID = 'mixed-development-full300-merger-v2'
CAMPAIGN = 'large-campaign-development-v2'
BASE_REF = '4050291345c43776e9c93528b6afe292c062ba7a'
MODELS = {
    'tinyllama11': ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'fe8a4ea1ffedaf415f4da2f062534de366a451e6'),
    'qwen05': ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775'),
    'falcon3_1b': ('tiiuae/Falcon3-1B-Instruct', '28ba2251970a01dd1edc7ba7dad2eb71216ccfdf'),
}
BASE_FILES = {
    'source': 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl',
    'base': 'results/baseline-single-record-recovery-v1-root/full300-after090.json',
    'checkpoint': 'results/continuation-20261028-reconciliation-v1/status.json',
    'guard': 'results/current-git-status-audit-20261028-v1/no-replay-guard.json',
    'prior_audit': 'results/current-git-status-audit-20261028-v1/audit.json',
}
# Filled from actual immutable Git byte reads, not generated model outcomes.
BASE_PINS = {
    'source': '3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1',
    'base': 'b25f56576ec2c47898b80cb95a59e1d70a15e99037a78b6b96a3f430ca4aaac7',
    'guard': 'ed7c17d68df5cc524ec8e441f3febb2a96e6d8108eda6f22f2ac154fae74db0c',
    'prior_audit': '9b1e2d87221b1b1d2e0c6a4907c31cee3fa25df604e553abba185efdcba4afa4',
    'checkpoint': 'f34d1a380385d570e6f7336f73ef41d2822ccf9733e324a7f8f9dc188f43d686',
}
SETTINGS = {'api_version': None, 'cpu_threads': 6, 'max_output_tokens': 1024,
            'request_options': None, 'seed': 20261024,
            'system': 'Write clear English prose. Return only the requested text, without a heading or commentary.',
            'temperature': 0.8, 'top_k': 50, 'top_p': 0.95}
RUNNER_SHA256 = '4d5ab74c78c58214c1b5e11a346b4339eb0c6d08e713e8909d0d5fd9947c7e02'
LEDGER_SHA256 = 'c04ca4e3141a9b9873054ab71c38545705353c8609c12ec0536eaabcd2e0b19b'
TERMINAL_CORE = {'activation-receipt.json', 'artifact-manifest.json', 'child.json',
                 'process.json', 'reservation.json', 'stdout.log', 'stderr.log',
                 'run/run.json', 'run/runtime.json', 'run/local_ledger/ledger.json'}
COMPLETED_CORE = TERMINAL_CORE | {'run/summary.json', 'run/local_ledger/campaign.lock',
                                  'run/run_guard/campaign.lock'}
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
HEX40 = re.compile(r'[0-9a-f]{40}\Z')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_sha(obj):
    return sha(json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode())


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def parse(data):
    def bad_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    value = json.loads(data, object_pairs_hook=unique_pairs, parse_constant=bad_constant)
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'nonfinite JSON number')
        elif isinstance(item, dict):
            for child in item.values(): finite(child)
        elif isinstance(item, list):
            for child in item: finite(child)
    finite(value)
    return value


def relative(path):
    require(isinstance(path, str) and path, 'empty/nonstring path')
    p = Path(path)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == path, 'unsafe/noncanonical path')
    return p


def number(value, name, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid integer ' + name)
    return value


def exact_keys(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys), label + ' keys differ')


def inventory(root):
    root = Path(root)
    for parent in (root, *root.parents):
        require(not parent.is_symlink(), 'symlink root/ancestor refused')
    require(root.is_dir(), 'input root absent')
    result = {}
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'symlink evidence refused')
        if path.is_file():
            result[str(path.relative_to(root))] = sha(path.read_bytes())
    return result


def bound_inputs(root, manifest):
    exact_keys(manifest, ['schema', 'campaign', 'base_ref', 'snapshot_ref', 'files_sha256',
                          'model_contracts', 'new_shards', 'reservation_scope'], 'manifest')
    require(manifest['schema'] == ID + '-input' and manifest['campaign'] == CAMPAIGN, 'wrong manifest identity')
    require(manifest['base_ref'] == BASE_REF, 'wrong historical base ref')
    require(isinstance(manifest['snapshot_ref'], str) and HEX40.fullmatch(manifest['snapshot_ref']), 'invalid snapshot commit')
    pins = manifest['files_sha256']
    require(isinstance(pins, dict) and pins, 'empty file inventory')
    for path, digest in pins.items():
        relative(path)
        require(isinstance(digest, str) and HEX64.fullmatch(digest), 'invalid file digest')
    require(inventory(root) == pins, 'missing, changed or orphan input bytes')
    for role, path in BASE_FILES.items():
        require(path in pins, 'missing pinned base input: ' + role)
        if role in BASE_PINS:
            require(pins[path] == BASE_PINS[role], 'historical source/base pin differs: ' + role)
    exact_keys(manifest['model_contracts'], MODELS, 'model contracts')
    require(isinstance(manifest['new_shards'], list), 'new_shards must be a list')
    scope = manifest['reservation_scope']
    exact_keys(scope, ['snapshot_ref', 'index_complete_for_supplied_roots', 'roots', 'files'], 'reservation scope')
    require(scope['snapshot_ref'] == manifest['snapshot_ref'] and scope['index_complete_for_supplied_roots'] is True, 'reservation census not bound')
    require(isinstance(scope['roots'], list) and len(set(scope['roots'])) == len(scope['roots']), 'invalid scope roots')
    for path in scope['roots']:
        require(str(relative(path)).startswith('runs/'), 'reservation scope outside runs')
    require(isinstance(scope['files'], list) and len(set(scope['files'])) == len(scope['files']), 'duplicate scope files')
    actual = sorted(p for p in pins if any(p.startswith(prefix + '/') for prefix in scope['roots']))
    require(sorted(scope['files']) == actual, 'reservation census/file inventory mismatch')
    return pins


def source_and_base(root):
    rows = [parse(line) for line in (root / BASE_FILES['source']).read_bytes().splitlines() if line.strip()]
    require(len(rows) == 100 and len({r['record_id'] for r in rows}) == 100, 'source denominator/identity changed')
    for row in rows:
        require(row['split'] == 'development' and row['prompt_sha256'] == sha(row['prompt'].encode()), 'source split/prompt changed')
    base = parse((root / BASE_FILES['base']).read_bytes())
    records = base['records']
    expected = {(alias, n): rows[n-1]['record_id'] for alias in MODELS for n in range(1, 101)}
    require(len(records) == 300, 'base denominator differs')
    keys = [(r['model_alias'], r['source_order']) for r in records]
    require(len(set(keys)) == 300 and set(keys) == set(expected), 'base identity grid differs')
    for row in records:
        require(row['record_id'] == expected[(row['model_alias'], row['source_order'])], 'base source binding differs')
    require(dict(Counter(r['status'] for r in records)) == {'completed': 89, 'reserved_outcome_unknown': 1, 'not_attempted': 210}, 'base counts differ')
    checkpoint = parse((root / BASE_FILES['checkpoint']).read_bytes())
    require(checkpoint['baseline']['completed'] == 89 and checkpoint['baseline']['unknown_source_orders'] == [82]
            and checkpoint['baseline']['blocked_intents_separate_from_counts'] == [91], 'checkpoint disagrees')
    audit = parse((root / BASE_FILES['prior_audit']).read_bytes())
    require(audit['baseline']['counts'] == base['counts'], 'prior audit differs')
    return rows, records


def model_contract(root, path, alias, sources, pins):
    relative(path)
    require(path in pins, 'unbound model contract')
    obj = parse((root / path).read_bytes())
    exact_keys(obj, ['schema', 'model_alias', 'model_id', 'revision', 'dtype', 'settings',
                     'runtime_path', 'runtime_sha256', 'context_report_path', 'model_config_path',
                     'execution_stratum', 'timeout_seconds_including_load'], 'model contract')
    require(obj['schema'] == ID + '-model-contract' and obj['model_alias'] == alias, 'wrong contract model')
    require((obj['model_id'], obj['revision']) == MODELS[alias], 'changed model/revision')
    require(obj['dtype'] == 'float32' and obj['settings'] == SETTINGS, 'changed scientific settings')
    require(obj['timeout_seconds_including_load'] == 1200 and type(obj['timeout_seconds_including_load']) is int, 'changed time stratum')
    require(isinstance(obj['execution_stratum'], str) and bool(obj['execution_stratum']), 'missing runtime stratum')
    paths = [path]
    for key in ['runtime_path', 'context_report_path', 'model_config_path']:
        relative(obj[key]); require(obj[key] in pins, 'unbound contract resource'); paths.append(obj[key])
    require(pins[obj['runtime_path']] == obj['runtime_sha256'], 'runtime pin mismatch')
    runtime = parse((root / obj['runtime_path']).read_bytes())
    require(isinstance(runtime.get('packages'), dict) and runtime.get('python'), 'incomplete runtime identity')
    config = parse((root / obj['model_config_path']).read_bytes())
    limit = number(config.get('max_position_embeddings'), 'model context', 1)
    context = parse((root / obj['context_report_path']).read_bytes())
    exact_keys(context, ['schema', 'model_id', 'revision', 'source_sha256', 'model_config_sha256',
                         'model_context_limit', 'records'], 'context report')
    require(context['schema'] == ID + '-context-report' and (context['model_id'], context['revision']) == MODELS[alias], 'context model identity differs')
    require(context['source_sha256'] == BASE_PINS['source'] and context['model_config_sha256'] == pins[obj['model_config_path']], 'context source/config binding differs')
    require(type(context['model_context_limit']) is int and context['model_context_limit'] == limit, 'wrong model-specific context limit')
    require(isinstance(context['records'], list) and len(context['records']) == 100, 'context report must include full100')
    contexts = {}
    for row in context['records']:
        exact_keys(row, ['source_order', 'record_id', 'prompt_sha256', 'input_tokens'], 'context record')
        n = number(row['source_order'], 'source_order', 1)
        require(n <= 100 and n not in contexts, 'duplicate/outside context row')
        require(row['record_id'] == sources[n-1]['record_id'] and row['prompt_sha256'] == sources[n-1]['prompt_sha256'], 'context prompt differs')
        count = number(row['input_tokens'], 'input tokens', 1)
        require(count + SETTINGS['max_output_tokens'] <= limit, 'context overflow')
        contexts[n] = count
    obj['model_context_limit'] = limit
    return obj, contexts, set(paths)


def shard_outcome(root, spec, contract, contexts, sources, pins):
    exact_keys(spec, ['model_alias', 'source_order', 'path'], 'shard index entry')
    alias, order, prefix = spec['model_alias'], number(spec['source_order'], 'source_order', 1), spec['path']
    require(alias in MODELS and order <= 100, 'unknown model/order')
    require(str(relative(prefix)).startswith('runs/'), 'invalid shard root')
    listed = {p[len(prefix)+1:]: v for p, v in pins.items() if p.startswith(prefix + '/')}
    require('reservation.json' in listed and 'git-preservation-receipt.json' in listed, 'reservation/preservation receipt absent')
    receipt = parse((root / prefix / 'git-preservation-receipt.json').read_bytes())
    require(receipt.get('readback_verified') is True and isinstance(receipt.get('commit'), str) and HEX40.fullmatch(receipt['commit']), 'invalid preservation receipt')
    raw = {p: v for p, v in listed.items() if p != 'git-preservation-receipt.json'}
    require(receipt.get('files_sha256') == raw, 'shard raw receipt inventory differs')
    src = sources[order-1]
    identity = canonical_sha(['local', MODELS[alias][0], src['record_id']])
    reservation = parse((root / prefix / 'reservation.json').read_bytes())
    require(reservation.get('source_order') == order and type(reservation.get('source_order')) is int
            and reservation.get('record_id') == src['record_id']
            and reservation.get('automatic_retries') == 0 and type(reservation.get('automatic_retries')) is int,
            'reservation source/retry differs')
    require(reservation.get('status') == 'reserved_outcome_unknown', 'reservation must be prior to terminal')
    require(reservation.get('identity') == identity, 'reservation identity absent/differs')
    outcome = 'reserved_outcome_unknown'
    attempt_path = 'run/attempts/' + identity + '.json'
    attempts = [p for p in raw if p.startswith('run/attempts/')]
    require(attempts in ([], [attempt_path]), 'orphan/duplicate shard attempt')
    attempt = parse((root / prefix / attempt_path).read_bytes()) if attempts else None
    has_terminal = 'process.json' in raw
    process = parse((root / prefix / 'process.json').read_bytes()) if has_terminal else None
    if process is not None:
        require(receipt.get('process_sha256') == raw['process.json'], 'process receipt binding differs')
        require(process.get('source_order') == order and type(process.get('source_order')) is int
                and process.get('identity') == identity, 'process identity differs')
        require('record_id' not in process or process['record_id'] == src['record_id'], 'process source differs')
        require(type(process.get('timeout_seconds')) is int and process['timeout_seconds'] == 1200
                and type(process.get('automatic_retries')) is int and process['automatic_retries'] == 0,
                'process time/retry stratum differs')
    if 'artifact-manifest.json' in raw:
        artifact = parse((root / prefix / 'artifact-manifest.json').read_bytes())
        expected_artifacts = {p: v for p, v in raw.items() if p != 'artifact-manifest.json'}
        require(artifact.get('files_sha256') == expected_artifacts, 'artifact manifest inventory differs')
    provenance = dict(shard_path=prefix, primary_commit=receipt['commit'], receipt_sha256=listed['git-preservation-receipt.json'],
                      model_id=MODELS[alias][0], revision=MODELS[alias][1], runtime_sha256=contract['runtime_sha256'],
                      execution_stratum=contract['execution_stratum'], model_context_limit=contract['model_context_limit'],
                      max_output_tokens=1024, timeout_seconds_including_load=1200, quality=None, transport=None)
    if attempt is not None:
        require(attempt.get('identity') == identity and attempt.get('model') == MODELS[alias][0]
                and attempt.get('provider') == 'local' and attempt.get('automatic_retries') == 0
                and type(attempt.get('automatic_retries')) is int, 'attempt identity/provider/retry differs')
        selected_record = {key: src[key] for key in ['genre', 'prompt_sha256', 'record_id',
                                                   'source_group_id', 'split', 'target_words', 'topic']}
        require(attempt.get('record') == selected_record, 'attempt source metadata binding differs')
        for p in ['run/run.json', 'run/runtime.json', 'run/local_ledger/ledger.json']:
            require(p in raw, 'attempt missing required run/runtime/ledger')
        run = parse((root / prefix / 'run/run.json').read_bytes())
        require((run.get('model'), run.get('revision')) == MODELS[alias] and run.get('dtype') == 'float32'
                and run.get('provider') == 'local' and run.get('settings') == SETTINGS, 'run model/settings differ')
        require(run.get('schema') == 'synsteg-budget-generation-run-v1'
                and run.get('runner_sha256') == RUNNER_SHA256 and run.get('ledger_code_sha256') == LEDGER_SHA256
                and run.get('split') == 'development' and run.get('automatic_retries') == 0
                and type(run.get('automatic_retries')) is int, 'run scientific code/split/retry differs')
        require(run.get('selected_records_count') == 1 and type(run.get('selected_records_count')) is int and len(run.get('selected_records', [])) == 1
                and run['selected_records'][0] == attempt['record'], 'run selected source differs')
        require(raw['run/runtime.json'] == contract['runtime_sha256'], 'actual shard runtime differs')
        ledger = parse((root / prefix / 'run/local_ledger/ledger.json').read_bytes())
        require(set(ledger.get('attempts', {})) == {identity}, 'ledger orphan/missing attempt')
        entry = ledger['attempts'][identity]
        require(entry.get('status') == attempt.get('status') and entry.get('result') == attempt.get('result'), 'attempt/ledger terminal differs')
        require(entry.get('metadata', {}).get('record') == attempt['record'], 'ledger source differs')
        state = attempt.get('status')
        require(state in ['completed', 'failed', 'reserved_outcome_unknown'], 'unknown attempt status')
        if state == 'completed' and has_terminal:
            require(COMPLETED_CORE | {attempt_path} <= set(raw), 'completed shard missing core artifacts')
            require(process.get('status') == 'completed_pending_remote_preservation'
                    and process.get('exit_code') == 0 and type(process.get('exit_code')) is int
                    and process.get('timed_out') is False and not process.get('interrupted'),
                    'completed attempt lacks clean process')
            result = attempt.get('result')
            require(isinstance(result, dict) and isinstance(result.get('cover_text'), str) and bool(result['cover_text']), 'completed output absent')
            require(result.get('dtype') == 'float32' and result.get('returned_model') == MODELS[alias][0]
                    and result.get('revision') == MODELS[alias][1], 'returned model identity differs')
            expected_seed = int(canonical_sha([20261024, MODELS[alias][0], src['record_id']])[:8], 16)
            require(type(result.get('seed')) is int and result['seed'] == expected_seed, 'per-record seed differs')
            usage = result.get('usage', {})
            input_tokens = number(usage.get('input_tokens'), 'actual input tokens', 1)
            require(input_tokens == contexts[order], 'actual tokenizer context differs from preflight')
            require(number(usage.get('output_tokens'), 'output tokens') <= 1024, 'output token overflow')
            require(result.get('stop_reason') in ('end_turn', 'max_tokens') and type(result.get('hit_output_limit')) is bool,
                    'missing stop classification')
            outcome = 'completed'
            provenance['cover_sha256'] = sha(result['cover_text'].encode())
            provenance['attempt_sha256'] = raw[attempt_path]
        elif state == 'failed' and has_terminal:
            require(TERMINAL_CORE | {attempt_path} <= set(raw), 'failed shard missing core artifacts')
            require(process.get('status') == 'stopped_requires_review_no_retry', 'failed attempt lacks stopped process')
            outcome = 'failed'
    if outcome == 'reserved_outcome_unknown':
        provenance['evidence_gap_reason'] = ('attempt_absent' if attempt is None else
                                             'terminal_process_absent' if not has_terminal else
                                             'attempt_outcome_unknown')
    return outcome, provenance, set(prefix + '/' + p for p in listed)


def merge(root, manifest):
    root = Path(root)
    pins = bound_inputs(root, manifest)
    sources, records = source_and_base(root)
    result = copy.deepcopy(records)
    positions = {(r['model_alias'], r['source_order']): i for i, r in enumerate(result)}
    guard = parse((root / BASE_FILES['guard']).read_bytes())
    blocked = {(r['model_alias'], r['source_order']) for r in guard['blocked_baseline_intents'] + guard['historical_unknown_baseline_orders']}
    contracts, allowed = {}, set(BASE_FILES.values())
    for alias, path in manifest['model_contracts'].items():
        if path is not None:
            obj, contexts, used = model_contract(root, path, alias, sources, pins)
            contracts[alias] = (obj, contexts); allowed |= used
    seen, paths, evidence = set(), set(), []
    for spec in manifest['new_shards']:
        alias, order = spec.get('model_alias'), spec.get('source_order')
        require(alias in MODELS and type(order) is int and 1 <= order <= 100, 'invalid indexed model/order')
        key = (alias, order)
        require(key not in seen and key not in blocked, 'duplicate or blocked reservation')
        require(result[positions[key]]['status'] == 'not_attempted', 'new shard overwrites historical attempted outcome')
        prefix = spec.get('path')
        relative(prefix)
        require(all(not (prefix == p or prefix.startswith(p + '/') or p.startswith(prefix + '/')) for p in paths), 'overlapping shard roots')
        require(any(prefix.startswith(p + '/') for p in manifest['reservation_scope']['roots']), 'shard outside census roots')
        require(alias in contracts, 'new model requires actual pinned runtime/context contract')
        outcome, provenance, used = shard_outcome(root, spec, *contracts[alias], sources, pins)
        allowed |= used; seen.add(key); paths.add(prefix)
        row = result[positions[key]]
        row.update(status=outcome, source_stratum=provenance['execution_stratum'], details=['new_preserved_shard'])
        evidence.append(dict(model_alias=alias, source_order=order, status=outcome, **provenance))
    require(set(pins) == allowed, 'unreferenced input/orphan reservation or attempt')
    counts = dict(Counter(r['status'] for r in result))
    return {'schema': ID + '-result', 'scientific_campaign': CAMPAIGN,
            'base_ref': BASE_REF, 'snapshot_ref': manifest['snapshot_ref'], 'full_denominator': 300,
            'model_denominators': {m: 100 for m in MODELS}, 'counts': counts,
            'model_counts': {m: dict(Counter(r['status'] for r in result if r['model_alias'] == m)) for m in MODELS},
            'records': result, 'new_shard_evidence': evidence,
            'scheduling_overlays': copy.deepcopy(guard['blocked_baseline_intents']),
            'inherited_base_raw_reaudited': False, 'base_snapshot_sha256': pins[BASE_FILES['base']],
            'global_remote_reservation_completeness_independently_proven': False,
            'scope': 'Exact supplied snapshot and explicit new-shard census; caller must freshly verify remote census completeness.',
            'new_model_calls': 0, 'new_tokenizer_calls': 0, 'closed_test_opened': False,
            'readiness': 'Offline merger only; synthetic future-arm fixtures are not model experiments.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    require(sha(raw) == args.expected_manifest_sha256, 'manifest hash not externally bound')
    require(not args.output.exists(), 'output exists')
    require(not args.output.resolve().is_relative_to(args.root.resolve()), 'output must be outside input root')
    require(args.output.resolve() != args.manifest.resolve(), 'output must differ from manifest')
    result = merge(args.root, parse(raw))
    result['input_manifest_sha256'] = sha(raw)
    result['merger_sha256'] = sha(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'counts': result['counts'], 'models': result['model_counts'], 'new_shards': len(result['new_shard_evidence'])}))


if __name__ == '__main__':
    main()
