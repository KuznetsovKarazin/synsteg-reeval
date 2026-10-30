#!/usr/bin/env python3
"""Independent read-only checks for one completed recovery shard; no NN imports."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

RUN = 'runs/large-campaign-development-v2/models/tinyllama11'
MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
REVISION = 'fe8a4ea1ffedaf415f4da2f062534de366a451e6'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':')).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(prior_root, protocol_path, shard_root, order):
    prior_root, protocol_path, shard_root = map(Path, (prior_root, protocol_path, shard_root))
    errors, checks, identities = [], [], {}

    def check(label, condition):
        checks.append(label)
        if not condition:
            errors.append(label)

    def read(path):
        data = path.read_bytes()
        identities[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    check('allowed source order', type(order) is int and 83 <= order <= 100)
    if errors:
        return {'status': 'failed', 'checks': len(checks), 'errors': errors}
    protocol = read(protocol_path)
    job = protocol['jobs'][order - 83]
    check('job order', job['source_order'] == order)
    prompt_path = protocol_path.parent / job['prompt_path']
    prompt_bytes = prompt_path.read_bytes()
    prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
    identities[str(prompt_path)] = prompt_sha
    rows = [json.loads(x) for x in prompt_bytes.decode().splitlines() if x.strip()]
    check('exactly one prompt', len(rows) == 1)
    if len(rows) != 1:
        return {'status': 'failed', 'checks': len(checks), 'errors': errors}
    row = rows[0]
    ident = canonical(['local', MODEL, row['record_id']])
    check('identity', job['identity'] == ident and job['record_id'] == row['record_id'])
    check('development only', row['split'] == 'development')
    check('prompt SHA', job['prompt_file_sha256'] == prompt_sha)
    original_run = read(prior_root / RUN / 'run.json')
    original_runtime = read(prior_root / RUN / 'runtime.json')
    original_ledger = read(prior_root / RUN / 'local_ledger/ledger.json')
    check('new identity absent from historical ledger', ident not in original_ledger['attempts'])
    historical_unknown = [e for e in original_ledger['attempts'].values() if e['status'] != 'completed']
    check('historical unknown retained', len(historical_unknown) == 1
          and historical_unknown[0]['status'] == 'reserved_outcome_unknown'
          and historical_unknown[0].get('result') is None)
    meta = {k: row[k] for k in ('record_id', 'source_group_id', 'split', 'topic', 'genre', 'target_words')}
    meta['prompt_sha256'] = hashlib.sha256(row['prompt'].encode()).hexdigest()
    check('original source-position metadata/prompt', len(original_run['selected_records']) == 100
          and original_run['selected_records'][order - 1] == meta)
    expected_run = copy.deepcopy(original_run)
    expected_run.update(prompts_sha256=prompt_sha, selected_records=[meta], selected_records_count=1)
    actual_run = read(shard_root / 'run/run.json')
    runtime = read(shard_root / 'run/runtime.json')
    ledger = read(shard_root / 'run/local_ledger/ledger.json')
    attempt = read(shard_root / 'run/attempts' / (ident + '.json'))
    summary = read(shard_root / 'run/summary.json')
    check('run plan parity except exact shard metadata', actual_run == expected_run)
    check('runtime parity', runtime == original_runtime)
    check('one ledger identity', set(ledger['attempts']) == {ident})
    if ident not in ledger['attempts']:
        return {'status': 'failed', 'checks': len(checks), 'errors': errors}
    entry = ledger['attempts'][ident]
    expected_fingerprint = canonical({'record': meta, 'provider': 'local', 'model': MODEL,
        'revision': REVISION, 'dtype': original_run['dtype'], 'settings': original_run['settings'],
        'pricing_sha256': None, 'prompts_sha256': prompt_sha, 'runtime_sha256': canonical(runtime),
        'implementation': {k: original_run[k] for k in ('runner_sha256', 'ledger_code_sha256', 'transport_code_sha256')}})
    check('ledger schema/cap', ledger.get('schema') == 'synsteg-budget-ledger-recovery-v1'
          and ledger.get('budget_nanodollars') == 20_000_000_000 and not ledger.get('blocked_reason'))
    check('ledger metadata', entry.get('metadata') == {'record': meta, 'provider': 'local', 'model': MODEL})
    check('attempt metadata', attempt.get('record') == meta and attempt.get('provider') == 'local'
          and attempt.get('model') == MODEL and attempt.get('schema') == 'synsteg-budget-generation-attempt-v1')
    check('both request fingerprints', entry.get('request_fingerprint') == expected_fingerprint
          and attempt.get('request_fingerprint') == expected_fingerprint)
    check('both identities', entry.get('identity') == ident and attempt.get('identity') == ident)
    check('both completed/no retries', entry.get('status') == attempt.get('status') == 'completed'
          and entry.get('automatic_retries') == attempt.get('automatic_retries') == 0)
    check('zero local accounted/reserved', entry.get('accounted_nanodollars') == entry.get('reserved_nanodollars') == 0)
    result = entry.get('result')
    check('matching result objects', isinstance(result, dict) and result == attempt.get('result'))
    if isinstance(result, dict):
        check('result model/revision/dtype', result.get('returned_model') == MODEL
              and result.get('revision') == REVISION and result.get('dtype') == 'float32')
        seed = int(canonical([20261024, MODEL, row['record_id']])[:8], 16)
        check('original per-record seed', result.get('seed') == seed)
        usage = result.get('usage', {})
        tokens = usage.get('output_tokens')
        check('token bounds', type(tokens) is int and 1 <= tokens <= 1024
              and type(usage.get('input_tokens')) is int and usage['input_tokens'] >= 1)
        check('termination parity', (result.get('stop_reason') == ('max_tokens' if tokens == 1024 else 'end_turn'))
              and result.get('hit_output_limit') is (tokens == 1024))
        check('nonempty cover', isinstance(result.get('cover_text'), str) and bool(result['cover_text'].strip()))
    check('terminal summary', summary.get('status') == 'selected_records_completed'
          and summary.get('exit_code') == 0 and summary.get('selected_records_count') == 1
          and summary.get('completed') == 1 and summary.get('resumed_completed') == 0
          and summary.get('requests_attempted_this_invocation') == 1
          and summary.get('max_new_records_this_invocation') == 1 and summary.get('automatic_retries') == 0)
    check('summary attempt list', summary.get('attempts') == [{'identity': ident, 'record_id': row['record_id'], 'status': 'completed'}])
    check('summary ledger count', summary.get('ledger', {}).get('attempt_count') == 1
          and summary.get('ledger', {}).get('unknown_or_failed_attempts') == 0)
    return {'schema': 'baseline-single-record-recovery-v1-independent-shard-audit',
            'status': 'passed' if not errors else 'failed', 'source_order': order,
            'record_id': row['record_id'], 'checks': len(checks), 'errors': errors,
            'input_sha256': identities, 'model_calls': 0, 'tokenizer_calls': 0,
            'quality_assessed': False, 'transport_assessed': False,
            'limitation': 'Integrity and scientific-setting parity checks; does not establish cover fidelity, delivery or bitwise NN reproducibility.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prior-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--shard', type=Path, required=True)
    p.add_argument('--source-order', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.prior_root, a.protocol, a.shard, a.source_order)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'checks', 'errors')}))
    raise SystemExit(0 if result['status'] == 'passed' else 2)
