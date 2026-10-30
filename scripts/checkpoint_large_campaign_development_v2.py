#!/usr/bin/env python3
"""Read-only full-denominator checkpoint; never import a model or retry a request."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=True,
                             separators=(',', ':')).encode())


def identity(model, record_id):
    return canonical(['local', model, record_id])


def project(selected, models, snapshots):
    """Join full fixed cohort with attempted-only ledgers; unknowns stay attempted."""
    records, issues = [], []
    for model in models:
        alias, model_id = model['alias'], model['model_id']
        snapshot = snapshots.get(alias)
        ledger = snapshot['ledger']['attempts'] if snapshot else {}
        attempts = snapshot['attempts'] if snapshot else {}
        expected_ids = [identity(model_id, row['record_id']) for row in selected]
        if set(ledger) - set(expected_ids) or set(attempts) - set(ledger):
            raise ValueError('Unplanned or orphan attempt: ' + alias)
        if set(ledger) != set(expected_ids[:len(ledger)]):
            raise ValueError('Attempted cohort is not the fixed-order prefix: ' + alias)
        if snapshot and snapshot['run']['selected_records'] != selected:
            raise ValueError('Full selected denominator changed: ' + alias)
        for meta, ident in zip(selected, expected_ids):
            row = {'model_alias': alias, 'model': model_id, 'revision': model['revision'],
                   **meta, 'identity': ident, 'status': 'not_attempted',
                   'quality_assessment': 'pending', 'stego_assessment': 'pending'}
            if ident in ledger:
                entry = ledger[ident]
                status = entry['status']
                if status not in {'completed', 'failed_or_unknown_no_retry',
                                  'reserved_outcome_unknown', 'accounting_bound_violation_stop'}:
                    raise ValueError('Unrecognized ledger status')
                if entry['metadata'] != {'record': meta, 'provider': 'local', 'model': model_id}:
                    raise ValueError('Ledger metadata mismatch')
                if entry['accounted_nanodollars'] != 0 or entry['reserved_nanodollars'] != 0:
                    raise ValueError('Local generation has a nonzero charge')
                row['status'] = status
                attempt = attempts.get(ident)
                if attempt is None:
                    issues.append({'identity': ident, 'issue': 'attempt_json_missing'})
                else:
                    for key, expected in [('identity', ident), ('record', meta), ('model', model_id),
                                          ('provider', 'local'), ('automatic_retries', 0),
                                          ('request_fingerprint', entry['request_fingerprint'])]:
                        if attempt.get(key) != expected:
                            raise ValueError('Attempt/ledger disagreement: ' + key)
                    if attempt.get('status') != status or attempt.get('result') != entry.get('result'):
                        issues.append({'identity': ident, 'issue': 'attempt_ledger_settlement_disagrees'})
                if status == 'completed':
                    result = entry['result']
                    if not isinstance(result.get('cover_text'), str) or not result['cover_text'].strip():
                        raise ValueError('Completed result has no retained text')
                    expected_seed = int(canonical([20261024, model_id, meta['record_id']])[:8], 16)
                    if (result['seed'] != expected_seed or result['returned_model'] != model_id
                            or result['revision'] != model['revision'] or result['dtype'] != 'float32'):
                        raise ValueError('Completed model/settings disagree')
                    usage = result['usage']
                    if (not 0 < usage['output_tokens'] <= 1024 or usage['input_tokens'] <= 0
                            or usage['input_tokens'] + 1024 > 2048):
                        raise ValueError('Unexpected TinyLlama token/context counts')
                    words = len(result['cover_text'].split())
                    row.update(cover_sha256=digest(result['cover_text'].encode()),
                               whitespace_words=words, target_word_ratio=words / meta['target_words'],
                               usage=usage, seed=expected_seed, elapsed_seconds=result['elapsed_seconds'],
                               stop_reason=result['stop_reason'], hit_output_limit=result['hit_output_limit'])
                else:
                    row['automatic_retry_allowed'] = False
            records.append(row)
    counts = dict(sorted(Counter(row['status'] for row in records).items()))
    return {'records': records, 'counts': counts, 'consistency_issues': issues,
            'planned_model_record_pairs': len(records),
            'attempted': len(records) - counts.get('not_attempted', 0),
            'completed_generations': counts.get('completed', 0)}


def checkpoint(root):
    config_path = root / 'configs/large_campaign_resume_v2.json'
    config = json.loads(config_path.read_text())
    cohort = config['development_candidate']
    prompts = root / cohort['source_prompts']
    if digest(prompts.read_bytes()) != cohort['source_prompts_sha256']:
        raise ValueError('Development prompts changed')
    source = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    selected = [{**{key: row[key] for key in
                 ['record_id', 'source_group_id', 'split', 'topic', 'genre', 'target_words']},
                 'prompt_sha256': digest(row['prompt'].encode())} for row in source]
    if (len(selected) != 100 or selected != cohort['selected_record_metadata']
            or any(row['split'] != 'development' for row in selected)):
        raise ValueError('Unexpected development cohort')
    snapshots, files = {}, {str(config_path.relative_to(root)): digest(config_path.read_bytes()),
                            str(prompts.relative_to(root)): digest(prompts.read_bytes())}
    previews = {item['alias']: item for item in config['preview_validation']}
    for model in cohort['local_models']:
        alias = model['alias']
        directory = root / 'runs/large-campaign-development-v2/models' / alias
        if not directory.exists():
            continue
        # This first checkpoint is deliberately scoped to the first TinyLlama batch.
        if alias != 'tinyllama11':
            raise ValueError('Unexpected active arm in first-batch checkpoint')
        run = json.loads((directory / 'run.json').read_text())
        if canonical(run) != previews[alias]['plan_sha256']:
            raise ValueError('Run differs from the recorded 100-record preview')
        for name, field in [('scripts/run_budget_generation.py', 'runner_sha256'),
                            ('src/synsteg_audit/budget_ledger.py', 'ledger_code_sha256')]:
            if digest((root / name).read_bytes()) != run[field]:
                raise ValueError('Frozen implementation changed')
        ledger = json.loads((directory / 'local_ledger/ledger.json').read_text())
        if len(ledger['attempts']) > 2:
            raise ValueError('First-batch cap exceeded')
        attempts = {p.stem: json.loads(p.read_text()) for p in (directory / 'attempts').glob('*.json')}
        runtime = json.loads((directory / 'runtime.json').read_text())
        for ident, entry in ledger['attempts'].items():
            fingerprint = canonical({'record': entry['metadata']['record'], 'provider': 'local',
                'model': model['model_id'], 'revision': model['revision'], 'dtype': run['dtype'],
                'settings': run['settings'], 'pricing_sha256': run['pricing_sha256'],
                'prompts_sha256': run['prompts_sha256'], 'runtime_sha256': canonical(runtime),
                'implementation': {name: run[name] for name in
                     ['runner_sha256', 'ledger_code_sha256', 'transport_code_sha256']}})
            if fingerprint != entry['request_fingerprint']:
                raise ValueError('Attempt fingerprint changed')
        snapshots[alias] = {'run': run, 'ledger': ledger, 'attempts': attempts}
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                files[str(path.relative_to(root))] = digest(path.read_bytes())
    result = project(selected, cohort['local_models'], snapshots)
    return {'schema': 'large-campaign-development-v2-first-batch-checkpoint-v1',
            'status': 'requires_review' if result['consistency_issues'] else 'verified',
            **result, 'input_sha256': files, 'new_model_calls_by_checkpoint': 0,
            'raw_token_ids_or_logits_retained': False, 'closed_test_opened': False,
            'paid_calls': 0, 'legacy_qwen_imported': 0,
            'historical_plan_local': 4800, 'historical_plan_claude': 300,
            'source_groups': 100,
            'limitation': 'Completed means returned generation, not factuality, useful stego delivery or security.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = checkpoint(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value[key] for key in
          ['status', 'planned_model_record_pairs', 'attempted', 'completed_generations', 'counts']}))


if __name__ == '__main__':
    main()
