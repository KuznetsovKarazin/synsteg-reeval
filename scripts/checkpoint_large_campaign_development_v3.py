#!/usr/bin/env python3
"""Inspect the full 300-record cohort while TinyLlama advances through its 100 rows.

This extends operational checkpointing only. Frozen generation code/settings,
first-batch evidence and the original projection implementation are unchanged.
No model is imported or called.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('runs/large-campaign-development-v2/models/tinyllama11')
FIRST_MANIFEST = Path('experiments/large-campaign-development-v2/execution-manifest.json')
FIRST_MANIFEST_SHA = '7b23df64b875799cc592640a9aadf7ea6bcd870b525335141169cf93a6003ff1'
PROJECTOR = Path('scripts/checkpoint_large_campaign_development_v2.py')
PROJECTOR_SHA = 'f47e99feb3c6b3aef4a415337405888fe6f598ce60f7e3395bd30781a5053945'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':')).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def first_manifest(root):
    path = root / FIRST_MANIFEST
    if file_sha(path) != FIRST_MANIFEST_SHA:
        raise ValueError('First execution manifest differs from the Git-frozen experiment')
    return load_json(path)


def run_file_hashes(root):
    directory = root / RUN
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink in run evidence')
        if path.is_file():
            result[str(path.relative_to(root))] = file_sha(path)
    return result


def inspect(root=ROOT):
    root = Path(root)
    frozen = first_manifest(root)
    config_path = root / 'configs/large_campaign_resume_v2.json'
    if file_sha(config_path) != frozen['frozen_files_sha256'][str(config_path.relative_to(root))]:
        raise ValueError('Frozen cohort configuration changed')
    config = load_json(config_path)
    cohort = config['development_candidate']
    prompts = root / cohort['source_prompts']
    if file_sha(prompts) != cohort['source_prompts_sha256']:
        raise ValueError('Frozen development prompts changed')
    source = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    selected = [{**{k: row[k] for k in
                 ['record_id', 'source_group_id', 'split', 'topic', 'genre', 'target_words']},
                 'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest()} for row in source]
    if (len(selected) != 100 or selected != cohort['selected_record_metadata']
            or any(row['split'] != 'development' for row in selected)):
        raise ValueError('Require the same complete 100-development cohort')
    if file_sha(root / PROJECTOR) != PROJECTOR_SHA:
        raise ValueError('Frozen full-denominator projector changed')
    spec = importlib.util.spec_from_file_location('frozen_campaign_projector_v2', root / PROJECTOR)
    projection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection)
    models = cohort['local_models']
    for model in models:
        if model['alias'] != 'tinyllama11' and (root / RUN.parent / model['alias']).exists():
            raise ValueError('Another model arm needs its own launch freeze before this utility can include it')
    model = next(item for item in models if item['alias'] == 'tinyllama11')
    directory = root / RUN
    run = load_json(directory / 'run.json')
    runtime = load_json(directory / 'runtime.json')
    ledger = load_json(directory / 'local_ledger/ledger.json')
    summary = load_json(directory / 'summary.json')
    if canonical(run) != frozen['expected_run_plan_sha256']:
        raise ValueError('Run plan changed')
    for path, field in [('scripts/run_budget_generation.py', 'runner_sha256'),
                        ('src/synsteg_audit/budget_ledger.py', 'ledger_code_sha256')]:
        if file_sha(root / path) != run[field]:
            raise ValueError('Frozen runner/ledger source changed')
    if (ledger['schema'] != 'synsteg-budget-ledger-recovery-v1'
            or ledger['budget_nanodollars'] != 20_000_000_000 or ledger.get('blocked_reason')):
        raise ValueError('Unexpected or accounting-blocked local ledger')
    attempts = {p.stem: load_json(p) for p in (directory / 'attempts').glob('*.json')}
    if len(ledger['attempts']) > 100:
        raise ValueError('Full TinyLlama denominator exceeded')
    for ident, entry in ledger['attempts'].items():
        if entry['identity'] != ident or entry['automatic_retries'] != 0:
            raise ValueError('Ledger identity or retry policy changed')
        expected = canonical({'record': entry['metadata']['record'], 'provider': 'local',
            'model': model['model_id'], 'revision': model['revision'], 'dtype': run['dtype'],
            'settings': run['settings'], 'pricing_sha256': run['pricing_sha256'],
            'prompts_sha256': run['prompts_sha256'], 'runtime_sha256': canonical(runtime),
            'implementation': {key: run[key] for key in
                ['runner_sha256', 'ledger_code_sha256', 'transport_code_sha256']}})
        if entry['request_fingerprint'] != expected:
            raise ValueError('Request fingerprint mismatch')
    projected = projection.project(selected, models,
        {'tinyllama11': {'run': run, 'ledger': ledger, 'attempts': attempts}})
    issues = projected['consistency_issues']
    if summary.get('status') not in {'batch_completed', 'selected_records_completed'}:
        issues.append({'issue': 'summary_not_successfully_terminal'})
    if summary.get('automatic_retries') != 0 or summary.get('max_new_records_this_invocation') != 2:
        issues.append({'issue': 'summary_retry_or_batch_policy_mismatch'})
    if summary.get('selected_records_count') != 100:
        issues.append({'issue': 'summary_denominator_mismatch'})
    if summary.get('completed') != projected['completed_generations']:
        issues.append({'issue': 'summary_completion_count_mismatch'})
    if summary.get('ledger', {}).get('attempt_count') != projected['attempted']:
        issues.append({'issue': 'summary_ledger_count_mismatch'})
    expected_summary = [(row['identity'], row['record_id'], row['status'])
                        for row in projected['records'] if row['status'] != 'not_attempted']
    actual_summary = [(row.get('identity'), row.get('record_id'), row.get('status'))
                      for row in summary.get('attempts', [])]
    if actual_summary != expected_summary:
        issues.append({'issue': 'summary_attempt_list_mismatch'})
    expected_paths = {str(RUN / p) for p in ['run.json', 'runtime.json', 'summary.json',
                     'local_ledger/ledger.json', 'local_ledger/campaign.lock', 'run_guard/campaign.lock']}
    expected_paths.update(str(RUN / 'attempts' / (ident + '.json')) for ident in ledger['attempts'])
    file_hashes = run_file_hashes(root)
    if set(file_hashes) - expected_paths:
        raise ValueError('Unexpected files in cumulative run evidence')
    if expected_paths - set(file_hashes):
        issues.append({'issue': 'missing_run_evidence', 'paths': sorted(expected_paths - set(file_hashes))})
    unknown = projected['attempted'] - projected['completed_generations']
    return {'schema': 'large-campaign-development-checkpoint-v3',
            'status': 'requires_review' if issues or unknown else 'verified',
            **projected, 'model_counts': {'tinyllama11': projected['completed_generations'],
                                       'qwen05': 0, 'falcon3_1b': 0},
            'run_files_sha256': file_hashes,
            'frozen_execution_manifest_sha256': FIRST_MANIFEST_SHA,
            'source_groups': 100, 'new_model_calls_by_checkpoint': 0,
            'closed_test_opened': False, 'paid_calls': 0, 'legacy_qwen_imported': 0,
            'raw_token_ids_or_logits_retained': False,
            'limitation': 'Completed means a retained generation; quality and useful stego delivery remain separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = inspect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value[key] for key in
                     ['status', 'planned_model_record_pairs', 'attempted', 'completed_generations', 'counts']}))


if __name__ == '__main__':
    main()
