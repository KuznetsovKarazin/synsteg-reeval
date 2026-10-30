#!/usr/bin/env python3
"""Read-only generation/review milestone, independent of transport evaluation.

Never reads the advancing live run, imports a model, or invokes any encoder.
The old clean-prefix aggregator is reused unchanged; blocked/error checkpoints
retain all300 source rows and explicit error evidence instead of being cleaned.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
AGG = 'results/large-campaign-continuation-v3-audit/aggregate_immutable_campaign.py'
AGG_SHA = '42fcdb153b63e7d0bef489f070c10d55333836d8195d472fa287f74cbf05a2ac'
MANIFEST = 'experiments/large-campaign-development-v2/execution-manifest.json'
MANIFEST_SHA = '7b23df64b875799cc592640a9aadf7ea6bcd870b525335141169cf93a6003ff1'
UTILITY = 'experiments/large-campaign-resume-v3/manifest.json'
UTILITY_SHA = 'd3cfc68320e330b89d0f5db2019cf6c33a2460b5c154cbc35f3fc7b6238e7101'
RUN = 'runs/large-campaign-development-v2/models/tinyllama11'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(root, relative, evidence, expected=None):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe path')
    path = root / rel
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'Missing/unsafe file: ' + str(rel))
    raw = path.read_bytes()
    require(expected is None or sha(raw) == expected, 'SHA mismatch: ' + str(rel))
    require(str(rel) not in evidence or evidence[str(rel)] == sha(raw), 'Input changed during inspection')
    evidence[str(rel)] = sha(raw)
    return raw


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def describe(values):
    return {'n': len(values), 'min': min(values) if values else None,
            'median': statistics.median(values) if values else None,
            'mean': statistics.fmean(values) if values else None,
            'max': max(values) if values else None}


def compose_records(checkpoint_records, config, attempts, reviews):
    """Keep failed/unknown/missing in the full denominator; no success imputation."""
    meta = {(m['alias'], r['record_id']): (m, r, order) for m in config['local_models']
            for order, r in enumerate(config['selected_record_metadata'], 1)}
    require(len(checkpoint_records) == len(meta) == 300, 'Full300 denominator required')
    require(len({(r['model_alias'], r['record_id']) for r in checkpoint_records}) == 300, 'Duplicate checkpoint row')
    rows = []
    for original in checkpoint_records:
        key = (original['model_alias'], original['record_id'])
        require(key in meta, 'Unexpected checkpoint identity')
        model, source, order = meta[key]
        require(all(original[k] == v for k, v in source.items()) and original['model'] == model['model_id'] and original['revision'] == model['revision'], 'Metadata differs from fixed cohort')
        require(original['status'] in ('completed', 'not_attempted', 'failed', 'unknown'), 'Unknown generation state')
        row = dict(original, source_order=order, attempt_sha256=None, retained_attempt_status=None,
                   retained_error=None, target_word_delta=None, target_word_ratio=None,
                   target_relation=None, length_pass=None,
                   length_pass_status='not_defined_no_frozen_numeric_tolerance',
                   text_review=reviews.get(original['record_id']) if original['model_alias'] == 'tinyllama11' else None)
        saved = attempts.get(original['identity'])
        if saved is not None:
            attempt = saved['attempt']
            retained_result = attempt.get('result') or {}
            row.update(attempt_sha256=saved['attempt_sha256'], retained_attempt_status=attempt.get('status'),
                       retained_error=({k: retained_result[k] for k in
                           ('error_type', 'error_code', 'generation_outcome_known')
                           if k in retained_result} or None))
            require(attempt.get('identity') == original['identity'], 'Attempt identity differs')
        if original['status'] == 'completed':
            require(saved is not None and saved['attempt']['status'] == 'completed', 'Missing/noncompleted retained attempt')
            text = saved['attempt']['result']['cover_text']
            words = len(text.split())
            require(sha(text.encode()) == original['cover_sha256'] and words == original['whitespace_words'], 'Cover hash/length differs')
            delta = words - original['target_words']
            row.update(target_word_delta=delta, target_word_ratio=words / original['target_words'],
                       target_relation='under' if delta < 0 else 'over' if delta > 0 else 'equal')
        rows.append(row)
    return rows


def inspect(root, through_batch=15, provisional=False):
    root = Path(root).resolve()
    require(type(through_batch) is int and 2 <= through_batch <= 50, 'Expected immutable batch02..50')
    require(through_batch == 15 or provisional, 'Other prefixes must be explicitly provisional')
    evidence = {}
    manifest = json.loads(read(root, MANIFEST, evidence, MANIFEST_SHA))
    utility = json.loads(read(root, UTILITY, evidence, UTILITY_SHA))
    frozen_sources = {**manifest['frozen_files_sha256'], **utility['source_sha256']}
    for path, digest in frozen_sources.items():
        read(root, path, evidence, digest)
    read(root, AGG, evidence, AGG_SHA)
    config = json.loads(read(root, 'configs/large_campaign_resume_v2.json', evidence))['development_candidate']
    source_raw = read(root, config['source_prompts'], evidence, config['source_prompts_sha256'])
    sources = {r['record_id']: r for r in map(json.loads, source_raw.splitlines())}
    require(len(sources) == 100, 'Expected all100 development sources')
    batch = f'results/large-campaign-development-v2/batches/batch{through_batch:02d}'
    checkpoint = json.loads(read(root, batch + '/after-checkpoint.json', evidence))
    require(checkpoint['planned_model_record_pairs'] == len(checkpoint['records']) == 300, 'Checkpoint denominator differs')
    require(not checkpoint['closed_test_opened'] and checkpoint['paid_calls'] == 0, 'Task boundary differs')
    attempts = {}
    for relative, digest in checkpoint['run_files_sha256'].items():
        raw = read(root, batch + '/after/' + relative, evidence, digest)
        if '/attempts/' in relative:
            attempt = json.loads(raw)
            attempts[Path(relative).stem] = {'attempt': attempt, 'attempt_sha256': sha(raw)}
    clean = None
    clean_error = None
    try:
        clean = module(root / AGG, 'frozen_milestone_aggregate').aggregate(root, through_batch)
        for path, digest in clean['input_sha256'].items():
            require(path not in evidence or evidence[path] == digest, 'Evidence changed')
            evidence[path] = digest
    except ValueError as exc:
        # Preserve failed/unknown primary evidence; never emit a clean aggregate.
        clean_error = str(exc)
    completed_attempts = {x['attempt']['record']['record_id']: x for x in attempts.values()
                          if x['attempt'].get('status') == 'completed'}
    read(root, 'scripts/summarize_development_text_reviews_v1.py', evidence)
    read(root, 'scripts/check_large_campaign_milestone_review_v1.py', evidence)
    helper = module(root / 'scripts/summarize_development_text_reviews_v1.py', 'text_review_summary')
    review_rows, review_summary = helper.collect_reviews(root, through_batch, completed_attempts, sources, evidence)
    rows = compose_records(checkpoint['records'], config, attempts, review_rows)
    counts = dict(Counter(r['status'] for r in rows))
    require(counts == checkpoint['counts'], 'Checkpoint status counts differ')
    completed = [r for r in rows if r['status'] == 'completed']
    runtime = json.loads(read(root, batch + '/after/' + RUN + '/runtime.json', evidence))
    settings = json.loads(read(root, batch + '/after/' + RUN + '/run.json', evidence))
    runtime_matches = read(root, batch + '/after/' + RUN + '/runtime.json', evidence) == read(root,
        'results/large-campaign-development-v2/batches/batch02/before/' + RUN + '/runtime.json', evidence)
    run_matches = read(root, batch + '/after/' + RUN + '/run.json', evidence) == read(root,
        'results/large-campaign-development-v2/batches/batch02/before/' + RUN + '/run.json', evidence)
    require(runtime_matches and run_matches, 'Frozen run/runtime differs from original pair')
    issues = list(checkpoint.get('consistency_issues', []))
    if clean_error:
        issues.append({'clean_prefix_audit': clean_error})
    result = {'schema': 'large-campaign-generation-review-milestone-v1',
        'status': 'verified_readonly_milestone' if clean is not None and not issues else 'blocked_requires_review',
        'provisional': provisional, 'through_batch': f'batch{through_batch:02d}',
        'intended_first30_target': 'batch15', 'full_model_record_denominator': 300,
        'full_source_group_denominator': 100, 'generation_status_counts': counts,
        'completed': len(completed), 'failed_or_unknown': sum(r['status'] in ('failed', 'unknown') for r in rows),
        'all_original_checkpoint_records_retained': True, 'integrity_issues': issues,
        'clean_prefix_audit': clean,
        'frozen_source_checks': {'verified_files': len(frozen_sources), 'utility_manifest_sha256': UTILITY_SHA,
            'generation_manifest_sha256': MANIFEST_SHA, 'original2_and_all_prior_attempts_immutable': clean is not None,
            'runtime_and_run_bytes_equal_original2': runtime_matches and run_matches},
        'recorded_actual_runtime': runtime,
        'recorded_actual_generation': {k: settings[k] for k in ('provider', 'model', 'revision', 'dtype', 'settings', 'automatic_retries')},
        'length': {'definition': 'len(cover_text.split())', 'formal_failure_count': None,
            'formal_acceptance_tolerance': None, 'reason': 'No numeric acceptance tolerance in frozen launch; no posthoc pass/fail.',
            'descriptive_target_relation_counts': dict(Counter(r['target_relation'] for r in completed)),
            'words': describe([r['whitespace_words'] for r in completed]),
            'target_word_ratio': describe([r['target_word_ratio'] for r in completed])},
        'existing_text_reviews': review_summary, 'records': rows, 'input_sha256': evidence,
        'transport_evaluation': {'performed_by_this_script': False, 'encoder_calls': 0, 'decoder_calls': 0,
            'original_fixed_candidate_stage': 'BLOCKED_RESOURCE_OR_STAGE_NOT_RUN',
            'public_first30': 'Separate frozen728f85c4 evaluator; no results inferred here.'},
        'live_run_read': False, 'model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False,
        'new_semantic_labels_assigned': False,
        'limitations': ['Existing review labels are descriptive agent observations, not blind human gold or exhaustive claim annotation.',
            'unsupported_by_excerpt is not verified falsehood in the world; counts are observations or covers with recorded observations.',
            'A completed generation, target length or successful transmission cannot establish factuality or security.',
            'Provisional prefixes are not whole-cohort model estimates; all remaining model-record pairs stay in the denominator.',
            'Raw token IDs were not saved, so token counts/stops rely on retained runtime evidence.',
            'Remote Git readback remains the coordinator responsibility.']}
    return result


def render_report(result):
    length = result['length']
    clean = result['clean_prefix_audit']
    runtime = result['recorded_actual_generation']
    return '\n'.join([
        '# Проверка development milestone', '',
        '**Предварительный срез.**' if result['provisional'] else '**Зафиксированный first30 milestone.**', '',
        f"Immutable {result['through_batch']}: {result['completed']}/300 completed; статусы: {json.dumps(result['generation_status_counts'], ensure_ascii=False)}.",
        f"Integrity: {result['status']}; исходные два результата и предыдущие попытки проверены: {result['frozen_source_checks']['original2_and_all_prior_attempts_immutable']}.", '',
        f"Модель {runtime['model']}, revision {runtime['revision']}, dtype {runtime['dtype']}; настройки и runtime побайтово совпали с исходной парой.",
        f"Фактические настройки: {json.dumps(runtime['settings'], ensure_ascii=False)}.", '',
        f"Длина: {json.dumps(length['descriptive_target_relation_counts'], ensure_ascii=False)} относительно target; ratio {json.dumps(length['target_word_ratio'], ensure_ascii=False)}.",
        'Формальный length-pass/fail не определён: frozen launch не задаёт числовой tolerance. Значения under/over описательные, не новая метрика качества.', '',
        'Существующие описательные обзоры:',
        '```json', json.dumps(result['existing_text_reviews'], ensure_ascii=False, indent=2), '```',
        'Числа отражают сохранённые сгруппированные наблюдения. Они не являются частотой ложных фактов или независимой человеческой оценкой.', '',
        f"Stop reasons: {json.dumps(clean['stop_reasons'], ensure_ascii=False) if clean else 'blocked by integrity issues'}.",
        f"Recorded tokens: {json.dumps(clean['recorded_tokens'], ensure_ascii=False) if clean else 'unavailable'}.",
        f"Generation/process seconds: {clean['timing']['generation_seconds_sum'] if clean else None} / {clean['timing']['process_wall_seconds_sum'] if clean else None}; Git, загрузки и ожидание не включены.", '',
        'Это проверка обычных генераций и ранее написанных обзоров. Public transport выполняется отдельно; original/fixed candidate stage остаётся blocked. Новых encoder/decoder/model calls здесь нет.', '',
        'Полные 300 строк, hashes, добавления к обзорам и все integrity issues находятся в соседнем report.json.', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--through-batch', type=int, default=15)
    parser.add_argument('--provisional', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output exists; never overwrite a milestone')
    result = inspect(args.root, args.through_batch, args.provisional)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'report_ru.md').write_text(render_report(result))
    manifest = {p.name: sha(p.read_bytes()) for p in args.output.iterdir() if p.is_file()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'provisional': result['provisional'],
                      'completed': result['completed'], 'full_denominator': 300}))


if __name__ == '__main__':
    main()
