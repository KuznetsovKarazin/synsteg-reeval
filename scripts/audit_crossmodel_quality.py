#!/usr/bin/env python3
"""Portable, model-free verification of retained crossmodel secondary scores.

Uses only stdlib utilities from the independent primary auditor. It never
imports the frozen quality summarizer, scorer, codec, or model packages.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

from audit_crossmodel_transfer import Checks, canonical, digest, read, sha, events, child_path

ARMS = ('cover', 'fixed_canonical', 'fixed_quality')
COMPARISONS = {'quality_vs_canonical': ('fixed_quality', 'fixed_canonical'),
               'canonical_vs_cover': ('fixed_canonical', 'cover'),
               'quality_vs_cover': ('fixed_quality', 'cover')}
JOBS = {'gpt2': ('gpt2', 'configs/neural/gpt2-full.json'),
        'qwen_base': ('qwen-base', 'configs/neural/crossmodel-qwen-base-ppl.json')}


def stats(values):
    values = sorted(values)
    count = len(values)
    total = math.fsum(values)
    return {'n': count, 'sum': total, 'mean': total / count if count else None,
            'median': (values[count // 2] if count % 2 else
                       (values[count // 2 - 1] + values[count // 2]) / 2) if count else None,
            'min': values[0] if count else None, 'max': values[-1] if count else None}


def paired_effect(left, right):
    if len(left) != len(right):
        raise ValueError('Unpaired measurements')
    deltas = [a['mean_nll'] - b['mean_nll'] for a, b in zip(left, right)]
    # Independently compute the same quantity through log PPL ratios as well.
    log_ratios = [math.log(a['ppl']) - math.log(b['ppl']) for a, b in zip(left, right)]
    if any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12) for a, b in zip(deltas, log_ratios)):
        raise ValueError('Mean NLL and log PPL ratio disagree')
    mean = math.fsum(deltas) / len(deltas) if deltas else None
    return {'document_mean_nll_delta': stats(deltas),
            'geometric_mean_ppl_ratio': math.exp(mean) if mean is not None else None,
            'document_ppl_delta': stats([a['ppl'] - b['ppl'] for a, b in zip(left, right)]),
            'lower': sum(x < -1e-12 for x in deltas),
            'equal_within_1e_12': sum(abs(x) <= 1e-12 for x in deltas),
            'higher': sum(x > 1e-12 for x in deltas)}


def validate_score(row, checks):
    detail, score = row['details'], row['score']
    n, predicted = detail['input_tokens'], detail['predicted_tokens']
    checks.require('full_text_coverage', type(n) is int and n >= 2
                   and type(predicted) is int and predicted == n - 1
                   and type(detail['scored_input_tokens']) is int and detail['scored_input_tokens'] == n
                   and detail['truncated'] is False
                   and detail['ppl_policy'] == 'sliding_window')
    checks.require('finite_score', all(type(x) in (int, float) and math.isfinite(x)
                   for x in (score, detail['sum_nll'], detail['mean_nll']))
                   and score >= 1 and detail['mean_nll'] >= 0 and detail['sum_nll'] >= 0)
    checks.equal('nll_coverage_arithmetic', detail['mean_nll'], detail['sum_nll'] / predicted)
    checks.equal('ppl_nll_identity', math.log(score), detail['mean_nll'])
    return {'mean_nll': detail['mean_nll'], 'ppl': score,
            'sum_nll': detail['sum_nll'], 'input_tokens': n, 'predicted_tokens': predicted}


def audit(repo, primary, secondary, primary_audit_path, expected_freeze, owned_proof=None):
    checks = Checks()
    primary_audit = read(primary_audit_path)
    checks.require('prior_primary_audit', primary_audit['status'] == 'pass_in_declared_scope'
                   and primary_audit['freeze_id'] == expected_freeze)
    current_primary = {p.relative_to(primary).as_posix(): sha(p) for p in sorted(primary.rglob('*')) if p.is_file()}
    checks.exact('primary_matches_independent_audit', current_primary, primary_audit['primary_files_sha256'])
    frozen, protocol = read(primary / 'freeze.json'), read(primary / 'protocol.json')
    checks.require('frozen_identity', frozen['freeze_id'] == expected_freeze
                   and digest({k: v for k, v in frozen.items() if k != 'freeze_id'}) == expected_freeze)
    summary_path = secondary / 'summary/measurements.json'
    checks.require('both_jobs_and_summary_complete', summary_path.is_file()
                   and all((secondary / folder / 'manifest.json').is_file() for folder, _ in JOBS.values()))
    before = {p.relative_to(secondary).as_posix(): sha(p) for p in sorted(secondary.rglob('*')) if p.is_file()}
    export = read(secondary / 'corpus/manifest.json')
    rows = events(secondary / 'corpus/corpus.jsonl')
    summary = read(summary_path)
    checks.require('two_quality_model_names', set(summary['quality_models']) == set(JOBS))
    checks.require('secondary_identity', export['freeze_id'] == summary['freeze_id'] == expected_freeze
                   and export['source_lock_id'] == summary['source_lock_id'] == frozen['source_lock_id']
                   and export['status'] == 'exported_after_completed_primary_evaluation'
                   and summary['status'] == 'complete_crossmodel_descriptive_secondary_measurements'
                   and export['evaluation_sha256'] == summary['evaluation_sha256'] == sha(primary / 'evaluation.json')
                   and export['corpus_sha256'] == summary['corpus_sha256'] == sha(secondary / 'corpus/corpus.jsonl'))
    checks.exact('primary_record_digests', export['all_model_records_sha256'], read(primary / 'evaluation.json')['all_model_records_sha256'])
    checks.exact('summary_record_digests', export['all_model_records_sha256'], summary['all_model_records_sha256'])
    expected_rows, common = {}, {}
    for model in protocol['models']:
        alias = model['alias']
        records = [read(primary / 'models' / alias / 'records' / (sample['id'] + '.json')) for sample in protocol['samples']]
        checks.require('source_records_hash', digest(records) == export['all_model_records_sha256'][alias])
        common[alias] = []
        for record in records:
            sample = record['sample']
            complete = all((record['variants'].get(v, {}).get('metrics') or {}).get('full_message_success') is True for v in ARMS[1:])
            if complete:
                common[alias].append(sample['id'])
            for arm in ARMS:
                text = record['cover_text'] if arm == 'cover' else record['variants'].get(arm, {}).get('stego_text')
                available = bool(text) if arm == 'cover' else (record['variants'].get(arm, {}).get('metrics') or {}).get('full_message_success') is True
                if available:
                    key = json.dumps([expected_freeze, alias, sample['id'], arm], separators=(',', ':'))
                    expected_rows[key] = {'id': key, 'model_alias': alias, 'sample_id': sample['id'],
                        'topic_id': sample['topic_id'], 'genre_id': sample['genre_id'], 'payload_bits': sample['payload_bits'],
                        'arm': arm, 'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
    indexed = {row['id']: row for row in rows}
    checks.require('corpus_unique_identities', len(indexed) == len(rows))
    checks.exact('export_matches_primary_texts_and_metadata', indexed, expected_rows)
    checks.require('planned_complete_pilot', len(common) == 3 and all(len(ids) == 16 for ids in common.values())
                   and len(rows) == export['n_records'] == summary['n_records'] == 144
                   and len({row['text_sha256'] for row in rows}) == export['n_unique_texts'] == summary['n_unique_texts'] == 139
                   and export['exclusions'] == summary['exclusions'] == [])
    checks.exact('common_cohort_ids', common, export['common_success_sample_ids'])
    matched = sorted(set.intersection(*(set(ids) for ids in common.values())))
    checks.exact('matched_cohort_ids', matched, export['matched_all_models_success_sample_ids'])
    source_files = {'helper_script_sha256': 'scripts/crossmodel_quality.py',
                    'runner_script_sha256': 'scripts/rescore_models.py',
                    'measurement_script_sha256': 'scripts/summarize_prospective_results.py'}
    source_evidence = {}
    for field, relative in source_files.items():
        expected = next(value for path, value in frozen['resource_sha256'].items() if path.endswith('/' + relative))
        checks.require('frozen_secondary_source', export[field] == sha(repo / relative) == expected)
        source_evidence[relative] = expected
    checks.require('summary_helper_identity', summary['helper_script_sha256'] == export['helper_script_sha256'])
    launcher = export['postfreeze_compatibility_launcher']
    checks.require('compatibility_launcher_identity', sha(child_path(repo, launcher['file'])) == launcher['sha256']
                   and launcher['frozen_helper_sha256'] == export['helper_script_sha256'])
    results, job_evidence = {}, {}
    for job, (folder, config_path) in JOBS.items():
        directory = secondary / folder
        manifest = read(directory / 'manifest.json')
        fingerprint = digest(manifest['identity'])
        expected_job = {'dtype': 'float32', 'precision_protocol': 'float32_reference',
                        'batch_size': 1, 'trust_remote_code': False, 'implementation_version': 2,
                        **read(repo / config_path)}
        frozen_job_sha = next(value for path, value in frozen['resource_sha256'].items() if path.endswith('/' + config_path))
        checks.require('frozen_job_file', sha(repo / config_path) == export['job_files_sha256'][job] == frozen_job_sha)
        checks.exact('job_identity', expected_job, export['expected_jobs'][job])
        checks.exact('scorer_identity', manifest['identity']['job'], expected_job)
        checks.require('completed_score_job', manifest['status'] == 'complete'
                       and manifest['job_fingerprint'] == fingerprint
                       and manifest['identity']['device'] == 'cpu'
                       and manifest['identity']['script_sha256'] == export['runner_script_sha256']
                       and manifest['corpus_sha256'] == export['corpus_sha256']
                       and manifest['n_records'] == len(rows)
                       and manifest['n_unique_texts'] == manifest['n_complete_unique_texts'] == export['n_unique_texts']
                       and manifest['scores_sha256'] == sha(directory / 'scores.jsonl'))
        scores = events(directory / 'scores.jsonl')
        score_index = {row['id']: row for row in scores}
        checks.require('score_row_set', len(score_index) == len(scores) == len(rows) and set(score_index) == set(indexed))
        measurements, expected_checkpoints = {}, set()
        for key, corpus_row in indexed.items():
            row = score_index[key]
            cache_key = digest({'job_fingerprint': fingerprint, 'text_sha256': corpus_row['text_sha256']})
            expected_checkpoints.add(cache_key)
            path = directory / 'checkpoints' / (cache_key + '.json')
            checkpoint = read(path)
            checks.require('score_cache_identity', row['text_sha256'] == checkpoint['text_sha256'] == corpus_row['text_sha256']
                           and row['cache_key'] == cache_key and checkpoint['job_fingerprint'] == fingerprint
                           and checkpoint['status'] == 'complete')
            checks.exact('checkpoint_score', [row['score'], row['details']], [checkpoint['score'], checkpoint['details']])
            measurements[key] = {k: corpus_row[k] for k in ('id', 'model_alias', 'sample_id', 'topic_id', 'genre_id', 'payload_bits', 'arm', 'text_sha256')} | validate_score(row, checks)
        checks.require('exact_checkpoint_set', {p.stem for p in (directory / 'checkpoints').glob('*.json')} == expected_checkpoints)
        retained_job = summary['quality_models'][job]
        checks.exact('summary_job_identity', retained_job['job'], expected_job)
        checks.require('summary_score_identity', retained_job['job_fingerprint'] == fingerprint
                       and retained_job['scores_sha256'] == manifest['scores_sha256']
                       and retained_job['score_manifest_sha256'] == sha(directory / 'manifest.json'))
        retained_measurements = {row['id']: row for row in retained_job['measurements']}
        checks.require('summary_measurements_once', len(retained_measurements) == len(retained_job['measurements']) == len(measurements))
        checks.exact('summary_measurements', retained_measurements, measurements)
        lookup = {(row['model_alias'], row['sample_id'], row['arm']): row for row in measurements.values()}
        results[job] = {}
        for cohort_name, ids_by_alias in [('per_model_common_success', common), ('matched_all_models_success', {alias: matched for alias in common})]:
            results[job][cohort_name] = {}
            checks.require('cohort_model_names', set(retained_job[cohort_name]) == set(common))
            for alias, ids in ids_by_alias.items():
                retained = retained_job[cohort_name][alias]
                checks.require('paired_cohort_count', retained['common_success_pairs'] == len(ids)
                               and retained['topic_clusters'] == len({lookup[alias, sid, 'cover']['topic_id'] for sid in ids}) == 4)
                paired_rows = {row['sample_id']: row for row in retained['paired_rows']}
                checks.require('paired_row_ids', len(paired_rows) == len(retained['paired_rows']) == len(ids) and set(paired_rows) == set(ids))
                checks.require('exact_comparison_names', set(retained['effects']) == set(COMPARISONS)
                               and set(retained['paired_cohort_arm_distributions']) == set(ARMS))
                for sid in ids:
                    checks.exact('paired_row_metadata',
                        [paired_rows[sid]['topic_id'], paired_rows[sid]['payload_bits'], paired_rows[sid]['identical_selector_text']],
                        [lookup[alias, sid, 'cover']['topic_id'], lookup[alias, sid, 'cover']['payload_bits'],
                         lookup[alias, sid, ARMS[1]]['text_sha256'] == lookup[alias, sid, ARMS[2]]['text_sha256']])
                effects = {}
                for name, (left, right) in COMPARISONS.items():
                    effects[name] = paired_effect([lookup[alias, sid, left] for sid in ids], [lookup[alias, sid, right] for sid in ids])
                    checks.equal('independent_paired_effect', effects[name], retained['effects'][name])
                    for sid in ids:
                        checks.equal('per_sample_nll_difference', paired_rows[sid][name + '_nll_delta'],
                                     lookup[alias, sid, left]['mean_nll'] - lookup[alias, sid, right]['mean_nll'])
                        checks.equal('per_sample_ppl_difference', paired_rows[sid][name + '_ppl_delta'],
                                     lookup[alias, sid, left]['ppl'] - lookup[alias, sid, right]['ppl'])
                identical = sum(lookup[alias, sid, ARMS[1]]['text_sha256'] == lookup[alias, sid, ARMS[2]]['text_sha256'] for sid in ids)
                checks.require('identical_selector_text_count', identical == retained['identical_selector_text_pairs'])
                for arm in ARMS:
                    for metric in ('mean_nll', 'ppl', 'predicted_tokens'):
                        checks.equal('paired_arm_distribution', stats([lookup[alias, sid, arm][metric] for sid in ids]),
                                     retained['paired_cohort_arm_distributions'][arm][metric])
                results[job][cohort_name][alias] = {'paired_ids': ids, 'pairs': len(ids), 'identical_selector_text_pairs': identical, 'effects': effects}
        job_evidence[job] = {'job_fingerprint': fingerprint, 'scores_sha256': manifest['scores_sha256'],
            'manifest_sha256': sha(directory / 'manifest.json'), 'records': len(scores), 'verified_checkpoints': len(expected_checkpoints),
            'input_token_count_range': [min(r['input_tokens'] for r in measurements.values()), max(r['input_tokens'] for r in measurements.values())],
            'all_rows_have_exact_n_minus_one_targets': True, 'any_truncated': False}
    runtime = {'status': 'not_requested_not_verified'}
    if owned_proof:
        proof = read(owned_proof)
        checks.require('secondary_owned_completion', proof['status'] == 'completed'
                       and [step['stage'] for step in proof['stages']] == ['export', 'gpt2', 'qwen_base', 'summary']
                       and all(step['status'] == 'completed' and type(step['actual_wait_returncode']) is int
                               and step['actual_wait_returncode'] == 0 for step in proof['stages']))
        runtime = {'status': 'four_successful_owned_stages_from_retained_evidence', 'sha256': sha(owned_proof),
                   'stages': [{'stage': step['stage'], 'exit_code': step['actual_wait_returncode'],
                               'elapsed_seconds': step['elapsed_seconds']} for step in proof['stages']]}
    after = {p.relative_to(secondary).as_posix(): sha(p) for p in sorted(secondary.rglob('*')) if p.is_file()}
    checks.exact('secondary_unchanged', before, after)
    checks.exact('primary_unchanged', current_primary, {p.relative_to(primary).as_posix(): sha(p) for p in sorted(primary.rglob('*')) if p.is_file()})
    return {'schema': 'synsteg-crossmodel-secondary-independent-audit-v1', 'status': 'pass_in_declared_scope',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'freeze_id': expected_freeze,
        'audit_script_sha256': sha(__file__), 'primary_audit_sha256': sha(primary_audit_path),
        'assertions': sum(checks.counts.values()), 'check_counts': dict(checks.counts),
        'n_records': len(rows), 'n_unique_texts': export['n_unique_texts'],
        'corpus_sha256': export['corpus_sha256'], 'summary_sha256': sha(summary_path),
        'frozen_secondary_sources': source_evidence, 'postfreeze_compatibility_launcher': launcher,
        'job_evidence': job_evidence, 'paired_effects': results, 'owned_execution': runtime,
        'primary_file_count': len(current_primary), 'secondary_file_count': len(before),
        'secondary_file_manifest_sha256': digest(before), 'primary_modified': False, 'secondary_modified': False,
        'limitations': ['Checkpoint, hash, full-target-count and arithmetic verification; no neural forward pass, tokenization, or model weight/environment reconstruction.',
            'The retained n-1 coverage claims and sums are checked; actual per-token losses are not independently recalculated.',
            'Ratios compare paired document mean NLL within each scorer and tokenizer. They do not compare raw perplexity across tokenizers.',
            'GPT-2 is also the selection model; Qwen base shares a generator family. These are descriptive proxies, not human semantic quality or security tests.',
            'Historical six-model scores and transmission summaries are outside this secondary audit; the independent primary audit covers delivery and full-book costs.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--secondary', type=Path, required=True)
    parser.add_argument('--primary-audit', type=Path, required=True)
    parser.add_argument('--expected-freeze-id', required=True)
    parser.add_argument('--owned-proof', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(root.resolve()) for root in (args.primary, args.secondary)):
        parser.error('Output must be a new file outside primary and secondary runs')
    report = audit(args.repo.resolve(), args.primary.resolve(), args.secondary.resolve(),
                   args.primary_audit, args.expected_freeze_id, args.owned_proof)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'assertions': report['assertions'], 'output': str(output)}))


if __name__ == '__main__':
    main()
