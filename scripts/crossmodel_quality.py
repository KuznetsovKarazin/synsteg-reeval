#!/usr/bin/env python3
"""Descriptive full-text language scores after frozen cross-generator evaluation.

GPT-2 is the selector's own model. Qwen base is independent from selection,
but belongs to the family of two generators. No cross-tokenizer PPL ranking,
new attack fit, threshold selection, hypothesis test or semantic claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
JOBS = {
    'gpt2': ROOT / 'configs/neural/gpt2-full.json',
    'qwen_base': ROOT / 'configs/neural/crossmodel-qwen-base-ppl.json',
}
VARIANTS = ('fixed_canonical', 'fixed_quality')
ARMS = ('cover', *VARIANTS)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


transfer = module('crossmodel_quality_transfer', ROOT / 'scripts/crossmodel_transfer_pilot.py')
neural = module('crossmodel_quality_neural', ROOT / 'scripts/rescore_models.py')
measure = module('crossmodel_quality_measure', ROOT / 'scripts/summarize_prospective_results.py')
canonical, digest, file_hash = transfer.canonical, transfer.digest, transfer.file_hash
read_json, distribution = transfer.read_json, measure.distribution


def load_completed(run):
    """Do not access generated records until the whole primary gate passes."""
    run = Path(run)
    for name in ('evaluation.json', 'evaluation_opened.json', 'events.jsonl'):
        if not (run / name).is_file():
            raise ValueError('Completed primary transfer evaluation required: ' + name)
    config, frozen, lock = transfer.verify(run)
    evaluation = read_json(run / 'evaluation.json')
    opening = read_json(run / 'evaluation_opened.json')
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines() if line.strip()]
    opened = [row for row in events if row.get('event') == 'transfer_evaluation_opened_once']
    completed = [row for row in events if row.get('event') == 'transfer_evaluation_completed']
    if (len(opened) != 1 or len(completed) != 1
            or completed[0].get('evaluation_sha256') != file_hash(run / 'evaluation.json')
            or completed[0]['time_unix'] < opened[0]['time_unix']
            or evaluation.get('status') != 'complete_frozen_attack_transfer_evaluation'):
        raise ValueError('Primary evaluation lacks one intact opening and completion')
    aliases = [model['alias'] for model in config['models']]
    expected_hashes = evaluation.get('all_model_records_sha256') or {}
    if set(expected_hashes) != set(aliases) or set(evaluation.get('models', {})) != set(aliases):
        raise ValueError('Primary model set differs from protocol')
    for value in (evaluation, opening, opened[0]):
        if (value.get('freeze_id') != frozen['freeze_id']
                or value.get('source_lock_id') != lock['lock_id']
                or value.get('all_model_records_sha256') != expected_hashes):
            raise ValueError('Primary evaluation identities differ')
    for alias in aliases:
        status = read_json(run / 'models' / alias / 'generation.json')
        if (status.get('complete') is not True or status.get('records') != len(config['samples'])
                or status.get('freeze_id') != frozen['freeze_id']
                or status.get('records_sha256') != expected_hashes[alias]):
            raise ValueError('Incomplete or altered generator: ' + alias)
        for cohort in evaluation['models'][alias]['cohorts'].values():
            if set(cohort['attacks']) != set(lock['attacks']):
                raise ValueError('Primary attack set differs')
            if any(a.get('status') not in {'complete', 'no_common_success_pairs'} for a in cohort['attacks'].values()):
                raise ValueError('Nonterminal primary attack result')
    # The first generated-record access happens here, after every gate above.
    records = {alias: transfer.read_records(run, alias, complete=True) for alias in aliases}
    for alias, rows in records.items():
        if digest(rows) != expected_hashes[alias]:
            raise ValueError('Generated records differ from completed evaluation')
        if any(row.get('sample') != sample or row.get('freeze_id') != frozen['freeze_id']
               or row.get('model_alias') != alias for row, sample in zip(rows, config['samples'])):
            raise ValueError('Record identity differs from frozen shared design')
    return config, evaluation, records


def jsonl(rows):
    return b''.join(canonical(row) + b'\n' for row in rows)


def make_export(run):
    config, evaluation, records = load_completed(run)
    rows, exclusions, common_ids = [], [], {}
    for alias, source in records.items():
        common_ids[alias] = []
        for record in source:
            sample, cover = record['sample'], record.get('cover_text')
            available = []
            for arm in ARMS:
                output = record.get('variants', {}).get(arm, {})
                metrics = output.get('metrics') or {}
                if arm == 'cover':
                    text = cover if isinstance(cover, str) and cover.strip() else None
                    reason = record.get('failure') or 'no_generated_cover'
                elif metrics.get('full_message_success') is True:
                    payload = record.get('synthetic_payload')
                    if (not isinstance(cover, str) or not cover.strip() or not isinstance(payload, str)
                            or set(payload) - {'0', '1'} or len(payload) != sample['payload_bits']
                            or output.get('decoded_payload') != payload
                            or metrics.get('requested_bits') != sample['payload_bits']):
                        raise ValueError('Full success lacks exact delivery evidence')
                    if (output.get('metadata', {}).get('framing') != 'external_length'
                            or output['metadata'].get('external_metadata_bits') != 32):
                        raise ValueError('Unexpected external framing')
                    text, reason = output.get('stego_text'), None
                    available.append(arm)
                else:
                    text = None
                    reason = record.get('failure') or output.get('error') or 'not_full_message_success'
                if text is None:
                    exclusions.append({'model_alias': alias, 'sample_id': sample['id'], 'arm': arm,
                                       'payload_bits': sample['payload_bits'], 'reason': reason})
                    continue
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('Exported arm has no nonempty text')
                rows.append({'id': json.dumps([evaluation['freeze_id'], alias, sample['id'], arm], separators=(',', ':')),
                             'model_alias': alias, 'sample_id': sample['id'], 'topic_id': sample['topic_id'],
                             'genre_id': sample['genre_id'], 'payload_bits': sample['payload_bits'], 'arm': arm,
                             'text': text, 'text_sha256': neural.text_hash(text)})
            if set(available) == set(VARIANTS):
                if digest(record['variants'][VARIANTS[0]]['key']) != digest(record['variants'][VARIANTS[1]]['key']):
                    raise ValueError('Paired selectors do not share one codebook')
                common_ids[alias].append(sample['id'])
    corpus = jsonl(rows)
    manifest = {
        'schema': 'synsteg-crossmodel-secondary-v1', 'status': 'exported_after_completed_primary_evaluation',
        'freeze_id': evaluation['freeze_id'], 'source_lock_id': evaluation['source_lock_id'],
        'evaluation_sha256': file_hash(Path(run) / 'evaluation.json'),
        'all_model_records_sha256': evaluation['all_model_records_sha256'],
        'helper_script_sha256': file_hash(__file__), 'runner_script_sha256': file_hash(ROOT / 'scripts/rescore_models.py'),
        'measurement_script_sha256': file_hash(ROOT / 'scripts/summarize_prospective_results.py'),
        'expected_jobs': {name: neural.validate_job(read_json(path)) for name, path in JOBS.items()},
        'job_files_sha256': {name: file_hash(path) for name, path in JOBS.items()},
        'corpus_sha256': hashlib.sha256(corpus).hexdigest(), 'n_records': len(rows),
        'n_unique_texts': len({row['text_sha256'] for row in rows}),
        'planned_attempts_by_model': {alias: len(source) for alias, source in records.items()},
        'common_success_sample_ids': common_ids,
        'matched_all_models_success_sample_ids': sorted(set.intersection(*(set(ids) for ids in common_ids.values()))),
        'exclusions': exclusions,
        'cohort_rule': 'All generated covers, successful outputs only; paired effects use both successful selectors. No missing quality score is imputed.',
    }
    return manifest, rows, corpus, records


def export_corpus(run, output):
    manifest, _, corpus, _ = make_export(run)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite secondary export')
    output.mkdir(parents=True)
    (output / 'manifest.json').write_bytes(canonical(manifest) + b'\n')
    (output / 'corpus.jsonl').write_bytes(corpus)
    return manifest


def validate_full_text_score(score, detail):
    nll, total = detail.get('mean_nll'), detail.get('sum_nll')
    n, predicted = detail.get('input_tokens'), detail.get('predicted_tokens')
    if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (nll, score, total))
            or nll < 0 or score < 1 or type(n) is not int or n < 2
            or detail.get('scored_input_tokens') != n or type(predicted) is not int or predicted != n - 1
            or detail.get('truncated') is not False or detail.get('ppl_policy') != 'sliding_window'
            or not math.isclose(total / predicted, nll, rel_tol=1e-10, abs_tol=1e-12)
            or not math.isclose(math.log(score), nll, rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError('Invalid, incomplete, or truncated full-text language score')


def verified_scores(directory, expected, rows, job_name):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    identity = manifest.get('identity') or {}
    fingerprint = neural.digest(identity)
    if (manifest.get('status') != 'complete' or manifest.get('n_records') != len(rows)
            or manifest.get('n_unique_texts') != expected['n_unique_texts']
            or manifest.get('n_complete_unique_texts') != expected['n_unique_texts']
            or manifest.get('corpus_sha256') != expected['corpus_sha256']
            or manifest.get('job_fingerprint') != fingerprint
            or identity.get('job') != expected['expected_jobs'][job_name]
            or identity.get('device') != 'cpu'
            or identity.get('script_sha256') != expected['runner_script_sha256']
            or manifest.get('scores_sha256') != file_hash(directory / 'scores.jsonl')):
        raise ValueError('Incomplete or mismatched score manifest: ' + job_name)
    scores = [json.loads(line) for line in (directory / 'scores.jsonl').read_text().splitlines() if line.strip()]
    indexed = {row['id']: row for row in scores}
    if len(indexed) != len(scores) or set(indexed) != {row['id'] for row in rows}:
        raise ValueError('Score identities differ from secondary corpus')
    measurements = []
    for row in rows:
        score = indexed[row['id']]
        key = neural.cache_key(fingerprint, row['text_sha256'])
        checkpoint = neural.read_checkpoint(directory / 'checkpoints' / (key + '.json'), fingerprint, row['text_sha256'])
        if (checkpoint is None or score.get('text_sha256') != row['text_sha256']
                or score.get('cache_key') != key or score.get('score') != checkpoint['score']
                or score.get('details') != checkpoint.get('details')):
            raise ValueError('Score and individual checkpoint differ')
        detail = score.get('details') or {}
        validate_full_text_score(score.get('score'), detail)
        measurements.append({k: row[k] for k in ('id', 'model_alias', 'sample_id', 'topic_id', 'genre_id', 'payload_bits', 'arm', 'text_sha256')} |
                            {'mean_nll': detail['mean_nll'], 'ppl': score['score'], 'sum_nll': detail['sum_nll'],
                             'input_tokens': detail['input_tokens'], 'predicted_tokens': detail['predicted_tokens']})
    return manifest, measurements


def pair_effects(measurements, aliases_and_ids):
    indexed = {(row['model_alias'], row['sample_id'], row['arm']): row for row in measurements}
    result = {}
    comparisons = {'quality_vs_canonical': ('fixed_quality', 'fixed_canonical'),
                   'canonical_vs_cover': ('fixed_canonical', 'cover'), 'quality_vs_cover': ('fixed_quality', 'cover')}
    for alias, ids in aliases_and_ids.items():
        paired = []
        for sample_id in ids:
            chosen = {arm: indexed[alias, sample_id, arm] for arm in ARMS}
            item = {'sample_id': sample_id, 'topic_id': chosen['cover']['topic_id'],
                    'payload_bits': chosen['cover']['payload_bits'],
                    'identical_selector_text': chosen[VARIANTS[0]]['text_sha256'] == chosen[VARIANTS[1]]['text_sha256']}
            for name, (left, right) in comparisons.items():
                item[name + '_nll_delta'] = chosen[left]['mean_nll'] - chosen[right]['mean_nll']
                item[name + '_ppl_delta'] = chosen[left]['ppl'] - chosen[right]['ppl']
            paired.append(item)
        effects = {}
        for name in comparisons:
            deltas = [row[name + '_nll_delta'] for row in paired]
            stats = distribution(deltas)
            effects[name] = {'document_mean_nll_delta': stats,
                             'geometric_mean_ppl_ratio': math.exp(stats['mean']) if deltas else None,
                             'document_ppl_delta': distribution(row[name + '_ppl_delta'] for row in paired),
                             'lower': sum(x < -1e-12 for x in deltas),
                             'equal_within_1e_12': sum(abs(x) <= 1e-12 for x in deltas),
                             'higher': sum(x > 1e-12 for x in deltas)}
        result[alias] = {'common_success_pairs': len(ids), 'topic_clusters': len({row['topic_id'] for row in paired}),
                         'identical_selector_text_pairs': sum(row['identical_selector_text'] for row in paired),
                         'effects': effects, 'paired_rows': paired,
                         'paired_cohort_arm_distributions': {arm: {metric: distribution(row[metric] for row in measurements
                              if row['model_alias'] == alias and row['arm'] == arm and row['sample_id'] in ids)
                              for metric in ('mean_nll', 'ppl', 'predicted_tokens')} for arm in ARMS}}
    return result


def transmission_measurements(records):
    result = {}
    for alias, source in records.items():
        for record in source:
            for variant in VARIANTS:
                output = record.get('variants', {}).get(variant, {})
                if (output.get('metrics') or {}).get('full_message_success') is not True:
                    continue
                # Rebuild dataclass order used by encode_decode, not sorted JSON order.
                original_key = transfer.local.FrozenCodebook.from_dict(output['key']).to_dict()
                encoded = json.dumps(original_key, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
                if (len(encoded) != output['metrics']['key_json_bytes']
                        or len(zlib.compress(encoded, 9)) != output['metrics']['key_zlib9_bytes']):
                    raise ValueError('Encoder-time codebook serialization cost differs')
        rows = [measure.measure_record(record, variant) for record in source for variant in VARIANTS]
        generation = [record.get('generation_metadata') or {} for record in source]
        result[alias] = {'variants': {variant: {
                            'overall': measure.aggregate([row for row in rows if row['variant'] == variant]),
                            'by_payload_bits': {str(bits): measure.aggregate([row for row in rows if row['variant'] == variant and row['payload_bits'] == bits])
                                                for bits in sorted({row['payload_bits'] for row in rows})}}
                        for variant in VARIANTS},
                        'selector_comparison': measure.selector_comparison(source, VARIANTS),
                        'generation_seconds': distribution(row['seconds'] for row in generation if row.get('seconds') is not None),
                        'full_attempt_seconds': distribution(record['finished_at'] - record['started_at'] for record in source),
                        'token_alignment': measure.token_alignment_summary(source, VARIANTS),
                        'records': rows}
    return result


def summarize(run, corpus_directory, score_directories):
    expected, rows, corpus, records = make_export(run)
    corpus_directory = Path(corpus_directory)
    if (read_json(corpus_directory / 'manifest.json') != expected
            or (corpus_directory / 'corpus.jsonl').read_bytes() != corpus):
        raise ValueError('Secondary export differs from completed primary data')
    if set(score_directories) != set(JOBS):
        raise ValueError('Both preregistered quality models are required')
    jobs = {}
    for name, directory in score_directories.items():
        manifest, measurements = verified_scores(directory, expected, rows, name)
        jobs[name] = {'job': expected['expected_jobs'][name], 'job_fingerprint': manifest['job_fingerprint'],
                      'scores_sha256': manifest['scores_sha256'], 'score_manifest_sha256': file_hash(Path(directory) / 'manifest.json'),
                      'measurements': measurements,
                      'per_model_common_success': pair_effects(measurements, expected['common_success_sample_ids']),
                      'matched_all_models_success': pair_effects(measurements, {alias: expected['matched_all_models_success_sample_ids']
                                                          for alias in expected['common_success_sample_ids']})}
    return {'status': 'complete_crossmodel_descriptive_secondary_measurements',
            'freeze_id': expected['freeze_id'], 'source_lock_id': expected['source_lock_id'],
            'evaluation_sha256': expected['evaluation_sha256'], 'all_model_records_sha256': expected['all_model_records_sha256'],
            'helper_script_sha256': expected['helper_script_sha256'], 'corpus_sha256': expected['corpus_sha256'],
            'n_records': expected['n_records'], 'n_unique_texts': expected['n_unique_texts'],
            'exclusions': expected['exclusions'], 'transmission': transmission_measurements(records), 'quality_models': jobs,
            'limitations': [
                'All planned attempts count for transmission reliability; paired quality effects condition on full delivery.',
                'Per-generator success cohorts may differ; the common intersection across all generators is reported separately.',
                'GPT-2 also selects replacements and is not an independent quality model.',
                'Qwen base is independent from the selector but shares the Qwen generator family; this may favor that family.',
                'Different tokenizers define different token averages; do not rank models by raw cross-tokenizer PPL.',
                'Full text scoring counts each causal target exactly once and does not silently truncate.',
                'Lower model NLL does not establish semantics, human naturalness, steganographic secrecy, or superiority.',
                'Key size is serialized document-specific codebook cost plus external length; it is not seed-only communication.',
                'No attacks, thresholds, generation settings or hypotheses were adjusted after the frozen primary experiment.']}


def markdown(report):
    def fmt(value):
        return 'n/a' if value is None else f'{value:.6f}' if isinstance(value, float) else str(value)
    lines = ['# Cross-generator transfer pilot: supplementary measurements', '',
             'Descriptive measurements after the completed primary evaluation; no tuning or attack retraining.', '',
             '| Generator / selector | Full deliveries | Delivered / requested bits | Goodput per all generated cover words | Median compressed codebook bytes |',
             '|---|---:|---:|---:|---:|']
    for alias, transmission in report['transmission'].items():
        for variant, item in transmission['variants'].items():
            row = item['overall']
            lines.append(f"| {alias} / {variant} | {row['full_success_samples']}/{row['attempted_samples']} | {row['delivered_payload_bits']}/{row['attempted_payload_bits']} | {fmt(row['full_message_goodput_per_all_generated_cover_word'])} | {fmt(row['successful_transmission_costs']['key_zlib9_bytes']['median'])} |")
    lines += ['', 'Codebook bytes and 32 external length bits are excluded from text goodput and must be accounted for separately.', '',
              '| Scorer / generator | Paired successes | Quality/canonical geometric PPL ratio | Lower / equal / higher | Quality/cover geometric PPL ratio |',
              '|---|---:|---:|---:|---:|']
    for name, job in report['quality_models'].items():
        for alias, row in job['per_model_common_success'].items():
            effect = row['effects']['quality_vs_canonical']
            lines.append(f"| {name} / {alias} | {row['common_success_pairs']} | {fmt(effect['geometric_mean_ppl_ratio'])} | {effect['lower']}/{effect['equal_within_1e_12']}/{effect['higher']} | {fmt(row['effects']['quality_vs_cover']['geometric_mean_ppl_ratio'])} |")
    lines += ['', 'Each ratio compares texts scored with the same model and tokenizer. The JSON also retains every score, paired delta, failed attempt and the matched success intersection across all generators.', '']
    lines += ['- ' + limitation for limitation in report['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--run', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    summary = sub.add_parser('summarize')
    for name in ('run', 'corpus', 'gpt2-scores', 'qwen-base-scores', 'output'):
        summary.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        report = export_corpus(args.run, args.output)
        print(json.dumps({name: report[name] for name in ('status', 'n_records', 'n_unique_texts', 'planned_attempts_by_model')}))
    else:
        report = summarize(args.run, args.corpus, {'gpt2': args.gpt2_scores, 'qwen_base': args.qwen_base_scores})
        if args.output.exists():
            raise FileExistsError('Refusing to overwrite final secondary measurements')
        args.output.mkdir(parents=True)
        (args.output / 'measurements.json').write_bytes(canonical(report) + b'\n')
        (args.output / 'measurements.md').write_text(markdown(report))
        print(json.dumps({'status': report['status'], 'n_records': report['n_records'], 'n_unique_texts': report['n_unique_texts']}))


if __name__ == '__main__':
    main()
