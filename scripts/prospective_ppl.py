#!/usr/bin/env python3
"""Gated secondary GPT-2 full-text measurements after primary test completion."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/prospective_v3_secondary_ppl_plan.md'
DEFAULT_JOB = ROOT / 'configs/neural/gpt2-full.json'
VARIANTS = ('fixed_canonical', 'fixed_quality')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


gate = module('prospective_ppl_gate', ROOT / 'scripts/summarize_prospective_results.py')
runner = module('prospective_ppl_runner', ROOT / 'scripts/rescore_models.py')


def jsonl(rows):
    return b''.join(gate.canonical(row) + b'\n' for row in rows)


def make_export(run, job_path=None):
    # The imported integrity gate is the first operation that accesses the run.
    protocol, evaluation, records = gate.load_completed_test(run)
    if set(protocol['variants']) != set(VARIANTS):
        raise ValueError('Secondary comparison requires exactly the two frozen selectors')
    job = runner.validate_job(gate.read_json(job_path or DEFAULT_JOB))
    if job != runner.validate_job(gate.read_json(DEFAULT_JOB)):
        raise ValueError('Secondary job differs from the prespecified full-text GPT-2 job')
    rows, exclusions, common_ids = [], [], []
    successful = {variant: 0 for variant in VARIANTS}
    for record in records:
        sample, cover = record['sample'], record.get('cover_text')
        sample_id = sample['id']
        cover_exists = isinstance(cover, str) and bool(cover.strip())
        common = []
        for arm in ('cover', *VARIANTS):
            output = record.get('variants', {}).get(arm, {})
            metrics = output.get('metrics') or {}
            if arm == 'cover':
                text = cover if cover_exists else None
                reason = record.get('failure') or 'no_generated_cover'
                framing, external_bits = None, None
            else:
                success = metrics.get('full_message_success') is True
                if success:
                    payload = record.get('synthetic_payload')
                    if (not cover_exists or not isinstance(payload, str)
                            or set(payload) - {'0', '1'}
                            or len(payload) != sample['payload_bits']
                            or output.get('decoded_payload') != payload
                            or metrics.get('requested_bits') != sample['payload_bits']):
                        raise ValueError('Successful output lacks exact complete delivery evidence')
                    text = output.get('stego_text')
                    metadata = output.get('metadata') or {}
                    framing, external_bits = metadata.get('framing'), metadata.get('external_metadata_bits')
                    if framing != 'external_length' or external_bits != 32:
                        raise ValueError('Successful output differs from declared external-length framing')
                    successful[arm] += 1
                    common.append(arm)
                else:
                    text, framing, external_bits = None, None, None
                reason = record.get('failure') or output.get('error') or 'not_full_message_success'
            if text is None:
                exclusions.append({'sample_id': sample_id, 'topic_id': sample['topic_id'],
                    'arm': arm, 'requested_payload_bits': sample['payload_bits'], 'reason': reason})
                continue
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Selected full-success output has no nonempty text')
            rows.append({'id': json.dumps([evaluation['freeze_id'], sample_id, arm], separators=(',', ':')),
                'dataset': 'prospective_v3_test_secondary_ppl', 'uuid': sample_id,
                'sample_id': sample_id, 'topic_id': sample['topic_id'], 'genre_id': sample['genre_id'],
                'arm': arm, 'requested_payload_bits': sample['payload_bits'], 'text': text,
                'text_sha256': runner.text_hash(text), 'framing': framing,
                'external_metadata_bits': external_bits,
                'evaluation_role': 'secondary_after_completed_primary_test'})
        if len(common) == len(VARIANTS):
            if len({gate.digest(record['variants'][v]['key']) for v in VARIANTS}) != 1:
                raise ValueError('Common successful selectors do not share one codebook')
            common_ids.append(sample_id)
    corpus = jsonl(rows)
    manifest = {'schema': 'prospective-secondary-ppl-export-v1',
        'status': 'exported_after_completed_primary_one_opening_evaluation',
        'freeze_id': evaluation['freeze_id'], 'lock_id': evaluation['lock_id'],
        'test_records_sha256': evaluation['test_records_sha256'],
        'evaluation_sha256': gate.file_hash(Path(run) / 'evaluation.json'),
        'plan_sha256': gate.file_hash(PLAN), 'export_script_sha256': gate.file_hash(__file__),
        'gate_script_sha256': gate.file_hash(ROOT / 'scripts/summarize_prospective_results.py'),
        'runner_script_sha256': gate.file_hash(ROOT / 'scripts/rescore_models.py'),
        'expected_job': job, 'corpus_sha256': hashlib.sha256(corpus).hexdigest(),
        'planned_attempts': len(records), 'generated_covers': sum(r['arm'] == 'cover' for r in rows),
        'successful_outputs': successful, 'common_success_samples': len(common_ids),
        'common_success_sample_ids': common_ids, 'n_records': len(rows),
        'n_unique_texts': len({r['text_sha256'] for r in rows}), 'exclusions': exclusions,
        'cohort_note': 'Every generated cover; only full-success variants. Pairing uses common success. Failure and metadata costs are not PPL outcomes.'}
    return manifest, rows, corpus


def export_corpus(run, output, job_path=None):
    manifest, rows, corpus = make_export(run, job_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite the secondary corpus')
    output.mkdir(parents=True)
    (output / 'corpus.jsonl').write_bytes(corpus)
    (output / 'manifest.json').write_bytes(gate.canonical(manifest) + b'\n')
    return manifest


def summarize(run, corpus_dir, scores_dir):
    expected, rows, corpus = make_export(run)
    corpus_dir, scores_dir = Path(corpus_dir), Path(scores_dir)
    if gate.read_json(corpus_dir / 'manifest.json') != expected:
        raise ValueError('Secondary export identity differs from completed primary records')
    if (corpus_dir / 'corpus.jsonl').read_bytes() != corpus:
        raise ValueError('Secondary corpus bytes differ from the verified export')
    if not rows:
        raise ValueError('No generated texts to score; retain the export failure ledger')
    manifest = gate.read_json(scores_dir / 'manifest.json')
    identity = manifest.get('identity') or {}
    fingerprint = runner.digest(identity)
    if (manifest.get('status') != 'complete'
            or manifest.get('n_records') != len(rows)
            or manifest.get('n_unique_texts') != expected['n_unique_texts']
            or manifest.get('n_complete_unique_texts') != expected['n_unique_texts']
            or manifest.get('corpus_sha256') != expected['corpus_sha256']
            or manifest.get('job_fingerprint') != fingerprint
            or identity.get('job') != expected['expected_job']
            or identity.get('script_sha256') != expected['runner_script_sha256']
            or manifest.get('scores_sha256') != gate.file_hash(scores_dir / 'scores.jsonl')):
        raise ValueError('Incomplete or mismatched secondary neural score manifest')
    scores = [json.loads(line) for line in (scores_dir / 'scores.jsonl').read_text().splitlines() if line.strip()]
    indexed = {row['id']: row for row in scores}
    if len(indexed) != len(scores) or set(indexed) != {r['id'] for r in rows}:
        raise ValueError('Secondary score record set differs from the corpus')
    measurements = []
    for row in rows:
        score = indexed[row['id']]
        key = runner.cache_key(fingerprint, row['text_sha256'])
        checkpoint = runner.read_checkpoint(scores_dir / 'checkpoints' / (key + '.json'), fingerprint, row['text_sha256'])
        if (checkpoint is None or score.get('text_sha256') != row['text_sha256']
                or score.get('cache_key') != key or score.get('score') != checkpoint['score']
                or score.get('details') != checkpoint.get('details')):
            raise ValueError('Secondary score or checkpoint identity differs')
        detail = score.get('details') or {}
        nll, ppl = detail.get('mean_nll'), score.get('score')
        total, predicted = detail.get('sum_nll'), detail.get('predicted_tokens')
        n = detail.get('input_tokens')
        if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in (nll, ppl, total))
                or nll < 0 or ppl < 1 or type(n) is not int or n < 2
                or detail.get('scored_input_tokens') != n or type(predicted) is not int or predicted != n - 1
                or detail.get('truncated') is not False or detail.get('ppl_policy') != 'sliding_window'
                or not math.isclose(total / predicted, nll, rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(math.log(ppl), nll, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError('Invalid or truncated full-text NLL/PPL details')
        measurements.append({k: row[k] for k in ('id', 'sample_id', 'topic_id', 'genre_id', 'arm',
            'requested_payload_bits', 'text_sha256')} | {'mean_nll': nll, 'ppl': ppl,
            'sum_nll': total, 'predicted_tokens': predicted, 'input_tokens': n})
    by_pair = {(row['sample_id'], row['arm']): row for row in measurements}
    paired_rows = []
    for sample_id in expected['common_success_sample_ids']:
        selected = {arm: by_pair[sample_id, arm] for arm in ('cover', *VARIANTS)}
        paired = {'sample_id': sample_id, 'topic_id': selected['cover']['topic_id'],
            'requested_payload_bits': selected['cover']['requested_payload_bits'], 'scores': selected,
            'identical_selector_text': selected[VARIANTS[0]]['text_sha256'] == selected[VARIANTS[1]]['text_sha256']}
        for metric in ('mean_nll', 'ppl'):
            paired['quality_minus_canonical_' + metric] = selected[VARIANTS[1]][metric] - selected[VARIANTS[0]][metric]
            for label, variant in zip(('canonical', 'quality'), VARIANTS):
                paired[label + '_minus_cover_' + metric] = selected[variant][metric] - selected['cover'][metric]
        paired_rows.append(paired)
    delta_names = [prefix + metric for prefix in ('quality_minus_canonical_', 'canonical_minus_cover_', 'quality_minus_cover_')
                   for metric in ('mean_nll', 'ppl')]
    return {'status': 'complete_secondary_descriptive_full_text_ppl',
        'freeze_id': expected['freeze_id'], 'lock_id': expected['lock_id'],
        'evaluation_sha256': expected['evaluation_sha256'], 'test_records_sha256': expected['test_records_sha256'],
        'plan_sha256': expected['plan_sha256'], 'script_sha256': gate.file_hash(__file__),
        'corpus_sha256': expected['corpus_sha256'], 'scores_sha256': manifest['scores_sha256'],
        'score_manifest_sha256': gate.file_hash(scores_dir / 'manifest.json'), 'job_fingerprint': fingerprint,
        'planned_attempts': expected['planned_attempts'], 'generated_covers': expected['generated_covers'],
        'common_success_samples': len(paired_rows), 'common_success_topics': len({x['topic_id'] for x in paired_rows}),
        'identical_selector_output_pairs': sum(x['identical_selector_text'] for x in paired_rows),
        'exclusions': expected['exclusions'], 'measurements': measurements, 'paired_rows': paired_rows,
        'available_arm_distributions': {arm: {metric: gate.distribution(x[metric] for x in measurements if x['arm'] == arm)
                                            for metric in ('mean_nll', 'ppl', 'predicted_tokens')}
                                        for arm in ('cover', *VARIANTS)},
        'paired_deltas': {name: gate.distribution(row[name] for row in paired_rows) for name in delta_names},
        'limitations': ['Secondary same-GPT-2 descriptive measurement; not independent human or semantic quality evidence.',
            'No hypothesis tests, attack refitting, threshold choice, or security claim.',
            'Each document has equal weight; arithmetic mean PPL is not exp(mean document NLL).',
            'Variant token counts can differ. Paired comparisons exclude delivery failures; the export ledger retains them.',
            'Codebook and external-length metadata costs remain outside language-model PPL.']}


def markdown(report):
    lines = ['# Secondary full-text GPT-2 measurements', '',
        'Descriptive analysis after the completed primary one-opening evaluation; the same GPT-2 also supplied local selector scores.', '',
        '| Arm | Texts | Mean / median token NLL | Mean / median PPL |', '|---|---:|---:|---:|']
    def fmt(x):
        return 'n/a' if x is None else f'{x:.6f}'
    for arm, row in report['available_arm_distributions'].items():
        lines.append(f"| {arm} | {row['mean_nll']['n']} | {fmt(row['mean_nll']['mean'])} / {fmt(row['mean_nll']['median'])} | {fmt(row['ppl']['mean'])} / {fmt(row['ppl']['median'])} |")
    lines += ['', f"Common fully successful pairs: {report['common_success_samples']} across {report['common_success_topics']} topics; planned attempts: {report['planned_attempts']}. Identical selector outputs: {report['identical_selector_output_pairs']}.", '',
              '| Paired difference | Mean | Median |', '|---|---:|---:|']
    for name, row in report['paired_deltas'].items():
        lines.append(f"| {name} | {fmt(row['mean'])} | {fmt(row['median'])} |")
    lines += ['', 'Negative differences indicate lower GPT-2 NLL/PPL. Available-arm distributions can have different cohorts; selector deltas use only common full success. JSON retains every paired value, sample/topic identity, token count and exclusion.', '']
    lines += ['- ' + text for text in report['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--run', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    summary = sub.add_parser('summarize')
    for name in ('run', 'corpus', 'scores', 'output'):
        summary.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        result = export_corpus(args.run, args.output)
        print(json.dumps({key: result[key] for key in ('status', 'planned_attempts', 'common_success_samples', 'n_records', 'n_unique_texts')}))
    else:
        result = summarize(args.run, args.corpus, args.scores)
        if args.output.exists():
            raise FileExistsError('Refusing to overwrite the secondary summary')
        args.output.mkdir(parents=True)
        (args.output / 'summary.json').write_bytes(gate.canonical(result) + b'\n')
        (args.output / 'summary.md').write_text(markdown(result), encoding='utf-8')
        print(json.dumps({key: result[key] for key in ('status', 'planned_attempts', 'common_success_samples', 'paired_deltas')}))


if __name__ == '__main__':
    main()
