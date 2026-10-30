#!/usr/bin/env python3
"""Independent arithmetic/cohort audit of retained shared-pilot predictions.

The caller must first establish the completed-opening/integrity gate. This
module loads no classifier and does not claim to reproduce model predictions.
It uses only the Python standard library and compares AUC by all positive /
negative pairs, giving ties one half, rather than using the runner's metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

FRAMINGS = ('plain_length16', 'aes_siv')
SELECTORS = ('canonical', 'quality')
FEATURES = ('char', 'word')
COHORTS = ('per_model_success', 'matched_all_models_success')
METRICS = ('accuracy', 'auc', 'fpr_at_development_threshold', 'tpr_at_development_threshold')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value):
    return type(value) in (float, int) and math.isfinite(value)


def prediction_metrics(rows, threshold):
    require(numeric(threshold), 'Nonfinite source development threshold')
    require(bool(rows), 'No prediction rows')
    require(all(type(r['label']) is int and r['label'] in (0, 1)
                and numeric(r['score']) and 0 <= r['score'] <= 1 for r in rows),
            'Invalid prediction label or probability')
    negative = [r['score'] for r in rows if r['label'] == 0]
    positive = [r['score'] for r in rows if r['label'] == 1]
    require(negative and positive, 'Both prediction classes are required')
    return {
        'accuracy': sum((r['score'] >= .5) == r['label'] for r in rows) / len(rows),
        'auc': sum((p > n) + .5 * (p == n) for p in positive for n in negative)
               / (len(positive) * len(negative)),
        'fpr_at_development_threshold': sum(n >= threshold for n in negative) / len(negative),
        'tpr_at_development_threshold': sum(p >= threshold for p in positive) / len(positive),
    }


def audit_predictions(run, config, evaluation, records, lock):
    """Rebuild all cohorts and validate saved predictions; never fit or infer."""
    run = Path(run).resolve()
    aliases = [m['alias'] for m in config['models']]
    sample_ids = [s['id'] for s in config['samples']]
    require(len(aliases) == len(set(aliases)) and len(sample_ids) == len(set(sample_ids)),
            'Duplicate declared model or sample')
    require(config['payload_bits'] == [8, 16] and config['framings'] == list(FRAMINGS)
            and config['selectors'] == list(SELECTORS), 'Unexpected prediction arm contract')
    require(set(records) == set(evaluation['models']) == set(aliases), 'Prediction model denominator changed')
    require(set(lock['attacks']) == {f'fixed_{s}__{f}' for s in SELECTORS for f in FEATURES},
            'Four frozen source attacks are required')
    for alias in aliases:
        require([r['sample']['id'] for r in records[alias]] == sample_ids,
                'Raw model sample order differs from declared panel')
    expected_cohorts = {f'{framing}__{bits}__{cohort}' for framing in FRAMINGS
                        for bits in config['payload_bits'] for cohort in COHORTS}
    for alias in aliases:
        require(set(evaluation['models'][alias]['cohorts']) == expected_cohorts,
                'Missing or extra declared detector cohort')
    summary = {'schema': 'synsteg-shared-dictionary-prediction-audit-v1',
               'status': 'passed', 'classifier_inference_recomputed': False,
               'scope': 'Frozen score-direction/threshold provenance, paired cohort membership, saved score hashes and arithmetic; classifier outputs are not independently rerun.',
               'cohorts_checked': 0, 'complete_attack_cells': 0,
               'empty_attack_cells': 0, 'prediction_rows_checked': 0, 'models': {a: {} for a in aliases}}
    expected_files = set()
    # Same text/classifier appears in a per-model and an intersection cohort.
    # Compare retained probabilities with numerical tolerance for batch effects.
    repeated_scores = {}
    for framing in FRAMINGS:
        for bits in config['payload_bits']:
            arm_ids = [f'{framing}__{selector}__{bits}' for selector in SELECTORS]
            successful = {alias: [r['sample']['id'] for r in records[alias]
                if all(r['variants'][arm]['full_success'] is True for arm in arm_ids)] for alias in aliases}
            common = set(sample_ids)
            for ids in successful.values():
                common.intersection_update(ids)
            for alias in aliases:
                for cohort_kind in COHORTS:
                    cohort_id = f'{framing}__{bits}__{cohort_kind}'
                    selected = successful[alias] if cohort_kind == 'per_model_success' else [sid for sid in sample_ids if sid in common]
                    cohort = evaluation['models'][alias]['cohorts'][cohort_id]
                    require(set(cohort) == {'pairs', 'sample_ids', 'attacks'}, 'Unexpected cohort fields')
                    require(type(cohort['pairs']) is int and cohort['pairs'] == len(selected)
                            and cohort['sample_ids'] == selected, 'Cohort membership or pair denominator changed')
                    attack_names = {f'{framing}__{selector}__{bits}__{feature}' for selector in SELECTORS for feature in FEATURES}
                    require(set(cohort['attacks']) == attack_names, 'Missing or extra cohort attack entry')
                    computed = {'pairs': len(selected), 'sample_ids': selected, 'attacks': {}}
                    summary['cohorts_checked'] += 1
                    for selector in SELECTORS:
                        for feature in FEATURES:
                            name = f'{framing}__{selector}__{bits}__{feature}'
                            saved = cohort['attacks'][name]
                            if not selected:
                                require(saved == {'status': 'no_common_success_pairs'}, 'Empty cohort was assigned detector metrics')
                                computed['attacks'][name] = dict(saved)
                                summary['empty_attack_cells'] += 1
                                continue
                            source_name = f'fixed_{selector}__{feature}'
                            source = lock['attacks'][source_name]
                            require(source['variant'] == f'fixed_{selector}' and source['features'] == feature
                                    and source['score_direction'] in (-1, 1) and source['decision_threshold'] == .5
                                    and numeric(source['threshold']), 'Source attack contract changed')
                            require(set(saved) == {'status', 'source_attack', *METRICS, 'source_development_threshold',
                                                   'source_score_direction', 'predictions_sha256'}, 'Unexpected detector report fields')
                            require(saved['status'] == 'complete' and saved['source_attack'] == source_name
                                    and saved['source_development_threshold'] == source['threshold']
                                    and saved['source_score_direction'] == source['score_direction'], 'Source attack direction or threshold changed')
                            relative = Path('predictions') / alias / cohort_id / (name + '.json')
                            path = (run / relative).resolve()
                            require(path.is_relative_to(run), 'Escaping prediction path')
                            raw = path.read_bytes()
                            require(hashlib.sha256(raw).hexdigest() == saved['predictions_sha256'], 'Prediction file checksum changed')
                            values = json.loads(raw)
                            require(isinstance(values, list) and len(values) == 2 * len(selected), 'Prediction row denominator changed')
                            require(all(set(r) == {'sample_id', 'label', 'score', 'prediction', 'prediction_at_source_dev_threshold'} for r in values),
                                    'Unexpected prediction row fields')
                            require([(r['sample_id'], r['label']) for r in values] == [(sid, label) for sid in selected for label in (0, 1)],
                                    'Prediction pairing, labels or order changed')
                            metrics = prediction_metrics(values, source['threshold'])
                            for row in values:
                                require(type(row['prediction']) is int and row['prediction'] == int(row['score'] >= .5)
                                        and type(row['prediction_at_source_dev_threshold']) is int
                                        and row['prediction_at_source_dev_threshold'] == int(row['score'] >= source['threshold']),
                                        'Saved threshold prediction disagrees with probability')
                                key = (alias, framing, bits, selector, feature, row['sample_id'], row['label'])
                                if key in repeated_scores:
                                    require(math.isclose(repeated_scores[key], row['score'], rel_tol=1e-12, abs_tol=1e-12),
                                            'Repeated cohort has inconsistent probability for the same text and classifier')
                                repeated_scores[key] = row['score']
                            for metric, value in metrics.items():
                                require(numeric(saved[metric]) and math.isclose(saved[metric], value, rel_tol=1e-12, abs_tol=1e-12),
                                        'Detector metric arithmetic changed: ' + metric)
                            expected_files.add(relative.as_posix())
                            summary['complete_attack_cells'] += 1
                            summary['prediction_rows_checked'] += len(values)
                            computed['attacks'][name] = {**metrics, 'source_attack': source_name,
                                'source_development_threshold': source['threshold'], 'source_score_direction': source['score_direction'],
                                'predictions_sha256': saved['predictions_sha256']}
                    summary['models'][alias][cohort_id] = computed
    actual_files = {path.relative_to(run).as_posix() for path in (run / 'predictions').rglob('*.json')}
    require(actual_files == expected_files, 'Missing or extra retained prediction files')
    summary['unique_prediction_files'] = len(expected_files)
    summary['unique_probabilities_checked'] = len(repeated_scores)
    return summary
