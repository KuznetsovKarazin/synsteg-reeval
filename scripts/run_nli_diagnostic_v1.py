#!/usr/bin/env python3
"""Run the frozen full-text NLI diagnostic locally and preserve every outcome.

This scorer is not a receiver-invariant codec gate or a semantic certificate.
No network, generation, model fallback, automatic retry or truncation is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import prepare_nli_diagnostic_v1 as prep


def probabilities(logits):
    """Numerically stable softmax; invalid logits never produce acceptance."""
    if (not isinstance(logits, (list, tuple)) or len(logits) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits)):
        raise ValueError('Exactly three finite numeric logits required')
    highest = max(logits)
    exponents = [math.exp(v - highest) for v in logits]
    total = sum(exponents)
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Invalid logit normalization')
    result = {prep.LABELS[str(i)]: exponents[i] / total for i in range(3)}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in result.values()):
        raise ValueError('Invalid probability output')
    return result


def threshold_pass(value):
    if (not isinstance(value, dict) or set(value) != set(prep.LABELS.values())
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in value.values())):
        return False
    return value['entailment'] >= prep.THRESHOLDS['entailment_min'] and value['contradiction'] <= prep.THRESHOLDS['contradiction_max']


def _result_shell(pair):
    result = {k: pair[k] for k in ('pair_id', 'review_id', 'cohort', 'source_group_id', 'alternative_word',
        'original_word', 'direction', 'is_identity', 'premise_sha256', 'hypothesis_sha256', 'source_text_sha256',
        'pair_tokens', 'tokenization_sha256', 'planned_status')}
    result.update(classifier_call_attempted=False, passes_thresholds=False, logits=None, probabilities=None)
    return result


def score_pair(pair, predictor):
    """Pure controller supports synthetic test predictors; production is fixed ONNX."""
    result = _result_shell(pair)
    if pair['planned_status'] == 'abstain_pair_too_long':
        if pair['pair_tokens'] <= prep.MAX_TOKENS:
            raise ValueError('Overlength abstention requires a genuinely long full pair')
        result['status'] = 'abstain_pair_too_long'
        return result
    if pair['planned_status'] != 'score_once' or not 0 < pair['pair_tokens'] <= prep.MAX_TOKENS:
        raise ValueError('Invalid planned score status/length')
    result['classifier_call_attempted'] = True
    try:
        logits = predictor(pair)
        value = probabilities(logits)
        result.update(status='scored', logits=logits, probabilities=value,
            predicted_label=prep.LABELS[str(max(range(3), key=lambda i: logits[i]))],
            passes_thresholds=threshold_pass(value))
    except Exception as error:
        # Every attempted classifier outcome remains in the full denominator.
        result.update(status='classifier_error', error_type=type(error).__name__)
    return result


def summarize(pairs, results):
    prep.validate_pair_plan(pairs)
    if len(results) != len(pairs) or len({r['pair_id'] for r in results}) != len(pairs):
        raise ValueError('Full unique result denominator required')
    by_id = {r['pair_id']: r for r in results}
    if set(by_id) != {p['pair_id'] for p in pairs}:
        raise ValueError('Unexpected/missing result identities')
    for pair in pairs:
        row = by_id[pair['pair_id']]
        if any(row.get(k) != pair[k] for k in ('review_id', 'alternative_word', 'direction', 'is_identity', 'pair_tokens', 'planned_status', 'premise_sha256', 'hypothesis_sha256', 'tokenization_sha256')):
            raise ValueError('Result provenance differs from frozen pair')
        if row.get('status') not in ('scored', 'abstain_pair_too_long', 'classifier_error', 'model_load_error'):
            raise ValueError('Unknown result status')
        if (pair['planned_status'] == 'abstain_pair_too_long') != (row['status'] == 'abstain_pair_too_long'):
            raise ValueError('Overlength plan/outcome mismatch')
        if row.get('classifier_call_attempted') != (row['status'] in ('scored', 'classifier_error')):
            raise ValueError('Classifier call accounting mismatch')
        if row['status'] == 'scored':
            expected = probabilities(row['logits'])
            if row['probabilities'] != expected or row['passes_thresholds'] != threshold_pass(expected):
                raise ValueError('Scores or threshold result altered')
        elif row.get('passes_thresholds') is not False:
            raise ValueError('Unscored outcome cannot pass')
    groups = {}
    for pair in pairs:
        groups.setdefault(pair['review_id'], []).append(by_id[pair['pair_id']])
    alternatives = []
    for review_id, group in groups.items():
        identities = [r for r in group if r['is_identity']]
        if len(identities) != 1:
            raise ValueError('Every position requires exactly one identity control')
        identity = identities[0]
        words = sorted({r['alternative_word'] for r in group if not r['is_identity']})
        for word in words:
            directed = [r for r in group if not r['is_identity'] and r['alternative_word'] == word]
            if len(directed) != 2 or {r['direction'] for r in directed} != {'original_to_variant', 'variant_to_original'}:
                raise ValueError('Every alternative requires both directions')
            if identity['status'] != 'scored':
                state = 'identity_control_unavailable'
            elif not identity['passes_thresholds']:
                state = 'identity_control_rejected'
            elif any(r['status'] != 'scored' for r in directed):
                state = 'directed_pair_unavailable'
            elif not all(r['passes_thresholds'] for r in directed):
                state = 'directed_pair_rejected'
            else:
                state = 'accepted_by_fixed_nli_criterion'
            alternatives.append({'review_id': review_id, 'alternative_word': word,
                'cohort': identity['cohort'], 'source_group_id': identity['source_group_id'],
                'identity_pair_id': identity['pair_id'], 'directed_pair_ids': [r['pair_id'] for r in directed],
                'status': state, 'accepted': state == 'accepted_by_fixed_nli_criterion'})
    if len(alternatives) != prep.EXPECTED_ALTERNATIVES:
        raise ValueError('All52 nonidentity alternatives required')
    summary = {'schema': prep.SCHEMA + '-summary', 'planned_positions': prep.EXPECTED_POSITIONS,
        'planned_pairs': prep.EXPECTED_PAIRS, 'planned_identity_controls': prep.EXPECTED_POSITIONS,
        'planned_nonidentity_alternatives': prep.EXPECTED_ALTERNATIVES,
        'pair_statuses': dict(Counter(r['status'] for r in results)),
        'identity_statuses': dict(Counter(r['status'] for r in results if r['is_identity'])),
        'identity_threshold_passes': sum(r['is_identity'] and r['passes_thresholds'] for r in results),
        'alternative_statuses': dict(Counter(a['status'] for a in alternatives)),
        'classifier_calls_attempted': sum(r['classifier_call_attempted'] for r in results),
        'classifier_calls_completed': sum(r['status'] == 'scored' for r in results),
        'generation_calls': 0, 'paid_calls': 0, 'closed_test_opened': False,
        'receiver_invariant_codec_gate': False, 'joint_substitution_orbit_tested': False,
        'semantic_safety_certified': False, 'human_validation': False,
        'cohorts_independent': False, 'threshold_tuning_performed': False,
        'alternatives': alternatives}
    return summary


class FixedOnnxPredictor:
    def __init__(self, resources):
        import numpy as np
        import onnxruntime as ort
        ort.disable_telemetry_events()
        self.np = np
        self.tokenizer = prep.load_tokenizer(resources)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.use_deterministic_compute = True
        self.session = ort.InferenceSession(str(Path(resources) / prep.MODEL_FILE),
            sess_options=options, providers=['CPUExecutionProvider'])
        if self.session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('Only the declared CPU provider is allowed')
        self.inputs = self.session.get_inputs()
        names = {i.name for i in self.inputs}
        if (not {'input_ids', 'attention_mask'} <= names or not names <= {'input_ids', 'attention_mask', 'token_type_ids'}
                or any(i.type != 'tensor(int64)' for i in self.inputs)):
            raise ValueError('Unexpected ONNX input names/types')
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise ValueError('Expected exactly one classifier output')
        self.output_name = outputs[0].name
        self.description = {'input_names': sorted(names), 'output_name': self.output_name,
            'input_types': {i.name: i.type for i in self.inputs}, 'providers': self.session.get_providers()}

    def __call__(self, pair):
        encoded = prep.tokenization(self.tokenizer, pair['premise'], pair['hypothesis'])
        if (len(encoded['input_ids']) != pair['pair_tokens'] or len(encoded['input_ids']) > prep.MAX_TOKENS
                or prep.sha(prep.canonical(encoded)) != pair['tokenization_sha256']):
            raise ValueError('Full tokenization differs from frozen preflight or exceeds context')
        feeds = {i.name: self.np.asarray([encoded[i.name]], dtype=self.np.int64) for i in self.inputs}
        outputs = self.session.run([self.output_name], feeds)
        value = self.np.asarray(outputs[0])
        if value.shape != (1, 3) or not self.np.isfinite(value).all():
            raise ValueError('Invalid classifier output shape or values')
        return [float(x) for x in value[0]]


def execute(root, prepared, resources, output, frozen_commit):
    root, prepared, resources, output = map(lambda p: Path(p).resolve(), (root, prepared, resources, output))
    if output.exists():
        raise FileExistsError('Existing diagnostic output must not be overwritten or resumed')
    if not isinstance(frozen_commit, str) or not re.fullmatch(r'[a-f0-9]{40}', frozen_commit):
        raise ValueError('Exact externally verified Git freeze commit required')
    protocol, pairs = prep.verify_preparation(root, prepared, resources)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    execution = {'schema': prep.SCHEMA + '-execution', 'status': 'running', 'frozen_commit': frozen_commit,
        'freeze_commit_validation': 'Operator verifies Git contents before invocation; this offline runner does not contact Git',
        'protocol_sha256': prep.sha((prepared / 'protocol.json').read_bytes()),
        'resources_sha256': prep.sha((prepared / 'resource-manifest.json').read_bytes()),
        'started_utc': started, 'runtime': {'python': sys.version, 'platform': platform.platform(),
            'machine': platform.machine(), 'packages': prep.package_versions(),
            'profile': protocol['runtime_profile']}, 'runner_requested_network_calls': 0,
        'network_isolation_scope': 'Offline APIs and ORT telemetry disabled; no OS-level network sandbox claim',
        'paid_calls': 0, 'generation_calls': 0}
    prep.write_exclusive(output / 'started.json', prep.formatted(execution))
    predictor, load_error = None, None
    if any(p['planned_status'] == 'score_once' for p in pairs):
        try:
            predictor = FixedOnnxPredictor(resources)
            execution['onnx_session'] = predictor.description
        except Exception as error:
            load_error = type(error).__name__
            execution['model_load_error_type'] = load_error
    results = []
    with (output / 'pairs.jsonl').open('xb') as stream:
        for pair in pairs:
            if load_error and pair['planned_status'] == 'score_once':
                # Create the complete provenance record without a model call.
                row = _result_shell(pair)
                row.update(status='model_load_error', error_type=load_error)
            else:
                row = score_pair(pair, predictor)
            results.append(row)
            stream.write(prep.canonical(row) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
    prep.verify_preparation(root, prepared, resources)
    summary = summarize(pairs, results)
    prep.write_exclusive(output / 'summary.json', prep.formatted(summary))
    execution.update({'status': 'completed_with_model_load_error' if load_error else 'completed',
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - clock,
        'classifier_calls_attempted': summary['classifier_calls_attempted'],
        'classifier_calls_completed': summary['classifier_calls_completed'], 'planned_pairs': len(pairs),
        'output_sha256': {name: prep.sha((output / name).read_bytes()) for name in ('pairs.jsonl', 'summary.json', 'started.json')}})
    prep.write_exclusive(output / 'execution.json', prep.formatted(execution))
    return {k: v for k, v in summary.items() if k != 'alternatives'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=prep.ROOT)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--resources', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frozen-commit', required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve(); prepared = args.prepared or root / prep.PREPARED
    result = execute(root, prepared, args.resources or root / prep.RESOURCE_DIR,
        args.output or prepared / 'scores', args.frozen_commit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
