#!/usr/bin/env python3
"""Matched generator feasibility pilot with previously frozen v3 attacks.

This is a transfer evaluation, not newly trained within-generator steganalysis.
All generated attempts remain in the reliability denominator. Attack evaluation
starts only after every declared model has a terminal record for every sample.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


base = module('crossmodel_existing_runner', ROOT / 'scripts/prospective_experiment.py')
local = module('crossmodel_existing_adapter', ROOT / 'scripts/prospective_local_adapter.py')
VARIANTS = ['fixed_canonical', 'fixed_quality']
SCHEMA = 'synsteg-crossmodel-transfer-v1'
canonical, digest, file_hash = base.canonical, base.digest, base.file_hash
read_json, atomic_json = base.read_json, base.atomic_json


def event(directory, name, **details):
    directory.mkdir(parents=True, exist_ok=True)
    base.event(directory, name, **details)


def safe_id(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]+', value) is not None


def validate_config(config):
    if config.get('schema') != SCHEMA or config.get('execution_enabled') is not True:
        raise ValueError('Unsupported or disabled transfer protocol')
    if config.get('paid_api_allowed') is not False:
        raise ValueError('Only local no-paid-API execution is supported')
    if config['method_config'].get('variants') != VARIANTS:
        raise ValueError('Exactly the two existing selectors are required')
    models, samples = config.get('models', []), config.get('samples', [])
    if len(models) < 2 or len({x['alias'] for x in models}) != len(models):
        raise ValueError('At least two uniquely named models are required')
    for model in models:
        if not safe_id(model['alias']) or not model.get('model_id'):
            raise ValueError('Invalid model identity')
        for revision in (model['revision'], model.get('tokenizer_revision', model['revision'])):
            if not re.fullmatch(r'[a-f0-9]{40}', revision):
                raise ValueError('Models and tokenizers require immutable revisions')
        if type(model.get('cpu_threads', 3)) is not int or model.get('cpu_threads', 3) < 1:
            raise ValueError('Invalid CPU thread count')
    if not samples or len({x['id'] for x in samples}) != len(samples):
        raise ValueError('Distinct nonempty samples are required')
    for sample in samples:
        if not all(safe_id(sample[k]) for k in ('id', 'topic_id', 'genre_id')):
            raise ValueError('Unsafe sample identity')
        if not isinstance(sample['prompt'], str) or not sample['prompt'].strip():
            raise ValueError('Nonempty prompt required')
        if type(sample['payload_bits']) is not int or sample['payload_bits'] <= 0:
            raise ValueError('Positive payload length required')
        if type(sample['generation_seed']) is not int or not 0 <= sample['generation_seed'] < 2**63:
            raise ValueError('Invalid generation seed')
    if type(config.get('payload_seed')) is not int:
        raise ValueError('Integer payload seed required')
    generation = config['generation']
    if (type(generation.get('max_new_tokens')) is not int or generation['max_new_tokens'] < 1
            or not 0 < generation['temperature'] or not 0 < generation['top_p'] <= 1
            or type(generation.get('top_k', 50)) is not int or generation.get('top_k', 50) < 0):
        raise ValueError('Invalid generation settings')
    if not 0 < config.get('duplicate_policy', {}).get('bigram_jaccard', .9) <= 1:
        raise ValueError('Invalid duplicate threshold')


def code_hashes():
    names = [Path(__file__), ROOT / 'scripts/prospective_experiment.py',
             ROOT / 'scripts/prospective_local_adapter.py',
             ROOT / 'src/synsteg_audit/fixed_codebook.py',
             ROOT / 'src/synsteg_audit/metrics.py', ROOT / 'src/synsteg_audit/__init__.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in names}


def resources(config):
    result = {}
    for item in config.get('external_resource_files', []) + config.get('additional_identity_files', []):
        path = Path(item['path']).resolve()
        actual = file_hash(path)
        if actual != item['sha256']:
            raise ValueError('Resource differs from declared hash: ' + str(path))
        result[str(path)] = actual
    return result


def validate_attack_lock(lock):
    if digest({k: v for k, v in lock.items() if k != 'lock_id'}) != lock.get('lock_id'):
        raise ValueError('Invalid source attack lock identity')
    expected = {v + '__' + feature for v in VARIANTS for feature in ('word', 'char')}
    if set(lock['attacks']) != expected:
        raise ValueError('Exactly four frozen v3 attacks required')
    for name, attack in lock['attacks'].items():
        if (attack['variant'] not in VARIANTS or attack['features'] not in ('word', 'char')
                or name != attack['variant'] + '__' + attack['features']
                or attack['score_direction'] not in (-1, 1)
                or attack.get('decision_threshold') != .5):
            raise ValueError('Unexpected source attack contract')


def prepare(config_path, run):
    config = read_json(config_path)
    validate_config(config)
    if (run / 'protocol.json').exists() or (run / 'freeze.json').exists():
        raise FileExistsError('Refusing to overwrite an existing design')
    source = Path(config['source_run']).resolve()
    lock_path = source / 'attack_lock.json'
    lock = read_json(lock_path)
    validate_attack_lock(lock)
    expected = config.get('source_attack_lock_sha256')
    if expected is not None and file_hash(lock_path) != expected:
        raise ValueError('Source attack lock differs from preregistered identity')
    resource_hashes = resources(config)
    copied = {}
    run.mkdir(parents=True, exist_ok=True)
    for name, attack in sorted(lock['attacks'].items()):
        artifact = (source / attack['artifact']).resolve()
        if not artifact.is_relative_to(source) or file_hash(artifact) != attack['artifact_sha256']:
            raise ValueError('Source attack artifact changed or escaped source directory')
        destination = run / 'source_attacks' / (name + '.joblib')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(artifact.read_bytes())
        copied[str(destination.relative_to(run))] = file_hash(destination)
    (run / 'source_attacks' / 'attack_lock.json').write_bytes(lock_path.read_bytes())
    copied['source_attacks/attack_lock.json'] = file_hash(lock_path)
    atomic_json(run / 'protocol.json', config)
    atomic_json(run / 'samples.json', config['samples'])
    frozen = {'schema': SCHEMA, 'protocol_sha256': file_hash(run / 'protocol.json'),
              'samples_sha256': file_hash(run / 'samples.json'), 'code_sha256': code_hashes(),
              'resource_sha256': resource_hashes, 'environment': base.environment(),
              'source_attack_files_sha256': copied, 'source_lock_id': lock['lock_id'],
              'frozen_at_unix': time.time(),
              'claim_scope': 'cross_generator_transfer_feasibility_not_within_generator_security'}
    frozen['freeze_id'] = digest(frozen)
    atomic_json(run / 'freeze.json', frozen)
    event(run, 'design_and_source_attacks_frozen_before_generation', freeze_id=frozen['freeze_id'])
    return frozen


def verify(run):
    frozen, config = read_json(run / 'freeze.json'), read_json(run / 'protocol.json')
    validate_config(config)
    if digest({k: v for k, v in frozen.items() if k != 'freeze_id'}) != frozen['freeze_id']:
        raise ValueError('Frozen manifest identity changed')
    if (file_hash(run / 'protocol.json') != frozen['protocol_sha256']
            or file_hash(run / 'samples.json') != frozen['samples_sha256']
            or read_json(run / 'samples.json') != config['samples']
            or code_hashes() != frozen['code_sha256']
            or resources(config) != frozen['resource_sha256']
            or base.environment() != frozen['environment']):
        raise ValueError('Frozen design, code, resources, or environment changed')
    for relative, expected in frozen['source_attack_files_sha256'].items():
        if file_hash(run / relative) != expected:
            raise ValueError('Frozen transferred attack changed')
    lock = read_json(run / 'source_attacks/attack_lock.json')
    validate_attack_lock(lock)
    if lock['lock_id'] != frozen['source_lock_id']:
        raise ValueError('Transferred attack lock identity changed')
    return config, frozen, lock


class NativeChatAdapter(local.LocalAdapter):
    """Use native chat rendering without inserting a second BOS token."""

    def generate_cover(self, sample, generation):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = generation['model']
        torch.set_num_threads(int(model.get('cpu_threads', 3)))
        torch.use_deterministic_algorithms(True)
        if self.generator is None:
            self.generator_tokenizer = AutoTokenizer.from_pretrained(
                model.get('tokenizer_id', model['model_id']),
                revision=model.get('tokenizer_revision', model['revision']), local_files_only=True)
            self.generator = AutoModelForCausalLM.from_pretrained(
                model['model_id'], revision=model['revision'], local_files_only=True,
                torch_dtype=torch.bfloat16).eval()
        torch.manual_seed(sample['generation_seed'])
        tokenizer = self.generator_tokenizer
        messages = [{'role': 'system', 'content': 'Write clear English prose. Return only the requested text, without a heading or commentary.'},
                    {'role': 'user', 'content': sample['prompt']}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
        kwargs = {'max_new_tokens': generation['max_new_tokens'], 'do_sample': True,
                  'temperature': generation['temperature'], 'top_p': generation['top_p'],
                  'top_k': generation.get('top_k', 50), 'repetition_penalty': 1.0}
        eos = self.generator.generation_config.eos_token_id
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else (eos[0] if isinstance(eos, list) else eos)
        if pad is None:
            raise ValueError('A declared PAD or EOS token is required')
        kwargs['pad_token_id'] = pad
        started = time.monotonic()
        with torch.inference_mode():
            output = self.generator.generate(**inputs, **kwargs)
        tokens = output[0, inputs['input_ids'].shape[1]:]
        template = tokenizer.get_chat_template()
        effective = self.generator.generation_config.to_dict()
        effective.update(kwargs)
        return {'text': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                'metadata': {'model': model, 'dtype': 'bfloat16', 'device': 'cpu',
                             'seed': sample['generation_seed'], 'generated_tokens': len(tokens),
                             'input_tokens': inputs['input_ids'].shape[1],
                             'hit_token_limit': len(tokens) == generation['max_new_tokens'],
                             'seconds': time.monotonic() - started,
                             'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
                             'rendered_prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                             'input_token_ids_sha256': digest(inputs['input_ids'][0].tolist()),
                             'output_token_ids': tokens.tolist(), 'eos_token_id': eos,
                             'pad_token_id': pad, 'add_special_tokens_after_template': False,
                             'effective_generation_config': effective}}


def payload_for(config, sample):
    rng = random.Random(int(digest([config['payload_seed'], sample['id']]), 16))
    return ''.join(str(rng.randrange(2)) for _ in range(sample['payload_bits']))


def read_records(run, alias, complete=False):
    samples = read_json(run / 'samples.json')
    directory = run / 'models' / alias / 'records'
    expected_paths = {sample['id'] + '.json' for sample in samples}
    if directory.exists() and {p.name for p in directory.glob('*.json')} - expected_paths:
        raise ValueError('Unexpected sample record')
    result = []
    for sample in samples:
        path = directory / (sample['id'] + '.json')
        if not path.exists():
            if complete:
                raise ValueError('Missing terminal record: ' + sample['id'])
            continue
        row = read_json(path)
        if row['sample'] != sample or row['model_alias'] != alias or row['attempt'] != 1:
            raise ValueError('Record identity differs from frozen sample')
        result.append(row)
    return result


def generate(run, alias, adapter=None, limit=None):
    config, frozen, _ = verify(run)
    models = {model['alias']: model for model in config['models']}
    if alias not in models:
        raise ValueError('Undeclared model')
    if (run / 'evaluation_opened.json').exists() or (run / 'evaluation.json').exists():
        raise ValueError('Evaluation has opened; generation is forbidden')
    if (run / 'models' / alias / 'generation.json').exists():
        rows = read_records(run, alias, complete=True)
        report = read_json(run / 'models' / alias / 'generation.json')
        if report['records_sha256'] != digest(rows):
            raise ValueError('Completed records changed')
        return report
    directory = run / 'models' / alias
    adapter = adapter or NativeChatAdapter()
    with base.writer_lock(directory):
        existing = read_records(run, alias)
        prior = [(r['sample']['id'], r['cover_text']) for r in existing
                 if r.get('cover_text') and 'quarantined_duplicate_of:' not in (r.get('failure') or '')]
        added = 0
        for sample in config['samples']:
            path = directory / 'records' / (sample['id'] + '.json')
            if path.exists():
                continue
            if limit is not None and added >= limit:
                break
            marker = directory / 'attempt_started' / (sample['id'] + '.json')
            row = {'sample': sample, 'model_alias': alias, 'freeze_id': frozen['freeze_id'],
                   'attempt': 1, 'status': 'failed', 'failure': None,
                   'cover_text': None, 'variants': {}}
            if marker.exists():
                row.update(failure='interrupted_attempt_not_retried',
                           started_at=read_json(marker)['time_unix'], finished_at=time.time())
                atomic_json(path, row)
                event(directory, 'attempt_interrupted_not_retried', sample_id=sample['id'])
                added += 1
                continue
            row['started_at'] = time.time()
            atomic_json(marker, {'time_unix': row['started_at'], 'freeze_id': frozen['freeze_id']})
            event(directory, 'attempt_started', sample_id=sample['id'])
            try:
                settings = copy.deepcopy(config['generation'])
                settings['model'] = copy.deepcopy(models[alias])
                generation = adapter.generate_cover(copy.deepcopy(sample), settings)
                row['generation_metadata'] = generation.get('metadata', {})
                cover = generation['text']
                if not isinstance(cover, str) or not cover.strip():
                    raise ValueError('Generator returned an empty cover')
                row['cover_text'] = cover
                collision = next((sid for sid, old in prior if base.near_duplicate(
                    cover, old, config.get('duplicate_policy', {}).get('bigram_jaccard', .9))), None)
                if collision is not None:
                    raise ValueError('quarantined_duplicate_of:' + collision)
                prior.append((sample['id'], cover))
                method = copy.deepcopy(config['method_config'])
                prepared = adapter.prepare_methods(cover, copy.deepcopy(sample), method)
                if set(prepared) != set(VARIANTS):
                    raise ValueError('Prepared selectors differ from frozen design')
                before = {name: digest(value) for name, value in prepared.items()}
                row['prepared_methods'], row['prepared_method_sha256'] = copy.deepcopy(prepared), before
                event(directory, 'methods_prepared_before_payload', sample_id=sample['id'], sha256=before)
                payload = payload_for(config, sample)
                row['synthetic_payload'] = payload
                row['payload_sha256'] = hashlib.sha256(payload.encode()).hexdigest()
                for variant in VARIANTS:
                    try:
                        state = copy.deepcopy(prepared[variant])
                        output = adapter.encode_decode(variant, state, payload, cover, copy.deepcopy(sample), method)
                        if digest(state) != before[variant]:
                            raise ValueError('Prepared state changed after payload')
                        metrics = base.delivery_metrics(payload, output['decoded_payload'],
                            used_bits=output['embedded_bits'], cover=cover, stego=output['stego_text'], key=output.get('key'))
                        row['variants'][variant] = {'stego_text': output['stego_text'], 'metrics': metrics,
                            'decoded_payload': output['decoded_payload'], 'key': output.get('key'),
                            'metadata': output.get('metadata', {}), 'error': None,
                            'prepared_state_sha256': before[variant]}
                    except Exception as error:
                        row['variants'][variant] = {'stego_text': None, 'metrics': None,
                            'error': f'{type(error).__name__}: {error}', 'prepared_state_sha256': before[variant]}
                row['status'] = 'completed'
            except Exception as error:
                row['failure'] = f'{type(error).__name__}: {error}'
            row['finished_at'] = time.time()
            atomic_json(path, row)
            event(directory, 'attempt_completed', sample_id=sample['id'], status=row['status'], failure=row['failure'])
            added += 1
            print(json.dumps({'model': alias, 'sample': sample['id'], 'terminal': True, 'status': row['status']}), flush=True)
        rows = read_records(run, alias)
        report = {'model_alias': alias, 'freeze_id': frozen['freeze_id'], 'records': len(rows),
                  'expected': len(config['samples']), 'complete': len(rows) == len(config['samples']),
                  'records_sha256': digest(rows), 'new_attempts': added}
        if report['complete']:
            atomic_json(directory / 'generation.json', report)
            event(directory, 'generation_completed', records_sha256=report['records_sha256'])
        return report


def evaluate(run):
    import joblib
    config, frozen, lock = verify(run)
    if (run / 'evaluation.json').exists():
        raise ValueError('Evaluation already completed once; do not reopen')
    with base.writer_lock(run / 'evaluation_control'):
        if (run / 'evaluation.json').exists():
            raise ValueError('Evaluation already completed once; do not reopen')
        generations = {}
        for model in config['models']:
            alias = model['alias']
            path = run / 'models' / alias / 'generation.json'
            if not path.exists():
                raise ValueError('All models must finish before evaluation opens')
            status = read_json(path)
            if not status.get('complete') or status.get('records') != len(config['samples']):
                raise ValueError('All declared attempts must be terminal')
            if (run / 'models' / alias / '.writer.lock').exists():
                raise ValueError('A model writer is still active')
            generations[alias] = status['records_sha256']
        marker = run / 'evaluation_opened.json'
        identity = {'freeze_id': frozen['freeze_id'], 'source_lock_id': lock['lock_id'],
                    'all_model_records_sha256': generations}
        if marker.exists():
            previous = read_json(marker)
            if any(previous.get(k) != v for k, v in identity.items()):
                raise ValueError('Cannot resume evaluation after any identity change')
            event(run, 'transfer_evaluation_resumed', **identity)
        else:
            atomic_json(marker, {**identity, 'opened_at_unix': time.time()})
            event(run, 'transfer_evaluation_opened_once', **identity)
        all_records = {m['alias']: read_records(run, m['alias'], complete=True) for m in config['models']}
        for alias, rows in all_records.items():
            if digest(rows) != generations[alias] or any(r['freeze_id'] != frozen['freeze_id'] for r in rows):
                raise ValueError('Records changed since generation completion')
        cohorts = {alias: base.common_success(rows, VARIANTS) for alias, rows in all_records.items()}
        common_ids = set.intersection(*({r['sample']['id'] for r in rows} for rows in cohorts.values()))
        report = {**identity, 'schema': SCHEMA, 'status': 'complete_frozen_attack_transfer_evaluation',
                  'claim_scope': 'cross_generator_transfer_feasibility_not_within_generator_security',
                  'matched_success_sample_ids': sorted(common_ids), 'models': {},
                  'limitations': [
                      'Attacks were trained on old Qwen v3 data and transferred unchanged; this is not within-generator security.',
                      'No training, threshold selection, score-direction change or best-model selection uses these outputs.',
                      'All planned attempts remain in reliability; steganalysis is conditional on complete delivery.',
                      'Per-model success cohorts may differ; a matched intersection is also reported.',
                      'Old development thresholds do not guarantee 5% false positives on these generators.',
                      'The receiver still requires the complete document-specific codebook and 32 external length bits; this is not a seed-only decoder.',
                      'A small matched pilot and procedural one-opening gate do not establish broad security or semantic quality.']}
        classifiers = {name: joblib.load(run / 'source_attacks' / (name + '.joblib')) for name in lock['attacks']}
        for alias, records in all_records.items():
            payloads = sorted({s['payload_bits'] for s in config['samples']})
            item = {'reliability': base.reliability(records, VARIANTS),
                    'reliability_by_payload_bits': {str(n): base.reliability(
                        [r for r in records if r['sample']['payload_bits'] == n], VARIANTS) for n in payloads},
                    'planned_attempts': len(records), 'generated_covers': sum(bool(r.get('cover_text')) for r in records),
                    'failures': [{'sample_id': r['sample']['id'], 'failure': r['failure'],
                                  'variant_errors': {v: r.get('variants', {}).get(v, {}).get('error') for v in VARIANTS}}
                                 for r in records if r.get('failure') or any(r.get('variants', {}).get(v, {}).get('error') for v in VARIANTS)],
                    'cohorts': {}}
            for cohort_name, rows in [('per_model_common_success', cohorts[alias]),
                                     ('matched_all_models_success', [r for r in cohorts[alias] if r['sample']['id'] in common_ids])]:
                result = {'pairs': len(rows), 'sample_ids': [r['sample']['id'] for r in rows], 'attacks': {}}
                for name, attack in lock['attacks'].items():
                    if not rows:
                        result['attacks'][name] = {'status': 'no_common_success_pairs'}
                        continue
                    texts, labels, identities = base.paired_rows(rows, attack['variant'])
                    scores = classifiers[name].predict_proba(texts)[:, 1]
                    if attack['score_direction'] == -1:
                        scores = 1 - scores
                    predictions = [{**identity, 'label': int(y), 'score': float(p),
                        'prediction': int(p >= .5), 'prediction_at_source_dev_threshold': int(p >= attack['threshold'])}
                        for identity, y, p in zip(identities, labels, scores)]
                    prediction_path = run / 'predictions' / alias / cohort_name / (name + '.json')
                    atomic_json(prediction_path, predictions)
                    result['attacks'][name] = {'status': 'complete',
                        **base.metrics_from_scores(labels, scores, attack['threshold']),
                        'source_development_threshold': attack['threshold'], 'source_score_direction': attack['score_direction'],
                        'predictions_sha256': file_hash(prediction_path)}
                item['cohorts'][cohort_name] = result
            report['models'][alias] = item
        atomic_json(run / 'evaluation.json', report)
        event(run, 'transfer_evaluation_completed', evaluation_sha256=file_hash(run / 'evaluation.json'))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'generate', 'evaluate', 'verify'))
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--model')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.command == 'prepare':
        if args.config is None:
            parser.error('prepare requires --config')
        with base.writer_lock(run):
            result = prepare(args.config, run)
    elif args.command == 'generate':
        if args.model is None:
            parser.error('generate requires --model')
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be positive')
        result = generate(run, args.model, limit=args.limit)
    elif args.command == 'evaluate':
        result = evaluate(run)
    else:
        _, frozen, _ = verify(run)
        result = {'verified': True, 'freeze_id': frozen['freeze_id']}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
