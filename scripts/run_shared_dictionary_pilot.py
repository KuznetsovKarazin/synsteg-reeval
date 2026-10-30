#!/usr/bin/env python3
"""Registered common-dictionary feasibility, with isolated receivers.

The public dictionary and protocol replace document-specific shared state.
Plain framing is an unauthenticated transport diagnostic; AES-SIV is a separate
authenticated arm. Every complete-frame capacity failure stays in the denominator.
No experiment here establishes semantic equivalence or steganographic secrecy.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synsteg_audit import shared_dictionary as codec


def receiver_main():
    """Fresh-process receiver; installed public code/resources only, then no I/O."""
    import socket
    from cryptography.hazmat.primitives.ciphers.aead import AESSIV
    # Complete lazy public-library setup before the inspection guard is installed.
    warm = AESSIV(bytes(64))
    warm.decrypt(warm.encrypt(b'', [b'public-warmup']), [b'public-warmup'])

    def guard(event, args):
        if (event in {'open', 'os.listdir', 'os.scandir', 'os.system', 'os.posix_spawn'}
                or event.startswith(('socket.', 'subprocess.'))):
            raise PermissionError('receiver external I/O forbidden: ' + event)
    sys.addaudithook(guard)
    checks = {}
    for name, operation in [('file', lambda: open('/receiver-forbidden')),
                            ('socket', socket.socket),
                            ('subprocess', lambda: subprocess.Popen(['forbidden']))]:
        try:
            operation()
        except PermissionError:
            checks[name] = True
        else:
            raise RuntimeError('Receiver I/O guard self-check failed')
    request = json.loads(sys.stdin.buffer.read())
    expected = {'key_hex', 'stego_text', 'public_dictionary', 'public_protocol'}
    if set(request) != expected:
        raise ValueError('Receiver request contains missing or additional state')
    dictionary = codec.PublicDictionary.from_dict(request['public_dictionary'])
    p = request['public_protocol']
    protocol = codec.Protocol(min_group_size=p['min_group_size'],
                              max_bits_per_slot=p['max_bits_per_slot'], framing=p['framing'])
    if p != protocol.to_dict():
        raise ValueError('Receiver protocol version or fields differ from installed public protocol')
    result = {'input_keys': sorted(request), 'io_guard_self_checks': checks,
              'isolation_scope': 'inspected-program Python audit guard, not an OS sandbox',
              'public_dictionary_sha256': dictionary.sha256, 'public_protocol_sha256': protocol.sha256}
    try:
        result.update(status='decoded', payload_hex=codec.decode(
            request['stego_text'], bytes.fromhex(request['key_hex']), dictionary, protocol).hex())
        code = 0
    except codec.CodecError as error:
        result.update(status='rejected', error_type=type(error).__name__, error=str(error))
        code = 3
    print(json.dumps(result, sort_keys=True))
    return code


# The receiver does not import any experiment generator, saved-cover, or scorer API.
if __name__ == '__main__' and len(sys.argv) == 2 and sys.argv[1] == 'receiver':
    raise SystemExit(receiver_main())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


transfer = module('shared_pilot_transfer', ROOT / 'scripts/crossmodel_transfer_pilot.py')
neural = module('shared_pilot_neural', ROOT / 'scripts/rescore_models.py')
quality_checks = module('shared_pilot_quality_checks', ROOT / 'scripts/crossmodel_quality.py')
base = transfer.base
canonical, digest, read_json, atomic_json = base.canonical, base.digest, base.read_json, base.atomic_json
SCHEMA = 'synsteg-shared-dictionary-pilot-v1'
FRAMINGS = ('plain_length16', 'aes_siv')
SELECTORS = ('canonical', 'quality')
JOBS = {'gpt2': ROOT / 'configs/neural/gpt2-full.json',
        'qwen_base': ROOT / 'configs/neural/crossmodel-qwen-base-ppl.json'}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(directory, name, **details):
    directory.mkdir(parents=True, exist_ok=True)
    base.event(directory, name, **details)


def public_protocol(config, framing, *, min_group_size=None):
    method = config['shared_method']
    return codec.Protocol(min_group_size=method['min_group_size'] if min_group_size is None else min_group_size,
                          max_bits_per_slot=method['max_bits_per_slot'], framing=framing)


def arm_id(framing, selector, bits):
    return f'{framing}__{selector}__{bits}'


def arms(config):
    result = [{'id': arm_id(framing, selector, bits), 'framing': framing,
               'selector': selector, 'payload_bits': bits, 'method': 'shared_dictionary'}
              for bits in config['payload_bits'] for framing in FRAMINGS for selector in SELECTORS]
    if config['include_fixed_quality_baseline']:
        result += [{'id': f'fixed_quality__{bits}', 'framing': 'external_length',
                    'selector': 'quality', 'payload_bits': bits, 'method': 'fixed_codebook'}
                   for bits in config['payload_bits']]
    return result


def validate_config(config):
    if config.get('schema') != SCHEMA or config.get('execution_enabled') is not True:
        raise ValueError('Unsupported or disabled shared-dictionary protocol')
    if config.get('paid_api_allowed') is not False:
        raise ValueError('This registered runner supports only local model inference')
    if config.get('framings') != list(FRAMINGS) or config.get('selectors') != list(SELECTORS):
        raise ValueError('Both declared framing modes and selectors are required')
    if config.get('payload_bits') != [8, 16]:
        raise ValueError('This pilot pairs exactly 8 and 16 bits on the same cover')
    if type(config.get('include_fixed_quality_baseline')) is not bool:
        raise ValueError('Explicit baseline inclusion required')
    if config.get('bundle_policy') != 'all-declared-covers-in-order-double-newline-aes-siv-canonical-v1':
        raise ValueError('Exactly one fixed all-cover authenticated canonical bundle per generator is required')
    if (config['shared_method'].get('min_group_size') != 1
            or config['shared_method'].get('max_bits_per_slot') != 1):
        raise ValueError('Main protocol is one bit per eligible dictionary word')
    if config['shared_method'].get('quality_cost') != 'gpt2-original-cover-single-replacement-full-text-mean-nll-v1':
        raise ValueError('Unexpected sender quality rule')
    models, samples = config.get('models', []), config.get('samples', [])
    if len(models) < 2 or len({m['alias'] for m in models}) != len(models):
        raise ValueError('Distinct matched generators required')
    for model in models:
        if not transfer.safe_id(model['alias']) or not model.get('model_id'):
            raise ValueError('Invalid generator identity')
        for revision in (model['revision'], model.get('tokenizer_revision', model['revision'])):
            if not re.fullmatch('[a-f0-9]{40}', revision):
                raise ValueError('Immutable model/tokenizer revisions required')
    if not samples or len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Distinct nonempty sample panel required')
    for sample in samples:
        if not all(transfer.safe_id(sample[k]) for k in ('id', 'topic_id', 'genre_id')):
            raise ValueError('Unsafe sample identity')
        if not isinstance(sample.get('prompt'), str) or not sample['prompt'].strip():
            raise ValueError('Nonempty prompt required')
        if type(sample.get('generation_seed')) is not int or not 0 <= sample['generation_seed'] < 2**63:
            raise ValueError('Invalid generator seed')
        if 'payload_bits' in sample:
            raise ValueError('Payload sizes are paired on each cover, not separate generations')
    if type(config.get('payload_seed')) is not int or not isinstance(config.get('public_demo_key_seed'), str):
        raise ValueError('Explicit public fixture seeds required')
    g = config['generation']
    if (type(g.get('max_new_tokens')) is not int or g['max_new_tokens'] < 1
            or not 0 < g.get('temperature', 0) or not 0 < g.get('top_p', 0) <= 1
            or type(g.get('top_k')) is not int or g['top_k'] < 0):
        raise ValueError('Invalid generation contract')
    if not 0 < config.get('duplicate_policy', {}).get('bigram_jaccard', .9) <= 1:
        raise ValueError('Invalid duplicate policy')


def code_hashes():
    paths = set(transfer.code_hashes()) | {
        'scripts/run_shared_dictionary_pilot.py', 'scripts/crossmodel_transfer_pilot.py',
        'scripts/rescore_models.py', 'scripts/crossmodel_quality.py',
        'scripts/summarize_prospective_results.py', 'src/synsteg_audit/shared_dictionary.py'}
    return {name: file_hash(ROOT / name) for name in sorted(paths)}


def environment():
    value = base.environment()
    value['packages']['cryptography'] = importlib.metadata.version('cryptography')
    value['unicode_database_version'] = unicodedata.unidata_version
    return value


def fixture_key(config):
    return hashlib.sha256(('PUBLIC-NONPRODUCTION-SHARED-KEY:' + config['public_demo_key_seed']).encode()).digest()


def payload_for(config, sample, bits):
    return hashlib.sha256(canonical([config['payload_seed'], sample['id']])).digest()[:bits // 8]


def prepare(config_path, run):
    config = read_json(config_path)
    validate_config(config)
    if (run / 'freeze.json').exists() or (run / 'protocol.json').exists():
        raise FileExistsError('Refusing to overwrite a prior design')
    dictionary_path = (ROOT / config['dictionary_file']).resolve()
    if not dictionary_path.is_relative_to(ROOT):
        raise ValueError('Dictionary must be an explicit repository public artifact')
    dictionary = codec.PublicDictionary.from_dict(read_json(dictionary_path))
    source = Path(config['source_run']).resolve()
    lock_path = source / 'attack_lock.json'
    lock = read_json(lock_path)
    transfer.validate_attack_lock(lock)
    if file_hash(lock_path) != config['source_attack_lock_sha256']:
        raise ValueError('Source attacks differ from declared identity')
    resource_hashes = transfer.resources(config)
    run.mkdir(parents=True, exist_ok=True)
    (run / 'public_dictionary.json').write_bytes(dictionary_path.read_bytes())
    copied = {}
    for name, attack in sorted(lock['attacks'].items()):
        original = (source / attack['artifact']).resolve()
        if not original.is_relative_to(source) or file_hash(original) != attack['artifact_sha256']:
            raise ValueError('Source attack artifact changed or escaped source directory')
        target = run / 'source_attacks' / (name + '.joblib')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        copied[str(target.relative_to(run))] = file_hash(target)
    target = run / 'source_attacks/attack_lock.json'
    target.write_bytes(lock_path.read_bytes())
    copied[str(target.relative_to(run))] = file_hash(target)
    atomic_json(run / 'protocol.json', config)
    atomic_json(run / 'samples.json', config['samples'])
    frozen = {'schema': SCHEMA, 'protocol_sha256': file_hash(run / 'protocol.json'),
              'samples_sha256': file_hash(run / 'samples.json'), 'code_sha256': code_hashes(),
              'resource_sha256': resource_hashes, 'environment': environment(),
              'dictionary_file_sha256': file_hash(run / 'public_dictionary.json'),
              'dictionary_sha256': dictionary.sha256,
              'source_attack_files_sha256': copied, 'source_lock_id': lock['lock_id'],
              'scorer_job_files_sha256': {name: file_hash(path) for name, path in JOBS.items()},
              'frozen_at_unix': time.time(), 'fixture_key_is_public_not_secret': True}
    frozen['freeze_id'] = digest(frozen)
    atomic_json(run / 'freeze.json', frozen)
    event(run, 'shared_design_dictionary_and_attacks_frozen_before_generation', freeze_id=frozen['freeze_id'])
    return frozen


def verify(run):
    frozen, config = read_json(run / 'freeze.json'), read_json(run / 'protocol.json')
    validate_config(config)
    if digest({k: v for k, v in frozen.items() if k != 'freeze_id'}) != frozen['freeze_id']:
        raise ValueError('Frozen manifest identity changed')
    if (file_hash(run / 'protocol.json') != frozen['protocol_sha256']
            or file_hash(run / 'samples.json') != frozen['samples_sha256']
            or read_json(run / 'samples.json') != config['samples']
            or code_hashes() != frozen['code_sha256'] or environment() != frozen['environment']
            or transfer.resources(config) != frozen['resource_sha256']
            or {name: file_hash(path) for name, path in JOBS.items()} != frozen['scorer_job_files_sha256']
            or file_hash(run / 'public_dictionary.json') != frozen['dictionary_file_sha256']):
        raise ValueError('Frozen design, code, resources, dictionary, or environment changed')
    dictionary = codec.PublicDictionary.from_dict(read_json(run / 'public_dictionary.json'))
    if dictionary.sha256 != frozen['dictionary_sha256']:
        raise ValueError('Canonical public dictionary identity changed')
    for relative, expected in frozen['source_attack_files_sha256'].items():
        if file_hash(run / relative) != expected:
            raise ValueError('Frozen transferred attack changed')
    lock = read_json(run / 'source_attacks/attack_lock.json')
    transfer.validate_attack_lock(lock)
    if lock['lock_id'] != frozen['source_lock_id']:
        raise ValueError('Frozen source attack identity changed')
    return config, frozen, dictionary, lock


def isolated_decode(stego_text, key, dictionary, protocol):
    request = {'key_hex': key.hex(), 'stego_text': stego_text,
               'public_dictionary': dictionary.to_dict(), 'public_protocol': protocol.to_dict()}
    # New process and empty cwd. Only explicit installed public code is importable.
    with tempfile.TemporaryDirectory(prefix='synsteg-shared-receiver-') as directory:
        result = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), 'receiver'],
            input=canonical(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=directory, env={'PATH': os.defpath, 'PYTHONIOENCODING': 'utf-8'}, timeout=30)
    response = json.loads(result.stdout)
    response['actual_returncode'] = result.returncode
    response['stderr'] = result.stderr.decode('utf-8', errors='replace')
    if result.returncode != 0 or response.get('status') != 'decoded':
        raise ValueError('Isolated receiver rejected: ' + canonical(response).decode())
    return bytes.fromhex(response['payload_hex']), response


class SharedAdapter(transfer.NativeChatAdapter):
    def __init__(self):
        super().__init__()
        self.full_scorer = None

    def prepare_shared(self, cover, dictionary, protocol):
        import torch
        analysis = codec.analyze_text(cover, dictionary, protocol)
        classes = {entry.class_id: entry.words for entry in dictionary.classes}
        job = neural.validate_job(read_json(JOBS['gpt2']))
        torch.set_num_threads(job['cpu_threads'])
        if self.full_scorer is None:
            self.full_scorer = neural.Scorer(job, 'cpu', allow_download=False)
            # Share this exact pinned FP32 model with the unchanged baseline adapter.
            self.scorer, self.scorer_tokenizer = self.full_scorer.model, self.full_scorer.tokenizer
        costs, evidence, cache = {}, [], {}
        for slot in analysis.slots:
            words = classes[slot.class_id]
            for word in words:
                rendered = word.upper() if slot.case_style == 'upper' else word.capitalize() if slot.case_style == 'title' else word
                identity = str(slot.index) + ':' + rendered
                if len(words) == 2:
                    costs[identity] = 0.0
                    evidence.append({'slot': slot.index, 'candidate': rendered, 'cost': 0.0,
                                     'kind': 'singleton_groups_no_quality_choice_not_a_model_score'})
                    continue
                changed = cover[:slot.start] + rendered + cover[slot.end:]
                sha = neural.text_hash(changed)
                if sha not in cache:
                    ppl, details = self.full_scorer.score(changed)
                    quality_checks.validate_full_text_score(ppl, details)
                    cache[sha] = {'ppl': ppl, 'details': details,
                                 'forward_passes': len(list(neural.windows(details['scored_input_tokens'],
                                                        job['context_length'], job['stride'])))}
                score = cache[sha]
                costs[identity] = score['details']['mean_nll']
                evidence.append({'slot': slot.index, 'candidate': rendered, 'cost': costs[identity],
                                 'kind': 'original_cover_single_replacement_mean_nll', 'text_sha256': sha,
                                 'score': score})
        return {'analysis': asdict(analysis), 'candidate_costs': costs, 'candidate_score_evidence': evidence,
                'unique_full_text_scoring_calls': len(cache),
                'model_forward_passes': sum(value['forward_passes'] for value in cache.values()),
                'score_policy': 'Frozen full-text GPT2 mean NLL on original-cover one-word alternatives; no contextual admission filter.'}

    def prepare_baseline(self, cover, sample, config):
        return self.prepare_methods(cover, sample, config['baseline_method_config'])['fixed_quality']


def read_records(run, alias, complete=False):
    config = read_json(run / 'protocol.json')
    directory = run / 'models' / alias / 'records'
    expected = {sample['id'] + '.json' for sample in config['samples']}
    if directory.exists() and {p.name for p in directory.glob('*.json')} - expected:
        raise ValueError('Unexpected sample record')
    result = []
    for sample in config['samples']:
        path = directory / (sample['id'] + '.json')
        if not path.exists():
            if complete:
                raise ValueError('Missing terminal sample record')
            continue
        row = read_json(path)
        if row['sample'] != sample or row['model_alias'] != alias or row['attempt'] != 1:
            raise ValueError('Record differs from frozen sample identity')
        if set(row['variants']) != {arm['id'] for arm in arms(config)}:
            raise ValueError('Incomplete arm ledger')
        result.append(row)
    return result


def generate(run, alias, adapter=None, limit=None):
    config, frozen, dictionary, _ = verify(run)
    models = {model['alias']: model for model in config['models']}
    if alias not in models:
        raise ValueError('Undeclared generator')
    if (run / 'evaluation_opened.json').exists() or (run / 'evaluation.json').exists():
        raise ValueError('Evaluation has opened; generation is forbidden')
    directory = run / 'models' / alias
    if (directory / 'generation.json').exists():
        report = read_json(directory / 'generation.json')
        bundle = read_json(directory / 'bundle.json')
        if (report['records_sha256'] != digest(read_records(run, alias, complete=True))
                or report['bundle_sha256'] != digest(bundle)
                or bundle['source_records_sha256'] != report['records_sha256']):
            raise ValueError('Completed records changed')
        return report
    adapter = adapter or SharedAdapter()
    key = fixture_key(config)
    with base.writer_lock(directory):
        previous = read_records(run, alias)
        prior = [(row['sample']['id'], row['cover_text']) for row in previous
                 if isinstance(row.get('cover_text'), str) and row['cover_text'].strip()
                 and 'quarantined_duplicate_of:' not in (row.get('failure') or '')]
        added = 0
        for sample in config['samples']:
            path = directory / 'records' / (sample['id'] + '.json')
            if path.exists():
                continue
            if limit is not None and added >= limit:
                break
            marker = directory / 'attempt_started' / (sample['id'] + '.json')
            row = {'sample': sample, 'model_alias': alias, 'freeze_id': frozen['freeze_id'], 'attempt': 1,
                   'status': 'failed', 'failure': None, 'cover_text': None,
                   'variants': {arm['id']: {**arm, 'full_success': False, 'stego_text': None,
                       'error': 'cover_or_preparation_unavailable'} for arm in arms(config)}}
            if marker.exists():
                row.update(failure='interrupted_attempt_not_retried', started_at=read_json(marker)['time_unix'],
                           finished_at=time.time(), public_fixture_key_hex=key.hex(),
                           payloads_hex={str(bits): payload_for(config, sample, bits).hex()
                                         for bits in config['payload_bits']})
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
                generated = adapter.generate_cover(copy.deepcopy(sample), settings)
                row['generation_metadata'] = generated.get('metadata', {})
                cover = generated['text']
                row['cover_text'] = cover if isinstance(cover, str) else None
                if not isinstance(cover, str):
                    row['unusable_generated_value'] = cover
                if not isinstance(cover, str) or not cover.strip():
                    raise ValueError('Empty generated cover')
                collision = next((sid for sid, old in prior if base.near_duplicate(cover, old,
                    config.get('duplicate_policy', {}).get('bigram_jaccard', .9))), None)
                if collision is not None:
                    raise ValueError('quarantined_duplicate_of:' + collision)
                prior.append((sample['id'], cover))
                protocol = public_protocol(config, 'plain_length16')
                preparation_started = time.monotonic()
                prepared = adapter.prepare_shared(cover, dictionary, protocol)
                row['shared_preparation_seconds'] = time.monotonic() - preparation_started
                row['prepared_shared'], row['prepared_shared_sha256'] = prepared, digest(prepared)
                row['capacity_ablations'] = {str(minimum): {
                    'capacity_bits': codec.analyze_text(cover, dictionary,
                        public_protocol(config, 'plain_length16', min_group_size=minimum)).capacity_bits,
                    'min_group_size': minimum, 'max_bits_per_slot': config['shared_method']['max_bits_per_slot']}
                    for minimum in (1, 2)}
                baseline, baseline_error = None, None
                if config['include_fixed_quality_baseline']:
                    try:
                        baseline = adapter.prepare_baseline(cover, copy.deepcopy(sample), copy.deepcopy(config))
                        row['prepared_baseline'], row['prepared_baseline_sha256'] = baseline, digest(baseline)
                    except Exception as error:
                        baseline_error = f'{type(error).__name__}: {error}'
                event(directory, 'methods_prepared_before_payload', sample_id=sample['id'],
                      shared_sha256=row['prepared_shared_sha256'], baseline_sha256=row.get('prepared_baseline_sha256'))
                row['public_fixture_key_hex'] = key.hex()
                row['payloads_hex'] = {str(bits): payload_for(config, sample, bits).hex() for bits in config['payload_bits']}
                for arm in arms(config):
                    output = row['variants'][arm['id']]
                    arm_started = time.monotonic()
                    payload = bytes.fromhex(row['payloads_hex'][str(arm['payload_bits'])])
                    try:
                        if arm['method'] == 'shared_dictionary':
                            p = public_protocol(config, arm['framing'])
                            costs = prepared['candidate_costs']
                            cost = (lambda slot, candidate: costs[str(slot.index) + ':' + candidate]) if arm['selector'] == 'quality' else None
                            encoded = codec.encode(cover, payload, key, dictionary, p,
                                                   selector=arm['selector'], candidate_cost=cost)
                            decoded, receiver = isolated_decode(encoded.stego_text, key, dictionary, p)
                            output.update(stego_text=encoded.stego_text, decoded_payload_hex=decoded.hex(),
                                receiver=receiver, encoding={k: v for k, v in asdict(encoded).items() if k != 'stego_text'},
                                external_document_metadata_bits=0, individual_codebook_bytes=0,
                                shared_secret_key_bytes=32, public_dictionary_file_bytes=(run / 'public_dictionary.json').stat().st_size,
                                public_dictionary_canonical_wire_bytes=len(canonical(dictionary.to_dict())),
                                public_dictionary_sha256=dictionary.sha256, protocol=p.to_dict())
                        else:
                            if baseline is None:
                                raise ValueError(baseline_error or 'Baseline preparation unavailable')
                            state = copy.deepcopy(baseline)
                            payload_bits = ''.join(f'{byte:08b}' for byte in payload)
                            actual = adapter.encode_decode('fixed_quality', state, payload_bits, cover,
                                copy.deepcopy(sample), copy.deepcopy(config['baseline_method_config']))
                            if digest(state) != row['prepared_baseline_sha256']:
                                raise ValueError('Baseline prepared codebook changed')
                            decoded = bytes(int(actual['decoded_payload'][i:i+8], 2)
                                            for i in range(0, len(actual['decoded_payload']), 8))
                            output.update(stego_text=actual['stego_text'], decoded_payload_hex=decoded.hex(),
                                metadata=actual['metadata'], key=actual['key'],
                                delivery_metrics=base.delivery_metrics(payload_bits, actual['decoded_payload'],
                                    used_bits=actual['embedded_bits'], cover=cover, stego=actual['stego_text'], key=actual['key']),
                                external_document_metadata_bits=32)
                        output.update(full_success=decoded == payload, error=None if decoded == payload else 'full_payload_mismatch')
                        if digest(prepared) != row['prepared_shared_sha256']:
                            raise ValueError('Shared sender preparation changed after payload')
                    except Exception as error:
                        output.update(full_success=False, stego_text=None, error=f'{type(error).__name__}: {error}')
                    output['encode_and_decode_seconds'] = time.monotonic() - arm_started
                row['status'] = 'completed'
            except Exception as error:
                row['failure'] = f'{type(error).__name__}: {error}'
            # Failures retain the same public fixtures; no failed attempt reaches an encoder.
            row.setdefault('public_fixture_key_hex', key.hex())
            row.setdefault('payloads_hex', {str(bits): payload_for(config, sample, bits).hex()
                                           for bits in config['payload_bits']})
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
            bundle = generate_bundle(run, alias, rows, config, frozen, dictionary)
            report['bundle_sha256'] = digest(bundle)
            atomic_json(directory / 'generation.json', report)
            event(directory, 'generation_completed', records_sha256=report['records_sha256'])
        return report


def generate_bundle(run, alias, rows, config, frozen, dictionary):
    """One predetermined concatenation, never a selected subset or an extra cover."""
    path = run / 'models' / alias / 'bundle.json'
    if path.exists():
        return read_json(path)
    key, protocol = fixture_key(config), public_protocol(config, 'aes_siv')
    ready = len(rows) == len(config['samples']) and all(isinstance(row.get('cover_text'), str)
            and row['cover_text'].strip() and
            'quarantined_duplicate_of:' not in (row.get('failure') or '') for row in rows)
    cover = '\n\n'.join(row['cover_text'] for row in rows) if ready else None
    sample = {'id': 'all_covers_bundle'}
    bundle = {'model_alias': alias, 'freeze_id': frozen['freeze_id'], 'sample_ids': [r['sample']['id'] for r in rows],
              'source_records_sha256': digest(rows), 'policy': config['bundle_policy'],
              'independent_container_count': 1, 'cover_text': cover, 'public_fixture_key_hex': key.hex(),
              'protocol': protocol.to_dict(), 'public_dictionary_sha256': dictionary.sha256,
              'public_dictionary_file_bytes': (run / 'public_dictionary.json').stat().st_size,
              'public_dictionary_canonical_wire_bytes': len(canonical(dictionary.to_dict())),
              'cover_words': len(re.findall(r'\S+', cover)) if cover else 0, 'variants': {}}
    if cover:
        bundle['capacity_bits'] = codec.analyze_text(cover, dictionary, protocol).capacity_bits
    for bits in config['payload_bits']:
        payload = payload_for(config, sample, bits)
        output = {'payload_bits': bits, 'payload_hex': payload.hex(), 'full_success': False,
                  'stego_text': None, 'error': 'all_declared_nonquarantined_covers_required'}
        started = time.monotonic()
        if cover:
            try:
                encoded = codec.encode(cover, payload, key, dictionary, protocol, selector='canonical')
                decoded, receiver = isolated_decode(encoded.stego_text, key, dictionary, protocol)
                output.update(stego_text=encoded.stego_text, decoded_payload_hex=decoded.hex(), receiver=receiver,
                    encoding={k: v for k, v in asdict(encoded).items() if k != 'stego_text'},
                    full_success=decoded == payload, error=None if decoded == payload else 'full_payload_mismatch',
                    external_document_metadata_bits=0, individual_codebook_bytes=0, shared_secret_key_bytes=32)
            except Exception as error:
                output['error'] = f'{type(error).__name__}: {error}'
        output['encode_and_isolated_decode_seconds'] = time.monotonic() - started
        output['goodput_bits_per_bundle_word'] = (bits if output['full_success'] else 0) / bundle['cover_words'] if cover else None
        bundle['variants'][str(bits)] = output
    atomic_json(path, bundle)
    event(run / 'models' / alias, 'fixed_all_cover_bundle_completed', bundle_sha256=digest(bundle))
    return bundle


def reliability(rows, arm):
    successes = [row for row in rows if row['variants'][arm['id']]['full_success']]
    total_words = sum(len(re.findall(r'\S+', row.get('cover_text') or '')) for row in rows)
    return {'attempted_samples': len(rows), 'full_success_samples': len(successes),
            'full_success_rate': len(successes) / len(rows) if rows else None,
            'requested_payload_bits': len(rows) * arm['payload_bits'],
            'delivered_payload_bits': len(successes) * arm['payload_bits'],
            'goodput_bits_per_all_generated_cover_words': len(successes) * arm['payload_bits'] / total_words if total_words else None,
            'capacity_failures': sum('CapacityError:' in (row['variants'][arm['id']].get('error') or '') for row in rows),
            'failures': [{'sample_id': row['sample']['id'], 'error': row['failure'] or row['variants'][arm['id']]['error']}
                         for row in rows if not row['variants'][arm['id']]['full_success']]}


def all_finished(run, config):
    hashes = {}
    freeze_id = read_json(run / 'freeze.json')['freeze_id']
    for model in config['models']:
        alias = model['alias']
        path = run / 'models' / alias / 'generation.json'
        if not path.exists():
            raise ValueError('All declared models must finish before evaluation')
        status = read_json(path)
        if (status.get('complete') is not True or status.get('records') != len(config['samples'])
                or status.get('freeze_id') != freeze_id
                or (run / 'models' / alias / '.writer.lock').exists()):
            raise ValueError('All declared attempts must be terminal and writers closed')
        rows = read_records(run, alias, complete=True)
        if digest(rows) != status['records_sha256'] or any(row['freeze_id'] != freeze_id for row in rows):
            raise ValueError('Records differ from completed generation')
        bundle = read_json(run / 'models' / alias / 'bundle.json')
        if (digest(bundle) != status['bundle_sha256'] or bundle['source_records_sha256'] != status['records_sha256']
                or bundle['freeze_id'] != freeze_id):
            raise ValueError('Completed bundle changed')
        hashes[alias] = status['records_sha256']
    return hashes


def evaluate(run):
    import joblib
    config, frozen, _, lock = verify(run)
    with base.writer_lock(run / 'evaluation_control'):
        if (run / 'evaluation.json').exists():
            raise ValueError('Evaluation already completed once')
        hashes = all_finished(run, config)
        identity = {'freeze_id': frozen['freeze_id'], 'source_lock_id': lock['lock_id'],
                    'all_model_records_sha256': hashes,
                    'all_model_bundle_sha256': {m['alias']: file_hash(run / 'models' / m['alias'] / 'bundle.json')
                                               for m in config['models']}}
        marker = run / 'evaluation_opened.json'
        if marker.exists():
            if any(read_json(marker).get(k) != v for k, v in identity.items()):
                raise ValueError('Cannot resume evaluation after identity change')
            event(run, 'shared_evaluation_resumed', **identity)
        else:
            atomic_json(marker, {**identity, 'opened_at_unix': time.time()})
            event(run, 'shared_evaluation_opened_once', **identity)
        records = {model['alias']: read_records(run, model['alias'], complete=True) for model in config['models']}
        for alias, rows in records.items():
            if digest(rows) != hashes[alias] or any(row['freeze_id'] != frozen['freeze_id'] for row in rows):
                raise ValueError('Records differ from completed generation')
            for row in rows:
                for arm in arms(config):
                    output = row['variants'][arm['id']]
                    if output['full_success'] and (output['decoded_payload_hex'] != row['payloads_hex'][str(arm['payload_bits'])]
                                                 or not output['stego_text']):
                        raise ValueError('Full delivery lacks exact retained payload evidence')
        classifiers = {name: joblib.load(run / 'source_attacks' / (name + '.joblib')) for name in lock['attacks']}
        report = {**identity, 'schema': SCHEMA, 'status': 'complete_shared_dictionary_feasibility', 'models': {},
                  'evaluation_resume_count': sum(json.loads(line)['event'] == 'shared_evaluation_resumed'
                      for line in (run / 'events.jsonl').read_text().splitlines() if line),
                  'claim_scope': 'small_transport_feasibility_and_frozen_attack_transfer_not_security',
                  'limitations': ['Plain framing has no authentication or encryption guarantee.',
                    'All fixture keys and payloads are public: these archived outputs offer no confidentiality.',
                    'AES-SIV authenticates the extracted message and invariant text, not every within-group synonym spelling.',
                    'The fixed public dictionary is a shared resource, not a document-specific codebook or free storage.',
                    'All capacity and generation failures remain in reliability denominators.',
                    'Transferred attacks were fitted on old codebook outputs, not this new algorithm.',
                    'Payload sizes share each cover and must not be pooled as independent documents.',
                    'No semantic, human-naturalness, significance or steganographic-security claim is established.']}
        for alias, rows in records.items():
            item = {'planned_covers': len(rows), 'generated_covers': sum(isinstance(r.get('cover_text'), str)
                    and bool(r['cover_text'].strip()) for r in rows),
                    'reliability': {arm['id']: reliability(rows, arm) for arm in arms(config)}, 'cohorts': {}}
            bundle = read_json(run / 'models' / alias / 'bundle.json')
            item['fixed_all_cover_bundle'] = {k: v for k, v in bundle.items() if k not in {'cover_text', 'variants'}}
            item['fixed_all_cover_bundle']['variants'] = {bits: {k: v for k, v in value.items()
                if k not in {'stego_text', 'receiver'}} for bits, value in bundle['variants'].items()}
            for bits in config['payload_bits']:
                for framing in FRAMINGS:
                    ids = [arm_id(framing, selector, bits) for selector in SELECTORS]
                    successful = {name: [r for r in values if all(r['variants'][arm]['full_success'] for arm in ids)]
                                  for name, values in records.items()}
                    matched_ids = set.intersection(*({r['sample']['id'] for r in values} for values in successful.values()))
                    for cohort, selected in [('per_model_success', successful[alias]),
                        ('matched_all_models_success', [r for r in successful[alias] if r['sample']['id'] in matched_ids])]:
                        cohort_id = f'{framing}__{bits}__{cohort}'
                        summary = {'pairs': len(selected), 'sample_ids': [r['sample']['id'] for r in selected], 'attacks': {}}
                        for selector in SELECTORS:
                            arm = arm_id(framing, selector, bits)
                            for feature in ('char', 'word'):
                                source_name = f'fixed_{selector}__{feature}'
                                attack = lock['attacks'][source_name]
                                name = f'{arm}__{feature}'
                                if not selected:
                                    summary['attacks'][name] = {'status': 'no_common_success_pairs'}
                                    continue
                                texts = [text for row in selected for text in (row['cover_text'], row['variants'][arm]['stego_text'])]
                                labels = [label for _ in selected for label in (0, 1)]
                                scores = classifiers[source_name].predict_proba(texts)[:, 1]
                                if attack['score_direction'] == -1:
                                    scores = 1 - scores
                                predictions = [{'sample_id': row['sample']['id'], 'label': label, 'score': float(score),
                                    'prediction': int(score >= .5), 'prediction_at_source_dev_threshold': int(score >= attack['threshold'])}
                                    for row, pair in zip(selected, zip(scores[::2], scores[1::2]))
                                    for label, score in enumerate(pair)]
                                destination = run / 'predictions' / alias / cohort_id / (name + '.json')
                                atomic_json(destination, predictions)
                                summary['attacks'][name] = {'status': 'complete', 'source_attack': source_name,
                                    **base.metrics_from_scores(labels, scores, attack['threshold']),
                                    'source_development_threshold': attack['threshold'], 'source_score_direction': attack['score_direction'],
                                    'predictions_sha256': file_hash(destination)}
                        item['cohorts'][cohort_id] = summary
            report['models'][alias] = item
        atomic_json(run / 'evaluation.json', report)
        event(run, 'shared_evaluation_completed', evaluation_sha256=file_hash(run / 'evaluation.json'))
        return report


def completed(run):
    config, frozen, dictionary, lock = verify(run)
    evaluation = read_json(run / 'evaluation.json')
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines() if line]
    openings = [row for row in events if row['event'] == 'shared_evaluation_opened_once']
    completions = [row for row in events if row['event'] == 'shared_evaluation_completed']
    if (len(openings) != 1 or len(completions) != 1
            or completions[0]['evaluation_sha256'] != file_hash(run / 'evaluation.json')
            or evaluation['freeze_id'] != frozen['freeze_id']
            or evaluation['source_lock_id'] != lock['lock_id']
            or evaluation['status'] != 'complete_shared_dictionary_feasibility'
            or (run / 'evaluation_control/.writer.lock').exists()
            or evaluation['all_model_records_sha256'] != all_finished(run, config)):
        raise ValueError('One intact completed evaluation required')
    if events.index(openings[0]) >= events.index(completions[0]):
        raise ValueError('Evaluation event order changed')
    resumptions = [row for row in events if row['event'] == 'shared_evaluation_resumed']
    if evaluation['evaluation_resume_count'] != len(resumptions):
        raise ValueError('Evaluation resumption count changed')
    marker = read_json(run / 'evaluation_opened.json')
    for name in ('freeze_id', 'source_lock_id', 'all_model_records_sha256', 'all_model_bundle_sha256'):
        if (marker[name] != evaluation[name] or openings[0][name] != evaluation[name]
                or any(row[name] != evaluation[name] for row in resumptions)):
            raise ValueError('Opening identity changed')
    if evaluation['all_model_bundle_sha256'] != {m['alias']: file_hash(run / 'models' / m['alias'] / 'bundle.json')
                                               for m in config['models']}:
        raise ValueError('Bundles changed after evaluation')
    records = {m['alias']: read_records(run, m['alias'], complete=True) for m in config['models']}
    if any(digest(rows) != evaluation['all_model_records_sha256'][alias] for alias, rows in records.items()):
        raise ValueError('Records changed after evaluation')
    return config, frozen, dictionary, evaluation, records


def make_export(run):
    config, frozen, _, evaluation, records = completed(run)
    rows, exclusions = [], []
    for alias, source in records.items():
        for record in source:
            sample = record['sample']
            valid_cover = isinstance(record.get('cover_text'), str) and bool(record['cover_text'].strip())
            outcomes = [('cover', 0, record.get('cover_text') if valid_cover else None)]
            if not valid_cover:
                exclusions.append({'model_alias': alias, 'sample_id': sample['id'], 'arm': 'cover',
                                   'reason': record['failure'] or 'no_generated_cover'})
            for arm in arms(config):
                output = record['variants'][arm['id']]
                if output['full_success']:
                    outcomes.append((arm['id'], arm['payload_bits'], output['stego_text']))
                else:
                    exclusions.append({'model_alias': alias, 'sample_id': sample['id'], 'arm': arm['id'],
                                       'reason': record['failure'] or output['error']})
            for arm, bits, text in outcomes:
                if not text:
                    continue
                rows.append({'id': json.dumps([frozen['freeze_id'], alias, sample['id'], arm], separators=(',', ':')),
                    'model_alias': alias, 'sample_id': sample['id'], 'topic_id': sample['topic_id'], 'genre_id': sample['genre_id'],
                    'payload_bits': bits, 'arm': arm, 'text': text, 'text_sha256': neural.text_hash(text)})
        bundle = read_json(run / 'models' / alias / 'bundle.json')
        outcomes = [('bundle_cover', 0, bundle.get('cover_text'))]
        if not bundle.get('cover_text'):
            exclusions.append({'model_alias': alias, 'sample_id': 'all_covers_bundle', 'arm': 'bundle_cover',
                               'reason': 'all_declared_nonquarantined_covers_required'})
        for bits, output in bundle['variants'].items():
            name = 'bundle_aes_siv__canonical__' + bits
            if output['full_success']:
                outcomes.append((name, int(bits), output['stego_text']))
            else:
                exclusions.append({'model_alias': alias, 'sample_id': 'all_covers_bundle', 'arm': name,
                                   'reason': output['error']})
        for arm, bits, text in outcomes:
            if text:
                rows.append({'id': json.dumps([frozen['freeze_id'], alias, 'all_covers_bundle', arm], separators=(',', ':')),
                    'model_alias': alias, 'sample_id': 'all_covers_bundle', 'topic_id': 'all_topics', 'genre_id': 'bundle',
                    'payload_bits': bits, 'arm': arm, 'text': text, 'text_sha256': neural.text_hash(text)})
    corpus = b''.join(canonical(row) + b'\n' for row in rows)
    manifest = {'schema': 'synsteg-shared-dictionary-secondary-v1', 'freeze_id': frozen['freeze_id'],
                'evaluation_sha256': file_hash(run / 'evaluation.json'),
                'all_model_records_sha256': evaluation['all_model_records_sha256'],
                'helper_script_sha256': file_hash(__file__), 'runner_script_sha256': file_hash(ROOT / 'scripts/rescore_models.py'),
                'expected_jobs': {name: neural.validate_job(read_json(path)) for name, path in JOBS.items()},
                'job_files_sha256': {name: file_hash(path) for name, path in JOBS.items()},
                'corpus_sha256': hashlib.sha256(corpus).hexdigest(), 'n_records': len(rows),
                'n_unique_texts': len({row['text_sha256'] for row in rows}), 'exclusions': exclusions}
    return manifest, rows, corpus, records


def export(run, output):
    manifest, _, corpus, _ = make_export(run)
    if output.exists():
        raise FileExistsError('Refusing to overwrite secondary corpus')
    output.mkdir(parents=True)
    atomic_json(output / 'manifest.json', manifest)
    (output / 'corpus.jsonl').write_bytes(corpus)
    return manifest


def summarize(run, corpus, scores):
    if set(scores) != set(JOBS):
        raise ValueError('Both frozen full-text scorers are required')
    config = read_json(run / 'protocol.json')
    expected, rows, serialized, records = make_export(run)
    if read_json(corpus / 'manifest.json') != expected or (corpus / 'corpus.jsonl').read_bytes() != serialized:
        raise ValueError('Secondary corpus differs from completed primary')
    result = {'schema': 'synsteg-shared-dictionary-secondary-v1', 'status': 'complete',
              'freeze_id': expected['freeze_id'], 'corpus_sha256': expected['corpus_sha256'],
              'n_records': len(rows), 'n_unique_texts': expected['n_unique_texts'],
              'exclusions': expected['exclusions'], 'quality_models': {}}
    for job, directory in scores.items():
        manifest, values = quality_checks.verified_scores(directory, expected, rows, job)
        index = {(v['model_alias'], v['sample_id'], v['arm']): v for v in values}
        effects = {}
        for alias, source in records.items():
            for bits in config['payload_bits']:
                for framing in FRAMINGS:
                    left, right = arm_id(framing, 'quality', bits), arm_id(framing, 'canonical', bits)
                    comparisons = {'quality_vs_canonical': (left, right), 'quality_vs_cover': (left, 'cover'),
                                   'canonical_vs_cover': (right, 'cover')}
                    if config['include_fixed_quality_baseline']:
                        comparisons['quality_vs_fixed_quality'] = (left, f'fixed_quality__{bits}')
                    for comparison, (a, b) in comparisons.items():
                        pairs = [(row['sample']['id'], index.get((alias, row['sample']['id'], a)),
                                  index.get((alias, row['sample']['id'], b))) for row in source]
                        pairs = [(sid, x, y) for sid, x, y in pairs if x is not None and y is not None]
                        deltas = [x['mean_nll'] - y['mean_nll'] for _, x, y in pairs]
                        effects[f'{alias}__{framing}__{bits}__{comparison}'] = {
                            'pairs': len(pairs), 'sample_ids': [sid for sid, _, _ in pairs],
                            'mean_log_ppl_ratio': sum(deltas) / len(deltas) if deltas else None,
                            'geometric_mean_ppl_ratio': math.exp(sum(deltas) / len(deltas)) if deltas else None,
                            'lower': sum(d < -1e-12 for d in deltas), 'equal': sum(abs(d) <= 1e-12 for d in deltas),
                            'higher': sum(d > 1e-12 for d in deltas)}
        result['quality_models'][job] = {'job': expected['expected_jobs'][job],
            'score_manifest_sha256': file_hash(directory / 'manifest.json'),
            'scores_sha256': file_hash(directory / 'scores.jsonl'), 'effects': effects, 'measurements': values}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'generate', 'evaluate', 'export', 'summarize'))
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--model')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--gpt2-scores', type=Path)
    parser.add_argument('--qwen-base-scores', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.command == 'prepare':
        if args.config is None:
            parser.error('prepare requires --config')
        with base.writer_lock(run):
            value = prepare(args.config, run)
    elif args.command == 'verify':
        _, frozen, _, _ = verify(run)
        value = {'verified': True, 'freeze_id': frozen['freeze_id']}
    elif args.command == 'generate':
        if args.model is None or (args.limit is not None and args.limit < 1):
            parser.error('generate requires model and a positive optional limit')
        value = generate(run, args.model, limit=args.limit)
    elif args.command == 'evaluate':
        value = evaluate(run)
    elif args.command == 'export':
        if args.output is None:
            parser.error('export requires --output')
        value = export(run, args.output.resolve())
    else:
        if any(x is None for x in (args.output, args.corpus, args.gpt2_scores, args.qwen_base_scores)):
            parser.error('summarize requires output, corpus and both score directories')
        if args.output.exists():
            raise FileExistsError('Refusing to overwrite secondary summary')
        value = summarize(run, args.corpus, {'gpt2': args.gpt2_scores, 'qwen_base': args.qwen_base_scores})
        atomic_json(args.output / 'measurements.json', value)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
