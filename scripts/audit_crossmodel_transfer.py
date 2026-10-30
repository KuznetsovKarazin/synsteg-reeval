#!/usr/bin/env python3
"""Independent, portable audit of a completed cross-generator transfer pilot.

Only the Python standard library is used. No model, classifier, frozen runner,
or codec is imported. Generated records are read only after a completed
one-opening gate. Missing external resources are explicitly unverified unless
--strict-resources is selected; repository source checks are always required.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import zlib

VARIANTS = ('fixed_canonical', 'fixed_quality')
COHORTS = ('per_model_common_success', 'matched_all_models_success')
OLD_FREEZE = 'e0575ce1ba14d9975c1386bed0b16c723b72afc3c9f64752377304ecf6272956'
OLD_LOCK = 'ea1eef91b6f856220b81599df3297692536ee4d0c4cd64db8823f0b0622712d9'
OLD_PROOF = 'results/prospective-pilot-cpu-v3/remote_verification/frozen_source_verification.json'
OLD_PROOF_SHA = '7dd973bd800e953fff54a830a0af4634912336a34c19998be90e3bbe57dbde61'
KEY_ORDER = ('version', 'cover_sha256', 'seed', 'max_bits_per_position',
             'min_group_size', 'protect_capitalized', 'candidate_filter_id',
             'scorer_id', 'separators', 'positions')
POSITION_ORDER = ('original', 'groups', 'scores')
TOKEN = re.compile(r'([^A-Za-z0-9_]*)([A-Za-z]+)([^A-Za-z0-9_]*)\Z')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('Duplicate JSON key: ' + key)
        obj[key] = value
    return obj


def reject_constant(value):
    raise ValueError('Nonfinite JSON number: ' + value)


def loads(text):
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def read(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def events(path):
    return [loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return type(left) is type(right) and left == right


class Checks:
    def __init__(self):
        self.counts = collections.Counter()

    def require(self, name, condition):
        if not condition:
            raise AssertionError(name)
        self.counts[name.split(':', 1)[0]] += 1

    def equal(self, name, left, right):
        self.require(name, same(left, right))

    def exact(self, name, left, right):
        self.require(name, canonical(left) == canonical(right))


def child_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe relative evidence path')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Evidence path escapes its root')
    return path


def ordered_key_bytes(book):
    if set(book) != set(KEY_ORDER):
        raise ValueError('Unexpected codebook fields')
    ordered = {key: book[key] for key in KEY_ORDER}
    positions = []
    for position in book['positions']:
        if set(position) != set(POSITION_ORDER):
            raise ValueError('Unexpected position fields')
        positions.append({key: position[key] for key in POSITION_ORDER})
    ordered['positions'] = positions
    return json.dumps(ordered, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def split_layout(text):
    spans = list(re.finditer(r'\S+', text))
    tokens, separators, previous_end = [], [], 0
    for match in spans:
        separators.append(text[previous_end:match.start()])
        tokens.append(match.group())
        previous_end = match.end()
    separators.append(text[previous_end:])
    return tokens, separators


def independent_decode(book, stego, expected_bits):
    """Decode retained group indices without calling the experimental codec."""
    if type(expected_bits) is not int or expected_bits < 0:
        raise ValueError('Invalid expected payload length')
    tokens, separators = split_layout(stego)
    if len(tokens) != len(book['positions']) or separators != book['separators']:
        raise ValueError('Stego layout differs from codebook')
    recovered, used = '', []
    for index, (token, position) in enumerate(zip(tokens, book['positions'])):
        groups = position['groups']
        if not groups or len(recovered) >= expected_bits:
            if token != position['original']:
                raise ValueError('Literal or unused tail changed')
            continue
        count = len(groups)
        if count < 2 or count & (count - 1):
            raise ValueError('Group count is not a positive bit width')
        matches = [g for g, forms in enumerate(groups) for form in forms if form == token]
        if len(matches) != 1:
            raise ValueError('Token lacks unique group membership')
        width = count.bit_length() - 1
        recovered += format(matches[0], '0' + str(width) + 'b')
        used.append((index, matches[0], token))
    if len(recovered) < expected_bits or set(recovered[expected_bits:]) - {'0'}:
        raise ValueError('Incomplete payload or nonzero padding')
    return recovered[:expected_bits], used


def inspect_book(book, cover, sample, method, checks):
    sid = sample['id']
    raw = ordered_key_bytes(book)
    positions, separators = book['positions'], book['separators']
    checks.require('book_layout:' + sid, len(separators) == len(positions) + 1
                   and all(isinstance(s, str) and (not s or s.isspace()) for s in separators)
                   and all(separators[1:-1]))
    checks.require('original_tokens:' + sid, all(isinstance(p['original'], str) and p['original']
                   and not any(c.isspace() for c in p['original']) for p in positions))
    reconstructed = ''.join(s + p['original'] for s, p in zip(separators, positions)) + separators[-1]
    checks.require('cover_reconstruction:' + sid, reconstructed == cover
                   and hashlib.sha256(cover.encode()).hexdigest() == book['cover_sha256'])
    expected = {'version': 'fixed-codebook-v1', 'seed': 'prospective-fixed-v1:' + sid,
                'max_bits_per_position': 1, 'min_group_size': 1, 'protect_capitalized': True,
                'candidate_filter_id': method['candidate_filter_id'], 'scorer_id': method['scorer_id']}
    for key, value in expected.items():
        checks.exact('book_identity:' + sid + ':' + key, book[key], value)
    capacity = 0
    for index, position in enumerate(positions):
        groups, scores = position['groups'], position['scores']
        count = len(groups)
        checks.require('group_width:' + sid, count in (0, 2) and len(scores) == count)
        capacity += 1 if count else 0
        forms = [form for group in groups for form in group]
        checks.require('candidate_forms:' + sid, all(isinstance(form, str) and form
                       and not any(c.isspace() for c in form) for form in forms))
        checks.require('unique_candidates:' + sid, len(forms) == len(set(forms)))
        original_match = TOKEN.fullmatch(position['original'])
        for group, values in zip(groups, scores):
            checks.require('group_scores:' + sid, len(group) >= 1 and len(group) == len(values)
                           and all(type(x) in (int, float) and math.isfinite(x) for x in values))
            checks.require('candidate_punctuation:' + sid, original_match is not None
                           and original_match.group(2).islower()
                           and not any(c.isalpha() for c in original_match.group(1) + original_match.group(3))
                           and all(isinstance(form, str) and TOKEN.fullmatch(form) is not None
                                   and (TOKEN.fullmatch(form).group(1), TOKEN.fullmatch(form).group(3))
                                   == (original_match.group(1), original_match.group(3)) for form in group))
        if groups:
            def order(form):
                key = json.dumps([book['version'], book['seed'], book['cover_sha256'], index, form], separators=(',', ':'))
                return hashlib.sha256(key.encode()).hexdigest(), form
            ranked = sorted(forms, key=order)
            checks.equal('seeded_partition:' + sid, groups, [ranked[0::2], ranked[1::2]])
    eligibility = {'positions': len(positions), 'eligible_positions': capacity,
                   'groups': sum(len(p['groups']) for p in positions),
                   'singleton_groups': sum(len(g) == 1 for p in positions for g in p['groups']),
                   'groups_with_quality_choice': sum(len(g) > 1 for p in positions for g in p['groups'])}
    return {'capacity_bits': capacity, 'eligibility': eligibility,
            'key_json_bytes': len(raw), 'key_zlib9_bytes': len(zlib.compress(raw, 9))}


def inspect_success(record, variant, book_info, expected_payload, checks):
    sample, output = record['sample'], record['variants'][variant]
    label = sample['id'] + ':' + variant
    book, metadata, metrics = output['key'], output['metadata'], output['metrics']
    bits = sample['payload_bits']
    checks.require('full_delivery:' + label, output.get('error') is None
                   and output['decoded_payload'] == expected_payload
                   and metrics['full_message_success'] is True)
    decoded, used = independent_decode(book, output['stego_text'], bits)
    checks.require('independent_decode:' + label, decoded == expected_payload)
    checks.exact('key_prepared_identity:' + label, book, record['prepared_methods'][variant]['codebook'])
    for key, expected in {'codebook_sha256': digest(book), 'selection': variant.removeprefix('fixed_'),
                          'framing': 'external_length', 'external_metadata_bits': 32,
                          'external_length_bits': 32, 'embedded_bits': bits, 'frame_bits': bits,
                          'padding_bits': 0, 'capacity_bits': book_info['capacity_bits'], 'used_positions': len(used)}.items():
        checks.equal('encoding_metadata:' + label + ':' + key, metadata[key], expected)
    checks.require('capacity_sufficient:' + label, book_info['capacity_bits'] >= bits and len(used) == bits)
    checks.equal('encoding_eligibility:' + label, metadata['eligibility'], book_info['eligibility'])
    for index, group_id, token in used:
        position = book['positions'][index]
        group, scores = position['groups'][group_id], position['scores'][group_id]
        selected = group[0] if variant == 'fixed_canonical' else min(
            zip(group, scores), key=lambda x: (x[1], x[0] != position['original'], x[0]))[0]
        checks.require('selector_choice:' + label, token == selected)
    cw, sw = len(record['cover_text'].split()), len(output['stego_text'].split())
    expected_metrics = {'requested_bits': bits, 'used_bits': bits, 'decoded_bits': bits,
                        'payload_eligible': True, 'full_exact': True, 'full_message_success': True,
                        'prefix_exact': True, 'nonempty_prefix_exact': True, 'correct_leading_bits': bits,
                        'missing_bits': 0, 'extra_bits': 0, 'truncated': False, 'bit_error_rate': 0.0,
                        'cover_words': cw, 'stego_words': sw, 'error': None,
                        'key_json_bytes': book_info['key_json_bytes'], 'key_zlib9_bytes': book_info['key_zlib9_bytes']}
    for key in ('requested_bpw_cover', 'used_bpw_cover'):
        expected_metrics[key] = bits / cw
    for key in ('used_bpw_stego', 'decoded_bpw_stego', 'correct_leading_bpw_stego', 'full_message_goodput_bpw_stego'):
        expected_metrics[key] = bits / sw
    checks.equal('delivery_metrics:' + label, metrics, expected_metrics)
    return {'key_json_bytes': book_info['key_json_bytes'], 'key_zlib9_bytes': book_info['key_zlib9_bytes'],
            'external_length_bits': 32, 'key_json_plus_length_bits': 8 * book_info['key_json_bytes'] + 32,
            'key_zlib9_plus_length_bits': 8 * book_info['key_zlib9_bytes'] + 32}


def full_success(record, variant):
    return (record.get('variants', {}).get(variant, {}).get('metrics') or {}).get('full_message_success') is True


def reliability(rows, variant):
    outputs = [r.get('variants', {}).get(variant, {}) for r in rows]
    successful = [r for r in rows if full_success(r, variant)]
    return {'attempted_samples': len(rows), 'full_success_samples': len(successful),
            'full_success_rate': len(successful) / len(rows) if rows else None,
            'failed_samples': len(rows) - len(successful),
            'generation_or_preparation_failures': sum(bool(r.get('failure')) for r in rows),
            'capacity_or_encoding_or_decode_failures': sum(bool(o.get('error')) for o in outputs),
            'successful_payload_bits': sum(r['sample']['payload_bits'] for r in successful),
            'attempted_payload_bits': sum(r['sample']['payload_bits'] for r in rows),
            'mean_successful_bpw_stego': statistics.mean(r['sample']['payload_bits'] /
                len(r['variants'][variant]['stego_text'].split()) for r in successful) if successful else None}


def prediction_metrics(rows, threshold):
    if not rows or any(type(r['label']) is not int or r['label'] not in (0, 1)
                       or type(r['score']) not in (int, float) or not math.isfinite(r['score'])
                       or not 0 <= r['score'] <= 1 for r in rows):
        raise ValueError('Invalid prediction rows')
    positive = [r['score'] for r in rows if r['label'] == 1]
    negative = [r['score'] for r in rows if r['label'] == 0]
    if not positive or not negative:
        raise ValueError('Both prediction classes required')
    return {'accuracy': sum((r['score'] >= .5) == r['label'] for r in rows) / len(rows),
            'auc': sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative)),
            'fpr_at_development_threshold': sum(n >= threshold for n in negative) / len(negative),
            'tpr_at_development_threshold': sum(p >= threshold for p in positive) / len(positive)}


def completed_headers(run, checks):
    # No generated-record directory is accessed until this function succeeds.
    required = ('evaluation.json', 'evaluation_opened.json', 'events.jsonl',
                'freeze.json', 'protocol.json', 'samples.json', 'source_attacks/attack_lock.json')
    checks.require('completed_gate_files', all((run / name).is_file() for name in required))
    evaluation, opening = read(run / 'evaluation.json'), read(run / 'evaluation_opened.json')
    frozen, config = read(run / 'freeze.json'), read(run / 'protocol.json')
    lock, ledger = read(run / 'source_attacks/attack_lock.json'), events(run / 'events.jsonl')
    opens = [e for e in ledger if e['event'] == 'transfer_evaluation_opened_once']
    ends = [e for e in ledger if e['event'] == 'transfer_evaluation_completed']
    checks.require('one_opening_completion', len(opens) == len(ends) == 1)
    checks.require('zero_evaluation_resumes', not any(e['event'] == 'transfer_evaluation_resumed' for e in ledger))
    checks.require('completed_report_hash', ends[0].get('evaluation_sha256') == sha(run / 'evaluation.json')
                   and evaluation.get('status') == 'complete_frozen_attack_transfer_evaluation')
    checks.require('opening_completion_order', opens[0]['time_unix'] <= ends[0]['time_unix'])
    aliases = [m['alias'] for m in config['models']]
    hashes = evaluation['all_model_records_sha256']
    checks.require('completed_model_set', set(hashes) == set(aliases) == set(evaluation['models']))
    for identity in (evaluation, opening, opens[0]):
        checks.require('opening_identity', identity['freeze_id'] == frozen['freeze_id']
                       and identity['source_lock_id'] == lock['lock_id']
                       and identity['all_model_records_sha256'] == hashes)
    for alias in aliases:
        status = read(run / 'models' / alias / 'generation.json')
        checks.require('all_models_terminal:' + alias, status.get('complete') is True
                       and status['records'] == status['expected'] == len(config['samples'])
                       and status['model_alias'] == alias and status['freeze_id'] == frozen['freeze_id']
                       and status['records_sha256'] == hashes[alias]
                       and not (run / 'models' / alias / '.writer.lock').exists())
    checks.require('evaluation_writer_absent', not (run / 'evaluation_control/.writer.lock').exists())
    return config, frozen, lock, evaluation, opening, opens[0], ends[0], ledger


def resource_checks(repo, frozen, strict, checks):
    suffix = '/scripts/crossmodel_quality.py'
    roots = [path[:-len(suffix)] for path in frozen['resource_sha256'] if path.endswith(suffix)]
    checks.require('resource_repo_root', len(roots) == 1)
    original_root = roots[0]
    checked, missing = [], []
    for original, expected in frozen['resource_sha256'].items():
        in_repo = original.startswith(original_root + '/')
        path = child_path(repo, original[len(original_root) + 1:]) if in_repo else Path(original)
        if not path.is_file():
            checks.require('required_resource_available:' + original, not (strict or in_repo))
            missing.append({'recorded_path': original, 'expected_sha256': expected})
            continue
        checks.require('resource_sha256:' + original, sha(path) == expected)
        checked.append({'recorded_path': original, 'resolved_path': str(path), 'sha256': expected,
                        'repository_remapped': in_repo})
    return {'verified_files': checked, 'unavailable_external_files': missing,
            'all_declared_resource_files_verified': not missing,
            'runtime_environment_recreated': False, 'model_weight_bytes_rehashed': False}


def attempt_ledger(row, marker, starts, terminals, preparations, bounds, checks):
    """Account for crash boundaries without pretending an absent event exists."""
    sid = row['sample']['id']
    checks.require('one_terminal:' + sid, len(terminals) == 1)
    end = terminals[0]
    interrupted = end['event'] == 'attempt_interrupted_not_retried'
    checks.require('start_event_count:' + sid, len(starts) in ((0, 1) if interrupted else (1,)))
    start_time = starts[0]['time_unix'] if starts else marker['time_unix']
    checks.require('attempt_order:' + sid, marker['freeze_id'] == bounds['freeze_id']
                   and bounds['frozen'] <= bounds['freeze_event'] <= marker['time_unix']
                   and marker['time_unix'] == row['started_at'] <= start_time
                   <= row['finished_at'] <= end['time_unix'] <= bounds['generation'] <= bounds['opening'])
    checks.require('interruption_retained:' + sid, not interrupted or (
        row['failure'] == 'interrupted_attempt_not_retried' and row['status'] == 'failed'
        and row['cover_text'] is None and row['variants'] == {} and 'prepared_methods' not in row))
    if not interrupted:
        checks.exact('terminal_record_identity:' + sid,
                     [end['status'], end['failure']], [row['status'], row['failure']])
    checks.require('preparation_event_count:' + sid, len(preparations) <= 1)
    if preparations:
        checks.require('preparation_event_order:' + sid, bool(starts)
                       and start_time <= preparations[0]['time_unix'] <= row['finished_at'])
        values = preparations[0]['sha256']
        checks.require('preparation_event_hash_shape:' + sid, set(values) == set(VARIANTS)
                       and all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in values.values()))
    if 'prepared_methods' in row:
        checks.require('one_preparation:' + sid, len(preparations) == 1)
    else:
        checks.require('no_spurious_preparation:' + sid,
                       row['failure'] is not None and (interrupted or not preparations))
    return {'interrupted': interrupted, 'start_event_absent': not starts,
            'preparation_state_unavailable_after_interruption': bool(interrupted and preparations)}


def inspect_generation_metadata(row, model, generation, template, checks):
    """Check retained generation identities; no tokenizer execution is implied."""
    sid = row['sample']['id']
    metadata = row.get('generation_metadata')
    if metadata is None:
        checks.require('generation_metadata_absence:' + sid,
                       row['cover_text'] is None and row['failure'] is not None)
        return None
    checks.exact('generation_model:' + sid, metadata['model'], model)
    checks.exact('generation_seed:' + sid, metadata['seed'], row['sample']['generation_seed'])
    checks.require('generation_device_dtype:' + sid, metadata['dtype'] == 'bfloat16'
                   and metadata['device'] == 'cpu' and metadata['add_special_tokens_after_template'] is False)
    tokens = metadata['output_token_ids']
    checks.require('generation_token_counts:' + sid, isinstance(tokens, list)
                   and all(type(token) is int and token >= 0 for token in tokens)
                   and type(metadata['generated_tokens']) is int and metadata['generated_tokens'] == len(tokens)
                   and len(tokens) <= generation['max_new_tokens']
                   and type(metadata['input_tokens']) is int and metadata['input_tokens'] > 0
                   and metadata['hit_token_limit'] is (len(tokens) == generation['max_new_tokens']))
    checks.require('generation_duration:' + sid, type(metadata['seconds']) in (int, float)
                   and math.isfinite(metadata['seconds']) and metadata['seconds'] >= 0)
    for key in ('chat_template_sha256', 'rendered_prompt_sha256', 'input_token_ids_sha256'):
        checks.require('generation_digest:' + sid + ':' + key,
                       isinstance(metadata[key], str) and re.fullmatch('[0-9a-f]{64}', metadata[key]) is not None)
    checks.require('native_template_identity:' + sid, template['revision'] == model['revision']
                   and metadata['chat_template_sha256'] == hashlib.sha256(template['chat_template'].encode()).hexdigest())
    checks.exact('native_eos_identity:' + sid, metadata['eos_token_id'], template['generation_eos_token_id'])
    checks.require('native_pad_identity:' + sid, type(metadata['pad_token_id']) is int and metadata['pad_token_id'] >= 0)
    checks.exact('native_pad_frozen_identity:' + sid, metadata['pad_token_id'], template['generation_pad_token_id'])
    expected = {'max_new_tokens': generation['max_new_tokens'], 'do_sample': True,
                'temperature': generation['temperature'], 'top_p': generation['top_p'],
                'top_k': generation.get('top_k', 50), 'repetition_penalty': 1.0,
                'pad_token_id': metadata['pad_token_id'], 'eos_token_id': metadata['eos_token_id']}
    for key, value in expected.items():
        checks.exact('effective_generation_setting:' + sid + ':' + key, metadata['effective_generation_config'][key], value)
    return {'sample_id': sid, 'template_sha256': metadata['chat_template_sha256'],
            'rendered_prompt_sha256': metadata['rendered_prompt_sha256'],
            'input_token_ids_sha256': metadata['input_token_ids_sha256'],
            'input_tokens': metadata['input_tokens'], 'generated_tokens': len(tokens)}


def runtime_proofs(directory, config, opening_time, completion_time, timelines, checks):
    if directory is None:
        return {'status': 'not_provided_not_verified', 'files': []}
    directory = Path(directory)
    proof, command_runs = [], set()
    for alias in [m['alias'] for m in config['models']] + ['evaluation']:
        path = directory / (alias + '-process.json')
        status = read(path)
        checks.require('owned_exit:' + alias, status.get('status') == 'completed'
                       and type(status.get('exit_code')) is int and status['exit_code'] == 0
                       and status['started_at_unix'] <= status['finished_at_unix'])
        command = status['command']
        checks.require('owned_command:' + alias, isinstance(command, list)
                       and any(str(x).endswith('crossmodel_transfer_pilot.py') for x in command)
                       and command.count('--run') == 1 and command.index('--run') + 1 < len(command))
        command_runs.add(command[command.index('--run') + 1])
        if alias == 'evaluation':
            checks.require('owned_evaluation_timing', status['alias'] == alias and 'evaluate' in command
                           and status['started_at_unix'] <= opening_time <= completion_time <= status['finished_at_unix'])
        else:
            checks.require('owned_generator_identity:' + alias, status['alias'] == alias
                           and 'generate' in command and '--model' in command
                           and command[command.index('--model') + 1] == alias
                           and status['started_at_unix'] <= timelines[alias]['first_started']
                           <= timelines[alias]['last_finished'] <= timelines[alias]['generation_completed']
                           <= status['finished_at_unix'] <= opening_time)
        proof.append({'path': str(path), 'sha256': sha(path), 'status': status})
    checks.require('owned_commands_same_run', len(command_runs) == 1)
    return {'status': 'all_four_owned_exits_verified_from_retained_evidence', 'files': proof,
            'limitation': 'Recorded owning wait/return-code evidence is checked; this audit does not independently observe past processes.'}


def audit_run(repo, run, expected_freeze, runtime_directory=None, strict_resources=False):
    repo, run = Path(repo).resolve(), Path(run).resolve()
    checks = Checks()
    config, frozen, lock, evaluation, opening, opened, ended, ledger = completed_headers(run, checks)
    checks.require('expected_freeze', frozen['freeze_id'] == expected_freeze
                   and digest({k: v for k, v in frozen.items() if k != 'freeze_id'}) == expected_freeze)
    checks.require('frozen_protocol_samples', sha(run / 'protocol.json') == frozen['protocol_sha256']
                   and sha(run / 'samples.json') == frozen['samples_sha256']
                   and read(run / 'samples.json') == config['samples'])
    for relative, expected in frozen['code_sha256'].items():
        checks.require('frozen_source:' + relative, sha(child_path(repo, relative)) == expected)
    resources = resource_checks(repo, frozen, strict_resources, checks)
    template_paths = [item['resolved_path'] for item in resources['verified_files']
                      if item['recorded_path'].endswith('/results/crossmodel-transfer-v1-2026-10-23/model_templates.json')]
    checks.require('frozen_native_template_inventory', len(template_paths) == 1)
    templates = read(template_paths[0])
    proof_path = repo / OLD_PROOF
    checks.require('old_source_proof', sha(proof_path) == OLD_PROOF_SHA)
    source_proof = read(proof_path)
    checks.require('old_seven_source_identity', source_proof['freeze_id'] == OLD_FREEZE
                   and source_proof['frozen_code_file_count'] == len(source_proof['frozen_code']) == 7)
    for source in source_proof['frozen_code']:
        checks.require('old_frozen_source:' + source['path'], sha(child_path(repo, source['path'])) == source['sha256'])
    expected_attacks = {v + '__' + feature for v in VARIANTS for feature in ('word', 'char')}
    checks.require('source_attack_lock', lock['freeze_id'] == OLD_FREEZE and lock['lock_id'] == OLD_LOCK
                   and frozen['source_lock_id'] == OLD_LOCK and set(lock['attacks']) == expected_attacks
                   and digest({k: v for k, v in lock.items() if k != 'lock_id'}) == OLD_LOCK
                   and sha(run / 'source_attacks/attack_lock.json') == config['source_attack_lock_sha256'])
    for relative, expected in frozen['source_attack_files_sha256'].items():
        checks.require('copied_attack_file:' + relative, sha(child_path(run, relative)) == expected)
    for name, attack in lock['attacks'].items():
        checks.require('attack_artifact_identity:' + name, sha(run / 'source_attacks' / (name + '.joblib')) == attack['artifact_sha256']
                       and name == attack['variant'] + '__' + attack['features']
                       and attack['score_direction'] in (-1, 1) and attack['decision_threshold'] == .5
                       and math.isfinite(attack['threshold']))
    samples = config['samples']
    topics, genres = {s['topic_id'] for s in samples}, {s['genre_id'] for s in samples}
    ids = [s['id'] for s in samples]
    aliases = [m['alias'] for m in config['models']]
    checks.require('planned_48_grid', len(aliases) == len(set(aliases)) == 3 and len(samples) == len(set(ids)) == 16
                   and len(topics) == 4 and len(genres) == 2
                   and {(s['topic_id'], s['genre_id'], s['payload_bits']) for s in samples}
                   == {(t, g, b) for t in topics for g in genres for b in (8, 16)})
    for topic in topics:
        for genre in genres:
            checks.require('matched_prompt_grid', len({s['prompt'] for s in samples
                           if s['topic_id'] == topic and s['genre_id'] == genre}) == 1)
    freeze_events = [e for e in ledger if e['event'] == 'design_and_source_attacks_frozen_before_generation']
    checks.require('one_freeze_event', len(freeze_events) == 1 and freeze_events[0]['freeze_id'] == expected_freeze)
    # Only now, with every completion and frozen identity checked, read raw records.
    initial_hashes = {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    all_records, summaries, costs, payloads, timelines, generation_identities = {}, {}, {}, {}, {}, {}
    for sample in samples:
        rng = random.Random(int(digest([config['payload_seed'], sample['id']]), 16))
        payloads[sample['id']] = ''.join(str(rng.randrange(2)) for _ in range(sample['payload_bits']))
    for alias in aliases:
        directory = run / 'models' / alias
        checks.require('exact_record_set:' + alias, {p.stem for p in (directory / 'records').glob('*.json')} == set(ids))
        checks.require('exact_attempt_set:' + alias, {p.stem for p in (directory / 'attempt_started').glob('*.json')} == set(ids))
        rows = [read(directory / 'records' / (sid + '.json')) for sid in ids]
        checks.require('records_digest:' + alias, digest(rows) == evaluation['all_model_records_sha256'][alias])
        model_events = events(directory / 'events.jsonl')
        starts = [e for e in model_events if e['event'] == 'attempt_started']
        terminals = [e for e in model_events if e['event'] in ('attempt_completed', 'attempt_interrupted_not_retried')]
        prepared_events = [e for e in model_events if e['event'] == 'methods_prepared_before_payload']
        generation_events = [e for e in model_events if e['event'] == 'generation_completed']
        checks.require('start_event_identities:' + alias,
                       all(count == 1 and sid in ids for sid, count in collections.Counter(e['sample_id'] for e in starts).items()))
        checks.require('preparation_event_identities:' + alias,
                       all(count == 1 and sid in ids for sid, count in collections.Counter(e['sample_id'] for e in prepared_events).items()))
        checks.require('one_terminal_each:' + alias, collections.Counter(e['sample_id'] for e in terminals) == collections.Counter(ids))
        checks.require('model_ledger_order:' + alias, all(a['time_unix'] <= b['time_unix'] for a, b in zip(model_events, model_events[1:])))
        checks.require('one_generation_completion:' + alias, len(generation_events) == 1
                       and generation_events[0]['records_sha256'] == digest(rows)
                       and generation_events[0]['time_unix'] <= opened['time_unix'])
        costs[alias] = []
        generation_identities[alias], interruption_evidence = [], []
        model = next(m for m in config['models'] if m['alias'] == alias)
        for sample, row in zip(samples, rows):
            sid = sample['id']
            marker = read(directory / 'attempt_started' / (sid + '.json'))
            checks.require('record_identity:' + alias + ':' + sid, row['sample'] == sample
                           and row['model_alias'] == alias and row['freeze_id'] == expected_freeze
                           and row['attempt'] == 1 and row['status'] in ('completed', 'failed'))
            preparations = [e for e in prepared_events if e['sample_id'] == sid]
            attempt = attempt_ledger(row, marker, [e for e in starts if e['sample_id'] == sid],
                [e for e in terminals if e['sample_id'] == sid], preparations,
                {'freeze_id': expected_freeze, 'frozen': frozen['frozen_at_unix'],
                 'freeze_event': freeze_events[0]['time_unix'], 'generation': generation_events[0]['time_unix'],
                 'opening': opened['time_unix']}, checks)
            if attempt['interrupted']:
                interruption_evidence.append({'sample_id': sid, **attempt})
            identity = inspect_generation_metadata(row, model, config['generation'], templates[model['model_id']], checks)
            if identity is not None:
                generation_identities[alias].append(identity)
            if 'prepared_methods' in row:
                state = row['prepared_methods']
                checks.exact('shared_prepared_book:' + alias + ':' + sid, state[VARIANTS[0]], state[VARIANTS[1]])
                state_hashes = {v: digest(state[v]) for v in VARIANTS}
                checks.equal('prepared_hashes:' + sid, state_hashes, row['prepared_method_sha256'])
                checks.equal('preparation_event_hashes:' + sid, state_hashes, preparations[0]['sha256'])
                expected_payload = payloads[sid]
                checks.require('paired_payload:' + alias + ':' + sid, row['synthetic_payload'] == expected_payload
                               and row['payload_sha256'] == hashlib.sha256(expected_payload.encode()).hexdigest())
                prepared = state[VARIANTS[0]]
                book = prepared['codebook']
                info = inspect_book(book, row['cover_text'], sample, config['method_config'], checks)
                checks.require('prepared_book_hash:' + sid, prepared['codebook_sha256'] == digest(book))
                checks.equal('prepared_capacity:' + sid, prepared['capacity_bits'], info['capacity_bits'])
                checks.equal('prepared_eligibility:' + sid, prepared['eligibility'], info['eligibility'])
                alignment = prepared['token_alignment']
                checks.require('alignment_counts:' + sid, all(type(alignment[k]) is int and alignment[k] >= 0
                               for k in ('candidate_pairs_considered', 'candidate_pairs_admitted', 'candidate_pairs_rejected'))
                               and alignment['candidate_pairs_considered'] == alignment['candidate_pairs_admitted'] + alignment['candidate_pairs_rejected'])
                for variant in VARIANTS:
                    output = row['variants'][variant]
                    checks.equal('retained_state_hash:' + sid, output['prepared_state_sha256'], state_hashes[variant])
                    if full_success(row, variant):
                        cost = inspect_success(row, variant, info, expected_payload, checks)
                        costs[alias].append({'sample_id': sid, 'variant': variant, **cost})
        for topic in topics:
            for genre in genres:
                selected_ids = {sample['id'] for sample in samples if sample['topic_id'] == topic and sample['genre_id'] == genre}
                prompt_identities = {(identity['rendered_prompt_sha256'], identity['input_token_ids_sha256'], identity['input_tokens'])
                                     for identity in generation_identities[alias] if identity['sample_id'] in selected_ids}
                checks.require('same_native_input_for_same_prompt:' + alias, len(prompt_identities) <= 1)
        all_records[alias] = rows
        item = evaluation['models'][alias]
        summaries[alias] = {v: reliability(rows, v) for v in VARIANTS}
        checks.equal('all_attempt_reliability:' + alias, summaries[alias], item['reliability'])
        for bits in (8, 16):
            values = {v: reliability([r for r in rows if r['sample']['payload_bits'] == bits], v) for v in VARIANTS}
            checks.equal('per_payload_reliability:' + alias, values, item['reliability_by_payload_bits'][str(bits)])
        checks.require('attempt_denominators:' + alias, item['planned_attempts'] == 16
                       and item['generated_covers'] == sum(bool(r.get('cover_text')) for r in rows))
        failure_list = [{'sample_id': r['sample']['id'], 'failure': r['failure'],
                         'variant_errors': {v: r.get('variants', {}).get(v, {}).get('error') for v in VARIANTS}}
                        for r in rows if r.get('failure') or any(r.get('variants', {}).get(v, {}).get('error') for v in VARIANTS)]
        checks.equal('failure_ledger:' + alias, failure_list, item['failures'])
        timelines[alias] = {'first_started': min(r['started_at'] for r in rows),
                            'last_finished': max(r['finished_at'] for r in rows),
                            'generation_completed': generation_events[0]['time_unix'],
                            'preserved_interruption_evidence': interruption_evidence,
                            'event_counts': dict(collections.Counter(e['event'] for e in model_events))}
    success_ids = {alias: [r['sample']['id'] for r in rows if all(full_success(r, v) for v in VARIANTS)]
                   for alias, rows in all_records.items()}
    matched = set.intersection(*(set(value) for value in success_ids.values()))
    checks.equal('matched_success_ids', sorted(matched), evaluation['matched_success_sample_ids'])
    attacks, expected_prediction_files = {}, set()
    for alias in aliases:
        checks.require('cohort_names:' + alias, set(evaluation['models'][alias]['cohorts']) == set(COHORTS))
        attacks[alias] = {}
        for cohort_name in COHORTS:
            chosen_ids = success_ids[alias] if cohort_name == COHORTS[0] else [sid for sid in success_ids[alias] if sid in matched]
            cohort = evaluation['models'][alias]['cohorts'][cohort_name]
            checks.require('cohort_identity:' + alias + ':' + cohort_name, cohort['pairs'] == len(chosen_ids)
                           and cohort['sample_ids'] == chosen_ids and set(cohort['attacks']) == expected_attacks)
            cohort_metrics = {}
            for name, attack in lock['attacks'].items():
                retained = cohort['attacks'][name]
                if not chosen_ids:
                    checks.equal('empty_cohort:' + alias + ':' + name, retained, {'status': 'no_common_success_pairs'})
                    continue
                path = run / 'predictions' / alias / cohort_name / (name + '.json')
                expected_prediction_files.add(path.relative_to(run).as_posix())
                predictions = read(path)
                expected_rows = {(sid, arm): (label, next(s['topic_id'] for s in samples if s['id'] == sid))
                                 for sid in chosen_ids for arm, label in (('cover', 0), ('stego', 1))}
                identities = {(r['sample_id'], r['arm']): (r['label'], r['topic_id']) for r in predictions}
                checks.require('prediction_identities:' + alias + ':' + name,
                               len(predictions) == len(expected_rows) and identities == expected_rows)
                metrics = prediction_metrics(predictions, attack['threshold'])
                checks.require('prediction_hash:' + alias + ':' + name, sha(path) == retained['predictions_sha256'])
                checks.require('retained_attack_settings:' + alias + ':' + name, retained['status'] == 'complete'
                               and retained['source_development_threshold'] == attack['threshold']
                               and retained['source_score_direction'] == attack['score_direction'])
                for key, value in metrics.items():
                    checks.equal('independent_metric:' + alias + ':' + name + ':' + key, value, retained[key])
                checks.require('prediction_decisions:' + alias + ':' + name, all(
                    r['prediction'] == int(r['score'] >= .5)
                    and r['prediction_at_source_dev_threshold'] == int(r['score'] >= attack['threshold']) for r in predictions))
                cohort_metrics[name] = {'pairs': len(chosen_ids), 'auc_score_pairs': len(chosen_ids) ** 2,
                                        'threshold': attack['threshold'], 'direction': attack['score_direction'], **metrics}
            attacks[alias][cohort_name] = {'pairs': len(chosen_ids), 'attacks': cohort_metrics}
    checks.require('exact_prediction_file_set', {p.relative_to(run).as_posix() for p in (run / 'predictions').rglob('*.json')} == expected_prediction_files)
    runtime = runtime_proofs(runtime_directory, config, opened['time_unix'], ended['time_unix'], timelines, checks)
    cost_summary = {}
    for alias, values in costs.items():
        books = [value for value in values if value['variant'] == 'fixed_canonical']
        cost_summary[alias] = {'shared_books': len(books), 'external_length_bits_per_message': 32}
        for field in ('key_json_bytes', 'key_zlib9_bytes', 'key_json_plus_length_bits', 'key_zlib9_plus_length_bits'):
            numbers = [value[field] for value in books]
            cost_summary[alias][field] = {'mean': statistics.mean(numbers) if numbers else None,
                                        'minimum': min(numbers) if numbers else None,
                                        'maximum': max(numbers) if numbers else None}
    final_hashes = {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    checks.equal('primary_unchanged_during_audit', initial_hashes, final_hashes)
    return {'schema': 'synsteg-crossmodel-independent-audit-v1', 'status': 'pass_in_declared_scope',
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'freeze_id': expected_freeze, 'source_lock_id': OLD_LOCK, 'audit_script_sha256': sha(__file__),
            'assertions': sum(checks.counts.values()), 'check_counts': dict(checks.counts),
            'primary_files_sha256': initial_hashes, 'resources': resources,
            'old_seven_source_files': source_proof['frozen_code'], 'runtime_exit_evidence': runtime,
            'attempts_by_model': {alias: len(rows) for alias, rows in all_records.items()},
            'retained_generation_identities': generation_identities,
            'reliability': summaries, 'success_sample_ids': success_ids, 'matched_success_sample_ids': sorted(matched),
            'full_book_communication_costs': costs, 'communication_cost_summary': cost_summary,
            'independent_prediction_metrics': attacks,
            'timeline': {'model_generation': timelines, 'opened_once': opened['time_unix'],
                         'evaluation_completed_once': ended['time_unix'],
                         'primary_events': dict(collections.Counter(e['event'] for e in ledger))},
            'primary_modified': False,
            'limitations': ['Pure standard-library audit of retained data; no model loading, inference, classifier refit, threshold tuning, or primary evaluation rerun.',
                'AUC is independently calculated from every positive/negative score pair, with half-credit for ties; saved classifier scores themselves are not regenerated.',
                'Independent decoding uses the full retained document-specific codebook and separate 32-bit length; the book reconstructs the original cover. This is not a seed-only receiver.',
                'Candidate grouping, selection, decoding and costs are checked; WordNet admission and neural quality scores are not independently regenerated or semantically certified.',
                'Retained model, revision, seed, native-template hash, sampling settings and token-count identities are checked. Prompt rendering, tokenization and generated token IDs are not independently regenerated.',
                'Repository sources and available declared resources are checked; missing external resources remain explicitly unverified. Frozen training/runtime environment and model weights are not recreated.',
                'Old Qwen-trained attacks measure transfer on conditional successful cohorts, not within-generator security. Small samples and procedural gates do not establish secrecy or operator blindness.']}


def markdown(report):
    lines = ['# Independent cross-generator transfer audit', '',
             'PASS within the declared audit scope; no model execution or primary data changes.', '',
             f"Verified assertions: {report['assertions']}. Freeze: `{report['freeze_id']}`.", '',
             '| Generator | Attempts | Canonical successes | Quality successes |', '|---|---:|---:|---:|']
    for alias, n in report['attempts_by_model'].items():
        values = report['reliability'][alias]
        lines.append(f"| {alias} | {n} | {values['fixed_canonical']['full_success_samples']} | {values['fixed_quality']['full_success_samples']} |")
    lines += ['', '| Generator / cohort / attack | Pairs | Accuracy | AUC | FPR | TPR |', '|---|---:|---:|---:|---:|---:|']
    for alias, cohorts in report['independent_prediction_metrics'].items():
        for name, cohort in cohorts.items():
            for attack, metrics in cohort['attacks'].items():
                lines.append(f"| {alias} / {name} / {attack} | {metrics['pairs']} | {metrics['accuracy']:.6f} | {metrics['auc']:.6f} | {metrics['fpr_at_development_threshold']:.6f} | {metrics['tpr_at_development_threshold']:.6f} |")
    resources = report['resources']
    lines += ['', '| Generator | Shared books | Mean JSON bytes | Mean zlib9 bytes |', '|---|---:|---:|---:|']
    for alias, cost in report['communication_cost_summary'].items():
        lines.append(f"| {alias} | {cost['shared_books']} | {cost['key_json_bytes']['mean']} | {cost['key_zlib9_bytes']['mean']} |")
    lines += ['', 'Add 32 external length bits to each serialized book. Each retained book reconstructs its original cover.']
    lines += ['', f"Declared resource files verified: {len(resources['verified_files'])}; unavailable external files: {len(resources['unavailable_external_files'])}.",
              'Runtime proof status: ' + report['runtime_exit_evidence']['status'] + '.', '']
    lines += ['- ' + limitation for limitation in report['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-freeze-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-directory', type=Path)
    parser.add_argument('--strict-resources', action='store_true')
    args = parser.parse_args()
    output, run = args.output.resolve(), args.run.resolve()
    if output == run or output.is_relative_to(run) or output.exists():
        parser.error('Output must be a new directory outside the primary run')
    report = audit_run(args.repo, run, args.expected_freeze_id, args.runtime_directory, args.strict_resources)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'audit.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'assertions': report['assertions'], 'output': str(output)}))


if __name__ == '__main__':
    main()
