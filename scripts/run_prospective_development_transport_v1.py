#!/usr/bin/env python3
"""Public dictionary transport on one immutable development snapshot.

Production requires an external Git-readback receipt and the separate activation
manifest. The feasibility protocol is never edited. No model dependencies.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'experiments/prospective-development-transport-v1/protocol.json'
ACTIVATION = 'experiments/prospective-development-transport-v1/activation.json'
PLAN_SHA = '1220950e061801e1aebeaf0c6773ef103618c2aa3c5689122b59daa3dde3f09b'
VERSION = 'prospective-development-transport-v1'
KEY = bytes(range(32))
IMPLEMENTATION_FILES = (
    'scripts/run_prospective_development_transport_v1.py',
    'scripts/audit_prospective_development_transport_v1.py',
    'tests/test_prospective_development_transport_v1.py',
    'tests/test_prospective_transport_audit_v1.py',
    'docs/prospective_development_transport_v1_execution_ru.md',
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b'\n')


def write_rows(path, rows):
    raw = b''.join(canonical(row) + b'\n' for row in rows)
    Path(path).write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))


def read_rows(path):
    return [json.loads(line) for line in gzip.decompress(Path(path).read_bytes()).splitlines()]


def safe_relative(value):
    path = Path(value)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe relative path')
    return path


def module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_identity():
    import cryptography
    from cryptography.hazmat.backends.openssl.backend import backend
    return {'python': sys.version, 'python_implementation': platform.python_implementation(),
            'platform': platform.platform(), 'cryptography': cryptography.__version__,
            'cryptography_openssl': backend.openssl_version_text()}


def dependencies(root, frozen_root):
    """Hash/import public resources only; never inspect or encode real covers."""
    raw = (root / PLAN).read_bytes()
    require(sha(raw) == PLAN_SHA, 'Feasibility protocol identity changed')
    plan = json.loads(raw)
    checked = {}
    for pin in plan['resource_pins']:
        path = frozen_root / safe_relative(pin['path'])
        require(path.is_file() and sha(path.read_bytes()) == pin['sha256'], 'Frozen dependency mismatch: ' + pin['path'])
        checked[pin['path']] = pin['sha256']
    for path, digest in [(plan['payloads']['map'], plan['payloads']['map_sha256']),
                         (plan['cover_cohort']['plan'], plan['cover_cohort']['plan_sha256'])]:
        require(sha((root / path).read_bytes()) == digest, 'Pinned input mismatch: ' + path)
    config = read_json(root / plan['cover_cohort']['plan'])['development_candidate']
    require(sha((root / config['source_prompts']).read_bytes()) == plan['cover_cohort']['source_prompts_sha256'], 'Source prompt file mismatch')
    inherited = read_json(frozen_root / 'experiments/matched_archive_v1/protocol.json')
    require(plan['arms']['public_common'] == inherited['public_common'], 'Public parameter drift')
    # Reuse only the frozen matched runner's pure serialization helpers by AST;
    # importing its archive/generation code is unnecessary.
    wanted = {'canonical', 'sha', 'write_json', 'write_rows', 'read_rows'}
    tree = ast.parse((frozen_root / 'scripts/run_matched_archive_v1.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    require({n.name for n in selected} == wanted, 'Missing matched pure helper')
    helper_namespace = dict(json=json, hashlib=hashlib, gzip=gzip)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<frozen-matched-pure-helpers>', 'exec'), helper_namespace)
    require(helper_namespace['canonical']({'x': 'é'}) == canonical({'x': 'é'}), 'Serialization drift')
    shared = module_from_path(frozen_root / 'src/synsteg_audit/shared_dictionary.py', '_prospective_frozen_shared')
    dictionaries, resources = {}, []
    for entry in plan['resource_accounting']['public_installation']:
        raw = (frozen_root / entry['path']).read_bytes()
        dictionary = shared.PublicDictionary.from_dict(json.loads(raw))
        require(dictionary.sha256 == entry['dictionary_sha256'], 'Dictionary semantic identity mismatch')
        require(len(raw) == entry['one_time_file_bytes'] and len(zlib.compress(raw, 9)) == entry['one_time_file_zlib9_bytes'], 'Dictionary byte accounting mismatch')
        dictionaries[entry['id']] = dictionary
        resources.append(dict(entry))
    payload = read_json(root / plan['payloads']['map'])
    ids = [r['record_id'] for r in config['selected_record_metadata']]
    require(len(ids) == len(set(ids)) == 100 and len(payload['rows']) == 300, 'Wrong source/payload denominator')
    for order, rid in enumerate(ids, 1):
        rows = [r for r in payload['rows'] if r['record_id'] == rid]
        require(len(rows) == 3 and {r['primary_payload_length'] for r in rows} == {128, 256, 512}, 'Payload strata differ')
        for row in rows:
            bits = row['payload_bits']
            require(row['source_order'] == order and len(bits) == row['primary_payload_length'] and set(bits) <= {'0', '1'}, 'Payload mapping malformed')
            require(sha(bits.encode()) == row['payload_sha256'] and row['secondary16_bits'] == bits[:16] and sha(bits[:16].encode()) == row['secondary16_sha256'], 'Payload hash mismatch')
    return plan, config, payload, shared, dictionaries, resources, checked, helper_namespace


def activation_preflight(root, frozen_root, receipt_path=None):
    dependency = dependencies(root, frozen_root)
    activation = read_json(root / ACTIVATION)
    require(activation['experiment_id'] == VERSION and activation['feasibility_protocol_sha256'] == PLAN_SHA, 'Activation identity mismatch')
    require(activation['runtime'] == runtime_identity(), 'Frozen runtime mismatch')
    require(set(activation['implementation_files_sha256']) == set(IMPLEMENTATION_FILES), 'Incomplete implementation freeze map')
    for relative, digest in activation['implementation_files_sha256'].items():
        require(sha((root / safe_relative(relative)).read_bytes()) == digest, 'Implementation file mismatch: ' + relative)
    if receipt_path is not None:
        receipt = read_json(receipt_path)
        require(receipt.get('remote_readback_verified') is True, 'External Git readback not verified')
        for key in ('implementation_freeze_commit', 'checkpoint_commit'):
            require(re.fullmatch('[0-9a-f]{40}', receipt.get(key, '')) is not None, 'Missing Git commit: ' + key)
        expected = {**activation['implementation_files_sha256'], ACTIVATION: sha((root / ACTIVATION).read_bytes()),
                    PLAN: PLAN_SHA, dependency[0]['payloads']['map']: dependency[0]['payloads']['map_sha256']}
        cp_path = activation['target']['checkpoint']
        expected[cp_path] = sha((root / cp_path).read_bytes())
        require(all(receipt.get('verified_files_sha256', {}).get(k) == v for k, v in expected.items()), 'Git receipt missing exact local file identities')
        require(receipt.get('observed_completed_covers_at_implementation_freeze') is not None, 'Prior-data visibility must be disclosed')
        observed = activation['observed_completed_covers_at_implementation_freeze']
        require(type(observed) is int and 8 <= observed <= 300 and receipt['observed_completed_covers_at_implementation_freeze'] == observed, 'Activation/receipt prior visibility differs')
    return dependency, activation


def snapshot_records(root, config, activation):
    target = activation['target']
    cp_path = root / safe_relative(target['checkpoint'])
    require(cp_path.name == 'after-checkpoint.json' and cp_path.parent.name == target['batch'], 'Wrong immutable batch path')
    cp_raw = cp_path.read_bytes()
    cp = json.loads(cp_raw)
    require(cp['status'] == 'verified' and not cp['consistency_issues'] and cp['planned_model_record_pairs'] == 300, 'Checkpoint not verified/full300')
    require(cp['completed_generations'] == target['completed_covers'] and cp['counts'] == target['cover_counts'], 'Target first30 checkpoint state differs')
    require(cp['model_counts'] == target['model_counts'] and not cp['closed_test_opened'] and cp['paid_calls'] == 0, 'Checkpoint scope differs')
    snapshot = cp_path.parent / 'after'
    for rel, digest in cp['run_files_sha256'].items():
        require(sha((snapshot / safe_relative(rel)).read_bytes()) == digest, 'Immutable snapshot SHA mismatch: ' + rel)
    expected = {(m['alias'], r['record_id']): (m, r, i) for m in config['local_models']
                for i, r in enumerate(config['selected_record_metadata'], 1)}
    require(len(cp['records']) == 300 and len({(r['model_alias'], r['record_id']) for r in cp['records']}) == 300, 'Duplicate/omitted checkpoint records')
    records = []
    for r in cp['records']:
        key = (r['model_alias'], r['record_id'])
        require(key in expected, 'Unexpected checkpoint record')
        model, meta, order = expected[key]
        require(all(r[k] == v for k, v in meta.items()) and r['model'] == model['model_id'] and r['revision'] == model['revision'], 'Checkpoint metadata drift')
        require(r['status'] in ('completed', 'not_attempted', 'failed', 'unknown'), 'Unsupported cover state')
        row = dict(r, source_order=order, cover_text=None, attempt_sha256=None)
        if r['status'] == 'completed':
            rel = f"runs/large-campaign-development-v2/models/{r['model_alias']}/attempts/{r['identity']}.json"
            raw = (snapshot / rel).read_bytes()
            require(sha(raw) == cp['run_files_sha256'][rel], 'Unregistered attempt')
            attempt = json.loads(raw)
            text = attempt['result']['cover_text']
            require(attempt['status'] == 'completed' and attempt['identity'] == r['identity'] and attempt['record'] == meta, 'Attempt identity mismatch')
            require(sha(text.encode()) == r['cover_sha256'] and len(text.split()) == r['whitespace_words'], 'Cover content mismatch')
            row.update(cover_text=text, attempt_sha256=sha(raw))
        records.append(row)
    completed = [(r['model_alias'], r['source_order']) for r in records if r['status'] == 'completed']
    require(completed == [('tinyllama11', n) for n in range(1, 31)], 'Must use unchanged first30 TinyLlama covers')
    return records, cp, cp_raw, snapshot


def base_cell(record, payload, grid, arm, public):
    bits = payload['payload_bits'] if grid == 'primary' else payload['secondary16_bits']
    row_id = sha(canonical([record['model_alias'], record['record_id'], payload['primary_payload_length'], grid, arm]))
    complete = record['status'] == 'completed'
    row = {'row_id': row_id, 'model_alias': record['model_alias'], 'record_id': record['record_id'],
           'source_group_id': record['source_group_id'], 'source_order': record['source_order'],
           'identity': record['identity'], 'attempt_sha256': record['attempt_sha256'],
           'cover_sha256': record.get('cover_sha256') if complete else None,
           'cover_words': len(record['cover_text'].split()) if complete else None,
           'parent_payload_length': payload['primary_payload_length'], 'grid': grid, 'arm': arm,
           'requested_payload_bits': len(bits), 'requested_payload_bytes': len(bits) // 8,
           'payload_sha256': sha(bits.encode()), 'cover_status': record['status'],
           'resource_status': 'ready' if public else 'BLOCKED_RESOURCE_OR_STAGE_NOT_RUN',
           'execution_status': 'not_run_cover_unavailable' if not complete else 'not_run_resource_blocked',
           'capacity_bits': None, 'delivered_payload_bits': None, 'useful_decoded_payload_bytes': None,
           'book_json_bytes': None, 'book_zlib9_bytes': None, 'per_text_map_bytes': None,
           'max_payload_bytes': None, 'can_frame_empty': None, 'can_send_one_byte': None,
           'raw_payload_fits': None, 'frame_bits': None, 'padding_bits': None,
           'used_positions': None, 'changed_positions': None, 'normalization_sha256': None,
           'stego_sha256': None, 'stego_words': None, 'decoded_bits': None, 'decoded_sha256': None,
           'full_exact': None, 'fresh_receiver_exact': None, 'error': None,
           'encoder_called': None,
           'delivered_bits_per_source_word': None, 'delivered_bits_per_stego_word': None}
    return row, bits


def evaluate(records, payload_map, plan, shared, dictionaries):
    """Pure evaluator: called on real inputs only after external production gate.

    Tests call this function with declared synthetic text fixtures, never real
    development covers. It does not run the receiver or write files.
    """
    payloads = {(r['record_id'], r['primary_payload_length']): r for r in payload_map['rows']}
    cells, jobs, expected = [], [], {}
    fixed = [a['id'] for a in plan['arms']['fixed']]
    for record in records:
        for parent_length in (128, 256, 512):
            payload = payloads[(record['record_id'], parent_length)]
            for grid in ('primary', 'secondary16'):
                arms = (['original_native'] if grid == 'primary' else []) + fixed + plan['arms']['public_arms']
                for arm in arms:
                    public = arm.startswith('public_')
                    row, bits = base_cell(record, payload, grid, arm, public)
                    if not public:
                        row.update(resource_contract='missing_author_generated_per_cover_candidates',
                                   frame_overhead_bits=None, external_length_bits=None,
                                   minimum_nonempty_frame_bits=None, per_receiver_key_bytes=None,
                                   dictionary_sha256=None, protocol_sha256=None)
                    else:
                        _, version, framing = arm.split('_', 2)
                        protocol = shared.Protocol(min_group_size=1, max_bits_per_slot=2, framing=framing)
                        dictionary = dictionaries[version]
                        row.update(resource_contract='stego+32byte_key+public_dictionary_and_protocol',
                                   frame_overhead_bits=protocol.frame_overhead_bits, external_length_bits=0,
                                   minimum_nonempty_frame_bits=protocol.frame_overhead_bits + 8,
                                   per_receiver_key_bytes=32, dictionary_sha256=dictionary.sha256,
                                   protocol_sha256=protocol.sha256)
                        if record['status'] == 'completed':
                            row.update(execution_status='implementation_error', delivered_payload_bits=0,
                                       useful_decoded_payload_bytes=0, full_exact=False, fresh_receiver_exact=False,
                                       book_json_bytes=0, book_zlib9_bytes=0, per_text_map_bytes=0, encoder_called=False)
                            try:
                                analysis = shared.analyze_text(record['cover_text'], dictionary, protocol)
                                row.update(capacity_bits=analysis.capacity_bits, max_payload_bytes=analysis.max_payload_bytes,
                                           can_frame_empty=analysis.capacity_bits >= protocol.frame_overhead_bits,
                                           can_send_one_byte=analysis.capacity_bits >= protocol.frame_overhead_bits + 8,
                                           raw_payload_fits=analysis.capacity_bits >= len(bits),
                                           normalization_sha256=analysis.normalization_sha256)
                                raw = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
                                row['encoder_called'] = True
                                encoded = shared.encode(record['cover_text'], raw, KEY, dictionary, protocol, selector='canonical')
                                row.update(execution_status='awaiting_receiver', frame_bits=encoded.frame_bits,
                                           padding_bits=encoded.padding_bits, used_positions=encoded.consumed_slots,
                                           changed_positions=encoded.changed_slots, stego_sha256=sha(encoded.stego_text.encode()),
                                           stego_words=len(encoded.stego_text.split()))
                                jobs.append({'row_id': row['row_id'], 'kind': 'public', 'dictionary_version': version,
                                             'stego_text': encoded.stego_text, 'fixture_key_hex': KEY.hex(),
                                             'protocol': {'min_group_size': 1, 'max_bits_per_slot': 2, 'framing': framing}})
                                expected[row['row_id']] = bits
                            except shared.CapacityError as exc:
                                row.update(execution_status='insufficient_capacity', error=str(exc))
                            except Exception as exc:
                                row.update(execution_status='implementation_error', error=f'{type(exc).__name__}: {exc}')
                    cells.append(row)
    require(len(cells) == len(records) * 51 and len({r['row_id'] for r in cells}) == len(cells), 'Cell grid incomplete/duplicated')
    return cells, jobs, expected


def receiver(directory):
    """Receiver sees only public resources and jobs; no cover or expected bits."""
    shared = module_from_path(directory / 'shared_dictionary.py', '_prospective_recipient_shared')
    dictionaries = {v: shared.PublicDictionary.from_dict(read_json(directory / f'dictionary_{v}.json')) for v in ('v1', 'v2')}
    outcomes = []
    for job in read_rows(directory / 'receiver_jobs.jsonl.gz'):
        require(set(job) == {'row_id', 'kind', 'dictionary_version', 'stego_text', 'fixture_key_hex', 'protocol'}, 'Unpermitted receiver field')
        require(job['kind'] == 'public', 'Unsupported receiver job')
        result = {'row_id': job['row_id'], 'status': 'error', 'decoded_bits': None, 'error': None}
        try:
            raw = shared.decode(job['stego_text'], bytes.fromhex(job['fixture_key_hex']), dictionaries[job['dictionary_version']], shared.Protocol(**job['protocol']))
            result.update(status='decoded', decoded_bits=''.join(f'{b:08b}' for b in raw))
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        outcomes.append(result)
    write_rows(directory / 'receiver_results.jsonl.gz', outcomes)


def finalize(cells, expected, outcomes):
    decoded = {r['row_id']: r for r in outcomes}
    require(len(decoded) == len(outcomes) and set(decoded) == set(expected), 'Receiver membership mismatch')
    for row in cells:
        if row['row_id'] in expected:
            outcome = decoded[row['row_id']]
            exact = outcome['status'] == 'decoded' and outcome['decoded_bits'] == expected[row['row_id']]
            value = outcome['decoded_bits']
            row.update(execution_status='full_exact' if exact else 'decoder_mismatch', full_exact=exact,
                       fresh_receiver_exact=exact, delivered_payload_bits=len(expected[row['row_id']]) if exact else 0,
                       useful_decoded_payload_bytes=len(expected[row['row_id']]) // 8 if exact else 0,
                       decoded_bits=len(value) if value is not None else None,
                       decoded_sha256=sha(value.encode()) if value is not None else None,
                       error=None if exact else (outcome['error'] or 'Decoded payload differs'))
        if row['delivered_payload_bits'] is not None:
            row['delivered_bits_per_source_word'] = row['delivered_payload_bits'] / row['cover_words'] if row['cover_words'] else None
            row['delivered_bits_per_stego_word'] = row['delivered_payload_bits'] / row['stego_words'] if row['stego_words'] else None
    require(not any(r['execution_status'] == 'awaiting_receiver' for r in cells), 'Unfinished receiver rows')


def summarize(cells, records, resource_entries):
    groups = defaultdict(list)
    for row in cells:
        groups[(row['grid'], row['arm'], row['parent_payload_length'], 'all')].append(row)
        groups[(row['grid'], row['arm'], row['parent_payload_length'], row['model_alias'])].append(row)
    result = []
    for (grid, arm, parent, model), rows in sorted(groups.items()):
        measured = [r for r in rows if r['delivered_payload_bits'] is not None]
        caps = [r['capacity_bits'] for r in measured if r['capacity_bits'] is not None]
        result.append({'grid': grid, 'arm': arm, 'parent_payload_length': parent, 'model_alias': model,
                       'planned_cells': len(rows), 'executed_cells': len(measured),
                       'cover_status_counts': dict(Counter(r['cover_status'] for r in rows)),
                       'resource_status_counts': dict(Counter(r['resource_status'] for r in rows)),
                       'execution_status_counts': dict(Counter(r['execution_status'] for r in rows)),
                       'full_exact': sum(r['full_exact'] is True for r in rows),
                       'requested_bits_all_planned': sum(r['requested_payload_bits'] for r in rows),
                       'requested_bits_executed': sum(r['requested_payload_bits'] for r in measured),
                       'confirmed_delivered_bits': sum(r['delivered_payload_bits'] for r in measured),
                       'confirmed_useful_decoded_bytes': sum(r['useful_decoded_payload_bytes'] for r in measured),
                       'unmeasured_cells': len(rows) - len(measured),
                       'zero_capacity_executed': sum(r['capacity_bits'] == 0 for r in measured),
                       'mean_capacity_bits_executed': statistics.mean(caps) if caps else None,
                       'median_capacity_bits_executed': statistics.median(caps) if caps else None})
    completed = sum(r['status'] == 'completed' for r in records)
    resources = {'public_dictionaries': [dict(x,
        installation_bytes_per_completed_cover=x['one_time_file_bytes'] / completed if completed else None,
        installation_zlib9_bytes_per_completed_cover=x['one_time_file_zlib9_bytes'] / completed if completed else None,
        installation_bytes_per_planned_cover=x['one_time_file_bytes'] / 300,
        amortization_completed_cover_denominator=completed,
        amortization_planned_cover_denominator=300) for x in resource_entries],
        'per_public_receiver_secret_key_bytes': 32, 'fixture_key_hex': KEY.hex(), 'fixture_key_is_public': True,
        'public_per_text_map_bytes': 0, 'blocked_original_fixed_map_bytes': None,
        'dictionary_installation_shared_across_framings_payloads_and_cells': True,
        'key_reused_across_fixture_cells_not_new_key_per_cell': True,
        'protocol_identifiers_and_transport_packaging_excluded_from_byte_counts': True}
    summary = {'experiment_id': VERSION, 'planned_model_record_pairs': 300, 'source_groups': 100,
               'completed_covers': completed, 'cover_status_counts': dict(Counter(r['status'] for r in records)),
               'total_cells': len(cells), 'primary_cells': sum(r['grid'] == 'primary' for r in cells),
               'secondary16_cells': sum(r['grid'] == 'secondary16' for r in cells),
               'resource_status_counts': dict(Counter(r['resource_status'] for r in cells)),
               'execution_status_counts': dict(Counter(r['execution_status'] for r in cells)), 'groups': result,
               'model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False,
               'limitations': ['Missing and blocked are unmeasured, not zero capacity or capacity failure.',
                   'Payload/arm cells are repeated mechanical conditions on100sources, not independent texts.',
                   'Known first8covers/all100sources preceded the analysis plan; actual prior visibility appears in Git receipt.',
                   'Public dictionary transport does not certify meaning, factuality, security or original full pipeline replication.']}
    return summary, resources


def run(root, frozen_root, output, receipt_path):
    require(not output.exists(), 'Output exists; preserve completed or partial stage')
    dependency, activation = activation_preflight(root, frozen_root, receipt_path)
    plan, config, payload, shared, dictionaries, resource_entries, resource_hashes, helpers = dependency
    require(output == (root / activation['target']['output']).resolve(), 'Output differs from frozen milestone target')
    records, cp, cp_raw, snapshot = snapshot_records(root, config, activation)
    receipt = read_json(receipt_path)
    require(len(records) == 300, 'Wrong denominator')
    # All validation is finished before the exclusive output directory is made.
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write_json(output / 'status.json', {'status': 'running', 'experiment_id': VERSION})
    (output / 'input_checkpoint.json').write_bytes(cp_raw)
    shutil.copyfile(root / plan['cover_cohort']['plan'], output / 'input_config.json')
    shutil.copyfile(root / plan['payloads']['map'], output / 'payload-map.json')
    shutil.copyfile(root / PLAN, output / 'feasibility-protocol.json')
    shutil.copyfile(root / ACTIVATION, output / 'activation.json')
    shutil.copyfile(receipt_path, output / 'git-readback-receipt.json')
    for rel in cp['run_files_sha256']:
        destination = output / 'input_snapshot' / safe_relative(rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / rel, destination)
    helpers['write_rows'](output / 'records.jsonl.gz', records)
    cells, jobs, expected = evaluate(records, payload, plan, shared, dictionaries)
    helpers['write_rows'](output / 'cells_before_receiver.jsonl.gz', cells)
    recipient = output / 'recipient'
    recipient.mkdir()
    shutil.copyfile(frozen_root / 'src/synsteg_audit/shared_dictionary.py', recipient / 'shared_dictionary.py')
    for version in ('v1', 'v2'):
        shutil.copyfile(frozen_root / f'data/shared_dictionary_recovery_{version}.json', recipient / f'dictionary_{version}.json')
    helpers['write_rows'](recipient / 'receiver_jobs.jsonl.gz', jobs)
    process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--receiver', str(recipient.resolve())],
                             cwd=recipient, capture_output=True, text=True, check=False)
    write_json(output / 'receiver_process.json', {'returncode': process.returncode, 'stdout': process.stdout,
               'stderr': process.stderr, 'job_count': len(jobs), 'payload_or_cover_argument': False,
               'receiver_cwd': 'recipient', 'receiver_resource_files': sorted(p.name for p in recipient.iterdir())})
    require(process.returncode == 0, 'Receiver process failed; partial files preserved')
    finalize(cells, expected, read_rows(recipient / 'receiver_results.jsonl.gz'))
    helpers['write_rows'](output / 'cells.jsonl.gz', cells)
    summary, resources = summarize(cells, records, resource_entries)
    write_json(output / 'summary.json', summary)
    write_json(output / 'resources.json', resources)
    execution = {'experiment_id': VERSION, 'activation_sha256': sha((root / ACTIVATION).read_bytes()),
        'feasibility_protocol_sha256': PLAN_SHA, 'payload_map_sha256': plan['payloads']['map_sha256'],
        'config_sha256': plan['cover_cohort']['plan_sha256'], 'checkpoint_sha256': sha(cp_raw),
        'source_checkpoint': activation['target']['checkpoint'], 'runtime': runtime_identity(),
        'implementation_freeze_commit': receipt['implementation_freeze_commit'], 'checkpoint_commit': receipt['checkpoint_commit'],
        'observed_completed_covers_at_implementation_freeze': receipt['observed_completed_covers_at_implementation_freeze'],
        'resource_files_sha256': resource_hashes, 'model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False,
        'encoder_calls': sum(r['encoder_called'] is True for r in cells),
        'receiver_jobs': len(jobs), 'wall_seconds': time.monotonic() - start,
        'fresh_receiver_has_cover_or_payload': False, 'semantics_or_security_inferred': False}
    write_json(output / 'execution.json', execution)
    require((root / activation['target']['checkpoint']).read_bytes() == cp_raw, 'Source checkpoint changed')
    for rel, digest in cp['run_files_sha256'].items():
        require(sha((snapshot / rel).read_bytes()) == digest, 'Source snapshot changed during evaluation')
    status = {'status': 'completed_pending_independent_audit', 'experiment_id': VERSION,
              'completed_covers': len([r for r in records if r['status'] == 'completed']),
              'full_denominator': 300, 'total_cells': len(cells),
              'unexpected_errors': sum(r['execution_status'] in ('decoder_mismatch', 'implementation_error') for r in cells),
              'encoder_calls': execution['encoder_calls'], 'receiver_jobs': len(jobs),
              'model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False}
    write_json(output / 'status.json', status)
    hashes = {str(p.relative_to(output)): sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'output_manifest.json', hashes)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--frozen-root', type=Path, default=ROOT / 'runtime/prospective-development-transport-v1/frozen')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--git-receipt', type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--receiver', type=Path)
    args = parser.parse_args()
    if args.receiver:
        receiver(args.receiver.resolve())
    elif args.preflight:
        dep, activation = activation_preflight(args.root, args.frozen_root)
        print(json.dumps({'status': 'pass', 'resource_files': len(dep[6]), 'runtime': runtime_identity(),
                          'actual_cover_encode_decode_calls': 0, 'target': activation['target']}, indent=2))
    else:
        if not args.output or not args.git_receipt:
            parser.error('Production requires --output and --git-receipt; no implicit activation')
        print(json.dumps(run(args.root.resolve(), args.frozen_root.resolve(), args.output.resolve(), args.git_receipt.resolve())))


if __name__ == '__main__':
    main()
