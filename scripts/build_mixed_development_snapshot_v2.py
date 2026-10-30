#!/usr/bin/env python3
"""Stage exact Tiny92..100/Qwen1+Qwen2..100 evidence; no inference/network/Git.

A draft is NOT a remote audit. The caller must verify its entire census and
receipt map against actual Git before enabling the two explicit readback flags.
"""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import tempfile

ID = 'mixed-development-snapshot-builder-v2'
PREDECESSOR_BUILDER_SHA = '0a5f1929f665b3ebd8d46742b888b764d8b55af5a514d137cbf3a0db1dc1a468'
MERGER_SHA = '9640117b1806b3a6b86774bd754800349ad599b8a2d276540a1cf4f63c60dd03'
TINY_ROOT = 'runs/baseline-single-record-recovery-v2'
QWEN_ROOT = 'runs/development-other-models-single-record-v1'
QWEN_SEQUENTIAL_ROOT = 'runs/development-qwen-sequential-v1'
ROOTS = [TINY_ROOT, QWEN_ROOT, QWEN_SEQUENTIAL_ROOT]
TINY_CONTEXT = 'results/large-campaign-development-v2/preflight/all-100-context-checks.json'
TINY_CONTEXT_SHA = '6788c9bb54df452b9f6cc1788b6de7d653a4c81577965ebc59a11f3976eef4fd'
TINY_CONFIG = 'runtime/large-campaign-development-v2/hf/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json'
TINY_CONFIG_SHA = '486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9'
TINY_RUNTIME_SHA = 'b9902f9a50d45933b0797788814ce9793347197c2c4b52bd517a0e0d8427f03a'
QWEN_PRE = 'results/development-other-models-single-record-v1/qwen05-preflight'
QWEN_FILES = [QWEN_PRE + '/' + n for n in ['model-contract.json', 'runtime.json', 'config.json', 'all100-context.json']]


def merger_module(path):
    import hashlib
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MERGER_SHA:
        raise ValueError('frozen merger hash differs')
    spec = importlib.util.spec_from_file_location('verified_full300_merger_v2', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def new_output(m, root, output):
    for parent in (output, *output.parents):
        m.require(not parent.is_symlink(), 'symlink output/ancestor refused')
    m.require(not output.exists() and not output.resolve().is_relative_to(Path(root).resolve()),
              'output must be new and outside source root')


def publish_new_directory(source, output):
    """Linux atomic no-replace publication; fail closed if unavailable."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise RuntimeError('atomic no-replace directory publication requires Linux renameat2')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, bytes(source), -100, bytes(output), 1) != 0:
        raise OSError(ctypes.get_errno(), 'atomic no-replace snapshot publication failed')


def bound_read(m, root, path, expected=None):
    relative = m.relative(path)
    actual = root / relative
    for parent in (actual, *actual.parents):
        m.require(not parent.is_symlink(), 'symlink source refused')
    data = actual.read_bytes()
    if expected is not None:
        m.require(m.sha(data) == expected, 'source hash differs: ' + path)
    return data


def census(m, root):
    files, operational = {}, {}
    for prefix in ROOTS:
        directory = root / prefix
        for parent in (directory, *directory.parents):
            m.require(not parent.is_symlink(), 'symlink census root/ancestor refused')
        if not directory.exists():
            continue
        for relative, digest in m.inventory(directory).items():
            path = prefix + '/' + relative
            (operational if relative == 'supervisor.lock' else files)[path] = digest
    return files, operational


def shard_identity(m, prefix):
    for order in range(92, 101):
        if prefix == TINY_ROOT + f'/order{order:03d}':
            return 'tinyllama11', order
    if prefix == QWEN_ROOT + '/qwen05/order001':
        return 'qwen05', 1
    for order in range(2, 101):
        if prefix == QWEN_SEQUENTIAL_ROOT + f'/order{order:03d}':
            return 'qwen05', order
    raise ValueError('unsupported shard outside Tiny92..100/Qwen1+orderedQwen2..100: ' + prefix)


def draft_request(m, root, snapshot_ref):
    """Construct an explicitly UNVERIFIED draft from current local bytes."""
    files, _ = census(m, root)
    prefixes = sorted(p.rsplit('/', 1)[0] for p in files if p.endswith('/reservation.json'))
    receipts = {}
    for prefix in prefixes:
        shard_identity(m, prefix)
        path = prefix + '/git-preservation-receipt.json'
        receipt = m.parse(bound_read(m, root, path))
        receipts[prefix] = {'commit': receipt.get('commit'), 'sha256': files[path]}
    qwen = None
    if any(shard_identity(m, prefix)[0] == 'qwen05' for prefix in prefixes):
        qwen = {'commit': None, 'files_sha256': {p: m.sha(bound_read(m, root, p)) for p in QWEN_FILES}}
    return {'schema': ID + '-request', 'snapshot_ref': snapshot_ref,
            'primary_receipts': receipts, 'qwen_preflight': qwen,
            'remote_census': {'snapshot_ref': snapshot_ref, 'readback_verified': False,
                              'branches_and_prs_checked': False, 'files_sha256': files}}


def tiny_context(m, raw, sources, config_sha, limit):
    value = m.parse(raw)
    m.require(value.get('schema') == 'synsteg-development-context-checks-v2'
              and (value.get('model'), value.get('revision')) == m.MODELS['tinyllama11']
              and value.get('prompts_sha256') == m.BASE_PINS['source']
              and type(value.get('denominator')) is int and value['denominator'] == 100,
              'Tiny context identity differs')
    rows = value.get('rows')
    m.require(isinstance(rows, list) and len(rows) == 100, 'Tiny context denominator differs')
    records = []
    for n, (row, source) in enumerate(zip(rows, sources), 1):
        m.require(type(row.get('source_order')) is int and row['source_order'] == n, 'Tiny context order differs')
        for key in ['record_id', 'source_group_id', 'target_words', 'prompt_sha256']:
            m.require(row.get(key) == source[key] and type(row.get(key)) is type(source[key]), 'Tiny context source metadata differs')
        count = m.number(row.get('input_tokens'), 'Tiny input tokens', 1)
        m.require(type(row.get('max_output_tokens')) is int and row['max_output_tokens'] == 1024
                  and type(row.get('model_context_tokens')) is int and row['model_context_tokens'] == limit
                  and type(row.get('remaining_after_requested_output')) is int
                  and row['remaining_after_requested_output'] == limit - count - 1024
                  and row['context_fits'] is True and count + 1024 <= limit,
                  'Tiny context arithmetic differs')
        m.require(isinstance(row.get('chat_template_render_sha256'), str)
                  and m.HEX64.fullmatch(row['chat_template_render_sha256']), 'Tiny template render hash absent')
        records.append({k: row[k] for k in ['source_order', 'record_id', 'prompt_sha256', 'input_tokens']})
    return {'schema': m.ID + '-context-report', 'model_id': m.MODELS['tinyllama11'][0],
            'revision': m.MODELS['tinyllama11'][1], 'source_sha256': m.BASE_PINS['source'],
            'model_config_sha256': config_sha, 'model_context_limit': limit, 'records': records}


def stage(m, root, request):
    root = Path(root)
    m.exact_keys(request, ['schema', 'snapshot_ref', 'primary_receipts', 'qwen_preflight', 'remote_census'], 'builder request')
    m.require(request['schema'] == ID + '-request', 'wrong builder request')
    snapshot = request['snapshot_ref']
    m.require(isinstance(snapshot, str) and m.HEX40.fullmatch(snapshot), 'invalid snapshot ref')
    remote = request['remote_census']
    m.exact_keys(remote, ['snapshot_ref', 'readback_verified', 'branches_and_prs_checked', 'files_sha256'], 'remote census')
    m.require(remote['snapshot_ref'] == snapshot and remote['readback_verified'] is True
              and remote['branches_and_prs_checked'] is True, 'draft/unverified remote census')
    initial, operational = census(m, root)
    m.require(initial == remote['files_sha256'], 'local/remote new-run census differs')
    receipts = request['primary_receipts']
    m.require(isinstance(receipts, dict) and bool(receipts), 'no supplied primary receipts')
    discovered = {p.rsplit('/', 1)[0] for p in initial if p.endswith('/reservation.json')}
    m.require(set(receipts) == discovered, 'orphan/missing reservation in receipt map')
    has_successor = any(p.startswith(QWEN_SEQUENTIAL_ROOT + '/') for p in discovered)
    m.require(not has_successor or QWEN_ROOT + '/qwen05/order001' in discovered,
              'Qwen successor evidence requires preserved original firstQwen1 in the same census')
    payloads, inputs, contracts, shards = {}, {}, dict.fromkeys(m.MODELS), []
    for role, path in m.BASE_FILES.items():
        payloads[path] = bound_read(m, root, path, m.BASE_PINS[role]); inputs[path] = m.BASE_PINS[role]
    sources = [m.parse(row) for row in payloads[m.BASE_FILES['source']].splitlines() if row.strip()]
    raw_paths = set()
    tiny_prefixes = []
    for prefix, binding in sorted(receipts.items()):
        alias, order = shard_identity(m, prefix)
        m.exact_keys(binding, ['commit', 'sha256'], 'primary receipt binding')
        m.require(isinstance(binding['commit'], str) and m.HEX40.fullmatch(binding['commit']), 'invalid primary commit')
        receipt_path = prefix + '/git-preservation-receipt.json'
        receipt_raw = bound_read(m, root, receipt_path, binding['sha256'])
        receipt = m.parse(receipt_raw)
        m.require(receipt.get('readback_verified') is True and receipt.get('commit') == binding['commit'], 'primary receipt differs')
        actual = {p[len(prefix)+1:]: digest for p, digest in initial.items() if p.startswith(prefix + '/')}
        m.require('process.json' in actual, 'terminal process not preserved yet')
        m.require(receipt.get('files_sha256') == {p: digest for p, digest in actual.items() if p != 'git-preservation-receipt.json'},
                  'primary receipt exact raw inventory differs')
        for relative, digest in actual.items():
            path = prefix + '/' + relative
            payloads[path] = bound_read(m, root, path, digest); inputs[path] = digest; raw_paths.add(path)
        shards.append({'model_alias': alias, 'source_order': order, 'path': prefix})
        if alias == 'tinyllama11': tiny_prefixes.append(prefix)
    m.require(raw_paths == set(initial), 'orphan raw file outside indexed shards')
    if tiny_prefixes:
        context_raw = bound_read(m, root, TINY_CONTEXT, TINY_CONTEXT_SHA)
        config_raw = bound_read(m, root, TINY_CONFIG, TINY_CONFIG_SHA)
        limit = m.number(m.parse(config_raw).get('max_position_embeddings'), 'Tiny context limit', 1)
        runtime_path = tiny_prefixes[0] + '/run/runtime.json'
        runtime_raw = bound_read(m, root, runtime_path, TINY_RUNTIME_SHA)
        context = tiny_context(m, context_raw, sources, TINY_CONFIG_SHA, limit)
        base = 'results/' + ID + '/derived-tiny-contract'
        cp, rp, xp, mp = [base + '/' + n for n in ['config.json', 'runtime.json', 'all100-context.json', 'model-contract.json']]
        payloads[cp], payloads[rp], payloads[xp] = config_raw, runtime_raw, encoded(context)
        payloads[mp] = encoded({'schema': m.ID + '-model-contract', 'model_alias': 'tinyllama11',
                              'model_id': m.MODELS['tinyllama11'][0], 'revision': m.MODELS['tinyllama11'][1],
                              'dtype': 'float32', 'settings': m.SETTINGS, 'runtime_path': rp,
                              'runtime_sha256': TINY_RUNTIME_SHA, 'context_report_path': xp, 'model_config_path': cp,
                              'execution_stratum': 'baseline-single-record-recovery-v2', 'timeout_seconds_including_load': 1200})
        contracts['tinyllama11'] = mp
        inputs[TINY_CONTEXT], inputs[TINY_CONFIG] = TINY_CONTEXT_SHA, TINY_CONFIG_SHA
    has_qwen = any(s['model_alias'] == 'qwen05' for s in shards)
    qwen = request['qwen_preflight']
    if has_qwen:
        m.exact_keys(qwen, ['commit', 'files_sha256'], 'Qwen preflight binding')
        m.require(isinstance(qwen['commit'], str) and m.HEX40.fullmatch(qwen['commit']), 'Qwen preflight commit missing')
        m.require(set(qwen['files_sha256']) == set(QWEN_FILES), 'Qwen contract inventory differs')
        for path, digest in qwen['files_sha256'].items():
            payloads[path] = bound_read(m, root, path, digest); inputs[path] = digest
        contracts['qwen05'] = QWEN_PRE + '/model-contract.json'
    else:
        m.require(qwen is None, 'Qwen contract supplied without actual Qwen shard')
    manifest = {'schema': m.ID + '-input', 'campaign': m.CAMPAIGN, 'base_ref': m.BASE_REF,
                'snapshot_ref': snapshot, 'files_sha256': {p: m.sha(v) for p, v in payloads.items()},
                'model_contracts': contracts, 'new_shards': shards,
                'reservation_scope': {'snapshot_ref': snapshot, 'index_complete_for_supplied_roots': True,
                                      'roots': ROOTS, 'files': sorted(raw_paths)}}
    m.require(census(m, root)[0] == initial, 'new-run files changed during staging')
    provenance = {'schema': ID + '-provenance', 'source_inputs_sha256': inputs,
                  'caller_remote_census_assertion': remote, 'local_operational_files_excluded': operational,
                  'merger_sha256': MERGER_SHA, 'predecessor_builder_sha256': PREDECESSOR_BUILDER_SHA,
                  'reservation_roots': ROOTS, 'qwen_successor_supported_orders': list(range(2,101)),
                  'model_calls': 0, 'tokenizer_calls': 0,
                  'actual_new_shard_count': len(shards), 'inherited_89_raw_reaudited': False,
                  'independent_global_remote_census_proof': False,
                  'scope': 'Offline exact-byte staging; remote readback/census authenticity remains caller responsibility.'}
    return payloads, manifest, provenance


def build(m, root, request_raw, expected_sha, output):
    m.require(m.sha(request_raw) == expected_sha, 'request hash differs')
    output = Path(output)
    new_output(m, root, output)
    payloads, manifest, provenance = stage(m, Path(root), m.parse(request_raw))
    with tempfile.TemporaryDirectory(prefix='mixed300-verify-') as directory:
        evidence = Path(directory)
        for path, data in payloads.items():
            target = evidence / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        result = m.merge(evidence, manifest)
    manifest_raw = encoded(manifest)
    result.update(input_manifest_sha256=m.sha(manifest_raw), merger_sha256=MERGER_SHA)
    provenance.update(request_sha256=expected_sha, input_manifest_sha256=m.sha(manifest_raw), counts=result['counts'])
    output.parent.mkdir(parents=True, exist_ok=True)
    new_output(m, root, output)
    with tempfile.TemporaryDirectory(prefix='mixed300-publish-', dir=output.parent) as directory:
        staged = Path(directory) / 'bundle'; staged.mkdir()
        for path, data in payloads.items():
            target = staged / 'evidence' / path; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write(data)
        for name, data in [('input-manifest.json', manifest_raw), ('build-request.json', request_raw),
                           ('full300.json', encoded(result)), ('build-provenance.json', encoded(provenance))]:
            with (staged/name).open('xb') as stream: stream.write(data)
        m.require(m.inventory(staged/'evidence') == manifest['files_sha256'], 'staged output bytes differ')
        new_output(m, root, output)
        publish_new_directory(staged, output)
    return {'counts': result['counts'], 'new_shards': len(result['new_shard_evidence']),
            'manifest_sha256': m.sha(manifest_raw), 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--merger', type=Path, default=Path(__file__).with_name('merge_large_development_full300_v2.py'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--expected-request-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draft-snapshot-ref')
    args = parser.parse_args(); m = merger_module(args.merger)
    if args.draft_snapshot_ref:
        m.require(args.request is None and args.expected_request_sha256 is None, 'draft/build options mixed')
        draft = draft_request(m, args.root, args.draft_snapshot_ref)
        new_output(m, args.root, args.output)
        with args.output.open('xb') as stream: stream.write(encoded(draft))
        print(json.dumps({'status': 'unverified_draft_requires_actual_remote_readback', 'path': str(args.output)}))
    else:
        m.require(args.request is not None and args.expected_request_sha256, 'request and external hash required')
        print(json.dumps(build(m, args.root, args.request.read_bytes(), args.expected_request_sha256, args.output)))


if __name__ == '__main__':
    main()
