#!/usr/bin/env python3
"""Preserve first80 milestone/audits as exact <=192KiB Git archive parts.

Read-only toward scientific inputs/results. No encoder, decoder or model imports.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'prospective-development-transport-first80-v1'
BASE = 'results/' + VERSION
MILESTONE = BASE + '/milestones/tinyllama-first80'
AUDITS = BASE + '/audits'
BINDING = BASE + '/input-binding'
BINDING_NAMES = ('records.jsonl.gz', 'input_bound_grid.jsonl.gz', 'manifest.json')
ART = BASE + '-artifacts'
HELPER = 'scripts/package_first80_transport_results_v1.py'
LIMIT = 192 * 1024
MAX_BYTES = 256 * 1024 * 1024
SCHEMA = 'prospective-first80-results-chunked-artifact-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def local(root, name):
    require(type(name) is str and name, 'empty path')
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and '..' not in rel.parts and str(rel) == name and '\\' not in name, 'unsafe path')
    root = Path(root).resolve()
    path = root / name
    require(path.resolve().is_relative_to(root), 'path outside root')
    for part in (path, *path.parents):
        if part == root:
            break
        require(not part.is_symlink(), 'symlink refused')
    return path


def write_once(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def source_files(root):
    root = Path(root).resolve()
    milestone = local(root, MILESTONE)
    expected_raw = (milestone / 'output_manifest.json').read_bytes()
    expected = strict_json(expected_raw)
    actual = {p.relative_to(milestone).as_posix() for p in milestone.rglob('*') if p.is_file()}
    require(actual == set(expected) | {'output_manifest.json'}, 'milestone exact inventory')
    for name, digest in expected.items():
        require(sha(local(milestone, name).read_bytes()) == digest, 'milestone source hash: ' + name)
    audit = local(root, AUDITS + '/tinyllama-first80.json')
    audited = strict_json(audit.read_bytes())
    require(audited['status'] == 'passed' and audited['output_manifest_sha256'] == sha(expected_raw) and audited['cells'] == 15300 and audited['input_records'] == 300, 'passed full independent audit required')
    binding = local(root, BINDING)
    require({p.name for p in binding.iterdir()} == set(BINDING_NAMES), 'binding exact inventory')
    for name in BINDING_NAMES:
        require(local(root, BINDING + '/' + name).read_bytes() == local(milestone, 'input_binding/' + name).read_bytes(), 'binding differs from audited milestone: ' + name)
    paths = sorted([MILESTONE + '/' + name for name in actual] + [p.relative_to(root).as_posix() for p in local(root, AUDITS).rglob('*') if p.is_file()] + [BINDING + '/' + name for name in BINDING_NAMES])
    require(len(paths) == len(set(paths)) and 1 <= len(paths) <= 2000, 'primary inventory size')
    files = {name: local(root, name).read_bytes() for name in paths}
    require(sum(map(len, files.values())) <= MAX_BYTES, 'primary byte bound')
    return files, sha(expected_raw), sha(audit.read_bytes())


def package(root=ROOT):
    root = Path(root).resolve()
    out = local(root, ART)
    require(not out.exists(), 'artifact already exists; no overwrite')
    files, output_hash, audit_hash = source_files(root)
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in files.items():
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(entry, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    raw = memory.getvalue()
    require(len(raw) <= MAX_BYTES, 'archive byte bound')
    out.mkdir(parents=True, exist_ok=False)
    chunks = []
    for offset in range(0, len(raw), LIMIT):
        block = raw[offset:offset + LIMIT]
        path = ART + f'/parts/primary.zip.part{offset // LIMIT:03d}'
        write_once(local(root, path), block)
        chunks.append({'path': path, 'offset': offset, 'bytes': len(block), 'sha256': sha(block)})
    manifest = {'schema': SCHEMA, 'experiment_id': VERSION,
                'scope': 'Entire immutable first80 milestone including nested inherited evidence, exact original three input binding files and all saved audit files; mutable current handoff/status and later quality reviews excluded.',
                'source_output_manifest_sha256': output_hash, 'independent_audit_sha256': audit_hash,
                'files': {n: {'bytes': len(b), 'sha256': sha(b)} for n, b in files.items()},
                'file_count': len(files), 'uncompressed_bytes': sum(map(len, files.values())),
                'archive': {'format': 'ZIP_DEFLATED9', 'bytes': len(raw), 'sha256': sha(raw)},
                'chunks': chunks, 'chunk_count': len(chunks), 'chunk_bytes_max': LIMIT,
                'restore_helper': {'path': HELPER, 'sha256': sha(local(root, HELPER).read_bytes())},
                'full_archive_not_stored_as_one_git_blob': True,
                'encoder_calls': 0, 'decoder_calls': 0, 'model_calls': 0}
    write_once(local(root, ART + '/manifest.json'), canonical(manifest) + b'\n')
    require(source_files(root)[0] == files, 'scientific source changed during packaging')
    return manifest


def restore(destination, root=ROOT):
    root, destination = Path(root).resolve(), Path(destination)
    require(not destination.exists() and not destination.is_symlink(), 'restore destination exists; no overwrite')
    raw_manifest = local(root, ART + '/manifest.json').read_bytes()
    m = strict_json(raw_manifest)
    fields = {'schema', 'experiment_id', 'scope', 'source_output_manifest_sha256', 'independent_audit_sha256', 'files', 'file_count', 'uncompressed_bytes', 'archive', 'chunks', 'chunk_count', 'chunk_bytes_max', 'restore_helper', 'full_archive_not_stored_as_one_git_blob', 'encoder_calls', 'decoder_calls', 'model_calls'}
    require(set(m) == fields and m['schema'] == SCHEMA and m['experiment_id'] == VERSION, 'artifact schema')
    require(m['restore_helper'] == {'path': HELPER, 'sha256': sha(local(root, HELPER).read_bytes())}, 'restore helper identity')
    require(type(m['chunk_bytes_max']) is int and m['chunk_bytes_max'] == LIMIT and type(m['chunk_count']) is int and 0 < m['chunk_count'] == len(m['chunks']) <= (MAX_BYTES + LIMIT - 1) // LIMIT, 'chunk denominator')
    require(type(m['file_count']) is int and 0 < m['file_count'] == len(m['files']) <= 2000 and type(m['uncompressed_bytes']) is int and 0 < m['uncompressed_bytes'] <= MAX_BYTES, 'file denominator')
    require(all(type(m[k]) is int and m[k] == 0 for k in ('encoder_calls', 'decoder_calls', 'model_calls')) and m['full_archive_not_stored_as_one_git_blob'] is True, 'artifact execution scope')
    require(set(m['archive']) == {'format', 'bytes', 'sha256'} and m['archive']['format'] == 'ZIP_DEFLATED9' and type(m['archive']['bytes']) is int and 0 < m['archive']['bytes'] <= MAX_BYTES, 'archive descriptor')
    require({p.name for p in local(root, ART + '/parts').iterdir()} == {f'primary.zip.part{i:03d}' for i in range(m['chunk_count'])}, 'exact parts inventory')
    combined = bytearray()
    for i, item in enumerate(m['chunks']):
        require(set(item) == {'path', 'offset', 'bytes', 'sha256'} and item['path'] == ART + f'/parts/primary.zip.part{i:03d}', 'part schema/path')
        require(type(item['offset']) is int and item['offset'] == len(combined) and type(item['bytes']) is int and 0 < item['bytes'] <= LIMIT, 'part offset/size')
        raw = local(root, item['path']).read_bytes()
        require(len(raw) == item['bytes'] and sha(raw) == item['sha256'], 'part identity')
        combined.extend(raw)
        require(len(combined) <= MAX_BYTES, 'assembled byte bound')
    require(len(combined) == m['archive']['bytes'] and sha(combined) == m['archive']['sha256'], 'archive identity')
    pending = {}
    with zipfile.ZipFile(io.BytesIO(combined)) as archive:
        entries = archive.infolist()
        require(len(entries) == m['file_count'] and [e.filename for e in entries] == sorted(m['files']), 'archive exact ordered members')
        for entry in entries:
            name = entry.filename
            require(name.startswith(MILESTONE + '/') or name.startswith(AUDITS + '/') or name in {BINDING + '/' + x for x in BINDING_NAMES}, 'archive member scope')
            local(destination, name)
            item = m['files'][name]
            require(set(item) == {'bytes', 'sha256'} and type(item['bytes']) is int and 0 <= item['bytes'] <= MAX_BYTES, 'member descriptor')
            require(entry.file_size == item['bytes'] and entry.compress_type == zipfile.ZIP_DEFLATED and stat.S_ISREG(entry.external_attr >> 16), 'member size/type')
            raw = archive.read(entry)
            require(len(raw) == item['bytes'] and sha(raw) == item['sha256'], 'member identity')
            pending[name] = raw
    require(sum(map(len, pending.values())) == m['uncompressed_bytes'], 'uncompressed byte count')
    require(sha(pending[MILESTONE + '/output_manifest.json']) == m['source_output_manifest_sha256'] and sha(pending[AUDITS + '/tinyllama-first80.json']) == m['independent_audit_sha256'], 'scientific source identity')
    for name in BINDING_NAMES:
        require(pending[BINDING + '/' + name] == pending[MILESTONE + '/input_binding/' + name], 'restored binding differs from milestone')
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in pending.items():
        write_once(local(destination, name), raw)
    return {'schema': 'prospective-first80-results-fresh-restore-v1', 'status': 'passed',
            'artifact_manifest_sha256': sha(raw_manifest), 'archive_sha256': m['archive']['sha256'],
            'verified_members': len(pending), 'verified_bytes': m['uncompressed_bytes'],
            'verified_chunks': m['chunk_count'], 'all_original_bytes_exact': True,
            'encoder_calls': 0, 'decoder_calls': 0, 'model_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('package', 'restore'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        parser.error('receipt exists')
    if args.mode == 'package':
        result = package(args.root)
    else:
        if args.destination is None:
            parser.error('--destination required')
        result = restore(args.destination, args.root)
    if args.receipt:
        write_once(args.receipt, canonical(result) + b'\n')
    print(json.dumps({'status': result.get('status', 'packaged'),
                      'files': result.get('file_count', result.get('verified_members')),
                      'chunks': result.get('chunk_count', result.get('verified_chunks')),
                      'encoder_calls': 0, 'decoder_calls': 0, 'model_calls': 0}))
