#!/usr/bin/env python3
"""Preserve three pre-encode binding files through <=192KiB Git chunks.

No codec, decoder, model, generation, or mutable source imports. A restored
logical input binding has exactly the same bytes as the original preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'prospective-development-transport-first50-v1'
BINDING = 'results/' + VERSION + '/input-binding'
ART = 'results/' + VERSION + '/input-binding-artifacts'
MANIFEST = ART + '/manifest.json'
HELPER = 'scripts/restore_first50_input_binding_transport_v1.py'
NAMES = ('input_bound_grid.jsonl.gz', 'manifest.json', 'records.jsonl.gz')
LIMIT = 192 * 1024
TOTAL_LIMIT = 32 * 1024 * 1024
SCHEMA = 'prospective-first50-input-binding-git-transport-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def zero_calls(value):
    return all(type(value[name]) is int and value[name] == 0 for name in ('encode_calls', 'decode_calls', 'model_calls'))


def local_file(root, name):
    require(type(name) is str and name != '', 'empty transport path')
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and '..' not in rel.parts and str(rel) == name and '\\' not in name, 'unsafe transport path')
    root = Path(root).resolve()
    path = root / name
    require(path.resolve().is_relative_to(root), 'transport path escapes root')
    for part in (path, *path.parents):
        if part == root:
            break
        require(not part.is_symlink(), 'transport symlink refused')
    return path


def write_once(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def validate_logical(files):
    require(set(files) == set(NAMES), 'exact three logical binding files required')
    require(sum(len(raw) for raw in files.values()) <= TOTAL_LIMIT, 'logical input byte limit')
    manifest = strict_json(files['manifest.json'])
    require(manifest['schema'] == 'prospective-first50-input-binding-v1' and manifest['experiment_id'] == VERSION, 'logical binding identity')
    require(manifest['files_sha256'] == {name: sha(files[name]) for name in NAMES if name != 'manifest.json'}, 'logical binding hashes')
    expected = {'model_record_pairs': 300, 'cells': 15300, 'completed_covers': 50, 'reused_public_conditions': 960, 'new_public_conditions': 240}
    require(all(type(manifest[k]) is int and manifest[k] == value for k, value in expected.items()), 'logical binding fixed denominator')
    require(zero_calls(manifest) and manifest['new_outcomes_known'] is False, 'logical binding must precede encoding')


def build(root=ROOT):
    root = Path(root)
    directory = local_file(root, ART)
    require(not directory.exists(), 'transport output exists; no overwrite')
    source = local_file(root, BINDING)
    require({p.name for p in source.iterdir()} == set(NAMES), 'input binding file membership')
    files = {name: local_file(root, BINDING + '/' + name).read_bytes() for name in NAMES}
    validate_logical(files)
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name in NAMES:
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(entry, files[name])
    raw = output.getvalue()
    require(len(raw) <= TOTAL_LIMIT, 'archive byte limit')
    chunks = []
    directory.mkdir(parents=True, exist_ok=False)
    for offset in range(0, len(raw), LIMIT):
        block = raw[offset:offset + LIMIT]
        path = ART + f'/binding.zip.part{offset // LIMIT:03d}'
        write_once(local_file(root, path), block)
        chunks.append({'path': path, 'offset': offset, 'bytes': len(block), 'sha256': sha(block)})
    manifest = {'schema': SCHEMA, 'experiment_id': VERSION, 'chunk_bytes_max': LIMIT,
                'archive': {'format': 'ZIP_STORED', 'bytes': len(raw), 'sha256': sha(raw)},
                'chunks': chunks, 'chunk_count': len(chunks),
                'logical_files': {BINDING + '/' + name: {'bytes': len(files[name]), 'sha256': sha(files[name])} for name in NAMES},
                'logical_binding_manifest_sha256': sha(files['manifest.json']),
                'helper': {'path': HELPER, 'sha256': sha(local_file(root, HELPER).read_bytes())},
                'logical_bytes_unchanged': True, 'encode_calls': 0, 'decode_calls': 0, 'model_calls': 0}
    write_once(local_file(root, MANIFEST), canonical(manifest) + b'\n')
    return manifest


def restore(destination, root=ROOT):
    root = Path(root)
    destination = Path(destination)
    require(not destination.exists() and not destination.is_symlink(), 'restore destination exists; no overwrite')
    manifest_raw = local_file(root, MANIFEST).read_bytes()
    manifest = strict_json(manifest_raw)
    fields = {'schema', 'experiment_id', 'chunk_bytes_max', 'archive', 'chunks', 'chunk_count', 'logical_files', 'logical_binding_manifest_sha256', 'helper', 'logical_bytes_unchanged', 'encode_calls', 'decode_calls', 'model_calls'}
    require(set(manifest) == fields and manifest['schema'] == SCHEMA and manifest['experiment_id'] == VERSION and type(manifest['chunk_bytes_max']) is int and manifest['chunk_bytes_max'] == LIMIT, 'transport manifest schema')
    require(manifest['helper'] == {'path': HELPER, 'sha256': sha(local_file(root, HELPER).read_bytes())}, 'transport helper identity')
    require(manifest['logical_bytes_unchanged'] is True and zero_calls(manifest), 'transport scope')
    require(set(manifest['logical_files']) == {BINDING + '/' + name for name in NAMES}, 'logical file membership')
    archive = manifest['archive']
    require(set(archive) == {'format', 'bytes', 'sha256'} and archive['format'] == 'ZIP_STORED' and type(archive['bytes']) is int and 0 < archive['bytes'] <= TOTAL_LIMIT, 'archive schema/size')
    chunks = manifest['chunks']
    require(type(chunks) is list and type(manifest['chunk_count']) is int and 0 < len(chunks) == manifest['chunk_count'] <= (TOTAL_LIMIT + LIMIT - 1) // LIMIT, 'chunk count')
    require({p.name for p in local_file(root, ART).iterdir()} == {'manifest.json'} | {f'binding.zip.part{i:03d}' for i in range(len(chunks))}, 'transport directory membership')
    joined = bytearray()
    for index, item in enumerate(chunks):
        require(set(item) == {'path', 'offset', 'bytes', 'sha256'}, 'chunk closed schema')
        require(item['path'] == ART + f'/binding.zip.part{index:03d}' and type(item['offset']) is int and item['offset'] == len(joined) and type(item['bytes']) is int and 0 < item['bytes'] <= LIMIT, 'chunk path/offset/size')
        block = local_file(root, item['path']).read_bytes()
        require(len(block) == item['bytes'] and sha(block) == item['sha256'], 'chunk identity')
        joined.extend(block)
        require(len(joined) <= TOTAL_LIMIT, 'assembled byte bound')
    require(len(joined) == archive['bytes'] and sha(joined) == archive['sha256'], 'archive identity')
    files = {}
    with zipfile.ZipFile(io.BytesIO(joined)) as archive_file:
        entries = archive_file.infolist()
        require(len(entries) == 3 and [e.filename for e in entries] == list(NAMES), 'archive exact ordered members')
        for entry in entries:
            expected = manifest['logical_files'][BINDING + '/' + entry.filename]
            require(set(expected) == {'bytes', 'sha256'} and type(expected['bytes']) is int and 0 < expected['bytes'] <= TOTAL_LIMIT and re.fullmatch('[0-9a-f]{64}', expected['sha256']) is not None, 'logical descriptor')
            require(entry.compress_type == zipfile.ZIP_STORED and entry.file_size == expected['bytes'] and stat.S_ISREG(entry.external_attr >> 16), 'archive member type/size')
            data = archive_file.read(entry)
            require(len(data) == expected['bytes'] and sha(data) == expected['sha256'], 'logical file identity')
            files[entry.filename] = data
    validate_logical(files)
    require(sha(files['manifest.json']) == manifest['logical_binding_manifest_sha256'], 'inner manifest identity')
    destination.mkdir(parents=True, exist_ok=False)
    for name in NAMES:
        write_once(local_file(destination, BINDING + '/' + name), files[name])
    return {'schema': 'prospective-first50-input-binding-reassembly-v1', 'status': 'passed',
            'transport_manifest_sha256': sha(manifest_raw), 'archive_sha256': manifest['archive']['sha256'],
            'chunk_count': len(chunks), 'logical_files_verified': 3,
            'files_sha256': {BINDING + '/' + name: sha(files[name]) for name in NAMES},
            'all_logical_bytes_exact': True, 'encode_calls': 0, 'decode_calls': 0, 'model_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build', 'restore'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        parser.error('receipt exists; no overwrite')
    if args.mode == 'build':
        result = build(args.root)
    else:
        if args.destination is None:
            parser.error('--destination is required')
        result = restore(args.destination, args.root)
    if args.receipt:
        write_once(args.receipt, canonical(result) + b'\n')
    print(json.dumps({'status': result.get('status', 'built'), 'chunk_count': result['chunk_count'], 'logical_files': 3, 'encode_calls': 0, 'decode_calls': 0, 'model_calls': 0}))
