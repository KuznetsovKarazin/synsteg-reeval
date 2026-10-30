#!/usr/bin/env python3
"""Reconstruct an existing baseline ZIP from small hash-bound Git parts only.

No generation or decoding. The resulting ZIP is consumed by the already frozen
restore_native_continuation_artifact_v1.py with its original manifest.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def load(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Require nonempty relative POSIX path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or str(path) != value or '\\' in value:
        raise ValueError('Unsafe relative path')
    return value


def identity(data, entry):
    if (type(entry.get('bytes')) is not int or entry['bytes'] < 0
            or not isinstance(entry.get('sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])
            or len(data) != entry['bytes']
            or hashlib.sha256(data).hexdigest() != entry['sha256']):
        raise ValueError('Byte count or SHA256 differs')


def safe_target(root, relative):
    target = root / relative_path(relative)
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError('Symlink refused')
        current = current.parent
    if not target.resolve().is_relative_to(root):
        raise ValueError('Path escapes root')
    return target


def reassemble(root, manifest, output_root):
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    value = load(Path(manifest))
    if (value.get('schema') != 'baseline-primary-chunk-transport-v1'
            or type(value.get('chunk_bytes_max')) is not int
            or value['chunk_bytes_max'] != 196608
            or type(value.get('chunk_count')) is not int
            or not isinstance(value.get('chunks'), list)
            or not value['chunks'] or value['chunk_count'] != len(value['chunks'])):
        raise ValueError('Unexpected transport manifest')
    native_path = safe_target(root, value['native_manifest']['path'])
    native_bytes = native_path.read_bytes()
    identity(native_bytes, value['native_manifest'])
    native = load(native_path)
    if native.get('archive') != value.get('archive'):
        raise ValueError('Original archive identity changed')
    pieces, names, offset = [], set(), 0
    for entry in value['chunks']:
        name = relative_path(entry['path'])
        if name in names or type(entry.get('offset')) is not int or entry['offset'] != offset:
            raise ValueError('Duplicate chunk or discontinuous order')
        names.add(name)
        data = safe_target(root, name).read_bytes()
        identity(data, entry)
        if not 0 < len(data) <= value['chunk_bytes_max']:
            raise ValueError('Unexpected chunk size')
        pieces.append(data)
        offset += len(data)
    archive = b''.join(pieces)
    identity(archive, value['archive'])
    target = safe_target(output_root, value['archive']['path'])
    if target.exists():
        if not target.is_file() or target.read_bytes() != archive:
            raise ValueError('Existing different archive; refuse overwrite')
        written = False
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(archive)
            stream.flush()
        written = True
    return {'status': 'passed', 'chunk_count': len(pieces), 'archive': value['archive'],
            'archive_written': written, 'model_calls': 0, 'decoder_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('.'))
    args = parser.parse_args()
    print(json.dumps(reassemble(args.root, args.manifest, args.output_root), sort_keys=True))
