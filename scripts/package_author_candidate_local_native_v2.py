#!/usr/bin/env python3
"""Deterministically preserve every terminal/UNKNOWN native-v2 output byte.

No inference or tokenizer imports. Refuse active cooperative NN writer, links,
special files and source mutations. Raw Git parts are at most 192 KiB. An absent
result requires explicit incomplete-preservation mode, never an inferred success.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'results/author-candidate-local-native-v2/native-diagnostic-v2'
ARTIFACTS = 'results/author-candidate-local-native-v2-artifacts/native-diagnostic-v2'
SCHEMA = 'author-candidate-local-native-v2-artifacts'
PART_BYTES = 192 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def sha(raw): return hashlib.sha256(raw).hexdigest()
def json_bytes(value): return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def relative(name):
    if type(name) is not str or not name or '\\' in name:
        raise ValueError('unsafe relative path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix() != name or name == '.':
        raise ValueError('unsafe relative path')
    return p


def regular_bytes(path, *, limit=MAX_FILE_BYTES):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('regular bounded file required')
        chunks = []; count = 0
        while True:
            chunk = os.read(fd, min(65536, limit + 1 - count))
            if not chunk: break
            count += len(chunk)
            if count > limit: raise ValueError('file bound exceeded')
            chunks.append(chunk)
        after = os.fstat(fd); actual = path.lstat()
        identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if identity(before) != identity(after) or identity(before) != identity(actual) or count != before.st_size:
            raise ValueError('file changed while reading')
        return b''.join(chunks)
    finally:
        os.close(fd)


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    def constant(_): raise ValueError('nonfinite JSON value')
    return json.loads(regular_bytes(path), object_pairs_hook=pairs, parse_constant=constant)


def no_symlink_ancestors(path):
    for p in (Path(path).absolute(), *Path(path).absolute().parents):
        if p.is_symlink(): raise ValueError('symlinked path or ancestor')


def put(path, raw):
    path = Path(path); no_symlink_ancestors(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0: raise OSError('zero-progress write')
            view = view[count:]
        os.fsync(fd)
    finally: os.close(fd)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


def snapshot(root, base=BASE):
    root = Path(root).resolve(strict=True); primary = root / relative(base)
    no_symlink_ancestors(primary)
    if not primary.is_dir(): raise ValueError('primary directory missing')
    files = {}; directories = []
    def walk_error(error): raise error
    for current, dirs, names in os.walk(primary, followlinks=False, onerror=walk_error):
        current = Path(current); directories.append(current.relative_to(root).as_posix())
        for name in sorted(dirs + names):
            p = current / name; mode = p.lstat().st_mode
            if stat.S_ISDIR(mode): continue
            if not stat.S_ISREG(mode): raise ValueError('primary link or special file refused')
            raw = regular_bytes(p); files[p.relative_to(root).as_posix()] = {'bytes': len(raw), 'sha256': sha(raw)}
    if not files: raise ValueError('empty primary file tree')
    return dict(sorted(files.items())), sorted(directories)


def check_row(row):
    if (type(row) is not dict or set(row) != {'bytes', 'sha256'} or type(row['bytes']) is not int
            or row['bytes'] < 0 or type(row['sha256']) is not str
            or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None):
        raise ValueError('invalid byte identity')


def read_archive(index_path):
    index_path = Path(index_path); no_symlink_ancestors(index_path)
    index = read_json(index_path)
    if index.get('schema') != SCHEMA or type(index.get('part_bytes')) is not int or index['part_bytes'] != PART_BYTES:
        raise ValueError('artifact schema or part bound differs')
    parts = index.get('parts')
    if type(parts) is not list or not parts: raise ValueError('missing parts')
    raw_parts = []; total = 0
    for order, row in enumerate(parts):
        name = f'full.zip.part{order:03d}'
        if type(row) is not dict or set(row) != {'name', 'bytes', 'sha256'} or row['name'] != name:
            raise ValueError('part order/schema differs')
        check_row({k: row[k] for k in ('bytes', 'sha256')})
        if not 0 < row['bytes'] <= PART_BYTES or (order < len(parts) - 1 and row['bytes'] != PART_BYTES):
            raise ValueError('part size differs')
        total += row['bytes']
        if total > MAX_ARCHIVE_BYTES: raise ValueError('archive bound exceeded')
        data = regular_bytes(index_path.parent / name, limit=PART_BYTES)
        if len(data) != row['bytes'] or sha(data) != row['sha256']: raise ValueError('part bytes differ')
        raw_parts.append(data)
    raw = b''.join(raw_parts); archive = index['archive']
    check_row({k: archive[k] for k in ('bytes', 'sha256')})
    if len(raw) != archive['bytes'] or sha(raw) != archive['sha256']: raise ValueError('archive identity differs')
    return index, raw


def restore(index_path, destination):
    index, raw = read_archive(index_path); destination = Path(destination)
    no_symlink_ancestors(destination); dest = destination.resolve()
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError('fresh empty restoration destination required')
    base = relative(index['source_base']); files = index['files']; dirs = index['directories']
    if type(files) is not dict or type(dirs) is not list or len(dirs) != len(set(dirs)) or set(files) & set(dirs):
        raise ValueError('inventory differs')
    for name in list(files) + dirs:
        path = relative(name)
        if path != base and base not in path.parents: raise ValueError('member outside source tree')
    if base.as_posix() not in dirs: raise ValueError('base directory metadata absent')
    for name, row in files.items():
        check_row(row)
        if row['bytes'] > MAX_FILE_BYTES or relative(name).parent.as_posix() not in dirs:
            raise ValueError('file bound or parent metadata differs')
    campaign_files = {str(PurePosixPath(name).relative_to(base)): row for name, row in files.items()}
    if json_bytes(campaign_files) != json_bytes(index['campaign_files']): raise ValueError('campaign inventory differs')
    pending = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist(); names = [entry.filename for entry in infos]
        if len(names) != len(set(names)) or set(names) != set(files): raise ValueError('ZIP member set differs')
        for info in infos:
            row = files[info.filename]
            if info.file_size != row['bytes'] or not stat.S_ISREG(info.external_attr >> 16):
                raise ValueError('ZIP member size/type differs')
            data = archive.read(info)
            if len(data) != row['bytes'] or sha(data) != row['sha256']: raise ValueError('ZIP member bytes differ')
            pending.append((dest / relative(info.filename), data))
    for name in sorted(dirs, key=lambda x: (len(relative(x).parts), x)):
        (dest / relative(name)).mkdir(parents=True, exist_ok=True)
    for path, data in pending: put(path, data)
    actual_files, actual_dirs = snapshot(dest, base.as_posix())
    if actual_files != files or actual_dirs != sorted(dirs): raise ValueError('fresh restore inventory differs')
    return {'status': 'passed', 'files_verified': len(files), 'directories_verified': len(dirs),
            'archive_sha256': index['archive']['sha256'], 'model_calls': 0, 'tokenizer_calls': 0}


def package(root, output, *, base=BASE, allow_incomplete=False):
    root = Path(root).resolve(strict=True); primary = root / relative(base)
    output = Path(output); no_symlink_ancestors(output); output = output.resolve()
    if root not in output.parents: raise ValueError('artifact output must be inside repository')
    if output == primary or primary in output.parents: raise ValueError('artifact output inside primary')
    if output.exists(): raise FileExistsError('exclusive artifact output required')
    # Cooperative lease prevents concurrent model writers; not OS immutability.
    lease = os.open(root / 'results/project-nn.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(lease).st_mode): raise ValueError('NN lease not regular')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files, dirs = snapshot(root, base)
        terminal = (PurePosixPath(base) / 'result.json').as_posix() in files
        if not terminal and not allow_incomplete: raise ValueError('terminal absent; explicit incomplete preservation required')
        output.mkdir(parents=True)
        archive_path = output / 'full.zip'
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_LZMA) as archive:
            for name, row in files.items():
                data = regular_bytes(root / name)
                if len(data) != row['bytes'] or sha(data) != row['sha256']: raise ValueError('primary changed during archive')
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 25, 0, 0, 0))
                info.create_system = 3; info.compress_type = zipfile.ZIP_LZMA; info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        if snapshot(root, base) != (files, dirs): raise ValueError('primary changed during archive')
        raw = regular_bytes(archive_path, limit=MAX_ARCHIVE_BYTES); parts = []
        for i, start in enumerate(range(0, len(raw), PART_BYTES)):
            data = raw[start:start + PART_BYTES]; name = f'full.zip.part{i:03d}'
            put(output / name, data); parts.append({'name': name, 'bytes': len(data), 'sha256': sha(data)})
        index = {'schema': SCHEMA, 'source_base': base, 'archive': {'name': 'full.zip', 'bytes': len(raw),
            'sha256': sha(raw), 'compression': 'ZIP_LZMA', 'full_zip_local_only_upload_parts': True},
            'part_bytes': PART_BYTES, 'parts': parts, 'files': files, 'directories': dirs,
            'campaign_files': {str(PurePosixPath(name).relative_to(base)): row for name, row in files.items()},
            'primary_files': len(files), 'primary_bytes': sum(row['bytes'] for row in files.values()),
            'all_regular_files_and_empty_directories_preserved': True,
            'terminal_result_present': terminal, 'allow_incomplete_requested': allow_incomplete,
            'scientific_acceptance_inferred': False, 'model_calls': 0, 'tokenizer_calls': 0}
        put(output / 'index.json', json_bytes(index))
        with tempfile.TemporaryDirectory(prefix='candidate-v2-fresh-restore-') as tmp:
            fresh = restore(output / 'index.json', tmp)
        put(output / 'fresh-restore.json', json_bytes(fresh))
        uploads = ['index.json', 'fresh-restore.json'] + [row['name'] for row in parts]
        listing = {'schema': 'author-candidate-local-native-v2-upload-list', 'files': [
            {'path': (output / name).relative_to(root).as_posix(), 'bytes': len(regular_bytes(output / name)),
             'sha256': sha(regular_bytes(output / name)), 'encoding': 'binary' if name.startswith('full.zip.part') else 'utf-8'}
            for name in uploads], 'include_upload_list_itself': True, 'do_not_upload': ['full.zip'],
            'maximum_binary_part_bytes': PART_BYTES, 'full_primary_files': len(files), 'model_calls': 0}
        put(output / 'upload-files.json', json_bytes(listing))
        return {'status': 'passed', 'primary_files': len(files), 'archive_bytes': len(raw), 'parts': len(parts),
            'largest_part': max(row['bytes'] for row in parts), 'upload_list': str(output / 'upload-files.json'), 'model_calls': 0}
    finally: os.close(lease)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('package'); p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--allow-incomplete', action='store_true')
    p = sub.add_parser('restore'); p.add_argument('--index', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = (package(args.root, args.output, allow_incomplete=args.allow_incomplete) if args.command == 'package'
              else restore(args.index, args.destination))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
