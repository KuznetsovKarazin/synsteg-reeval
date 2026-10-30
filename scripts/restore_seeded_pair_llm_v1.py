#!/usr/bin/env python3
"""Verify and restore all saved LLM outputs without running a language model."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def restore(destination):
    manifest = json.loads((ROOT / 'results/seeded-pair-llm-v1-artifacts/manifest.json').read_text())
    archive = ROOT / manifest['archive']
    raw = archive.read_bytes()
    if len(raw) != manifest['bytes'] or hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise ValueError('Full output archive identity mismatch')
    entries = {item['path']: item for item in manifest['files']}
    if len(entries) != len(manifest['files']):
        raise ValueError('Duplicate declared output path')
    recovered = []
    with zipfile.ZipFile(archive) as stream:
        if set(stream.namelist()) != set(entries) or len(stream.namelist()) != len(entries):
            raise ValueError('Archive member set mismatch')
        for name, item in entries.items():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or path.parts[:2] != ('results', 'seeded-pair-llm-v1'):
                raise ValueError('Unexpected archive member path')
            data = stream.read(name)
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Output identity mismatch: ' + name)
            target = destination / name
            if destination.resolve() not in target.resolve().parents:
                raise ValueError('Destination escapes requested root')
            if target.exists() and (not target.is_file() or target.read_bytes() != data):
                raise ValueError('Existing output differs; refusing overwrite: ' + name)
            recovered.append((target, data))
    # All members and existing destinations are checked before any write.
    restored = 0
    for target, data in recovered:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(data)
        restored += 1
    return {'status': 'verified', 'verified_files': len(recovered), 'restored_files': restored,
            'archive_sha256': manifest['sha256'], 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination-root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(restore(args.destination_root)))


if __name__ == '__main__':
    main()
