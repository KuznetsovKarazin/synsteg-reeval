#!/usr/bin/env python3
"""Restore the four exact saved conformance JSON files; no tokenizer/model call."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def restore(destination):
    destination = Path(destination)
    manifest = json.loads((ROOT / 'results/canonical-packet-v3-artifacts/manifest.json').read_text())
    archive = ROOT / manifest['archive']
    raw = archive.read_bytes()
    if len(raw) != manifest['bytes'] or hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise ValueError('Archive size or SHA256 differs')
    expected = {row['path']: row for row in manifest['files']}
    if len(expected) != 4 or len(manifest['files']) != 4:
        raise ValueError('Require exactly four unique primary outputs')
    recovered = []
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != 4 or set(z.namelist()) != set(expected):
            raise ValueError('Archive member set differs')
        for name, item in expected.items():
            relative = PurePosixPath(name)
            if (relative.is_absolute() or '..' in relative.parts or
                    relative.parts[:2] != ('results', 'canonical-packet-v3-conformance')):
                raise ValueError('Unexpected archive member path')
            data = z.read(name)
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Primary output differs: ' + name)
            target = destination / name
            if destination.resolve() not in target.resolve().parents:
                raise ValueError('Destination escapes requested root')
            if target.exists() and (not target.is_file() or target.read_bytes() != data):
                raise ValueError('Existing output differs; refusing overwrite: ' + name)
            recovered.append((target, data))
    written = 0
    for target, data in recovered:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as out:
            out.write(data)
        written += 1
    return {'status': 'verified', 'verified_files': len(recovered), 'restored_files': written,
            'archive_sha256': manifest['sha256'], 'tokenizer_or_model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination-root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(restore(args.destination_root)))


if __name__ == '__main__':
    main()
