#!/usr/bin/env python3
"""Reassemble the exact hashed codebook artifact from Git-sized binary parts."""
import hashlib
import json
from pathlib import Path
import tempfile
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'results/matched-archive-v1-transport/manifest.json').read_text())
    if manifest['schema'] != 'synsteg-binary-transport-v1':
        raise ValueError('Unsupported transport manifest')
    target = ROOT / 'results/matched-archive-v1/books.jsonl.gz'
    if manifest['target'] != str(target.relative_to(ROOT)):
        raise ValueError('Unexpected artifact target')
    if target.exists():
        if target.stat().st_size != manifest['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError('Existing target differs; refusing overwrite')
        print('Existing artifact matches exact expected bytes')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='books-restore-', dir=target.parent)
    try:
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(fd, 'wb') as out:
            for part in manifest['parts']:
                path = ROOT / part['path']
                if path.resolve().parent != (ROOT / 'results/matched-archive-v1-transport').resolve():
                    raise ValueError('Unexpected part path')
                content = path.read_bytes()
                if len(content) != part['bytes'] or hashlib.sha256(content).hexdigest() != part['sha256']:
                    raise ValueError('Part identity mismatch: ' + part['path'])
                out.write(content)
                digest.update(content)
                total += len(content)
        if total != manifest['bytes'] or digest.hexdigest() != manifest['sha256']:
            raise ValueError('Combined artifact identity mismatch')
        # Atomic exclusive install: never overwrite a concurrently created file.
        os.link(temporary, target)
        print('Restored exact artifact: ' + manifest['sha256'])
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
