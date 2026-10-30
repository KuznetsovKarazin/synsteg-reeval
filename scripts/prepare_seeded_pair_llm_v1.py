#!/usr/bin/env python3
"""Restore only pinned public LLM resources; never generate or search keys."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def download(url, target, expected_sha, expected_bytes=None):
    if target.exists():
        if sha(target) != expected_sha or (expected_bytes is not None and target.stat().st_size != expected_bytes):
            raise ValueError('Existing public resource differs: ' + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    if sha(partial) != expected_sha or (expected_bytes is not None and partial.stat().st_size != expected_bytes):
        raise ValueError('Public download identity mismatch; partial retained')
    partial.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-root', type=Path, default=ROOT / 'resources/seeded-pair-llm-v1')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'experiments/seeded_pair_llm_v1/resource_fingerprint.json').read_text())
    spec = manifest['resource_spec']
    for kind in ('runtime', 'model'):
        item = spec[kind]
        path = args.resource_root / item.get('filename', item.get('archive'))
        download(item['url'], path, item['sha256'], item['bytes'])
        if kind == 'runtime' and not (args.resource_root / 'runtime').exists():
            with tarfile.open(path) as archive:
                archive.extractall(args.resource_root / 'runtime', filter='data')
    for relative, expected in manifest['downloaded_files'].items():
        path = args.resource_root / relative
        if '/' not in relative and relative.endswith('.h'):
            parent = 'include' if relative == 'llama.h' else 'ggml/include'
            url = 'https://raw.githubusercontent.com/ggml-org/llama.cpp/' + spec['runtime']['commit'] + '/' + parent + '/' + relative
            download(url, path, expected)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Pinned extracted resource mismatch: ' + relative)
    print(json.dumps({'status': 'verified', 'model_sha256': spec['model']['sha256'],
                      'runtime_sha256': spec['runtime']['sha256'],
                      'verified_files': len(manifest['downloaded_files']) + 2,
                      'generation_calls': 0, 'paid_calls': 0}))


if __name__ == '__main__':
    main()
