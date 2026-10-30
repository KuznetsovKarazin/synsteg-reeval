#!/usr/bin/env python3
"""Download the declared public GGUF artifact; no inference, keys or retries."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'results/long-summary-development-qwen15b-gguf-v1'


def main():
    spec = json.loads((STAGE / 'resources.json').read_text())['model']
    expected = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
    if spec['sha256'] != expected or spec['bytes'] != 1117320736:
        raise ValueError('Unexpected model resource declaration')
    target = ROOT / 'resources/qwen15b-gguf-v1' / spec['filename']
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.part')
    if target.exists() or partial.exists():
        raise FileExistsError('Refuse overwrite; inspect existing download explicitly')
    digest = hashlib.sha256()
    size = 0
    started = time.monotonic()
    with urllib.request.urlopen(spec['url'], timeout=60) as response, partial.open('xb') as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            digest.update(block)
            size += len(block)
            if size % (64 * 1024 * 1024) == 0:
                print(json.dumps({'downloaded_bytes': size}), flush=True)
    if size != spec['bytes'] or digest.hexdigest() != expected:
        raise ValueError('Artifact size/SHA mismatch; partial retained, inference forbidden')
    partial.rename(target)
    result = {'schema': 'qwen15b-public-download-v1', 'url': spec['url'],
              'sha256': digest.hexdigest(), 'bytes': size,
              'elapsed_seconds': time.monotonic() - started,
              'output': str(target.relative_to(ROOT)), 'inference_calls': 0,
              'paid_calls': 0}
    out = STAGE / 'preflight/model-download.json'
    with out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
