#!/usr/bin/env python3
"""Download only the exact frozen public checkpoints needed for this study."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('configs/neural/model_revisions.json'))
    parser.add_argument('--model', action='append')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    for entry in json.loads(args.manifest.read_text()):
        if args.model and entry['model_id'] not in args.model:
            continue
        names = entry['files']
        weights = [p for p in names if p.endswith('.safetensors') and '/' not in p]
        if not weights:
            weights = [p for p in names if p == 'pytorch_model.bin']
        if not weights:
            raise ValueError('No supported weights for ' + entry['model_id'])
        keep = weights + [p for p in names if '/' not in p and (
            p.endswith('.json') or p in {'vocab.txt', 'merges.txt', 'tokenizer.model', 'spiece.model'})]
        print('Downloading', entry['model_id'], entry['revision'], flush=True)
        path = snapshot_download(entry['model_id'], revision=entry['revision'], allow_patterns=keep, max_workers=2)
        print('Cached', entry['model_id'], path, flush=True)


if __name__ == '__main__':
    main()
