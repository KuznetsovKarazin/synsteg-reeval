#!/usr/bin/env python3
"""Offline tokenizer/config/cache check; does not load a model or generate text."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_budget_generation import SYSTEM, read_prompts


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'results/long-summary-development-v3')
    args = p.parse_args()
    from transformers import AutoTokenizer, AutoConfig, GenerationConfig
    expected = {
        'config.json': '18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45',
        'generation_config.json': 'e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6',
        'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
        'model.safetensors': 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
        'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
        'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
        'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}
    for name, value in expected.items():
        if sha(args.snapshot / name) != value:
            raise ValueError('Pinned model cache mismatch: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    generation = GenerationConfig.from_pretrained(args.snapshot, local_files_only=True)
    generation.update(do_sample=True, max_new_tokens=1536, temperature=0.8,
                      top_p=0.95, top_k=50, pad_token_id=tokenizer.eos_token_id)
    rows = read_prompts(args.output / 'prompts.jsonl', 'development', 20)
    checks = []
    for row in rows:
        text = tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True)
        count = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        if count + 1536 > config.max_position_embeddings:
            raise ValueError('Context would overflow; no truncation allowed')
        checks.append({'record_id': row['record_id'], 'input_tokens': count,
                       'output_token_cap': 1536, 'context': config.max_position_embeddings})
    report = {'schema': 'synsteg-long-summary-v3-preflight', 'status': 'passed',
        'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'revision': '7ae557604adf67be50417f59c2c2f167def9a775',
        'new_model_generations': 0, 'network_used': False, 'weight_sha256_verified': True,
        'model_files_sha256': expected, 'prompts_sha256': sha(args.output / 'prompts.jsonl'),
        'records': checks, 'runner_sha256': sha(ROOT / 'scripts/run_budget_generation.py')}
    outputs = {'tokenizer_preflight.json': report, 'effective_generation_config.json': generation.to_dict()}
    for name, obj in outputs.items():
        encoded = (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
        path = args.output / name
        if path.exists():
            if path.read_bytes() != encoded:
                raise ValueError('Refuse to replace preflight: ' + name)
        else:
            with path.open('xb') as stream:
                stream.write(encoded)
    print(json.dumps({'status': 'passed', 'records': len(checks), 'max_input_tokens': max(x['input_tokens'] for x in checks),
                      'output_token_cap': 1536, 'new_model_generations': 0}))


if __name__ == '__main__':
    main()
