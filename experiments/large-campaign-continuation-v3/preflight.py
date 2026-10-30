"""Read-only restoration preflight; writes only new continuation receipts."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/large-campaign-continuation-v3/preflight/model-preload.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError('Preserve previous preflight evidence')
    manifest = json.loads((ROOT / 'experiments/large-campaign-development-v2/execution-manifest.json').read_text())
    for path, expected in manifest['frozen_files_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    restoration = json.loads((OUT.parent / 'model-restore.json').read_text())
    if restoration['status'] != 'all_seven_resources_verified':
        raise ValueError('Resources not verified')
    os.environ.update({key: value.replace('{repository_root}', str(ROOT))
                       for key, value in manifest['environment'].items()})
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('frozen_runner', ROOT / 'scripts/run_budget_generation.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    saved_runtime = json.loads((ROOT / manifest['runner_output'] / 'runtime.json').read_text())
    if runner.local_runtime() != saved_runtime:
        raise ValueError('Restored runtime differs from saved execution')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    options = {'revision': manifest['revision'], 'local_files_only': True, 'trust_remote_code': False}
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'], **options)
    rows = runner.read_prompts(ROOT / 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl', 'development', 100)
    original = json.loads((ROOT / 'results/large-campaign-development-v2/preflight/all-100-context-checks.json').read_text())
    current = []
    for row, saved in zip(rows, original['rows']):
        prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': runner.SYSTEM},
                 {'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
        check = {'record_id': row['record_id'], 'input_tokens': ids.shape[1],
                 'chat_template_render_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        if any(saved[key] != value for key, value in check.items()):
            raise ValueError('Tokenization or fixed order drift')
        if check['input_tokens'] + 1024 > 2048:
            raise ValueError('Context does not fit')
        current.append(check)
    if len(rows) != len(current) or len(current) != len(original['rows']) or len(current) != 100:
        raise ValueError('Full development denominator changed')
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(manifest['model'], torch_dtype=torch.float32, **options).eval().cpu()
    receipt = {'schema': 'large-campaign-continuation-preflight-v3', 'status': 'passed',
               'runtime_equals_saved': True, 'all_100_tokenization_checks_equal_saved': True,
               'context_rows': current, 'parameter_count': sum(p.numel() for p in model.parameters()),
               'devices': sorted({str(p.device) for p in model.parameters()}),
               'dtypes': sorted({str(p.dtype) for p in model.parameters()}),
               'load_seconds': time.monotonic() - started,
               'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'generation_calls': 0, 'forward_calls': 0, 'reservations': 0,
               'source_sha256': digest(Path(__file__)), 'old_frozen_files_changed': False}
    with OUT.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'context_rows'}))


if __name__ == '__main__':
    main()
