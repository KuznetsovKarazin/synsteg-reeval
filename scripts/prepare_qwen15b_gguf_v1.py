#!/usr/bin/env python3
"""Freeze an eight-request development probe without downloading/loading a model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = 'long-summary-development-qwen15b-gguf-v1'
OUT = 'results/' + ID
PARENT = 'results/long-summary-development-v3/'
SOURCE = 'data/development-full-revisions-v3/source-windows.jsonl'
BASE_COMMIT = '026b5b94494d65461fb0f440f12ddfc99c8ee2fb'
SELECTED_INDICES = [2, 5, 6, 7, 8, 9, 10, 11]
MODEL_HASH = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
RUNTIME_HASH = 'ac75208179d00d86f378d5f10c65f9785028ce877cf57279f1a7b8dac54a25ff'
EXPECTED = {
    PARENT + 'plan.json': '17de8e335ab65107990e3697aa325eff9be814982f94e2ce1b7283bf77dd8ea4',
    PARENT + 'prompts.jsonl': '43532ebb81969db9c392177c344bb932e3f7f60a2a00b227e78c267211ce29bc',
    PARENT + 'review_rubric.json': '5d51ab2a6e4f410e7ca629e324b0beef87e553f87acd7777d9cdf4def2ede700',
    SOURCE: '0917500f53eefff3e984fb358158bd5853fddecd339b748ced70b5978e06e6b5',
}
CODE = ('scripts/prepare_qwen15b_gguf_v1.py', 'scripts/run_qwen15b_gguf_v1.py',
        'tests/test_qwen15b_gguf_v1.py', 'docs/qwen15b_gguf_v1_preregistration_ru.md')
SYSTEM = 'Write clear English prose. Return only the requested text, without a heading or commentary.'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def formatted(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def validate_parent(plan, prompts, windows):
    if len(plan) != 20 or len(prompts) != 15 or len(windows) != 20:
        raise ValueError('Required parent denominators: 20 plan, 15 prompts, 20 windows')
    if len({r['record_id'] for r in plan}) != 20 or len({r['source_group_id'] for r in plan}) != 20:
        raise ValueError('Duplicate parent identities')
    by_prompt = {p['record_id']: p for p in prompts}
    if len(by_prompt) != 15:
        raise ValueError('Duplicate prompts')
    selected = []
    for index, (item, window) in enumerate(zip(plan, windows), 1):
        if item['split'] != 'development' or window['split'] != 'development' or item['plan_index'] != index:
            raise ValueError('Development-only original order required')
        for k in ('record_id', 'source_group_id', 'source_window_sha256'):
            if item[k] != window[k]:
                raise ValueError('Parent/window identity differs: ' + k)
        text = window['source_window']
        if digest(text.encode()) != item['source_window_sha256'] or len(text.split()) != item['source_words']:
            raise ValueError('Source hash/word count mismatch')
        target = item['target_words']
        if target not in (250, 500) or window['original_long_target_words'] != target:
            raise ValueError('Invalid inherited target')
        gap = any('possible_antecedent_after_omitted_content' in x['flags'] for x in window['context_dependency_flags'])
        expected = ('review_required_context_gap' if gap else 'word_heuristic_met'
                    if len(text.split()) >= math.ceil(1.25 * target) else 'insufficient_selected_prose_words')
        if item['source_status'] != expected or window['status'] != expected:
            raise ValueError('Source-only gate mismatch')
        if expected == 'word_heuristic_met':
            p = by_prompt[item['record_id']]
            if p['split'] != 'development' or p['excerpt'] != text or p['excerpt_sha256'] != item['source_window_sha256']:
                raise ValueError('Prompt/source mismatch')
            if digest(p['prompt'].encode()) != item['prompt_sha256'] or p['prompt_sha256'] != item['prompt_sha256']:
                raise ValueError('Prompt text changed')
            if item['planned_action'] != 'generate_once':
                raise ValueError('Inherited action differs')
            if len(selected) < 8:
                selected.append(index)
        elif item['record_id'] in by_prompt or item['planned_action'] != 'abstain_source_gate':
            raise ValueError('Ineligible source has a prompt')
    if selected != SELECTED_INDICES:
        raise ValueError('First-eight source-only selection changed')
    return by_prompt


def build(root):
    for name, expected in EXPECTED.items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError('Frozen parent changed: ' + name)
    old_protocol = read(root / PARENT / 'protocol.prepared.json')
    if old_protocol['generator']['system'] != SYSTEM:
        raise ValueError('Frozen system instruction changed')
    old_plan = read(root / PARENT / 'plan.json')['records']
    source_prompts = rows(root / PARENT / 'prompts.jsonl')
    windows = rows(root / SOURCE)
    by_prompt = validate_parent(old_plan, source_prompts, windows)
    ledger, prompts = [], []
    for item in old_plan:
        row = dict(item)
        if item['source_status'] != 'word_heuristic_met':
            action = 'abstain_source_gate'
        elif item['plan_index'] in SELECTED_INDICES:
            action = 'generate_once'
        else:
            action = 'not_selected_fixed_eight_budget'
        row.update(parent_planned_action=item['planned_action'], planned_action=action,
                   generation_status_at_preparation='not_started' if action == 'generate_once' else action,
                   sampling_seed=20261024 + item['plan_index'])
        if action == 'generate_once':
            messages = [{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': by_prompt[item['record_id']]['prompt']}]
            row['messages_sha256'] = digest(formatted(messages))
            prompts.append({'record_id': item['record_id'], 'plan_index': item['plan_index'],
                            'messages': messages, 'messages_sha256': row['messages_sha256']})
        ledger.append(row)
    resources = {'model': {
        'repository': 'Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'revision': '91cad51170dc346986eccefdc2dd33a9da36ead9',
        'filename': 'qwen2.5-1.5b-instruct-q4_k_m.gguf', 'bytes': 1117320736, 'sha256': MODEL_HASH,
        'license': 'Apache-2.0',
        'url': 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf'},
        'runtime': {'tag': 'b11791', 'commit': 'ea9e3446b5c84a7f972cbbdea9df615fc4ff9159',
        'archive': 'llama-b11791-bin-ubuntu-x64.tar.gz', 'bytes': 22302292, 'sha256': RUNTIME_HASH,
        'license': 'MIT', 'url': 'https://github.com/ggml-org/llama.cpp/releases/download/b11791/llama-b11791-bin-ubuntu-x64.tar.gz'}}
    files = {'plan.json': formatted({'schema': ID + '-plan', 'planned_source_groups': 20,
              'selected_for_generation': 8, 'source_gates': 5, 'not_selected': 7, 'records': ledger}),
             'prompts.jsonl': b''.join((json.dumps(p, ensure_ascii=False, sort_keys=True) + '\n').encode() for p in prompts),
             'resources.json': formatted(resources)}
    protocol = {'schema': ID, 'status': 'prepared_requires_git_freeze', 'paid_calls': 0,
        'preparation_model_calls': 0, 'test_data_used': False, 'parent_git_commit': BASE_COMMIT,
        'implementation_base_git_commit': '0dd89f821e639ce66ed5c0708ae3a9db41644299',
        'input_hashes': {**EXPECTED, PARENT + 'protocol.prepared.json': digest((root / PARENT / 'protocol.prepared.json').read_bytes()),
                         OUT + '/preflight/runtime-preflight.json': digest((root / OUT / 'preflight/runtime-preflight.json').read_bytes())},
        'code_hashes': {p: digest((root / p).read_bytes()) for p in CODE},
        'files': {p: digest(data) for p, data in files.items()},
        'selection': {'original_plan_rows': 20, 'source_eligible': 15, 'selected': 8,
                      'source_gates': 5, 'not_selected': 7, 'selected_plan_indices': SELECTED_INDICES,
                      'rule': 'First eight source-gate-eligible rows in frozen original order; no replacements'},
        'context_tokens': 8192, 'max_output_tokens': 1536, 'reserved_tokens': 1,
        'token_gate': 'Reject if complete rendered prompt token count + 1536 + 1 > 8192; never truncate or shift',
        'sampling': {'temperature': .8, 'top_k': 50, 'top_p': .95, 'min_p': 0, 'typical_p': 1,
                     'repeat_penalty': 1.1, 'repeat_last_n': 8192, 'presence_penalty': 0,
                     'frequency_penalty': 0, 'dry_multiplier': 0, 'xtc_probability': 0,
                     'mirostat': 0, 'samplers': ['penalties', 'temperature', 'top_k', 'top_p'],
                     'seed_rule': '20261024 + original one-based plan index'},
        'runtime': {'threads': 6, 'parallel': 1, 'gpu_layers': 0, 'kv_type': 'f16',
                    'flash_attention': 'off', 'host': '127.0.0.1', 'cache_prompt': False,
                    'context_shift': False, 'startup_timeout_seconds': 120,
                    'per_record_timeout_seconds': 600, 'automatic_retries': 0,
                    'process_policy': 'Fresh local server per record; 600 seconds includes startup; kill entire owned process group in finally (up to6 seconds cleanup)'},
        'outputs': ['all20 ledger rows', 'started marker before each generation request', 'raw HTTP request/response',
                    'exact rendered prompt and token IDs', 'model/runtime hashes', 'server stdout/stderr',
                    'all errors/timeouts/empty/limit hits', 'whitespace word count and soft length band'],
        'required_before_execution': ['Root review and durable Git commit of this protocol, source-bound inputs and runner/tests',
            'Pass exact freeze commit explicitly; verify frozen files and pinned artifacts again before execution'],
        'analysis': {'generation_denominator': 8, 'source_ledger_denominator': 20,
            'word_count': 'len(text.split())', 'soft_band': 'Inherited target +/-10%, inclusive',
            'quality_review': 'Read every available whole output against its whole supplied source; preserve unavailable rows and disagreements',
            'rubric': PARENT + 'review_rubric.json',
            'rubric_denominator_override': 'Use 8 selected and 20 ledger; prior 15 denominator applies only to frozen parent',
            'automated_factual_pass': False, 'automatic_codec_evaluation': False},
        'limitations': ['Exploratory reused development cohort, not representative or blinded to earlier-stage history',
            'Size, quantization, runtime and sampling implementation differ; no causal model-size comparison',
            'Seed does not promise bitwise text reproducibility across hardware/compilers',
            'Source-only heuristic does not certify truth, sufficient factual material or summary quality',
            'No replacement, retries, output selection, output padding, closed test or paid API calls']}
    files['protocol.json'] = formatted(protocol)
    return files


def verify(root):
    for name, data in build(root).items():
        if (root / OUT / name).read_bytes() != data:
            raise ValueError('Prepared file differs: ' + name)
    return {'status': 'verified', 'planned': 20, 'selected': 8, 'source_gates': 5, 'not_selected': 7, 'model_calls': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == 'prepare':
        files = build(root)
        output = root / OUT
        output.mkdir(parents=True, exist_ok=True)
        if any((output / name).exists() for name in files):
            raise FileExistsError('Prepared files are immutable; do not overwrite')
        for name, data in files.items():
            with (output / name).open('xb') as stream:
                stream.write(data)
    print(json.dumps(verify(root), indent=2))


if __name__ == '__main__':
    main()
