#!/usr/bin/env python3
"""Portable development campaign: prepare, cache, run, audit and checkpoint.

No train/validation/test inference is exposed here. Every model uses the same
content-only development order. The Claude runner owns the single budget ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synsteg_audit.budget_ledger import CampaignLedger, BudgetError
DEFAULT_CONFIG = ROOT / 'configs/recovery_development_v1.json'
CAMPAIGN = ROOT / 'runs/recovery-development-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def stable_file(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Frozen file differs; preserve this campaign and use a new version: {path}')
        return
    with path.open('xb') as stream:
        stream.write(content)


def run_child(arguments):
    command = [sys.executable, *[str(x) for x in arguments]]
    print('Running:', subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def configuration(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('schema') != 'synsteg-recovery-development-campaign-v1'
            or value.get('campaign_id') != 'recovery-development-v1'
            or value.get('scope') != 'development_only'
            or value.get('test_execution_enabled') is not False):
        raise ValueError('Only the named development campaign is supported')
    return value


def development_order(records):
    """Thirty-record prefix: three/topic and 15 per length; no model output."""
    dev = [r for r in records if r['split'] == 'development']
    if len(dev) != 100 or len({r['source_group_id'] for r in dev}) != 100:
        raise ValueError('Require 100 distinct development source groups')
    topics = defaultdict(list)
    for row in dev:
        topics[row['topic']].append(row)
    if len(topics) != 10 or any(len(rows) != 10 for rows in topics.values()):
        raise ValueError('Require ten development cells per topic')
    prefix, remainder = [], []
    order_key = lambda r: hashlib.sha256(('recovery-dev-order-v1:' + r['record_id']).encode()).hexdigest()
    for index, topic in enumerate(sorted(topics)):
        selected = []
        for length in (250, 500):
            rows = sorted((r for r in topics[topic] if r['target_words'] == length), key=order_key)
            if len(rows) != 5 or len({r['genre'] for r in rows}) != 5:
                raise ValueError('Require five genres per topic and requested length')
            count = 2 if (index % 2 == 0) == (length == 250) else 1
            selected.extend(rows[:count])
            remainder.extend(rows[count:])
        prefix.extend(sorted(selected, key=order_key))
    return prefix + sorted(remainder, key=order_key)


def asset_binding(config_path, cfg):
    source = ROOT / cfg['source_directory']
    manifest_path = source / 'manifest.json'
    if not manifest_path.exists():
        raise ValueError('Source snapshot is absent; run prepare first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = cfg.get('source_manifest_sha256')
    if expected and sha(manifest_path) != expected:
        raise ValueError('Source manifest differs from the committed campaign identity')
    if manifest.get('test_generation_allowed') is not False or manifest.get('records') != 1600:
        raise ValueError('Expected the separately identified 1600-source development-only snapshot')
    prompts = source / 'prompts.jsonl'
    if sha(prompts) != manifest['prompts_sha256']:
        raise ValueError('Source prompts hash mismatch')
    dictionary = ROOT / cfg['dictionary']
    if sha(dictionary) != cfg['dictionary_file_sha256']:
        raise ValueError('Public dictionary file hash mismatch')
    files = [config_path, dictionary, manifest_path, prompts, ROOT / cfg['pricing'],
             ROOT / 'requirements-neural.txt', ROOT / 'requirements-shared-dictionary.txt',
             ROOT / 'pyproject.toml',
             Path(__file__), ROOT / 'scripts/run_budget_generation.py',
             ROOT / 'src/synsteg_audit/budget_ledger.py',
             ROOT / 'scripts/run_claude_smoke.py', ROOT / 'scripts/check_claude_access.py',
             ROOT / 'scripts/audit_recovery_development.py',
             ROOT / 'scripts/run_shared_dictionary_pilot.py',
             ROOT / 'src/synsteg_audit/shared_dictionary.py']
    return {'schema': 'synsteg-development-assets-v1', 'test_execution_enabled': False,
            'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}}


def prepare(config_path, cfg):
    source = ROOT / cfg['source_directory']
    builder = ROOT / 'scripts/build_wikipedia_panel_recovery.py'
    args = [builder, '--output-dir', source]
    if (source / 'manifest.json').exists():
        args.append('--verify')
    if run_child(args):
        raise ValueError('Source collection/validation failed; no generation started')
    binding = asset_binding(config_path, cfg)
    records = [json.loads(line) for line in (source / 'prompts.jsonl').read_text(encoding='utf-8').splitlines()]
    dev = development_order(records)
    prompts = b''.join((json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8') for r in dev)
    binding['development_prompts_sha256'] = hashlib.sha256(prompts).hexdigest()
    binding['development_order_rule'] = 'recovery-dev-order-v1; 30-prefix=3/topic,15/length'
    stable_file(CAMPAIGN / 'development-prompts.jsonl', prompts)
    # Materialize every planned model before the first generation. If budget or
    # network failure stops the campaign early, unstarted models stay visible.
    plan_hashes = {}
    for provider in ('local', 'claude'):
        for output, arguments in commands(cfg, provider, None):
            completed = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if completed.returncode:
                raise ValueError('Cannot prepare model plan: ' + completed.stderr.decode('utf-8', errors='replace'))
            preview = json.loads(completed.stdout)
            if preview.get('status') != 'preview_only_no_requests':
                raise ValueError('Unexpected generation preview status')
            stable_file(output / 'run.json', encoded(preview['plan']))
            plan_hashes[output.name] = sha(output / 'run.json')
    binding['model_plan_hashes'] = plan_hashes
    stable_file(CAMPAIGN / 'assets.json', encoded(binding))
    print(json.dumps({'status': 'prepared_development_only', 'development_records': len(dev),
                      'prefix_lengths': dict(Counter(r['target_words'] for r in dev[:30])),
                      'campaign': str(CAMPAIGN), 'paid_requests_sent': 0}, indent=2))


def verify_prepared(config_path, cfg):
    frozen = json.loads((CAMPAIGN / 'assets.json').read_text(encoding='utf-8'))
    current = asset_binding(config_path, cfg)
    if current['files'] != frozen['files'] or sha(CAMPAIGN / 'development-prompts.jsonl') != frozen['development_prompts_sha256']:
        raise ValueError('Campaign code/resources changed after prepare; do not continue this run')
    for alias, digest in frozen['model_plan_hashes'].items():
        if sha(CAMPAIGN / 'models' / alias / 'run.json') != digest:
            raise ValueError('A planned model manifest changed after prepare')
    return frozen


def selected_models(cfg, provider, selected):
    rows = (cfg['local_models'] if provider == 'local' else
            [{'alias': model, 'model_id': model} for model in cfg['claude_models']])
    if selected:
        rows = [r for r in rows if selected in (r['alias'], r['model_id'])]
        if not rows:
            raise ValueError('Unknown model selector for this provider')
    return rows


def cache_models(cfg, selected):
    from huggingface_hub import snapshot_download
    for model in selected_models(cfg, 'local', selected):
        print('Caching pinned model:', model['model_id'], model['revision'], flush=True)
        cached = Path(snapshot_download(repo_id=model['model_id'], revision=model['revision'], token=False,
                          allow_patterns=['*.safetensors', '*.json', '*.model', 'merges.txt', 'vocab.txt'], max_workers=2))
        inventory = {}
        for path in sorted(p for p in cached.rglob('*') if p.is_file()):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            inventory[path.relative_to(cached).as_posix()] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
        stable_file(CAMPAIGN / 'model_assets' / (model['alias'] + '.json'),
                    encoded({'model_id': model['model_id'], 'revision': model['revision'], 'files': inventory}))


def commands(cfg, provider, selected):
    for model in selected_models(cfg, provider, selected):
        count = cfg['local_records_per_model'] if provider == 'local' else cfg['claude_records_per_model']
        if model['model_id'] == 'claude-haiku-5-5':
            count = cfg['haiku_development_records']
        output = CAMPAIGN / 'models' / model['alias']
        args = [ROOT / 'scripts/run_budget_generation.py', '--provider', provider, '--model', model['model_id'],
                '--prompts', CAMPAIGN / 'development-prompts.jsonl', '--output', output,
                '--split', 'development', '--max-records', count,
                '--max-output-tokens', cfg[provider + '_max_output_tokens']]
        if provider == 'claude':
            args += ['--pricing', ROOT / cfg['pricing'], '--campaign-ledger', CAMPAIGN / 'ledger',
                     '--budget-usd', cfg['campaign_budget_usd']]
        else:
            args += ['--revision', model['revision'], '--dtype', cfg['dtype'],
                     '--cpu-threads', cfg['cpu_threads'], '--seed', cfg['seed']]
        yield output, args


def checkpoint(cfg):
    """A local integrity-checked snapshot; not a claim of remote backup."""
    with ExitStack() as stack:
        # Same acquisition order as generation: ledger before per-run guard.
        stack.enter_context(CampaignLedger(CAMPAIGN / 'ledger', cfg['campaign_budget_usd']).locked())
        for item in cfg['local_models']:
            stack.enter_context(CampaignLedger(CAMPAIGN / 'models' / item['alias'] / 'local_ledger').locked())
        for alias in sorted([m['alias'] for m in cfg['local_models']] + cfg['claude_models']):
            stack.enter_context(CampaignLedger(CAMPAIGN / 'models' / alias / 'run_guard').locked())
        return checkpoint_locked(cfg)


def checkpoint_locked(cfg):
    files = sorted(p for p in CAMPAIGN.rglob('*') if p.is_file() and not p.name.endswith('.lock'))
    if not files:
        raise ValueError('No campaign files to checkpoint')
    content = [(p, p.read_bytes()) for p in files]
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    if key and any(key.encode() in data for _, data in content):
        raise ValueError('Credential found in checkpoint input; no archive created')
    inventory = {p.relative_to(ROOT).as_posix(): hashlib.sha256(data).hexdigest() for p, data in content}
    name = 'recovery-development-v1-' + str(time.time_ns()) + '.zip'
    destination = ROOT / 'deliveries' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in content:
            archive.writestr(path.relative_to(ROOT).as_posix(), data)
        archive.writestr('SHA256SUMS.json', encoded(inventory))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() or any(hashlib.sha256(archive.read(name)).hexdigest() != digest for name, digest in inventory.items()):
            raise ValueError('Checkpoint verification failed')
    print(json.dumps({'checkpoint': str(destination), 'sha256': sha(destination), 'files': len(files),
                      'remote_backup_verified': False}, indent=2))


def generate(cfg, provider, selected, execute):
    if execute and provider == 'claude' and not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError('ANTHROPIC_API_KEY is missing; no paid request sent')
    for output, args in commands(cfg, provider, selected):
        if execute:
            args += ['--execute']
        code = run_child(args)
        if execute:
            checkpoint(cfg)
        if code:
            summary_path = output / 'summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
            if summary.get('status') == 'model_access_unverified_no_generation':
                print('Model unavailable without generation:', output.name, flush=True)
                continue
            return code
    return 0


def audit(cfg):
    runs = sorted(p.parent for p in (CAMPAIGN / 'models').glob('*/run.json'))
    if not runs:
        raise ValueError('No generated development plans to audit')
    output = CAMPAIGN / 'audits' / str(time.time_ns())
    args = [ROOT / 'scripts/audit_recovery_development.py', '--dictionary', ROOT / cfg['dictionary'], '--output', output]
    for run in runs:
        args += ['--run', run]
    code = run_child(args)
    if code == 0:
        checkpoint(cfg)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'cache-models', 'local', 'claude', 'audit', 'status', 'checkpoint'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--model')
    parser.add_argument('--execute', action='store_true', help='Execute the already authorized local/paid development stage')
    args = parser.parse_args(argv)
    try:
        path = args.config.resolve()
        cfg = configuration(path)
        if args.stage == 'prepare':
            prepare(path, cfg)
            return 0
        verify_prepared(path, cfg)
        if args.stage in ('local', 'claude'):
            return generate(cfg, args.stage, args.model, args.execute)
        if args.stage == 'cache-models':
            cache_models(cfg, args.model)
        elif args.stage == 'audit':
            return audit(cfg)
        elif args.stage == 'checkpoint':
            checkpoint(cfg)
        else:
            summaries = {p.parent.name: json.loads(p.read_text(encoding='utf-8')) for p in (CAMPAIGN / 'models').glob('*/summary.json')}
            planned = {}
            for p in (CAMPAIGN / 'models').glob('*/run.json'):
                plan = json.loads(p.read_text(encoding='utf-8'))
                planned[p.parent.name] = {'planned': plan['selected_records_count'],
                                         'status': summaries.get(p.parent.name, {}).get('status', 'not_started')}
            print(json.dumps({'status': 'prepared_development_only', 'planned_models': planned, 'model_summaries': summaries,
                              'test_execution_enabled': False}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, BudgetError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
