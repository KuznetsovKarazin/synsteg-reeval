"""Synthetic offline evaluator checks; none are research-quality observations."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('long_summary_audit_test', ROOT / 'scripts/audit_long_summary_development_v3.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.formatted(value))


def lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join(audit.previous.canonical(v) + b'\n' for v in values))


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / 'project'; prepared = root / audit.PREPARED
    for name in (*audit.CODE_FILES, 'scripts/run_budget_generation.py', 'src/synsteg_audit/budget_ledger.py'):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    dictionaries = {v: audit.base.PublicDictionary.from_mapping(
        {'quality': ['fine', 'good'], 'time': ['clock', 'time']}, 'synthetic-' + v) for v in ('v1', 'v2')}
    for v, dictionary in dictionaries.items():
        write(root / audit.RESOURCE_FILES[v], dictionary.to_dict())
    policy = audit.context.ProtectedPhrasePolicy.from_phrases(dictionaries['v1'], ['over time'], version='synthetic-guard')
    write(root / audit.RESOURCE_FILES['guard'], policy.to_dict())
    plan, windows, prompts = [], [], []
    for i in range(20):
        text = 'Synthetic source material. ' + 'good ' * 400
        digest = audit.sha(text.encode()); eligible = i < 15
        item = {'plan_index': i + 1, 'record_id': f'r{i}', 'source_group_id': f'g{i}', 'split': 'development',
                'source_status': 'word_heuristic_met' if eligible else 'insufficient_selected_prose_words',
                'source_words': len(text.split()), 'source_window_sha256': digest, 'target_words': 250,
                'preferred_word_band': [225, 275], 'planned_action': 'generate_once' if eligible else 'abstain_source_gate',
                'derived_sampling_seed': i, 'attribution': {'license': 'Synthetic fixture'}}
        windows.append({'record_id': f'r{i}', 'source_group_id': f'g{i}', 'split': 'development',
                        'source_window': text, 'source_window_sha256': digest,
                        'status': item['source_status'], 'original_long_target_words': 250})
        if eligible:
            prompt = f'Synthetic source-bound instruction {i}.'
            item['prompt_sha256'] = audit.sha(prompt.encode())
            prompts.append({**item, 'topic': 'fixture', 'genre': 'fixture', 'prompt': prompt,
                            'excerpt': text, 'excerpt_sha256': digest})
        else:
            item['prompt_sha256'] = None
        plan.append(item)
    lines(root / audit.SOURCE_WINDOWS, windows)
    write(prepared / 'plan.json', {'records': plan})
    lines(prepared / 'prompts.jsonl', prompts)
    write(prepared / 'review_rubric.json', {'factual_ratings': None})
    generator = {'provider': 'local', 'model': 'synthetic/model', 'revision': 'synthetic-revision', 'dtype': 'float32',
                 'max_output_tokens': 1536, 'system': 'Synthetic test system', 'master_seed': 1,
                 'cpu_threads': 1, 'temperature': .8, 'top_p': .95, 'top_k': 50}
    protocol = {'schema': 'synsteg-long-summary-development-v3', 'planned_source_groups': 20,
                'planned_generations': 15, 'paid_calls_allowed': False, 'test_data_used': False, 'generator': generator,
                'input_hashes': {name: audit.sha((root / name).read_bytes()) for name in
                    (*audit.CODE_FILES, *audit.RESOURCE_FILES.values(), audit.SOURCE_WINDOWS)},
                'files': {name: audit.sha((prepared / name).read_bytes()) for name in
                    ('plan.json', 'prompts.jsonl', 'review_rubric.json')}}
    write(prepared / 'protocol.prepared.json', protocol)
    run = {'schema': 'synsteg-budget-generation-run-v1', 'provider': 'local', 'model': generator['model'],
           'revision': generator['revision'], 'dtype': generator['dtype'], 'split': 'development',
           'selected_records': [{k: p[k] for k in audit.META_KEYS} for p in prompts], 'selected_records_count': 15,
           'prompts_sha256': audit.sha((prepared / 'prompts.jsonl').read_bytes()), 'automatic_retries': 0,
           'settings': {k: generator[k] for k in ('max_output_tokens', 'system', 'cpu_threads', 'temperature', 'top_p', 'top_k')},
           'runner_sha256': audit.sha((root / 'scripts/run_budget_generation.py').read_bytes()),
           'ledger_code_sha256': audit.sha((root / 'src/synsteg_audit/budget_ledger.py').read_bytes()),
           'transport_code_sha256': None, 'pricing_sha256': None}
    run['settings']['seed'] = generator['master_seed']
    write(prepared / 'run/run.json', run)
    write(prepared / 'run/runtime.json', {'schema': 'synthetic-runtime'})
    return root, prepared, plan, run, dictionaries, policy


def attempt(workspace, index, state='completed', text='No lexical slots here.', stop='end_turn'):
    root, prepared, plan, run, _, _ = workspace
    meta = run['selected_records'][index]; rid = meta['record_id']
    identity = audit.canonical_hash([run['provider'], run['model'], rid])
    fingerprint = audit.canonical_hash({'record': meta, 'provider': run['provider'], 'model': run['model'],
        'revision': run['revision'], 'dtype': run['dtype'], 'settings': run['settings'],
        'pricing_sha256': run['pricing_sha256'], 'prompts_sha256': run['prompts_sha256'],
        'runtime_sha256': audit.canonical_hash(audit.read(prepared / 'run/runtime.json')),
        'implementation': {k: run[k] for k in ('runner_sha256', 'ledger_code_sha256', 'transport_code_sha256')}})
    value = {'schema': 'synsteg-budget-generation-attempt-v1', 'identity': identity,
        'record': meta, 'provider': run['provider'], 'model': run['model'], 'status': state,
        'automatic_retries': 0, 'request_fingerprint': fingerprint,
        'result': {'cover_text': text, 'stop_reason': stop, 'returned_model': run['model'],
                   'revision': run['revision'], 'seed': plan[index]['derived_sampling_seed'],
                   'hit_output_limit': False, 'usage': {'input_tokens': 1, 'output_tokens': 1}, 'elapsed_seconds': 0.1}
                  if state == 'completed' else None}
    path = prepared / 'run/attempts' / (identity + '.json')
    write(path, value)
    return path


def test_complete_denominators_retain_missing_failure_refusal_empty_and_zero_capacity(workspace):
    root, prepared, *_ = workspace
    attempt(workspace, 0)
    attempt(workspace, 1, state='failed_or_unknown_no_retry')
    attempt(workspace, 2, text='I decline this fixture.', stop='refusal')
    attempt(workspace, 3, text='   ')
    result = audit.execute(root, prepared, prepared / 'audit')
    assert result['planned_source_groups'] == 20 and result['planned_generations'] == 15
    assert result['availability_counts'] == {'available': 1, 'failed_or_unknown_no_retry': 1,
        'model_refusal': 1, 'empty_output': 1, 'not_attempted': 11, 'source_gate_abstention': 5}
    rows = audit.jsonlines(prepared / 'audit/records.jsonl')
    assert len(rows) == 20 and all(set(r['methods']) == set(audit.METHODS) for r in rows)
    assert all(r['source_reference_capacity']['base_v1']['capacity_bits'] > 0 for r in rows)
    for method in audit.METHODS:
        arms = result['methods'][method]['arms']
        assert all(a['full_success'] == 0 and sum(a['statuses_all20'].values()) == 20 for a in arms.values())
        assert rows[0]['methods'][method]['capacity_bits'] == 0
        assert not any(a['encoding_called'] for a in rows[0]['methods'][method]['arms'].values())
    assert result['factual_quality'] == 'not_assessed'
    with pytest.raises(FileExistsError):
        audit.execute(root, prepared, prepared / 'audit')


def test_real_independent_receivers_and_protected_spellings(workspace):
    root, _, _, _, dictionaries, policy = workspace
    text = 'over time ' + 'good ' * 180
    result = audit.measure(root, text, 'available', 'public-group', dictionaries, policy)
    for method in audit.METHODS:
        for arm in result[method]['arms'].values():
            assert arm['status'] == 'decoded' and arm['full_success']
            assert arm['receiver']['io_guard_self_checks'] == {'file': True, 'socket': True, 'subprocess': True}
            assert arm['receiver']['input_keys'] == sorted(['key_hex', 'stego_text', 'public_dictionary', 'public_protocol'] +
                (['public_policy'] if method == 'context_guard_v1' else []))
            if method == 'context_guard_v1':
                assert arm['stego_text'].split()[:2] == ['over', 'time']
    assert result['base_v1']['capacity_bits'] == 181
    assert result['context_guard_v1']['capacity_bits'] == 180


@pytest.mark.parametrize('mutation', ['source', 'prompt', 'code', 'dictionary'])
def test_changed_frozen_input_is_rejected_before_evaluation(workspace, mutation):
    root, prepared, *_ = workspace
    names = {'source': root / audit.SOURCE_WINDOWS, 'prompt': prepared / 'prompts.jsonl',
             'code': root / audit.CODE_FILES[0], 'dictionary': root / audit.RESOURCE_FILES['v2']}
    path = names[mutation]; path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed'):
        audit.execute(root, prepared, prepared / 'audit')
    assert not (prepared / 'audit').exists()


@pytest.mark.parametrize('mutation', ['provider', 'record', 'fingerprint', 'duplicate'])
def test_wrong_attempt_identity_is_rejected(workspace, mutation):
    root, prepared, *_ = workspace
    path = attempt(workspace, 0); value = audit.read(path)
    if mutation == 'provider': value['provider'] = 'claude'
    elif mutation == 'record': value['record']['prompt_sha256'] = '0' * 64
    elif mutation == 'fingerprint': value['request_fingerprint'] = '0' * 64
    else: shutil.copy2(path, path.with_name('duplicate.json'))
    if mutation != 'duplicate': write(path, value)
    with pytest.raises(ValueError): audit.execute(root, prepared, prepared / 'audit')
    assert not (prepared / 'audit').exists()


def test_independent_receiver_failure_never_becomes_success(workspace, monkeypatch):
    root, _, _, _, dictionaries, policy = workspace
    def fail(*args, **kwargs): raise ValueError('Synthetic receiver failed')
    monkeypatch.setattr(audit.previous, 'isolated_receive', fail)
    result = audit.measure(root, 'good ' * 180, 'available', 'public-group', dictionaries, policy)
    assert all(a['encoding_called'] and not a['full_success'] and a['status'] == 'codec_or_receiver_failure'
               for m in result.values() for a in m['arms'].values())


def test_concurrent_source_change_prevents_publication(workspace, monkeypatch):
    root, prepared, *_ = workspace
    original = audit.measure
    changed = False
    def mutate(*args, **kwargs):
        nonlocal changed
        if not changed:
            path = root / audit.SOURCE_WINDOWS
            path.write_bytes(path.read_bytes() + b'\n')
            changed = True
        return original(*args, **kwargs)
    monkeypatch.setattr(audit, 'measure', mutate)
    with pytest.raises(ValueError, match='Input changed during audit'):
        audit.execute(root, prepared, prepared / 'audit')
    assert not (prepared / 'audit').exists()
