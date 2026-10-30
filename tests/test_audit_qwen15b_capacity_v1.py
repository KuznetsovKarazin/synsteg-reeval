"""Independent synthetic capacity contracts; no model or natural-cover evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_qwen15b_capacity_v1 as cap

FREEZE = 'b' * 40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cap.previous.formatted(value))


def synthetic_plan():
    rows = []
    for index in range(1, 21):
        action = ('generate_once' if index in cap.genprep.SELECTED_INDICES else
                  'abstain_source_gate' if index in (1, 3, 4, 12, 13) else 'not_selected_fixed_eight_budget')
        rows.append({'plan_index': index, 'record_id': f'synthetic-{index}',
            'source_group_id': f'synthetic-source-{index}', 'split': 'development',
            'planned_action': action, 'preferred_word_band': [225, 275],
            'sampling_seed': 20261024 + index, 'source_window_sha256': 'c' * 64})
    return rows


def synthetic_dictionary():
    return cap.base.PublicDictionary.from_mapping({'synthetic': ['amber', 'azure']}, version='synthetic-capacity-v1')


def synthetic_tree(tmp_path, monkeypatch):
    root = tmp_path / 'synthetic-tree'
    stage = root / cap.STAGE
    stage.mkdir(parents=True)
    for name in cap.CODE:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    dictionary = synthetic_dictionary()
    for name in cap.DICTIONARIES.values():
        write(root / name, dictionary.to_dict())
    policy = cap.universal.UniversalPolicy.from_dictionaries(dictionary, dictionary,
        version='morphology-complete-tag-spectrum-v3-reference-v1', radius=1, orbit_cap=4096)
    write(root / cap.POLICY, policy.to_dict())
    write(stage / 'plan.json', {'records': synthetic_plan()})
    write(stage / 'protocol.json', {'synthetic_generator_only': True})
    write(stage / 'resources.json', {'model': {'sha256': 'd' * 64}})
    write(stage / 'preflight/runtime-preflight.json', {'runtime': {
        'server_path': 'synthetic/server', 'files': [{'path': 'synthetic/server', 'sha256': 'a' * 64}]}})
    (stage / 'prompts.jsonl').write_bytes(b'')
    monkeypatch.setattr(cap.genprep, 'verify', lambda root: {'synthetic': True})
    cap.prepare(root, stage)
    return root, stage


def native_response(**updates):
    response = {'content': 'Synthetic amber prose.\r\nExact whitespace stays.  ',
        'tokens': [1, 2, 3], 'tokens_predicted': 3, 'tokens_evaluated': 12,
        'truncated': False, 'stop_type': 'eos', 'stopped_eos': True,
        'stopped_limit': False, 'stopped_word': False}
    response.update(updates)
    return response


def write_selected(stage, ledger, index, response=None, status=None, text=None, attempted=True):
    original = cap.read(stage / 'plan.json')['records'][index - 1]
    item = {**original, 'generation_request_attempted': attempted}
    directory = stage / 'run/attempts' / f'{index:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    if response is not None:
        item.update(cap.genrun.completion_summary(response, original))
        if response.get('truncated') is True:
            item['status'] = 'protocol_violation_truncated'
        write(directory / 'completion.response.json', response)
        (directory / 'cover.txt').write_bytes(item['text'].encode())
    else:
        item['status'] = status
        if text is not None:
            item.update(text=text, text_sha256=cap.sha(text.encode()))
            (directory / 'cover.txt').write_bytes(text.encode())
    ledger[index - 1] = item
    write(directory / 'attempt.json', item)


def save_ledger(stage, ledger):
    write(stage / 'run/ledger.json', ledger)
    write(stage / 'run/summary.json', {**cap.genrun.summarize(ledger),
        'finished_at': '2026-10-24T00:00:00Z', 'post_run_hashes_verified': True})


def completed_run(root, stage):
    plan = cap.read(stage / 'plan.json')['records']
    ledger = cap.genrun.initial_ledger(plan)
    for index in cap.genprep.SELECTED_INDICES:
        write_selected(stage, ledger, index, response=native_response())
    save_ledger(stage, ledger)
    write(stage / 'run/execution.json', {'freeze_commit': FREEZE,
        'protocol_sha256': cap.sha((stage / 'protocol.json').read_bytes()),
        'model_sha256': 'd' * 64, 'server_sha256': 'a' * 64})
    return ledger


def normalized_row(text='amber', availability='available'):
    return {'audit_record_id': 'synthetic:1', 'record_id': 'synthetic-1', 'plan_index': 1,
        'source_group_id': 'synthetic-source-1', 'split': 'development', 'cohort': 'qwen15b_gguf_v1',
        'planned_action': 'generate_once', 'generation_status': 'generated',
        'generation_request_attempted': True, 'text': text, 'text_sha256': cap.sha(text.encode()) if text is not None else None,
        'availability': availability, 'generation_metadata': {}}


def test_preparation_and_verification_are_noninference_and_exclusive(tmp_path, monkeypatch):
    root, stage = synthetic_tree(tmp_path, monkeypatch)
    protocol, plan, dictionaries, policy = cap.verify(root, stage)
    assert len(plan) == 20 and protocol['all_arm_rows'] == 180
    assert protocol['base_protocol'] == {'min_group_size': 1, 'max_bits_per_slot': 1}
    assert set(dictionaries) == set(cap.METHODS)
    assert not (stage / 'run').exists()
    assert not (stage / 'capacity-policy.json').exists()
    before = (stage / 'capacity-protocol.json').read_bytes()
    with pytest.raises(FileExistsError):
        cap.prepare(root, stage)
    assert (stage / 'capacity-protocol.json').read_bytes() == before


@pytest.mark.parametrize('fault', ['source', 'code', 'methods', 'arms', 'base_protocol', 'policy'])
def test_frozen_capacity_inputs_and_parameters_are_bound(tmp_path, monkeypatch, fault):
    root, stage = synthetic_tree(tmp_path, monkeypatch)
    if fault in ('source', 'code', 'policy'):
        name = cap.DICTIONARIES['base_v1'] if fault == 'source' else cap.CODE[0] if fault == 'code' else cap.POLICY
        with (root / name).open('ab') as stream:
            stream.write(b'\n')
    else:
        protocol = cap.read(stage / 'capacity-protocol.json')
        if fault == 'methods': protocol['methods'] = ['base_v1']
        elif fault == 'arms': protocol['arms'].pop()
        else: protocol['base_protocol']['max_bits_per_slot'] = 2
        write(stage / 'capacity-protocol.json', protocol)
    with pytest.raises(ValueError):
        cap.verify(root, stage)


def test_complete_ledger_retains20_and_exact_crlf_text(tmp_path, monkeypatch):
    root, stage = synthetic_tree(tmp_path, monkeypatch)
    completed_run(root, stage)
    rows, evidence = cap.load_ledger(root, stage, FREEZE)
    assert len(rows) == 20
    assert sum(r['availability'] == 'available' for r in rows) == 8
    assert sum(r['availability'] == 'source_gate_abstention' for r in rows) == 5
    assert sum(r['availability'] == 'not_selected_fixed_eight_budget' for r in rows) == 7
    assert rows[1]['text'] == 'Synthetic amber prose.\r\nExact whitespace stays.  '
    assert all(cap.sha((root / name).read_bytes()) == digest for name, digest in evidence.items())


@pytest.mark.parametrize('fault', ['missing_row', 'duplicate_row', 'order', 'plan_field', 'incomplete',
    'wrong_freeze', 'wrong_model', 'wrong_server', 'wrong_protocol', 'not_completed', 'summary_counts', 'text_hash',
    'cover', 'attempt', 'raw_response', 'gated_text', 'unselected_attempt'])
def test_ledger_provenance_and_full_denominator_fail_closed(tmp_path, monkeypatch, fault):
    root, stage = synthetic_tree(tmp_path, monkeypatch)
    ledger = completed_run(root, stage)
    if fault == 'missing_row': ledger.pop()
    elif fault == 'duplicate_row': ledger[-1] = deepcopy(ledger[0])
    elif fault == 'order': ledger[0], ledger[1] = ledger[1], ledger[0]
    elif fault == 'plan_field': ledger[1]['source_group_id'] = 'changed'
    elif fault == 'incomplete': ledger[1]['status'] = 'generation_request_started'
    elif fault == 'text_hash': ledger[1]['text_sha256'] = 'e' * 64
    elif fault == 'gated_text': ledger[0].update(text='Forbidden output', text_sha256=cap.sha(b'Forbidden output'))
    elif fault == 'unselected_attempt': ledger[13]['generation_request_attempted'] = True
    write(stage / 'run/ledger.json', ledger)
    if fault in ('wrong_freeze', 'wrong_model', 'wrong_server', 'wrong_protocol'):
        execution = cap.read(stage / 'run/execution.json')
        execution[{'wrong_freeze': 'freeze_commit', 'wrong_model': 'model_sha256',
                   'wrong_server': 'server_sha256', 'wrong_protocol': 'protocol_sha256'}[fault]] = 'f' * 64
        write(stage / 'run/execution.json', execution)
    elif fault in ('not_completed', 'summary_counts'):
        summary = cap.read(stage / 'run/summary.json')
        if fault == 'not_completed': summary['post_run_hashes_verified'] = False
        else: summary['generation_requests_attempted'] = 7
        write(stage / 'run/summary.json', summary)
    elif fault == 'cover': (stage / 'run/attempts/02/cover.txt').write_bytes(b'different cover')
    elif fault == 'attempt':
        attempt = cap.read(stage / 'run/attempts/02/attempt.json'); attempt['status'] = 'timeout'
        write(stage / 'run/attempts/02/attempt.json', attempt)
    elif fault == 'raw_response': write(stage / 'run/attempts/02/completion.response.json', native_response(content='changed'))
    with pytest.raises(ValueError):
        cap.load_ledger(root, stage, FREEZE)


@pytest.mark.parametrize('status', ['empty_output', 'response_schema_uncertain', 'protocol_violation_truncated',
                                  'timeout', 'request_error', 'server_error', 'context_budget_reject'])
def test_failed_outputs_are_preserved_but_never_measured_or_encoded(tmp_path, monkeypatch, status):
    root, stage = synthetic_tree(tmp_path, monkeypatch)
    ledger = completed_run(root, stage)
    if status == 'empty_output':
        write_selected(stage, ledger, 2, response=native_response(content=' \r\n', tokens=[], tokens_predicted=0))
    elif status == 'response_schema_uncertain':
        write_selected(stage, ledger, 2, response=native_response(tokens=None))
    elif status == 'protocol_violation_truncated':
        write_selected(stage, ledger, 2, response=native_response(truncated=True))
    else:
        write_selected(stage, ledger, 2, status=status, text='Preserved partial evidence.', attempted=status != 'context_budget_reject')
    save_ledger(stage, ledger)
    rows, _ = cap.load_ledger(root, stage, FREEZE)
    failed = rows[1]
    assert failed['availability'] == 'generation_' + status
    assert failed['text'] is not None
    _, _, dictionaries, policy = cap.verify(root, stage)
    def forbidden(*args, **kwargs):
        raise AssertionError('Unavailable output must not reach any codec analysis')
    monkeypatch.setattr(cap.base, 'analyze_text', forbidden)
    monkeypatch.setattr(cap.universal, 'analyze_text', forbidden)
    evaluated = cap.evaluate_record(root, failed, dictionaries, policy)
    for method in evaluated['methods'].values():
        assert method['capacity_bits'] is None
        assert all(a['encoding_called'] is False and a['mechanical_success'] is False for a in method['arms'].values())


def test_limit_stop_without_context_truncation_remains_available(tmp_path, monkeypatch):
    root, stage = synthetic_tree(tmp_path, monkeypatch)
    ledger = completed_run(root, stage)
    write_selected(stage, ledger, 2, response=native_response(stop_type='limit', stopped_eos=False, stopped_limit=True))
    save_ledger(stage, ledger)
    rows, _ = cap.load_ledger(root, stage, FREEZE)
    assert rows[1]['availability'] == 'available'
    assert rows[1]['generation_metadata']['stopped_limit'] is True


def test_raw_synthetic_capacity_uses_one_bit_and_exact_frame_thresholds():
    dictionary = synthetic_dictionary()
    row = normalized_row(' '.join(['amber'] * 23))
    result = cap.evaluate_raw(ROOT, row, dictionary)
    assert result['capacity_bits'] == 23
    assert [a['required_frame_bits'] for a in result['arms'].values()] == [24, 32, 152]
    assert all(a['status'] == 'insufficient_capacity' and not a['encoding_called'] for a in result['arms'].values())


def test_actual_synthetic_plain_roundtrips_preserve_changes_and_separate_semantics():
    dictionary = synthetic_dictionary()
    dictionaries = {name: dictionary for name in cap.METHODS}
    policy = cap.universal.UniversalPolicy.from_dictionaries(dictionary, dictionary,
        version='morphology-complete-tag-spectrum-v3-reference-v1', radius=1, orbit_cap=4096)
    row = normalized_row('Protected.\t' + ' '.join(['amber'] * 32) + '\nEnd.')
    result = cap.evaluate_record(ROOT, row, dictionaries, policy)
    for name, method in result['methods'].items():
        assert method['capacity_bits'] == 32
        assert method['semantic_safety_certified'] is False
        assert method['known_unsafe_reference'] is (name == 'base_v1')
        for arm_id in ('plain_length16__8', 'plain_length16__16'):
            arm = method['arms'][arm_id]
            assert arm['status'] == 'decoded' and arm['mechanical_success'] is True
            assert arm['payload_hex'] == arm['decoded_payload_hex']
            assert arm['semantic_success'] is None
            assert arm['stego_sha256'] == cap.sha(arm['stego_text'].encode())
            assert len(arm['token_changes']) == arm['changed_slots']
            assert arm['stego_text'].startswith('Protected.\t') and arm['stego_text'].endswith('\nEnd.')
            assert arm['receiver']['io_guard_self_checks'] == {'file': True, 'socket': True, 'subprocess': True}
        assert method['arms']['aes_siv__8']['status'] == 'insufficient_capacity'


def test_actual_synthetic_aes_roundtrip_at_exact152_bit_boundary():
    row = normalized_row(' '.join(['amber'] * 152))
    result = cap.evaluate_raw(ROOT, row, synthetic_dictionary())
    arm = result['arms']['aes_siv__8']
    assert result['capacity_bits'] == arm['required_frame_bits'] == 152
    assert arm['mechanical_success'] is True and arm['status'] == 'decoded'
    assert arm['payload_hex'] == arm['decoded_payload_hex']


def evaluated_twenty(root, stage):
    rows, _ = cap.load_ledger(root, stage, FREEZE)
    _, _, dictionaries, policy = cap.verify(root, stage)
    return [cap.evaluate_record(root, row, dictionaries, policy) for row in rows]


def test_summary_keeps_all180_and_gates_are_null_not_zero(tmp_path, monkeypatch):
    root, stage = synthetic_tree(tmp_path, monkeypatch)
    completed_run(root, stage)
    records = evaluated_twenty(root, stage)
    summary = cap.summarize(records)
    assert (summary['all_arm_rows'], summary['selected_arm_rows'], summary['source_gate_arm_rows'], summary['not_selected_arm_rows']) == (180, 72, 45, 63)
    assert summary['all180_outcomes'] == {'source_gate_abstention': 45, 'insufficient_capacity': 72, 'not_selected_fixed_eight_budget': 63}
    for method in summary['methods'].values():
        assert (method['planned_denominator'], method['selected_denominator'], method['available_denominator']) == (20, 8, 8)
        assert method['gross_capacity_bits']['count'] == 8
    for record in records:
        if record['planned_action'] != 'generate_once':
            assert all(m['capacity_bits'] is None for m in record['methods'].values())


@pytest.mark.parametrize('fault', ['missing_record', 'duplicate_record', 'missing_method', 'missing_arm',
                                  'unavailable_zero', 'unavailable_success'])
def test_summary_rejects_denominator_or_unavailable_relabeling(tmp_path, monkeypatch, fault):
    root, stage = synthetic_tree(tmp_path, monkeypatch)
    completed_run(root, stage)
    records = evaluated_twenty(root, stage)
    if fault == 'missing_record': records.pop()
    elif fault == 'duplicate_record': records[-1] = deepcopy(records[0])
    elif fault == 'missing_method': records[1]['methods'].pop('base_v1')
    elif fault == 'missing_arm': records[1]['methods']['base_v1']['arms'].pop('aes_siv__8')
    elif fault == 'unavailable_zero': records[0]['methods']['base_v1']['capacity_bits'] = 0
    else: records[0]['methods']['base_v1']['arms']['plain_length16__8']['mechanical_success'] = True
    with pytest.raises(ValueError):
        cap.summarize(records)


def test_synthetic_execute_writes180_rows_and_refuses_overwrite(tmp_path, monkeypatch):
    root, stage = synthetic_tree(tmp_path, monkeypatch)
    completed_run(root, stage)
    output = stage / 'synthetic-capacity-audit'
    summary = cap.execute(root, stage, output, FREEZE)
    assert summary['all_arm_rows'] == 180
    assert len((output / 'records.jsonl').read_bytes().splitlines()) == 20
    assert len((output / 'arms.jsonl').read_bytes().splitlines()) == 180
    execution = cap.read(output / 'execution.json')
    assert all(cap.sha((output / name).read_bytes()) == digest for name, digest in execution['output_sha256'].items())
    before = (output / 'execution.json').read_bytes()
    with pytest.raises(FileExistsError):
        cap.execute(root, stage, output, FREEZE)
    assert (output / 'execution.json').read_bytes() == before
