"""Synthetic tests only: no real development cover encoding or decoding."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prospective_runner_test', ROOT / 'scripts/run_prospective_development_transport_v1.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
FROZEN = ROOT / 'runtime/prospective-development-transport-v1/frozen'


@pytest.fixture(scope='module')
def dependency():
    return runner.dependencies(ROOT, FROZEN)


def fixture_records(config, dictionary):
    records = []
    # A declared synthetic repeated-word stress fixture, never an observed cover.
    good = ' '.join([dictionary.classes[0].words[0]] * 800)
    for model in config['local_models']:
        for order, meta in enumerate(config['selected_record_metadata'], 1):
            status = 'not_attempted'
            text = None
            if model['alias'] == 'tinyllama11' and order <= 3:
                status = 'completed'
                text = {1: good, 2: 'zzzzzz ' * 20, 3: 'dexterous ' * 20}[order]
            if model['alias'] == 'tinyllama11' and order == 4:
                status = 'failed'
            if model['alias'] == 'tinyllama11' and order == 5:
                status = 'unknown'
            identity = runner.sha(runner.canonical(['synthetic', model['alias'], meta['record_id']]))
            records.append(dict(meta, model_alias=model['alias'], source_order=order,
                identity=identity, status=status, cover_text=text,
                attempt_sha256=runner.sha(('synthetic-' + identity).encode()) if text else None,
                cover_sha256=runner.sha(text.encode()) if text else None))
    return records


@pytest.fixture(scope='module')
def evaluation(dependency):
    plan, config, payload, shared, dictionaries, resources, *_ = dependency
    records = fixture_records(config, dictionaries['v2'])
    cells, jobs, expected = runner.evaluate(records, payload, plan, shared, dictionaries)
    return records, cells, jobs, expected


def test_full_grid_denominator_and_status_axes(evaluation):
    records, cells, jobs, expected = evaluation
    assert len(records) == 300 and len(cells) == 15300
    assert len({r['row_id'] for r in cells}) == 15300
    assert sum(r['grid'] == 'primary' for r in cells) == 8100
    assert sum(r['resource_status'] == 'ready' for r in cells) == 7200
    assert sum(r['resource_status'] == 'BLOCKED_RESOURCE_OR_STAGE_NOT_RUN' for r in cells) == 8100
    assert sum(r['execution_status'] == 'not_run_cover_unavailable' for r in cells) == 297 * 51
    assert sum(r['execution_status'] == 'not_run_resource_blocked' for r in cells) == 3 * 27
    assert sum(r['encoder_called'] is True for r in cells) == 3 * 24
    assert set(expected) == {j['row_id'] for j in jobs}


def test_missing_blocked_are_null_not_zero(evaluation):
    _, cells, _, _ = evaluation
    for row in cells:
        if row['execution_status'].startswith('not_run_'):
            for key in ('capacity_bits', 'delivered_payload_bits', 'useful_decoded_payload_bytes',
                        'book_json_bytes', 'book_zlib9_bytes', 'per_text_map_bytes',
                        'full_exact', 'fresh_receiver_exact', 'encoder_called'):
                assert row[key] is None
        if row['cover_status'] != 'completed':
            assert row['cover_words'] is None and row['cover_sha256'] is None


def test_zero_capacity_retained_as_executed_refusal(evaluation):
    _, cells, _, _ = evaluation
    zero = [r for r in cells if r['model_alias'] == 'tinyllama11' and r['source_order'] == 2 and r['arm'].startswith('public_')]
    assert len(zero) == 24
    for row in zero:
        assert row['execution_status'] == 'insufficient_capacity'
        assert row['capacity_bits'] == row['delivered_payload_bits'] == 0
        assert row['useful_decoded_payload_bytes'] == row['per_text_map_bytes'] == 0
        assert row['max_payload_bytes'] is None and row['can_frame_empty'] is False
        assert row['requested_payload_bits'] in (16, 128, 256, 512)


def test_exact_byte_capacity_and_frame_costs(evaluation):
    _, cells, _, _ = evaluation
    for row in cells:
        if row['cover_status'] == 'completed' and row['arm'].startswith('public_'):
            overhead = 144 if row['arm'].endswith('aes_siv') else 16
            cap = row['capacity_bits']
            assert row['frame_overhead_bits'] == overhead
            assert row['minimum_nonempty_frame_bits'] == overhead + 8
            assert row['max_payload_bytes'] == (min(65535, (cap - overhead) // 8) if cap >= overhead else None)
            assert row['can_send_one_byte'] == (cap >= overhead + 8)
            assert row['per_receiver_key_bytes'] == 32
            assert row['book_json_bytes'] == row['book_zlib9_bytes'] == row['per_text_map_bytes'] == 0


def test_receiver_new_process_exact_and_no_sender_fields(evaluation, tmp_path):
    _, original_cells, jobs, expected = evaluation
    cells = copy.deepcopy(original_cells)
    recipient = tmp_path / 'recipient'
    recipient.mkdir()
    shutil.copyfile(FROZEN / 'src/synsteg_audit/shared_dictionary.py', recipient / 'shared_dictionary.py')
    for version in ('v1', 'v2'):
        shutil.copyfile(FROZEN / f'data/shared_dictionary_recovery_{version}.json', recipient / f'dictionary_{version}.json')
    runner.write_rows(recipient / 'receiver_jobs.jsonl.gz', jobs)
    for job in jobs:
        assert set(job) == {'row_id', 'kind', 'dictionary_version', 'stego_text', 'fixture_key_hex', 'protocol'}
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/run_prospective_development_transport_v1.py'), '--receiver', str(recipient)], cwd=recipient, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    outcomes = runner.read_rows(recipient / 'receiver_results.jsonl.gz')
    runner.finalize(cells, expected, outcomes)
    for row in cells:
        if row['row_id'] in expected:
            assert row['execution_status'] == 'full_exact'
            assert row['useful_decoded_payload_bytes'] * 8 == row['requested_payload_bits'] == row['delivered_payload_bits']
            assert row['decoded_sha256'] == row['payload_sha256']
    assert not any(r['execution_status'] == 'awaiting_receiver' for r in cells)


def test_receiver_payload_mismatch_is_not_success(evaluation):
    _, initial, _, expected = evaluation
    cells = copy.deepcopy(initial)
    outcomes = [{'row_id': k, 'status': 'decoded', 'decoded_bits': bits, 'error': None} for k, bits in expected.items()]
    assert outcomes
    outcomes[0]['decoded_bits'] = '1' if outcomes[0]['decoded_bits'][0] == '0' else '0'
    runner.finalize(cells, expected, outcomes)
    row = next(r for r in cells if r['row_id'] == outcomes[0]['row_id'])
    assert row['execution_status'] == 'decoder_mismatch' and row['full_exact'] is False
    assert row['delivered_payload_bits'] == row['useful_decoded_payload_bytes'] == 0


def test_missing_or_duplicate_receiver_outcome_rejected(evaluation):
    _, initial, _, expected = evaluation
    outcomes = [{'row_id': k, 'status': 'decoded', 'decoded_bits': bits, 'error': None} for k, bits in expected.items()]
    for bad in (outcomes[:-1], outcomes + outcomes[:1]):
        with pytest.raises(ValueError, match='membership'):
            runner.finalize(copy.deepcopy(initial), expected, bad)


def test_summary_full_denominators_and_amortization(dependency, evaluation):
    records, initial, _, expected = evaluation
    cells = copy.deepcopy(initial)
    runner.finalize(cells, expected, [{'row_id': k, 'status': 'decoded', 'decoded_bits': b, 'error': None} for k, b in expected.items()])
    summary, resources = runner.summarize(cells, records, dependency[5])
    assert summary['completed_covers'] == 3 and summary['total_cells'] == 15300
    assert summary['cover_status_counts'] == {'not_attempted': 295, 'completed': 3, 'failed': 1, 'unknown': 1}
    for group in summary['groups']:
        assert group['planned_cells'] == (300 if group['model_alias'] == 'all' else 100)
        assert group['executed_cells'] + group['unmeasured_cells'] == group['planned_cells']
        if group['arm'] == 'original_native':
            assert group['executed_cells'] == 0 and group['mean_capacity_bits_executed'] is None
    for entry in resources['public_dictionaries']:
        assert entry['installation_bytes_per_completed_cover'] == entry['one_time_file_bytes'] / 3
        assert entry['installation_bytes_per_planned_cover'] == entry['one_time_file_bytes'] / 300


def test_no_clobber_before_any_read_or_encoding(tmp_path, monkeypatch):
    output = tmp_path / 'already-there'
    output.mkdir()
    monkeypatch.setattr(runner, 'activation_preflight', lambda *a: pytest.fail('must refuse before preflight'))
    with pytest.raises(ValueError, match='Output exists'):
        runner.run(tmp_path, tmp_path, output, tmp_path / 'receipt.json')


@pytest.mark.parametrize('attack', ['not_verified', 'bad_commit', 'bad_hash', 'missing_observed', 'wrong_observed'])
def test_external_git_gate_rejects_bad_receipt(tmp_path, monkeypatch, attack):
    root = tmp_path
    (root / runner.ACTIVATION).parent.mkdir(parents=True)
    (root / 'source.py').write_text('safe synthetic fixture\n')
    (root / 'payload.json').write_text('{}\n')
    (root / 'checkpoint.json').write_text('{}\n')
    activation = {'experiment_id': runner.VERSION, 'feasibility_protocol_sha256': runner.PLAN_SHA,
        'runtime': {'synthetic': True}, 'implementation_files_sha256': {'source.py': runner.sha((root / 'source.py').read_bytes())},
        'observed_completed_covers_at_implementation_freeze': 13, 'target': {'checkpoint': 'checkpoint.json'}}
    runner.write_json(root / runner.ACTIVATION, activation)
    plan = {'payloads': {'map': 'payload.json', 'map_sha256': runner.sha((root / 'payload.json').read_bytes())}}
    monkeypatch.setattr(runner, 'dependencies', lambda *a: (plan,))
    monkeypatch.setattr(runner, 'runtime_identity', lambda: {'synthetic': True})
    monkeypatch.setattr(runner, 'IMPLEMENTATION_FILES', ('source.py',))
    verified = {**activation['implementation_files_sha256'], runner.ACTIVATION: runner.sha((root / runner.ACTIVATION).read_bytes()),
        runner.PLAN: runner.PLAN_SHA, 'payload.json': plan['payloads']['map_sha256'],
        'checkpoint.json': runner.sha((root / 'checkpoint.json').read_bytes())}
    receipt = {'remote_readback_verified': True, 'implementation_freeze_commit': 'a' * 40,
        'checkpoint_commit': 'b' * 40, 'verified_files_sha256': verified,
        'observed_completed_covers_at_implementation_freeze': 13}
    if attack == 'not_verified': receipt['remote_readback_verified'] = False
    if attack == 'bad_commit': receipt['implementation_freeze_commit'] = 'a'
    if attack == 'bad_hash': receipt['verified_files_sha256']['source.py'] = '0' * 64
    if attack == 'missing_observed': del receipt['observed_completed_covers_at_implementation_freeze']
    if attack == 'wrong_observed': receipt['observed_completed_covers_at_implementation_freeze'] = 12
    path = root / 'receipt.json'
    runner.write_json(path, receipt)
    with pytest.raises(ValueError): runner.activation_preflight(root, root, path)


def test_public_dependency_identity_failure_aborts(dependency, tmp_path):
    shutil.copytree(FROZEN, tmp_path / 'frozen')
    (tmp_path / 'frozen/data/shared_dictionary_recovery_v2.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen dependency mismatch'):
        runner.dependencies(ROOT, tmp_path / 'frozen')
