"""Exercise real frozen runner state transitions using synthetic generators only."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import checkpoint_large_campaign_development_v3 as audit
import resume_large_campaign_development_v3 as supervisor


@pytest.fixture
def restored(tmp_path):
    paths = ['configs/large_campaign_resume_v2.json', str(audit.FIRST_MANIFEST), str(audit.PROJECTOR),
             'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl',
             'scripts/run_budget_generation.py', 'scripts/run_claude_smoke.py',
             'src/synsteg_audit/budget_ledger.py', 'src/synsteg_audit/__init__.py']
    paths.extend(str(p.relative_to(REPO)) for p in (REPO / audit.RUN).rglob('*') if p.is_file())
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    return tmp_path


def receipt(root, count):
    return {'schema': 'large-campaign-verified-git-checkpoint-v3', 'repository': supervisor.REPO,
            'commit': 'c' * 40, 'readback_verified': True, 'expected_completed': count,
            'run_files_sha256': audit.run_file_hashes(root)}


def load_runner(root):
    sys.path.insert(0, str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location('synthetic_frozen_runner', root / 'scripts/run_budget_generation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_run(root, expected, *, fail_on=None, supplied_receipt=None):
    runner = load_runner(root)
    previous = supplied_receipt or receipt(root, expected)
    supervisor.install_locked_lease_guard(runner, root, previous, expected)
    calls = []
    def generate(row, limit):
        calls.append(row['record_id'])
        if len(calls) == fail_on:
            raise RuntimeError('Synthetic failure; no language model')
        model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        return {'cover_text': 'Synthetic retained output.', 'stop_reason': 'end_turn',
                'usage': {'input_tokens': 250, 'output_tokens': 20}, 'hit_output_limit': False,
                'seed': int(audit.canonical([20261024, model, row['record_id']])[:8], 16),
                'elapsed_seconds': .001, 'returned_model': model,
                'revision': 'fe8a4ea1ffedaf415f4da2f062534de366a451e6', 'dtype': 'float32'}
    kwargs = dict(prompts=root / 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl',
        output=root / audit.RUN, provider='local', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        revision='fe8a4ea1ffedaf415f4da2f062534de366a451e6', max_records=100, max_new_records=2,
        max_output_tokens=1024, split='development', dtype='float32', cpu_threads=6,
        seed=20261024, execute=True, local_generator=generate,
        runtime_reader=lambda: audit.load_json(root / audit.RUN / 'runtime.json'))
    code, summary = runner.run(**kwargs)
    return code, summary, calls


def test_restored_original_pair_has_the_complete_300_denominator(restored):
    value = audit.inspect(restored)
    assert value['status'] == 'verified'
    assert value['counts'] == {'completed': 2, 'not_attempted': 298}
    assert value['model_counts'] == {'tinyllama11': 2, 'qwen05': 0, 'falcon3_1b': 0}


def test_two_batches_advance_without_regenerating_or_changing_original_pair(restored, tmp_path):
    before = tmp_path / 'saved-before'
    supervisor.capture(restored, before)
    code, summary, calls_a = fake_run(restored, 2)
    assert code == 0 and len(calls_a) == 2
    assert summary['completed'] == 4 and summary['resumed_completed'] == 2
    assert summary['requests_attempted_this_invocation'] == 2
    assert supervisor.compare_old_evidence(before, restored) == []
    code, summary, calls_b = fake_run(restored, 4)
    assert code == 0 and len(calls_b) == 2 and set(calls_a).isdisjoint(calls_b)
    assert summary['completed'] == 6 and summary['resumed_completed'] == 4
    value = audit.inspect(restored)
    assert value['status'] == 'verified' and value['counts'] == {'completed': 6, 'not_attempted': 294}
    assert supervisor.compare_old_evidence(before, restored) == []


def test_stale_lease_checked_inside_original_ledger_lock_before_any_reservation(restored):
    stale = receipt(restored, 2)
    fake_run(restored, 2)
    previous_files = audit.run_file_hashes(restored)
    with pytest.raises(ValueError, match='Stale count'):
        fake_run(restored, 2, supplied_receipt=stale)
    assert audit.run_file_hashes(restored) == previous_files


def test_failure_stays_in_attempted_denominator_and_blocks_next_batch(restored):
    code, summary, calls = fake_run(restored, 2, fail_on=2)
    assert code == 3 and len(calls) == 2
    value = audit.inspect(restored)
    assert value['status'] == 'requires_review'
    assert value['counts'] == {'completed': 3, 'failed_or_unknown_no_retry': 1, 'not_attempted': 296}
    with pytest.raises(ValueError, match='failed/unknown'):
        supervisor.verify_lease(restored, receipt(restored, 4), 4)


def test_changed_runtime_receipt_is_rejected_before_generation(restored):
    prior = receipt(restored, 2)
    p = restored / audit.RUN / 'runtime.json'
    obj = audit.load_json(p)
    obj['packages']['torch'] = 'different'
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='fingerprint'):
        supervisor.verify_lease(restored, prior, 2)


def test_missing_attempt_is_not_relabelled_as_not_attempted(restored):
    p = next((restored / audit.RUN / 'attempts').glob('*.json'))
    p.unlink()
    value = audit.inspect(restored)
    assert value['status'] == 'requires_review' and value['attempted'] == 2
    assert value['counts']['not_attempted'] == 298


def test_external_runner_lock_blocks_resume_without_any_changes(restored):
    runner = load_runner(restored)
    before = audit.run_file_hashes(restored)
    with runner.CampaignLedger(restored / audit.RUN / 'local_ledger').locked():
        with pytest.raises(runner.BudgetError, match='locked'):
            fake_run(restored, 2)
    assert audit.run_file_hashes(restored) == before


def test_false_or_incomplete_git_receipt_is_rejected(restored):
    prior = receipt(restored, 2)
    prior['readback_verified'] = False
    with pytest.raises(ValueError, match='verified Git'):
        supervisor.verify_lease(restored, prior, 2)
    prior = receipt(restored, 2)
    prior['run_files_sha256'].pop(next(iter(prior['run_files_sha256'])))
    with pytest.raises(ValueError, match='Local run differs'):
        supervisor.verify_lease(restored, prior, 2)


def test_unstarted_other_model_does_not_disappear_from_denominator(restored):
    (restored / audit.RUN.parent / 'qwen05').mkdir()
    with pytest.raises(ValueError, match='Another model arm'):
        audit.inspect(restored)


def test_existing_batch_directory_is_never_reused(restored, monkeypatch):
    out = restored / supervisor.BATCHES / 'batch02'
    out.mkdir(parents=True)
    (out / 'evidence.txt').write_text('retained')
    source = restored / 'receipt.json'
    source.write_text(json.dumps(receipt(restored, 2)))
    monkeypatch.setattr(supervisor, 'verify_utility', lambda root: {})
    monkeypatch.setattr(supervisor, 'verify_frozen_environment', lambda root: audit.first_manifest(root))
    with pytest.raises(ValueError, match='already exists'):
        supervisor.execute(restored, 'batch02', 2, source, 'a' * 40)
    assert (out / 'evidence.txt').read_text() == 'retained'


def test_prior_ledger_entry_mutation_is_detected_separately_from_file_byte_changes(restored, tmp_path):
    before = tmp_path / 'before'
    supervisor.capture(restored, before)
    path = restored / audit.RUN / 'local_ledger/ledger.json'
    obj = audit.load_json(path)
    next(iter(obj['attempts'].values()))['created_at_unix'] += 1
    path.write_text(json.dumps(obj))
    assert any('Prior ledger entry changed' in issue for issue in supervisor.compare_old_evidence(before, restored))


def test_completed_100_cannot_request_another_batch(restored):
    with pytest.raises(ValueError, match='2 through 98'):
        supervisor.verify_lease(restored, receipt(restored, 100), 100)
