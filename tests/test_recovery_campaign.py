import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recovery_campaign', ROOT / 'scripts/run_recovery_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def records():
    return [dict(record_id=f'{topic}-{genre}-{length}', source_group_id=f'{topic}-{genre}-{length}',
                 topic=f'topic-{topic:02}', genre=f'genre-{genre}', target_words=length, split='development')
            for topic in range(10) for genre in range(5) for length in (250, 500)]


def test_balanced_content_only_order_independent_of_input_order():
    rows = records()
    ordered = campaign.development_order(rows)
    assert ordered == campaign.development_order(list(reversed(rows)))
    assert len({r['record_id'] for r in ordered}) == 100
    assert sum(r['target_words'] == 250 for r in ordered[:30]) == 15
    assert all(sum(r['topic'] == f'topic-{n:02}' for r in ordered[:30]) == 3 for n in range(10))


def test_reject_duplicate_groups():
    rows = records()
    rows[-1]['source_group_id'] = rows[0]['source_group_id']
    with pytest.raises(ValueError, match='distinct'):
        campaign.development_order(rows)


def test_commands_share_ledger_and_do_not_offer_test():
    cfg = campaign.configuration(ROOT / 'configs/recovery_development_v1.json')
    calls = list(campaign.commands(cfg, 'claude', None))
    assert len(calls) == 14
    for output, args in calls:
        assert args[args.index('--campaign-ledger') + 1] == campaign.CAMPAIGN / 'ledger'
        assert args[args.index('--budget-usd') + 1] == '20'
        assert args[args.index('--split') + 1] == 'development'
        assert '--execute' not in args
    local = list(campaign.commands(cfg, 'local', 'tinyllama11'))
    assert len(local) == 1
    assert local[0][1][local[0][1].index('--max-output-tokens') + 1] == 1024


def test_checkpoint_is_readable_with_persistent_lock_files(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'ROOT', tmp_path)
    monkeypatch.setattr(campaign, 'CAMPAIGN', tmp_path / 'runs/recovery-development-v1')
    campaign.CAMPAIGN.mkdir(parents=True)
    (campaign.CAMPAIGN / 'assets.json').write_text('{}')
    cfg = {'campaign_budget_usd': '20', 'local_models': [], 'claude_models': ['model']}
    campaign.checkpoint(cfg)
    archives = list((tmp_path / 'deliveries').glob('*.zip'))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        inventory = json.loads(archive.read('SHA256SUMS.json'))
        assert list(inventory) == ['runs/recovery-development-v1/assets.json']
        assert not any(n.endswith('.lock') for n in archive.namelist())


def test_checkpoint_does_not_copy_active_run_or_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'ROOT', tmp_path)
    monkeypatch.setattr(campaign, 'CAMPAIGN', tmp_path / 'runs/recovery-development-v1')
    campaign.CAMPAIGN.mkdir(parents=True)
    (campaign.CAMPAIGN / 'assets.json').write_text('private-key-fixture')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'private-key-fixture')
    cfg = {'campaign_budget_usd': '20', 'local_models': [], 'claude_models': ['model']}
    with campaign.CampaignLedger(campaign.CAMPAIGN / 'models/model/run_guard').locked():
        with pytest.raises(campaign.BudgetError, match='locked'):
            campaign.checkpoint(cfg)
    with pytest.raises(ValueError, match='Credential'):
        campaign.checkpoint(cfg)
    assert not list(tmp_path.rglob('*.zip'))


def test_frozen_file_rejects_edits(tmp_path):
    path = tmp_path / 'record.json'
    campaign.stable_file(path, b'first')
    campaign.stable_file(path, b'first')
    with pytest.raises(ValueError, match='Frozen file differs'):
        campaign.stable_file(path, b'edited')
