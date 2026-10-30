"""Independent bookkeeping/immutability controls using wholly synthetic inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('morphology_audit', ROOT / 'scripts/audit_morphology_development_v3.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.previous.formatted(data))


def synthetic_rows():
    rows = []
    for cohort, count in audit.previous.COUNTS.items():
        for i in range(count):
            group = i % 2 if cohort == 'historical_covers' else i
            gate = cohort == 'long_summary_covers' and i < 5
            text = None if gate else 'plain prose'
            row = {'cohort': cohort, 'cohort_index': i, 'audit_record_id': f'{cohort}:{i}',
                'record_id': f'record-{group}', 'source_group_id': f'group-{group}', 'split': 'development',
                'text': text, 'text_sha256': audit.sha(text.encode()) if text else None,
                'availability': 'source_gate_abstention' if gate else 'available'}
            if cohort == 'long_summary_covers':
                row['planned_action'] = 'abstain_source_gate' if gate else 'generate_once'
            rows.append(row)
    return rows


def write_rows(root, rows):
    path = root / audit.INPUTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join(audit.previous.canonical(r) + b'\n' for r in rows))


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    write_rows(root, synthetic_rows())
    dictionary = audit.base.PublicDictionary.from_mapping({'pace': ['quick', 'rapid']}, 'synthetic-morphology-v3')
    reference = audit.base.PublicDictionary.from_mapping({'pace': ['quick', 'rapid']}, 'synthetic-v2')
    policy = audit.universal.UniversalPolicy.from_dictionaries(reference, reference)
    write_json(root / audit.DICTIONARY, dictionary.to_dict())
    write_json(root / audit.REFERENCE_DICTIONARY, reference.to_dict())
    write_json(root / audit.REFERENCE_POLICY, policy.to_dict())
    for name in audit.CODE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Synthetic hash-bound code fixture; never executed.\n')
    for name in ['annotations.json', 'provenance.json', 'trials.jsonl']:
        write_json(root / 'results/morphology-recovery-v3' / name, {'synthetic': True})
    write_json(root / 'data/morphology-recovery-v3/source/manifest.json', {'synthetic': True})
    prepared = root / audit.STAGE
    audit.prepare(root, prepared)
    return root, prepared


def test_all_synthetic_planned_records_and_source_gates_retained(setup):
    root, prepared = setup
    protocol, rows, _, _ = audit.verify(root, prepared)
    assert len(rows) == 144
    assert sum(r['availability'] == 'source_gate_abstention' for r in rows) == 5
    result = audit.execute(root, prepared, prepared / 'audit', 'a' * 40)
    assert result['planned_records'] == 144 and result['arms'] == 432
    assert result['all_outcomes'] == {'insufficient_capacity': 417, 'source_gate_abstention': 15}


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'closed_split', 'wrong_hash', 'gate_has_text', 'null_has_hash'])
def test_input_denominator_and_provenance_rejections(tmp_path, fault):
    rows = synthetic_rows()
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows[-1]['audit_record_id'] = rows[0]['audit_record_id']
    elif fault == 'closed_split':
        rows[0]['split'] = 'test'
    elif fault == 'wrong_hash':
        rows[0]['text_sha256'] = '0' * 64
    elif fault == 'gate_has_text':
        gate = next(r for r in rows if r['availability'] == 'source_gate_abstention')
        gate.update(text='plain prose', text_sha256=audit.sha(b'plain prose'))
    else:
        next(r for r in rows if r['availability'] == 'source_gate_abstention')['text_sha256'] = '0' * 64
    write_rows(tmp_path, rows)
    with pytest.raises(ValueError):
        audit.validate_inputs(tmp_path / audit.INPUTS)


@pytest.mark.parametrize('manifest', ['code_files', 'source_files'])
def test_missing_manifest_maps_rejected(setup, manifest):
    root, prepared = setup
    p = audit.read(prepared / 'protocol.json')
    p[manifest] = {}
    write_json(prepared / 'protocol.json', p)
    with pytest.raises(ValueError):
        audit.verify(root, prepared)


def test_removing_build_and_bundle_provenance_is_rejected(setup):
    root, prepared = setup
    p = audit.read(prepared / 'protocol.json')
    p['source_files'] = {n: h for n, h in p['source_files'].items() if n in audit.SOURCE}
    write_json(prepared / 'protocol.json', p)
    with pytest.raises(ValueError):
        audit.verify(root, prepared)


@pytest.mark.parametrize('directory', ['results/morphology-recovery-v3', 'data/morphology-recovery-v3/source'])
def test_new_unbound_file_changes_inventory_and_is_rejected(setup, directory):
    root, prepared = setup
    write_json(root / directory / 'unexpected-later-file.json', {'changed': True})
    with pytest.raises(ValueError):
        audit.verify(root, prepared)


@pytest.mark.parametrize('target', ['input', 'dictionary', 'policy'])
def test_modified_bound_resource_rejected(setup, target):
    root, prepared = setup
    path = {'input': root / audit.INPUTS, 'dictionary': root / audit.DICTIONARY, 'policy': prepared / 'policy.json'}[target]
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        audit.verify(root, prepared)


@pytest.mark.parametrize('state', ['model_refusal', 'not_attempted', 'failed', 'unknown', 'empty_output', 'source_gate_abstention'])
def test_each_unavailable_state_preserves_three_arms(state):
    dictionary = audit.base.PublicDictionary.from_mapping({'pace': ['quick', 'rapid']}, 'unavailable-fixture')
    policy = audit.universal.UniversalPolicy.from_dictionaries(dictionary, dictionary)
    row = {'audit_record_id': 'fixture', 'cohort': 'source_excerpts', 'source_group_id': 'fixture',
           'text': None, 'text_sha256': None, 'availability': state}
    result = audit.previous.evaluate(ROOT, row, {'morph_v3': dictionary}, {'morph_v3': policy})
    arms = result['methods']['universal_morph_v3']['arms']
    assert len(arms) == 3
    assert all(a['status'] == state and not a['encoding_called'] and not a['mechanical_success'] for a in arms.values())


def test_existing_preparation_and_output_are_not_overwritten(setup):
    root, prepared = setup
    before = (prepared / 'protocol.json').read_bytes()
    with pytest.raises(FileExistsError):
        audit.prepare(root, prepared)
    assert (prepared / 'protocol.json').read_bytes() == before
    output = prepared / 'already_exists'
    output.mkdir()
    with pytest.raises(FileExistsError):
        audit.execute(root, prepared, output, 'a' * 40)


@pytest.mark.parametrize('commit', [None, '', 'main', 'a' * 39, 'g' * 40])
def test_nonexact_freeze_identity_is_rejected(setup, commit):
    root, prepared = setup
    with pytest.raises(ValueError):
        audit.execute(root, prepared, prepared / 'new', commit)
