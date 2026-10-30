"""Integrity and filesystem-safety checks for durable run snapshots."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location('checkpoint', Path(__file__).resolve().parents[1] / 'scripts/package_run_checkpoint.py')
cp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cp)


@pytest.fixture
def source(tmp_path):
    p = tmp_path / 'run'
    (p / 'checkpoints').mkdir(parents=True)
    (p / 'manifest.json').write_text('{"status":"complete"}\n')
    (p / 'scores.jsonl').write_bytes(bytes(range(256)) * 400)
    (p / 'checkpoints/x.json').write_text('{"score":0.3}')
    return p


def test_roundtrip_deterministic_chunks_and_no_overwrite(source, tmp_path):
    a, b, target = tmp_path / 'a', tmp_path / 'b', tmp_path / 'restored'
    first = cp.create_snapshot(source, a, quiescent=True)
    second = cp.create_snapshot(source, b, quiescent=True)
    assert first == second
    assert all(p['bytes'] <= 45000 for p in first['parts'])
    cp.restore_snapshot(a, target)
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()} == {
        p.relative_to(target): p.read_bytes() for p in target.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='overwrite'):
        cp.restore_snapshot(a, target)
    with pytest.raises(ValueError, match='overwrite'):
        cp.create_snapshot(source, a, quiescent=True)


def test_requires_quiescence_and_rejects_locks_and_symlinks(source, tmp_path):
    with pytest.raises(ValueError, match='quiescent'):
        cp.create_snapshot(source, tmp_path / 'a')
    (source / '.writer.lock').write_text('123')
    with pytest.raises(ValueError, match='Lock file'):
        cp.create_snapshot(source, tmp_path / 'a', quiescent=True)
    (source / '.writer.lock').unlink()
    (source / 'link').symlink_to(source / 'scores.jsonl')
    with pytest.raises(ValueError, match='Symbolic link'):
        cp.create_snapshot(source, tmp_path / 'a', quiescent=True)


@pytest.mark.parametrize('damage', ['chunk', 'whole_hash', 'file_hash', 'order'])
def test_corruption_never_publishes_output(source, tmp_path, damage):
    p = tmp_path / 'snapshot'
    m = cp.create_snapshot(source, p, quiescent=True)
    if damage == 'chunk':
        (p / m['parts'][0]['path']).write_bytes(b'broken')
    elif damage == 'whole_hash':
        m['archive_sha256'] = '0' * 64
    elif damage == 'file_hash':
        m['files']['scores.jsonl']['sha256'] = '0' * 64
    else:
        m['parts'][0]['path'] = '../other.bin'
    (p / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):
        cp.restore_snapshot(p, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('name,kind', [('../escape','file'), ('/escape','file'), ('ok','symlink'), ('ok','hardlink')])
def test_even_hash_consistent_malicious_tar_is_rejected(source, tmp_path, name, kind):
    p = tmp_path / 'snapshot'
    m = cp.create_snapshot(source, p, quiescent=True)
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as tar:
        item = tarfile.TarInfo(name)
        if kind == 'file':
            item.size = 1
            tar.addfile(item, io.BytesIO(b'x'))
        else:
            item.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
            item.linkname = '../escape'
            tar.addfile(item)
    raw = gzip.compress(data.getvalue(), mtime=0)
    (p / 'part-00000.bin').write_bytes(raw)
    m.update(archive_bytes=len(raw), archive_sha256=hashlib.sha256(raw).hexdigest(),
             parts=[{'path':'part-00000.bin','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}],
             files={'ok':{'bytes':1 if kind == 'file' else 0,'sha256':hashlib.sha256(b'x' if kind == 'file' else b'').hexdigest()}})
    (p / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):
        cp.restore_snapshot(p, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    assert not (tmp_path / 'escape').exists()


def test_optional_cache_exclusion_is_explicit(source, tmp_path):
    (source / 'writing.tmp').write_text('temporary')
    m = cp.create_snapshot(source, tmp_path / 'snapshot', quiescent=True, exclude_checkpoints=True)
    assert m['exclude_checkpoints']
    assert m['excluded_paths'] == ['checkpoints/x.json', 'writing.tmp']
    assert 'scores.jsonl' in m['files']


def test_nested_snapshot_refusal_does_not_write_source(source):
    with pytest.raises(ValueError, match='outside'):
        cp.create_snapshot(source, source / 'new-parent' / 'snapshot', quiescent=True)
    assert not (source / 'new-parent').exists()


def test_large_inventory_is_bounded_and_verified(tmp_path):
    source = tmp_path / 'many'
    source.mkdir()
    for index in range(600):
        (source / (hashlib.sha256(str(index).encode()).hexdigest() + '.json')).write_text(str(index))
    snapshot = tmp_path / 'snapshot'
    m = cp.create_snapshot(source, snapshot, quiescent=True)
    header = json.loads((snapshot / 'manifest.json').read_text())
    assert 'files' not in header and header['file_count'] == 600
    assert all(p.stat().st_size <= 45000 for p in snapshot.iterdir())
    restored = tmp_path / 'restored'
    cp.restore_snapshot(snapshot, restored)
    assert len(list(restored.iterdir())) == 600
    part = snapshot / header['file_inventory']['parts'][0]['path']
    part.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='inventory'):
        cp.restore_snapshot(snapshot, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
