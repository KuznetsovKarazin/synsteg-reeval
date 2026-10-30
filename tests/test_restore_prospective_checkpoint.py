"""Corrupt and malicious stage archives must never publish a restored run."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "restore_stage", Path(__file__).resolve().parents[1] / "scripts/restore_prospective_checkpoint.py")
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, data, kind in entries:
            item = tarfile.TarInfo(name)
            if kind in {"symlink", "hardlink", "directory"}:
                item.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE,
                             "directory": tarfile.DIRTYPE}[kind]
                item.linkname = "../escape"
                archive.addfile(item)
            else:
                if kind == "sparse":
                    item.type = tarfile.GNUTYPE_SPARSE
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))


@pytest.fixture
def case(tmp_path):
    frozen = {"protocol_sha256": "1" * 64, "code_sha256": {"example.py": "2" * 64}}
    frozen["freeze_id"] = sha(json.dumps(frozen, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode())
    files = {"freeze.json": json.dumps(frozen).encode(), "records/train/example.json": b'{"score":1}\n'}
    entries = [("pilot/" + name, data, "file") for name, data in files.items()]
    archive = tmp_path / "stage.tar.gz"
    write_archive(archive, entries)
    manifest = {"schema": "prospective-stage-checkpoint-v1", "stage": "train_generation",
                "run_name": "pilot", "freeze_id": frozen["freeze_id"], "source_commit": "a" * 40,
                "archive_bytes": archive.stat().st_size, "archive_sha256": sha(archive.read_bytes()),
                "files_sha256": {name: sha(data) for name, data in files.items()}}
    manifest_path = tmp_path / "stage.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, archive, manifest, files, entries, tmp_path / "restored"


def update_archive(case, entries):
    manifest_path, archive, manifest, *_ = case
    write_archive(archive, entries)
    manifest.update(archive_bytes=archive.stat().st_size, archive_sha256=sha(archive.read_bytes()))
    manifest_path.write_text(json.dumps(manifest))


def test_roundtrip_and_existing_destination_refusal(case):
    manifest_path, archive, manifest, files, _, output = case
    result = restore.restore_stage(manifest_path, archive, output)
    assert result["freeze_id"] == manifest["freeze_id"] and result["files"] == len(files)
    assert {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()} == files
    with pytest.raises(ValueError, match="overwrite"):
        restore.restore_stage(manifest_path, archive, output)
    empty = output.parent / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        restore.restore_stage(manifest_path, archive, empty)


@pytest.mark.parametrize("damage", ["archive", "member_hash", "freeze_id", "freeze_digest"])
def test_integrity_failures_leave_no_output(case, damage):
    manifest_path, archive, manifest, files, entries, output = case
    if damage == "archive":
        archive.write_bytes(archive.read_bytes() + b"corrupt")
    elif damage == "member_hash":
        manifest["files_sha256"]["records/train/example.json"] = "0" * 64
    elif damage == "freeze_id":
        manifest["freeze_id"] = "0" * 64
    else:
        freeze = json.loads(files["freeze.json"])
        freeze["protocol_sha256"] = "9" * 64
        bad = json.dumps(freeze).encode()
        manifest["files_sha256"]["freeze.json"] = sha(bad)
        update_archive(case, [("pilot/freeze.json", bad, "file"), entries[1]])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        restore.restore_stage(manifest_path, archive, output)
    assert not output.exists()


@pytest.mark.parametrize("damage", ["extra", "missing", "duplicate", "traversal", "wrong_prefix",
                                     "symlink", "hardlink", "directory", "sparse"])
def test_hash_consistent_invalid_members_leave_no_output(case, damage):
    manifest_path, archive, _, _, entries, output = case
    if damage == "missing":
        changed = entries[:1]
    elif damage == "duplicate":
        changed = entries + [entries[1]]
    else:
        name = {"extra": "pilot/extra.json", "traversal": "pilot/../escape",
                "wrong_prefix": "other/records/train/example.json"}.get(damage, entries[1][0])
        kind = damage if damage in {"symlink", "hardlink", "directory", "sparse"} else "file"
        changed = [entries[0], (name, b"x", kind)]
    update_archive(case, changed)
    with pytest.raises((ValueError, tarfile.TarError)):
        restore.restore_stage(manifest_path, archive, output)
    assert not output.exists() and not (output.parent / "escape").exists()


@pytest.mark.parametrize("with_content", [False, True])
def test_destination_created_during_verification_is_not_overwritten(case, monkeypatch, with_content):
    manifest_path, archive, _, _, _, output = case
    publish = restore.publish_no_overwrite
    def concurrent_destination(staged, destination):
        destination.mkdir()
        if with_content:
            (destination / "owner.txt").write_text("keep")
        publish(staged, destination)
    monkeypatch.setattr(restore, "publish_no_overwrite", concurrent_destination)
    with pytest.raises(FileExistsError):
        restore.restore_stage(manifest_path, archive, output)
    assert output.is_dir()
    assert sorted(p.name for p in output.iterdir()) == (["owner.txt"] if with_content else [])
    if with_content:
        assert (output / "owner.txt").read_text() == "keep"


def test_symlink_destination_parent_is_rejected(case):
    manifest_path, archive, _, _, _, output = case
    link = output.parent / "linked"
    link.symlink_to(output.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic link"):
        restore.restore_stage(manifest_path, archive, link / "new")
    assert not (output.parent / "new").exists()
