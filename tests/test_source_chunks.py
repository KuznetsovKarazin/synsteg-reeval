"""The Git chunk transport must preserve source identity and fail atomically."""
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest import mock
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("import_source", ROOT / "scripts/import_source.py")
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)


@pytest.fixture
def chunks(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("synsteg_reproduction/example.txt", b"submitted bytes")
    raw = archive.read_bytes()
    archive.unlink()
    parts_dir = archive.with_name(archive.name + ".parts")
    parts_dir.mkdir()
    parts = []
    for index, offset in enumerate(range(0, len(raw), 64)):
        data = raw[offset:offset + 64]
        name = f"part-{index:03d}.bin"
        (parts_dir / name).write_bytes(data)
        parts.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    digest = hashlib.sha256(raw).hexdigest()
    manifest = {"format_version": 1, "archive_name": archive.name,
                "archive_bytes": len(raw), "archive_sha256": digest,
                "chunk_bytes": 64, "parts": parts}
    (parts_dir / "manifest.json").write_text(json.dumps(manifest))
    with mock.patch.object(importer, "EXPECTED_SHA256", digest):
        yield archive, parts_dir, manifest, raw


def test_reconstruction_equals_original_and_imports(chunks):
    archive, _, _, raw = chunks
    result = importer.import_source(archive, archive.parent / "source")
    assert archive.read_bytes() == raw
    assert result["file_count"] == 1
    assert (archive.parent / "source/example.txt").read_bytes() == b"submitted bytes"
    assert not list(archive.parent.glob(".archive-*"))


@pytest.mark.parametrize("failure", ["missing", "corrupt", "reordered", "wrong_whole_hash"])
def test_bad_chunks_do_not_publish_partial_archive(chunks, failure):
    archive, parts_dir, manifest, _ = chunks
    first = parts_dir / manifest["parts"][0]["path"]
    if failure == "missing":
        first.unlink()
    elif failure == "corrupt":
        raw = first.read_bytes()
        first.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    elif failure == "reordered":
        manifest["parts"][0], manifest["parts"][1] = manifest["parts"][1], manifest["parts"][0]
    else:
        manifest["archive_sha256"] = "0" * 64
    (parts_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        importer.assemble_archive(archive)
    assert not archive.exists()
    assert not list(archive.parent.glob(".archive-*"))


def test_existing_archive_is_never_overwritten(chunks):
    archive, _, _, _ = chunks
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        importer.assemble_archive(archive)
    assert archive.read_bytes() == b"changed"


def test_direct_zip_import_still_supported(chunks):
    archive, parts_dir, _, raw = chunks
    archive.write_bytes(raw)
    (parts_dir / "manifest.json").unlink()
    assert importer.import_source(archive, archive.parent / "source")["file_count"] == 1


def test_submitted_chunks_reconstruct_exact_source(tmp_path):
    parts_dir = ROOT / "data/source/synsteg_reproduction.zip.parts"
    if not parts_dir.is_dir():
        pytest.skip("submitted chunks unavailable")
    manifest = json.loads((parts_dir / "manifest.json").read_text())
    digest = hashlib.sha256()
    size = 0
    for index, part in enumerate(manifest["parts"]):
        assert part["path"] == f"part-{index:03d}.bin"
        raw = (parts_dir / part["path"]).read_bytes()
        assert len(raw) == part["bytes"]
        assert hashlib.sha256(raw).hexdigest() == part["sha256"]
        digest.update(raw)
        size += len(raw)
    assert size == manifest["archive_bytes"] == 31728727
    assert digest.hexdigest() == manifest["archive_sha256"] == importer.EXPECTED_SHA256
