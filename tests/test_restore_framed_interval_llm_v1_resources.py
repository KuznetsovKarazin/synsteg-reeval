"""Small filesystem/mock-network contracts; no public downloads or model calls."""
import dataclasses
import gzip
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    'scoped_qwen_resource_restore', ROOT / 'scripts/restore_framed_interval_llm_v1_resources.py')
M = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = M
MODULE_SPEC.loader.exec_module(M)


def identity(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def archive_bytes(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, kind, value in entries:
            member = tarfile.TarInfo(name)
            if kind == 'file':
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
            else:
                member.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
                member.linkname = value
                archive.addfile(member)
    return gzip.compress(stream.getvalue(), mtime=0)


@pytest.fixture
def resources():
    library, model, bridge = b'frozen library', b'frozen model', b'Git bridge'
    entries = [('llama-test/libx.so.1', 'file', library),
               ('llama-test/libx.so', 'symlink', 'libx.so.1'),
               ('llama-test/unrelated-program', 'file', b'never extracted')]
    archive = archive_bytes(entries)
    payloads = {'runtime.tar.gz': archive, 'model.gguf': model, 'bridge': bridge,
                'runtime/llama-test/libx.so.1': library,
                'runtime/llama-test/libx.so': library}
    files = {name: identity(data) for name, data in payloads.items()}
    downloads = {'runtime.tar.gz': {**files['runtime.tar.gz'],
                                  'url': 'https://github.com/example/runtime.tar.gz'},
                 'model.gguf': {**files['model.gguf'],
                                'url': 'https://huggingface.co/example/model.gguf'}}
    return M.ResourceSpec(files, downloads, 'runtime.tar.gz', 'bridge', 'runtime/llama-test'), payloads


def write(base, name, data):
    target = base / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def network(spec, payloads, calls):
    by_url = {item['url']: payloads[name] for name, item in spec.downloads.items()}
    def open_mock(url, timeout):
        calls.append(url)
        assert timeout == 60
        return io.BytesIO(by_url[url])
    return open_mock


def changed_archive(spec, payloads, entries):
    payloads = dict(payloads)
    payloads['runtime.tar.gz'] = archive_bytes(entries)
    files = {**spec.files, 'runtime.tar.gz': identity(payloads['runtime.tar.gz'])}
    downloads = {**spec.downloads, 'runtime.tar.gz': {
        **spec.downloads['runtime.tar.gz'], **files['runtime.tar.gz']}}
    return dataclasses.replace(spec, files=files, downloads=downloads), payloads


def test_production_scope_is_exactly_41_channel_resources_without_temporary_fragments():
    spec = M.load_spec(ROOT)
    assert len(spec.files) == 41 and set(spec.downloads) == {M.MODEL, M.ARCHIVE}
    assert spec.files[spec.bridge]['sha256'] == M.BRIDGE_SHA256
    assert not any(Path(name).name.startswith('.') for name in spec.files)


def test_default_missing_verification_creates_nothing_and_never_uses_network(tmp_path, resources):
    spec, payloads = resources
    base, calls = tmp_path / 'absent', []
    result = M.restore(base, spec, opener=network(spec, payloads, calls))
    assert result['status'] == 'missing_resources_no_changes'
    assert len(result['missing_files']) == 5 and not base.exists() and calls == []


def test_missing_private_bridge_blocks_before_large_public_downloads(tmp_path, resources):
    spec, payloads = resources
    base, calls = tmp_path / 'absent', []
    with pytest.raises(ValueError, match='private Git bridge'):
        M.restore(base, spec, allow_download=True, opener=network(spec, payloads, calls))
    assert calls == [] and not base.exists()


def test_existing_wrong_file_blocks_all_network_and_preserves_bytes(tmp_path, resources):
    spec, payloads = resources
    write(tmp_path, 'bridge', payloads['bridge'])
    wrong = write(tmp_path, 'model.gguf', b'wrong')
    calls = []
    with pytest.raises(ValueError, match='Existing resource differs'):
        M.restore(tmp_path, spec, allow_download=True, opener=network(spec, payloads, calls))
    assert wrong.read_bytes() == b'wrong' and calls == []
    assert not list(tmp_path.glob('.framed-resource-stage-*'))


def test_complete_correct_files_and_allowlisted_soname_symlink_are_read_only(tmp_path, resources):
    spec, payloads = resources
    for name, data in payloads.items():
        if name != 'runtime/llama-test/libx.so':
            write(tmp_path, name, data)
    alias = tmp_path / 'runtime/llama-test/libx.so'
    alias.symlink_to('libx.so.1')
    before = {name: ((tmp_path / name).lstat().st_ino, (tmp_path / name).lstat().st_mtime_ns)
              for name in payloads}
    calls = []
    result = M.restore(tmp_path, spec, allow_download=True, opener=network(spec, payloads, calls))
    assert result['status'] == 'verified' and result['verified_files'] == 5
    assert result['restored_files'] == [] and calls == [] and alias.is_symlink()
    assert before == {name: ((tmp_path / name).lstat().st_ino, (tmp_path / name).lstat().st_mtime_ns)
                      for name in payloads}


def test_success_materializes_only_allowlisted_verified_regular_files(tmp_path, resources):
    spec, payloads = resources
    write(tmp_path, 'bridge', payloads['bridge'])
    calls = []
    result = M.restore(tmp_path, spec, allow_download=True, opener=network(spec, payloads, calls))
    assert result['status'] == 'verified' and result['verified_files'] == 5
    assert len(result['restored_files']) == 4 and len(calls) == 2
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()} == set(payloads)
    assert not (tmp_path / 'runtime/llama-test/libx.so').is_symlink()
    assert all((tmp_path / p).read_bytes() == data for p, data in payloads.items())
    assert result['model_calls'] == result['compile_calls'] == 0
    assert result['ready_for_inference'] is False


@pytest.mark.parametrize('oversized', [False, True])
def test_bad_public_download_is_not_published_and_partial_is_retained(tmp_path, resources, oversized):
    spec, payloads = resources
    write(tmp_path, 'bridge', payloads['bridge'])
    bad = b'X' * (spec.files[spec.archive]['bytes'] + (1 if oversized else 0))
    calls = []
    def open_bad(url, timeout):
        calls.append(url)
        return io.BytesIO(bad)
    with pytest.raises(ValueError, match='(identity mismatch|exceeded)'):
        M.restore(tmp_path, spec, allow_download=True, opener=open_bad)
    partials = list(tmp_path.glob('.framed-resource-stage-*/download-*.partial'))
    assert len(calls) == len(partials) == 1 and partials[0].read_bytes() == bad
    assert not (tmp_path / spec.archive).exists() and not (tmp_path / 'model.gguf').exists()


@pytest.mark.parametrize('extra', [
    ('../escape', 'file', b'bad'),
    ('/absolute-escape', 'file', b'bad'),
    ('llama-test/libx.so.1', 'file', b'frozen library'),
])
def test_archive_traversal_absolute_paths_and_duplicate_allowlisted_members_refused(tmp_path, resources, extra):
    spec, payloads = resources
    entries = [('llama-test/libx.so.1', 'file', b'frozen library'),
               ('llama-test/libx.so', 'symlink', 'libx.so.1'), extra]
    spec, payloads = changed_archive(spec, payloads, entries)
    write(tmp_path, 'bridge', payloads['bridge'])
    calls = []
    with pytest.raises(ValueError, match='(Unsafe|Duplicate)'):
        M.restore(tmp_path, spec, allow_download=True, opener=network(spec, payloads, calls))
    assert len(calls) == 1 and not (tmp_path / spec.archive).exists()
    assert not (tmp_path / 'runtime').exists() and not (tmp_path.parent / 'escape').exists()


@pytest.mark.parametrize('link', ['../../outside', '/outside', 'unrelated-program'])
def test_archive_symlink_must_resolve_to_identical_allowlisted_member(tmp_path, resources, link):
    spec, payloads = resources
    entries = [('llama-test/libx.so.1', 'file', b'frozen library'),
               ('llama-test/libx.so', 'symlink', link),
               ('llama-test/unrelated-program', 'file', b'frozen library')]
    spec, payloads = changed_archive(spec, payloads, entries)
    write(tmp_path, 'bridge', payloads['bridge'])
    calls = []
    with pytest.raises(ValueError, match='(Unsafe|outside identical allowlist)'):
        M.restore(tmp_path, spec, allow_download=True, opener=network(spec, payloads, calls))
    assert len(calls) == 1 and not (tmp_path / 'runtime').exists()


def test_archive_library_hash_is_checked_before_model_download_or_publication(tmp_path, resources):
    spec, payloads = resources
    entries = [('llama-test/libx.so.1', 'file', b'changedlibrary'),
               ('llama-test/libx.so', 'symlink', 'libx.so.1')]
    spec, payloads = changed_archive(spec, payloads, entries)
    write(tmp_path, 'bridge', payloads['bridge'])
    calls = []
    with pytest.raises(ValueError, match='(byte size|hash differs)'):
        M.restore(tmp_path, spec, allow_download=True, opener=network(spec, payloads, calls))
    assert len(calls) == 1 and not (tmp_path / 'model.gguf').exists()
    assert not (tmp_path / spec.archive).exists()


def test_existing_symlink_escape_is_rejected_without_network(tmp_path, resources):
    spec, payloads = resources
    base = tmp_path / 'resources'
    base.mkdir()
    outside = write(tmp_path, 'outside', payloads['bridge'])
    (base / 'bridge').symlink_to(outside)
    calls = []
    with pytest.raises(ValueError, match='(Unsafe|allowlisted)'):
        M.restore(base, spec, allow_download=True, opener=network(spec, payloads, calls))
    assert calls == [] and outside.read_bytes() == payloads['bridge']


def test_safe_dot_prefix_and_allowlisted_hardlink_restore_as_regular_files(tmp_path, resources):
    spec, payloads = resources
    entries = [('./llama-test/libx.so.1', 'file', b'frozen library'),
               ('./llama-test/libx.so', 'hardlink', './llama-test/libx.so.1')]
    spec, payloads = changed_archive(spec, payloads, entries)
    write(tmp_path, 'bridge', payloads['bridge'])
    result = M.restore(tmp_path, spec, allow_download=True, opener=network(spec, payloads, []))
    assert result['status'] == 'verified'
    assert (tmp_path / 'runtime/llama-test/libx.so').read_bytes() == b'frozen library'
    assert not (tmp_path / 'runtime/llama-test/libx.so').is_symlink()


def test_executable_git_bridge_mode_is_a_production_precondition(tmp_path, resources):
    spec, payloads = resources
    spec = dataclasses.replace(spec, bridge_executable_required=True)
    bridge = write(tmp_path, 'bridge', payloads['bridge'])
    bridge.chmod(0o600)
    calls = []
    with pytest.raises(ValueError, match='not executable'):
        M.restore(tmp_path, spec, allow_download=True, opener=network(spec, payloads, calls))
    assert calls == [] and bridge.stat().st_mode & 0o777 == 0o600


def test_symlinked_resource_ancestor_is_rejected_before_any_write(tmp_path, resources):
    spec, payloads = resources
    actual = tmp_path / 'actual'
    actual.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match='ancestor'):
        M.restore(link / 'new-root', spec, allow_download=True,
                  opener=network(spec, payloads, []))
    assert list(actual.iterdir()) == []


def test_concurrent_creation_before_publication_is_never_replaced(tmp_path, resources):
    spec, payloads = resources
    write(tmp_path, 'bridge', payloads['bridge'])
    def competing_writer():
        write(tmp_path, 'model.gguf', b'concurrent writer')
    with pytest.raises(FileExistsError, match='Concurrent'):
        M.restore(tmp_path, spec, allow_download=True, opener=network(spec, payloads, []),
                  before_publish=competing_writer)
    assert (tmp_path / 'model.gguf').read_bytes() == b'concurrent writer'
    assert not (tmp_path / spec.archive).exists()


def test_atomic_publication_refuses_a_race_after_the_precheck(tmp_path, resources, monkeypatch):
    spec, payloads = resources
    write(tmp_path, 'bridge', payloads['bridge'])
    original_link = M.os.link
    calls = []
    def racing_link(source, target, **kwargs):
        calls.append(target)
        Path(target).write_bytes(b'racing owner')
        return original_link(source, target, **kwargs)
    monkeypatch.setattr(M.os, 'link', racing_link)
    with pytest.raises(FileExistsError):
        M.restore(tmp_path, spec, allow_download=True, opener=network(spec, payloads, []))
    assert len(calls) == 1 and Path(calls[0]).read_bytes() == b'racing owner'


@pytest.mark.parametrize('url', [
    'http://huggingface.co/model', 'https://localhost/model',
    'https://huggingface.co.evil.example/model', 'https://user:password@huggingface.co/model',
    'https://huggingface.co:444/model', 'file:///tmp/model',
])
def test_insecure_or_unrelated_redirect_is_refused(url):
    with pytest.raises(ValueError, match='insecure'):
        M.validate_public_url(url)
