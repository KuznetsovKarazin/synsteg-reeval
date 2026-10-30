#!/usr/bin/env python3
"""Verify/restore only the 41 frozen Qwen channel resources, without inference.

Default is read-only verification. The private Git bridge must already have been
restored by the coordinator. Optional downloads use two fixed public URLs only;
old manifests and the older preparation helper are never edited.

--freeze-commit records a caller-provided declaration. Its syntax is checked,
but it is not remote proof: the coordinator must actually verify the new
utility freeze in Git before authorizing downloads. No inference is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import tarfile
import tempfile
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('experiments/seeded_pair_llm_v1/execution_manifest.json')
MANIFEST_SHA256 = 'c877bfdd27a64abd126d6d22580277b3163159bc35a515679a5423e412f453e6'
RESOURCE_PREFIX = 'resources/seeded-pair-llm-v1/'
BRIDGE = 'seeded_pair_llama_bridge'
BRIDGE_SHA256 = '7b37da550a10d8f0c457e838e574bcf8c97159e9717ba88bd536e7e208eefe77'
ARCHIVE = 'llama-b11791-bin-ubuntu-x64.tar.gz'
MODEL = 'qwen2.5-1.5b-instruct-q4_k_m.gguf'
PUBLIC_DOWNLOADS = {
    ARCHIVE: {'bytes': 22302292,
              'sha256': 'ac75208179d00d86f378d5f10c65f9785028ce877cf57279f1a7b8dac54a25ff',
              'url': 'https://github.com/ggml-org/llama.cpp/releases/download/b11791/' + ARCHIVE},
    MODEL: {'bytes': 1117320736,
            'sha256': '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e',
            'url': 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/'
                   '91cad51170dc346986eccefdc2dd33a9da36ead9/' + MODEL},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Unsafe resource/archive path')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name.rstrip('/'):
        raise ValueError('Unsafe resource/archive path: ' + name)
    return str(path)


@dataclass(frozen=True)
class ResourceSpec:
    files: dict
    downloads: dict
    archive: str
    bridge: str
    runtime_prefix: str
    manifest_sha256: str = 'synthetic-test-fixture'
    resource_identity_sha256: str | None = None
    bridge_executable_required: bool = False

    def validate(self):
        for name, item in self.files.items():
            safe_relative(name)
            if (set(item) != {'bytes', 'sha256'} or type(item['bytes']) is not int
                    or item['bytes'] < 0 or not re.fullmatch('[a-f0-9]{64}', item['sha256'])):
                raise ValueError('Invalid resource identity: ' + name)
        if self.archive not in self.files or self.bridge not in self.files:
            raise ValueError('Archive and coordinator-restored bridge must be declared')
        safe_relative(self.runtime_prefix)
        for name, item in self.downloads.items():
            if name not in self.files or {k: item[k] for k in ('bytes', 'sha256')} != self.files[name]:
                raise ValueError('Download identity differs from resource manifest')
            validate_public_url(item['url'])
        if self.archive not in self.downloads or self.bridge in self.downloads:
            raise ValueError('Only public artifacts may be downloaded; bridge comes from Git')
        expected = set(self.downloads) | {self.bridge} | {
            p for p in self.files if p.startswith(self.runtime_prefix + '/')}
        if set(self.files) != expected:
            raise ValueError('Resource outside the scoped restore contract')


def load_spec(root=ROOT):
    path = Path(root) / MANIFEST
    if path.is_symlink() or not path.is_file() or sha256(path) != MANIFEST_SHA256:
        raise ValueError('Original execution manifest missing or changed')
    obj = json.loads(path.read_text())
    scoped = obj['channel_resources']['files']
    if len(scoped) != 41 or any(not p.startswith(RESOURCE_PREFIX) for p in scoped):
        raise ValueError('Expected exactly the original 41 channel resources')
    files = {p[len(RESOURCE_PREFIX):]: value for p, value in scoped.items()}
    for name, expected in PUBLIC_DOWNLOADS.items():
        if files[name] != {k: expected[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Pinned public resource identity differs')
    if files[BRIDGE] != {'bytes': 50976, 'sha256': BRIDGE_SHA256}:
        raise ValueError('Pinned private Git bridge identity differs')
    spec = ResourceSpec(files, PUBLIC_DOWNLOADS, ARCHIVE, BRIDGE,
                        'runtime/llama-b11791', MANIFEST_SHA256,
                        obj['resources_sha256'], True)
    spec.validate()
    return spec


def validate_public_url(url):
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or '').lower()
    allowed = (host in {'github.com', 'objects.githubusercontent.com',
                       'release-assets.githubusercontent.com', 'huggingface.co'}
               or host.endswith('.huggingface.co') or host.endswith('.hf.co'))
    if (parsed.scheme != 'https' or parsed.username or parsed.password
            or parsed.port not in (None, 443) or not allowed):
        raise ValueError('Refusing non-public or insecure download/redirect URL')


class PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def default_open(url, timeout=60):
    validate_public_url(url)
    opener = urllib.request.build_opener(PublicRedirectHandler())
    return opener.open(urllib.request.Request(url, headers={'User-Agent': 'SynSteg-resource-restore-v1'}),
                       timeout=timeout)


def destination(base, relative):
    safe_relative(relative)
    base = Path(base)
    if any(parent.is_symlink() for parent in [base, *base.absolute().parents]):
        raise ValueError('Resource root/ancestor must not be a symlink')
    current = base
    for part in PurePosixPath(relative).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlinked resource parent: ' + relative)
        if current.exists() and not current.is_dir():
            raise ValueError('Non-directory resource parent: ' + relative)
    return base / relative


def inspect_files(base, spec):
    """Read only; permit existing SONAME links only to identical allowlisted files."""
    spec.validate()
    base = Path(base)
    verified, missing = [], []
    for relative, expected in sorted(spec.files.items()):
        path = destination(base, relative)
        if not os.path.lexists(path):
            missing.append(relative)
            continue
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
                target_relative = target.relative_to(base.resolve()).as_posix()
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError('Unsafe or broken resource symlink: ' + relative) from exc
            if target_relative not in spec.files or spec.files[target_relative] != expected:
                raise ValueError('Resource link target is not an identical allowlisted file')
            destination(base, target_relative)
            path = target
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError('Resource is not a regular file: ' + relative)
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('Existing resource differs; refusing all downloads/writes: ' + relative)
        if (relative == spec.bridge and spec.bridge_executable_required
                and not os.access(path, os.X_OK)):
            raise ValueError('Exact Git bridge is not executable; coordinator must restore Git mode 100755')
        verified.append(relative)
    return {'verified': verified, 'missing': missing}


def download_to_stage(item, stage, opener):
    validate_public_url(item['url'])
    fd, name = tempfile.mkstemp(prefix='download-', suffix='.partial', dir=stage)
    path = Path(name)
    digest, total = hashlib.sha256(), 0
    # The partial file intentionally survives every failure for explicit review.
    with os.fdopen(fd, 'wb') as output, opener(item['url'], timeout=60) as response:
        if hasattr(response, 'geturl'):
            validate_public_url(response.geturl())
        while True:
            chunk = response.read(min(1024 * 1024, item['bytes'] - total + 1))
            if not chunk:
                break
            total += len(chunk)
            output.write(chunk)
            digest.update(chunk)
            if total > item['bytes']:
                raise ValueError('Public download exceeded its frozen byte limit')
    if total != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise ValueError('Public download identity mismatch; partial retained: ' + str(path))
    return path


def extract_allowlisted(archive_path, spec, stage, missing):
    """Materialize only required verified libraries; never use extract/extractall."""
    runtime = {name: value for name, value in spec.files.items()
               if name.startswith(spec.runtime_prefix + '/')}
    archive_names = {name.removeprefix('runtime/'): name for name in runtime}
    output = {}
    with tarfile.open(archive_path, 'r:*') as archive:
        members = {}
        def archive_name(name):
            # GNU tar commonly prefixes safe members with ./; normalize only
            # this harmless spelling, retaining traversal/absolute-path checks.
            while name.startswith('./'):
                name = name[2:]
            return safe_relative(name or '.')

        for member in archive:
            name = archive_name(member.name)
            if name in archive_names:
                if name in members:
                    raise ValueError('Duplicate allowlisted archive member: ' + name)
                members[name] = member
        if set(members) != set(archive_names):
            raise ValueError('Archive lacks required allowlisted library members')

        def regular(name, seen=()):
            if name in seen:
                raise ValueError('Archive link cycle')
            member = members[name]
            if member.isfile():
                return member
            if not (member.issym() or member.islnk()):
                raise ValueError('Non-file allowlisted archive member')
            link = archive_name(member.linkname)
            target = (posixpath.join(posixpath.dirname(name), link)
                      if member.issym() else link)
            target = safe_relative(target)
            if (target not in members
                    or runtime[archive_names[target]] != runtime[archive_names[name]]):
                raise ValueError('Archive link target outside identical allowlist')
            return regular(target, seen + (name,))

        # Validate all link topology before producing any staged library.
        resolved = {name: regular(name) for name in members}
        for archive_name, relative in archive_names.items():
            if relative not in missing:
                continue
            expected = runtime[relative]
            member = resolved[archive_name]
            if member.size != expected['bytes']:
                raise ValueError('Archive library byte size differs')
            fd, temporary = tempfile.mkstemp(prefix='library-', dir=stage)
            path = Path(temporary)
            digest, total = hashlib.sha256(), 0
            with os.fdopen(fd, 'wb') as target, archive.extractfile(member) as source:
                while chunk := source.read(min(1024 * 1024, expected['bytes'] - total + 1)):
                    total += len(chunk)
                    target.write(chunk)
                    digest.update(chunk)
                    if total > expected['bytes']:
                        raise ValueError('Archive library exceeded frozen byte size')
            if total != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                raise ValueError('Archive library hash differs; staged file retained')
            output[relative] = path
    return output


def restore(base, spec, *, allow_download=False, opener=None, before_publish=None):
    """No automatic retries. Every existing resource is checked before network I/O."""
    base = Path(base)
    before = inspect_files(base, spec)
    report = {'schema': 'framed-interval-llm-resource-restore-v1',
              'manifest_sha256': spec.manifest_sha256, 'expected_files': len(spec.files),
              'frozen_resource_identity_sha256': spec.resource_identity_sha256,
              'verification_scope': 'file_bytes_and_executable_bridge_only',
              'verified_files': len(before['verified']), 'missing_files': before['missing'],
              'verified_file_identities': {p: spec.files[p] for p in before['verified']},
              'bridge_executable_verified': (spec.bridge in before['verified']
                                              and os.access(base / spec.bridge, os.X_OK)),
              'restored_files': [], 'downloads': 0, 'model_calls': 0, 'compile_calls': 0,
              'ready_for_inference': False, 'resident_backend_checked': False}
    if not before['missing']:
        return {**report, 'status': 'verified'}
    if not allow_download:
        return {**report, 'status': 'missing_resources_no_changes'}
    if spec.bridge in before['missing']:
        raise ValueError('Restore the exact private Git bridge before any public downloads')
    base.mkdir(parents=True, exist_ok=True)
    if base.is_symlink():
        raise ValueError('Resource root became a symlink')
    stage = Path(tempfile.mkdtemp(prefix='.framed-resource-stage-', dir=base))
    opener = opener or default_open
    staged = {}
    archive_path = base / spec.archive
    if spec.archive in before['missing']:
        archive_path = download_to_stage(spec.downloads[spec.archive], stage, opener)
        staged[spec.archive] = archive_path
        report['downloads'] += 1
    runtime_missing = [p for p in before['missing'] if p.startswith(spec.runtime_prefix + '/')]
    if runtime_missing:
        staged.update(extract_allowlisted(archive_path, spec, stage, runtime_missing))
    for relative, item in spec.downloads.items():
        if relative != spec.archive and relative in before['missing']:
            staged[relative] = download_to_stage(item, stage, opener)
            report['downloads'] += 1
    if set(staged) != set(before['missing']):
        raise ValueError('Missing resources were not fully staged')
    inspect_files(base, spec)  # Detect changed previously existing resources again.
    if before_publish is not None:
        before_publish()  # Test-only race injection; CLI never supplies this callback.
    for relative in staged:
        if os.path.lexists(destination(base, relative)):
            raise FileExistsError('Concurrent destination creation; never replace: ' + relative)
    for relative, temporary in sorted(staged.items()):
        target = destination(base, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        destination(base, relative)
        # Atomic no-replace publication, including races after the prior check.
        os.link(temporary, target, follow_symlinks=False)
        report['restored_files'].append(relative)
    after = inspect_files(base, spec)
    if after['missing']:
        raise ValueError('Incomplete resource publication; manual review required')
    for temporary in staged.values():
        temporary.unlink()
    stage.rmdir()
    return {**report, 'status': 'verified', 'missing_files': [],
            'verified_files': len(after['verified']),
            'verified_file_identities': {p: spec.files[p] for p in after['verified']}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--resource-root', type=Path)
    parser.add_argument('--download-missing', action='store_true')
    parser.add_argument('--freeze-commit', help='Caller declaration of a coordinator-verified Git freeze; not remote proof')
    parser.add_argument('--receipt', type=Path, help='Optional new receipt; existing evidence is never replaced')
    args = parser.parse_args(argv)
    if args.receipt and args.receipt.exists():
        parser.error('Receipt already exists; refusing overwrite')
    if args.download_missing and not re.fullmatch('[a-f0-9]{40}', args.freeze_commit or ''):
        parser.error('Downloads require the coordinator-verified --freeze-commit')
    base = args.resource_root or args.root / RESOURCE_PREFIX
    try:
        result = restore(base, load_spec(args.root), allow_download=args.download_missing)
        result['coordinator_freeze_commit'] = args.freeze_commit
        code = 0 if result['status'] == 'verified' else 2
    except Exception as exc:
        result = {'schema': 'framed-interval-llm-resource-restore-v1',
                  'status': 'blocked_requires_review_no_automatic_retry',
                  'error_type': type(exc).__name__, 'detail': str(exc),
                  'model_calls': 0, 'compile_calls': 0, 'ready_for_inference': False}
        code = 2
    if args.receipt:
        with args.receipt.open('x') as output:
            output.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
