#!/usr/bin/env python3
"""Root-owned public restore v3: stream outside synchronized workspace; zero automatic retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request

MANIFEST_SHA = '741711fb21acd08db2a81fbcdb857f17bd1f0bdb53f819a9dd859a1bf220513f'
ROOT = '/workspace/scratch/af7adce5db85/baseline92-continuation-v2/checkout'
RESERVE_BYTES = 2 * 1024**3
STAGING = Path('/tmp/synsteg-qwen05-restore-v3')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reject_links(path):
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink refused')


def safe_url(url):
    value = urllib.parse.urlsplit(url)
    host = value.hostname or ''
    require(value.scheme == 'https' and value.username is None and value.password is None
            and (host == 'huggingface.co' or host.endswith('.huggingface.co') or host.endswith('.hf.co')),
            'Nonofficial HTTPS redirect refused')
    return url


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_file(path, item):
    reject_links(path)
    require(path.is_file() and path.stat().st_size == item['bytes'], 'Size differs: '+item['name'])
    digest = hashlib.sha256(); blob = hashlib.sha1(b'blob '+str(item['bytes']).encode()+b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block); blob.update(block)
    require(not item['known_file_sha256'] or digest.hexdigest() == item['known_file_sha256'], 'SHA256 differs: '+item['name'])
    require(item['upstream_lfs_sha256'] or blob.hexdigest() == item['upstream_git_blob_sha1'], 'Git blob differs: '+item['name'])
    return {'name': item['name'], 'bytes': item['bytes'], 'sha256': digest.hexdigest(),
            'git_blob_sha1_verified': not bool(item['upstream_lfs_sha256']),
            'lfs_sha256_verified': bool(item['upstream_lfs_sha256'])}


def write_new(path, value):
    reject_links(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--receipt-directory', type=Path, required=True)
    parser.add_argument('--execute-download', action='store_true')
    args = parser.parse_args(); root = args.root.absolute()
    require(str(root) == ROOT, 'Wrong shared checkout')
    reject_links(args.manifest); raw = args.manifest.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, 'Public manifest differs')
    manifest = json.loads(raw); target = root/manifest['expected_snapshot_relative']
    reject_links(target)
    missing = [x for x in manifest['model_files'] if not (target/x['name']).exists()]
    needed = sum(x['bytes'] for x in missing); free = shutil.disk_usage(root).free
    require(free-needed >= RESERVE_BYTES, 'Insufficient space for assets plus 2 GiB reserve')
    existing = [check_file(target/x['name'], x) for x in manifest['model_files'] if (target/x['name']).exists()]
    plan = {'schema': 'qwen05-public-restore-v3', 'status': 'preview_no_requests',
            'resource_attempt': 3, 'staging_directory': str(STAGING),
            'generation_identity_consumed': False,
            'model_id': manifest['model_id'], 'revision': manifest['revision'], 'manifest_sha256': MANIFEST_SHA,
            'target': str(target), 'free_bytes_before': free, 'bytes_to_download': needed,
            'minimum_free_bytes': RESERVE_BYTES, 'automatic_retries': 0, 'authentication_used': False,
            'existing_verified': existing, 'missing_names': [x['name'] for x in missing],
            'model_loads': 0, 'tokenizer_calls': 0}
    if not args.execute_download:
        print(json.dumps(plan)); return
    receipt_dir = args.receipt_directory.absolute(); reject_links(receipt_dir)
    require(not receipt_dir.exists(), 'Receipt directory already consumed; do not replay')
    write_new(receipt_dir/'started.json', dict(plan, status='download_started', time_unix=time.time()))
    target.mkdir(parents=True, exist_ok=True)
    reject_links(STAGING)
    require(not STAGING.exists(), 'Staging directory already consumed; do not replay')
    STAGING.mkdir()
    require(STAGING.stat().st_dev == target.stat().st_dev, 'Require same-volume atomic publish; refuse copy fallback')
    opener = urllib.request.build_opener(Redirect())
    files = list(existing); requests = 0
    try:
        for item in missing:
            require(shutil.disk_usage(root).free-item['bytes'] >= RESERVE_BYTES, 'Reserve breached before next file')
            url = safe_url(item['url']); path = target/item['name']; temp = STAGING/(item['name']+'.download-partial')
            reject_links(temp); require(not path.exists() and not temp.exists(), 'No overwrite/retry')
            started = time.monotonic(); requests += 1
            request = urllib.request.Request(url, headers={'User-Agent': 'SynSteg-public-resource-reproduction/1.0'})
            with opener.open(request, timeout=60) as response, temp.open('xb') as output:
                require(response.status == 200, 'Unexpected HTTP status'); safe_url(response.url)
                total = 0
                while True:
                    require(time.monotonic()-started <= 900, 'Per-file wall limit exceeded')
                    block = response.read(1024*1024)
                    if not block:
                        break
                    total += len(block); require(total <= item['bytes'], 'Oversized response')
                    require(shutil.disk_usage(root).free-len(block) >= RESERVE_BYTES, 'Reserve breached')
                    output.write(block)
                output.flush(); os.fsync(output.fileno())
            record = check_file(temp, item)
            os.link(temp, path); temp.unlink()
            descriptor = os.open(target, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            record.update(url=url, http_status=200, elapsed_seconds=time.monotonic()-started)
            files.append(record); write_new(receipt_dir/(item['name']+'.receipt.json'), record)
        result = dict(plan, status='all_seven_resources_verified', files=files, actual_http_requests=requests,
                      free_bytes_after=shutil.disk_usage(root).free)
        write_new(receipt_dir/'complete.json', result); print(json.dumps(result))
    except BaseException as exc:
        write_new(receipt_dir/'failed.json', dict(plan, status='stopped_no_retry', error_type=type(exc).__name__,
                  verified_files=files, actual_http_requests=requests, partial_files_preserved=True))
        raise


if __name__ == '__main__':
    main()
