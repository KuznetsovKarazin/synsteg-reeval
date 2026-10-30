"""Offline resource-integrity tests; no HTTP or real asset requests."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('restore_qwen_v3', ROOT/'scripts/restore_qwen05_public_resources_v3.py')
restore = importlib.util.module_from_spec(spec); spec.loader.exec_module(restore)


class PublicResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='qwen-public-resource-fixture-')
        self.path = Path(self.temp.name)/'fixture.bin'; self.data = b'public synthetic fixture\n'
        self.path.write_bytes(self.data)
        self.item = {'name': 'fixture.bin', 'bytes': len(self.data), 'known_file_sha256': None,
            'upstream_lfs_sha256': None,
            'upstream_git_blob_sha1': hashlib.sha1(b'blob '+str(len(self.data)).encode()+b'\0'+self.data).hexdigest()}

    def tearDown(self):
        self.temp.cleanup()

    def test_git_blob_identity_and_actual_sha256(self):
        result = restore.check_file(self.path, self.item)
        self.assertEqual(result['sha256'], hashlib.sha256(self.data).hexdigest())
        self.assertTrue(result['git_blob_sha1_verified'])

    def test_lfs_pointer_blob_not_confused_with_weight_content(self):
        self.item.update(upstream_lfs_sha256=hashlib.sha256(self.data).hexdigest(),
            known_file_sha256=hashlib.sha256(self.data).hexdigest(), upstream_git_blob_sha1='0'*40)
        result = restore.check_file(self.path, self.item)
        self.assertTrue(result['lfs_sha256_verified']); self.assertFalse(result['git_blob_sha1_verified'])

    def test_size_mismatch_refused(self):
        self.item['bytes'] += 1
        with self.assertRaises(ValueError): restore.check_file(self.path, self.item)

    def test_content_hash_mismatch_refused(self):
        self.item['known_file_sha256'] = '0'*64
        with self.assertRaises(ValueError): restore.check_file(self.path, self.item)

    def test_same_size_bad_git_blob_refused(self):
        self.path.write_bytes(b'x'*len(self.data))
        with self.assertRaises(ValueError): restore.check_file(self.path, self.item)

    def test_symlink_refused(self):
        link = self.path.with_name('link'); link.symlink_to(self.path)
        with self.assertRaises(ValueError): restore.check_file(link, self.item)

    def test_official_https_hosts_only(self):
        for url in ['https://huggingface.co/a', 'https://cas-bridge.xethub.hf.co/a', 'https://cdn-lfs.huggingface.co/a']:
            self.assertEqual(restore.safe_url(url), url)
        for url in ['http://huggingface.co/a', 'https://huggingface.co.evil.example/a',
                    'https://hf.co.evil.example/a', 'https://user:pass@huggingface.co/a']:
            with self.assertRaises(ValueError): restore.safe_url(url)

    def test_new_receipt_never_overwrites(self):
        path = self.path.with_name('receipt.json'); restore.write_new(path, {'first': True})
        with self.assertRaises(FileExistsError): restore.write_new(path, {'first': False})


if __name__ == '__main__':
    unittest.main()
