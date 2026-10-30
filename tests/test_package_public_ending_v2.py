"""Synthetic preservation only; no live result inspection or native calls."""
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ending_preservation',ROOT/'scripts/package_public_ending_v2.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
# Source-only test fixtures may use a separate immutable helper checkout.
import os
HELPER=Path(os.environ.get('ENDING_V2_STORAGE_HELPER',ROOT/'scripts/package_author_candidate_local_native_v3.py'))

class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'repo';self.root.mkdir()
        self.canonical=self.base/'canonical.lock';self.canonical.touch()
        self.addCleanup(patch.stopall);patch.object(w,'CANONICAL_LOCK',str(self.canonical)).start()
        self.primary=self.root/w.BASES[0];self.primary.mkdir(parents=True)
        for name,data in {'generation/result.json':b'{"status":"unknown"}',
            'generation/initial-denominator.json':b'{"senders":8,"receivers":4}',
            'generation/receiver-install-inventory.json':b'{"files_sha256":{},"expected_plaintext_present":false}',
            'generation/o013-stego/sender/token-journal.jsonl':b'{"partial":',
            'generation/o013-stego/sender/wire.raw':b'\xff\x00\x80'}.items():
            p=self.primary/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        (self.primary/'generation/reserved-empty').mkdir()
        self.ledger=self.root/w.BASES[1];self.ledger.mkdir(parents=True)
        (self.ledger/'burned.json').write_bytes(b'{"state":"burned_before_generation"}')
        weights=self.root/'runtime'/w.ID/'receiver-install/resources/huge-model-marker';weights.parent.mkdir(parents=True);weights.write_bytes(b'not archived')
        self.terminal=self.root/'terminal.json';self.terminal.write_text('{"exit_code":3,"automatic_retries":0}')
    def package(self):return w.package(self.root,HELPER,self.terminal)
    def test_two_exact_trees_and_empty_dirs_restored_without_weights(self):
        storage=w.storage(HELPER)
        before=[storage.snapshot(self.root,base) for base in w.BASES]
        receipt=self.package();self.assertEqual(receipt['outer_process']['exit_code'],3)
        self.assertFalse(receipt['scientific_acceptance_inferred']);self.assertFalse(receipt['actual_Git_readback_completed'])
        self.assertTrue(receipt['receiver_install_inventory_present'])
        for i,label in enumerate(['results','sender-nonces']):
            index=self.root/'artifacts'/w.ID/'raw-v1'/label/'index.json'
            dest=self.base/('restore-'+label);storage.restore(index,dest)
            self.assertEqual(storage.snapshot(dest,w.BASES[i]),before[i])
            self.assertFalse((dest/'runtime'/w.ID/'receiver-install').exists())
            listing=storage.read_json(index.parent/'upload-files.json')
            self.assertFalse(any(x['path'].endswith('/full.zip') for x in listing['files']))
    def test_unknown_outer_exit_refuses_before_any_archive(self):
        for value in [None,True,'0']:
            self.terminal.write_text(json.dumps({'exit_code':value}))
            with self.assertRaises(ValueError):self.package()
            self.assertFalse((self.root/'artifacts').exists())
    def test_active_canonical_lease_refuses(self):
        with self.canonical.open('r+b') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):self.package()
        self.assertFalse((self.root/'artifacts').exists())
    def test_absent_and_empty_nonce_ledger_are_explicit_not_retried(self):
        shutil.rmtree(self.ledger)
        first=self.package();self.assertEqual(first['preserved'][1]['status'],'absent_at_snapshot')
        self.assertTrue(first['preserved'][1]['absence_does_not_authorize_retry'])
        shutil.rmtree(self.root/'artifacts');self.ledger.mkdir(parents=True)
        second=self.package();self.assertEqual(second['preserved'][1]['status'],'empty_directory_at_snapshot')
    def test_existing_output_is_not_overwritten(self):
        self.package()
        with self.assertRaises(FileExistsError):self.package()
    def test_wrong_storage_bytes_rejected_without_source_read(self):
        bad=self.base/'bad.py';bad.write_bytes(HELPER.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'frozen storage bytes differ'):w.package(self.root,bad,self.terminal)
        self.assertFalse((self.root/'artifacts').exists())

if __name__=='__main__':unittest.main()
