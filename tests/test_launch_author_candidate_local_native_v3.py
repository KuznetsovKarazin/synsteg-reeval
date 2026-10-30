"""Synthetic subprocess checks only; never invokes the native bridge or tokenizer."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('launch_v3',ROOT/'scripts/launch_author_candidate_local_native_v3.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class LaunchTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  (self.root/'results').mkdir();self.directory=self.root/'results/process/run';self.lock=self.root/'project.lock'
 def run_command(self,code,timeout=2,gate=None):return m.execute_command([sys.executable,'-c',code],self.root,self.directory,self.lock,timeout=timeout,gate=gate)
 def test_exit0_saved_but_not_scientific_acceptance(self):
  r=self.run_command('print("synthetic")');self.assertEqual(r['exit_code'],0);self.assertFalse(r['scientific_acceptance_inferred'])
  self.assertEqual((self.directory/'runner.stdout').read_bytes(),b'synthetic\n');self.assertTrue((self.directory/'owned-process.json').exists())
 def test_nonzero_preserved_no_retry(self):
  r=self.run_command('raise SystemExit(7)');self.assertEqual(r['exit_code'],7);self.assertFalse(r['automatic_retry'])
 def test_timeout_own_process_stopped(self):
  r=self.run_command('import time;time.sleep(3)',timeout=.1);self.assertTrue(r['timeout']);self.assertIsNotNone(r['exit_code']);self.assertFalse(r['automatic_retry'])
 def test_timeout_kills_descendant_that_ignores_term(self):
  child="import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);open('child-ready','w').write('ready');time.sleep(30)"
  code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);open('child-pid','w').write(str(p.pid));time.sleep(30)"
  r=self.run_command(code,timeout=1)
  self.assertTrue(r['timeout']);self.assertTrue((self.root/'child-ready').exists())
  pid=int((self.root/'child-pid').read_text());stat=Path('/proc')/str(pid)/'stat'
  if stat.exists():self.assertEqual(stat.read_text().split(') ')[1].split()[0],'Z')
 def test_existing_process_directory_refused(self):
  self.run_command('pass');before=(self.directory/'completion.json').read_bytes();self.assertRaises(FileExistsError,self.run_command,'pass');self.assertEqual((self.directory/'completion.json').read_bytes(),before)
 def test_active_external_lease_blocks_before_launch(self):
  fd=os.open(self.lock,os.O_CREAT|os.O_RDWR,0o600)
  try:
   fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);self.assertRaises(BlockingIOError,self.run_command,'pass');self.assertFalse(self.directory.exists())
  finally:os.close(fd)
 def test_failed_gate_creates_no_process_evidence(self):
  def reject():raise ValueError('mock gate reject')
  self.assertRaises(ValueError,self.run_command,'pass',gate=reject);self.assertFalse(self.directory.exists())
 def test_symlink_project_lease_refused(self):
  target=self.root/'target';target.write_bytes(b'preserve');self.lock.symlink_to(target)
  self.assertRaises(ValueError,self.run_command,'pass');self.assertEqual(target.read_bytes(),b'preserve')
if __name__=='__main__':unittest.main()
