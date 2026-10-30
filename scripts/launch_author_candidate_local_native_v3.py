#!/usr/bin/env python3
"""One bounded native-v3 process, under a separately specified project NN lease.

No model is loaded unless --execute and full source/activation gate pass.
An existing process directory prevents rerun even if no result was written.
Timeout/exception terminates only the owned child process group; never retries.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from synsteg_audit import author_candidate_local_native_v3 as native
PROCESS='results/author-candidate-local-native-v3-process/native-diagnostic-v3'
TIMEOUT=1320

def stop_owned(process):
    """Only the child session created here is addressed, never unrelated PIDs."""
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
    # The leader may exit on TERM while an owned descendant ignores it.
    # Kill the still-owned process group even after the leader has exited.
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait(timeout=10)

def execute_command(command,root,process_directory,lock_path,*,timeout=TIMEOUT,gate=None):
    """Generic process mechanism; CLI supplies only the frozen native command."""
    root=Path(root).resolve(strict=True);directory=Path(process_directory)
    lock_path=Path(lock_path)
    if lock_path.is_symlink() or lock_path.parent.is_symlink():raise ValueError('project lock path must be direct')
    lease=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    try:
        if not stat.S_ISREG(os.fstat(lease).st_mode):raise ValueError('project lock must be regular')
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if gate is not None:gate()
        native.mkdir_durable(directory.parent,exist_ok=True);native.mkdir_durable(directory)
        started=time.time();mono=time.monotonic()
        native.write_json(directory/'launch.json',{'schema':native.ID+'-process-launch','command':command,
            'root':str(root),'shared_project_nn_lock':str(lock_path.resolve()),'timeout_seconds':timeout,
            'started_at_unix':started,'automatic_retry':False,'outcome_known':False})
        process=None;error=None;timed_out=False
        with (directory/'runner.stdout').open('xb') as stdout,(directory/'runner.stderr').open('xb') as stderr:
            try:
                process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,start_new_session=True)
                native.write_json(directory/'owned-process.json',{'pid':process.pid,'new_session':True,'started_at_unix':time.time()})
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                timed_out=True;error={'type':type(exc).__name__,'detail':'bounded process deadline reached; no retry'}
                if process is not None:stop_owned(process)
            except BaseException as exc:
                error={'type':type(exc).__name__,'detail':str(exc)[:1024]}
                if process is not None:stop_owned(process)
            finally:
                if process is not None and process.poll() is None:stop_owned(process)
                stdout.flush();os.fsync(stdout.fileno());stderr.flush();os.fsync(stderr.fileno())
        result={'schema':native.ID+'-process-completion','pid':None if process is None else process.pid,
            'exit_code':None if process is None else process.returncode,'timeout':timed_out,'error':error,
            'elapsed_seconds':time.monotonic()-mono,'completed_at_unix':time.time(),
            'status':'process_exit0_requires_evidence_audit' if process is not None and process.returncode==0 and error is None else 'terminal_requires_review_no_retry',
            'scientific_acceptance_inferred':False,'automatic_retry':False,'paid_calls':0,'closed_test_opened':False}
        native.write_json(directory/'completion.json',result)
        return result
    finally:os.close(lease)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--activation',type=Path,required=True);p.add_argument('--intent',type=Path,required=True)
    p.add_argument('--readback',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:p.error('--execute and immutable source/activation/readback are required')
    root=a.root.resolve(strict=True);project=a.project_root.resolve(strict=True)
    lock=project/'results/project-nn.lock'
    if project==root:p.error('shared project root must differ from native root to avoid nested same-lock acquisition')
    command=[sys.executable,'-B',str(root/native.BASE/'run_native.py'),'--root',str(root),
        '--output',str(root/native.OUTPUT),'--activation',str(a.activation.resolve(strict=True)),
        '--intent',str(a.intent.resolve(strict=True)),'--readback',str(a.readback.resolve(strict=True)),'--execute']
    result=execute_command(command,root,root/PROCESS,lock,gate=lambda:native.verify_gate(root,a.activation,a.intent,a.readback))
    print(json.dumps(result,sort_keys=True))
    return 0 if result['status']=='process_exit0_requires_evidence_audit' else 2
if __name__=='__main__':raise SystemExit(main())
