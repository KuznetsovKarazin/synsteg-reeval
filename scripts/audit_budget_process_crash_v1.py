"""Offline process-death audit. Temporary synthetic ledgers only; no provider code.

Run from repository root: python scripts/audit_budget_process_crash_v1.py
Each crash exits a separate child process without Python cleanup.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT/'src/synsteg_audit/budget_ledger.py'

def module():
    spec=importlib.util.spec_from_file_location('audited_ledger', LEDGER)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def child(directory, action):
    mod=module()
    ledger=mod.CampaignLedger(directory, '1')
    with ledger.locked():
        if action=='contend':
            raise RuntimeError('Contender unexpectedly acquired held lock')
        if action=='before_reserve':
            os._exit(73)
        if action.endswith('_before_replace') or action.endswith('_after_replace'):
            original=mod.os.replace
            def replace(source, target):
                if action.endswith('_before_replace'):
                    os._exit(73)
                original(source, target)
                os._exit(73)
            mod.os.replace=replace
        if action.startswith('reserve'):
            ledger.reserve('attempt', 'fixed-request', mod.money('.4'), {'synthetic':True})
        elif action.startswith('finish'):
            ledger.finish('attempt', {'synthetic':'complete'}, mod.money('.1'))
        elif action=='unknown':
            ledger.finish('attempt', {'synthetic':'unknown'}, None)
        elif action=='overrun':
            try:
                ledger.finish('attempt', {'synthetic':'overrun'}, mod.money('1.1'))
            except mod.BudgetError:
                pass
        else:
            raise ValueError(action)
        os._exit(73)

def invoke(directory, action):
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(directory), action],capture_output=True,text=True,timeout=15)

def run():
    mod=module()
    results=[]
    cases={
        'before_reserve':('absent',0),
        'reserve_before_replace':('absent',0),
        'reserve_after_replace':('reserved_outcome_unknown',400000000),
        'reserve_returned':('reserved_outcome_unknown',400000000),
        'finish_before_replace':('reserved_outcome_unknown',400000000),
        'finish_after_replace':('completed',100000000),
        'finish_returned':('completed',100000000),
        'unknown':('failed_or_unknown_no_retry',400000000),
        'overrun':('accounting_bound_violation_stop',1100000000),
    }
    with tempfile.TemporaryDirectory(prefix='synsteg-crash-audit-') as temp:
        for action,(expected_status,expected_total) in cases.items():
            directory=Path(temp)/action
            ledger=mod.CampaignLedger(directory,'1')
            with ledger.locked():
                ledger.read()
                if action.startswith('finish') or action in {'unknown','overrun'}:
                    ledger.reserve('attempt','fixed-request',mod.money('.4'),{'synthetic':True})
            process=invoke(directory,action)
            checks={'abrupt_exit':process.returncode==73}
            # Reacquisition also checks that the child's OS lock was released.
            with ledger.locked():
                state=ledger.read()
                status=state['attempts'].get('attempt',{}).get('status','absent')
                total=ledger.total(state)
                checks.update(lock_released=True,valid_json=True,status_matches=status==expected_status,total_matches=total==expected_total)
                before=ledger.path.read_bytes()
                should_block=status not in {'absent','completed'}
                try:
                    ledger.reserve('next','next-fixed',mod.money('.2'),{'synthetic':True})
                    blocked=False
                except mod.BudgetError:
                    blocked=True
                checks['next_request_gate']=blocked==should_block
                if should_block:
                    checks['blocked_request_no_mutation']=before==ledger.path.read_bytes()
                if status=='reserved_outcome_unknown':
                    _,created=ledger.reserve('attempt','fixed-request',mod.money('.4'),{'synthetic':True})
                    checks['same_request_not_reissued']=not created
            results.append({'case':action,'status_after_crash':status,'accounted_nanodollars':total,'checks':checks,'passed':all(checks.values())})
        directory=Path(temp)/'held_lock'
        with mod.CampaignLedger(directory,'1').locked():
            process=invoke(directory,'contend')
            checks={'contender_blocked':process.returncode!=0 and 'Campaign is locked by another process' in process.stderr}
        results.append({'case':'live_process_lock_contention','checks':checks,'passed':all(checks.values())})
    return {'schema':'synsteg-process-crash-audit-v1','audited_commit':'80a4f6791f236f357a689a9dbd9c8449311de22a','ledger_sha256':hashlib.sha256(LEDGER.read_bytes()).hexdigest(),'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'platform':platform.platform(),'python':platform.python_version(),'status':'passed' if all(r['passed'] for r in results) else 'failed','scenarios':len(results),'scenarios_passed':sum(r['passed'] for r in results),'checks':sum(len(r['checks']) for r in results),'results':results,'paid_calls':0,'model_generations':0,'test_data_accessed':False,'limits':['Synthetic process-termination experiment on the reported OS; not Windows execution.','Does not simulate machine power loss, filesystem corruption, provider billing, or the full generation runner.','Crashes immediately around os.replace test visibility of old/new complete JSON; post-replace crash precedes directory fsync and is not proof of power-loss durability.','Temporary ledgers and orphan temporary files are deleted after inspection; real campaign ledgers are untouched.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',nargs=2,metavar=('DIRECTORY','ACTION'))
    args=parser.parse_args()
    if args.child:
        child(*args.child)
    else:
        result=run()
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result['status']=='passed' else 1)
