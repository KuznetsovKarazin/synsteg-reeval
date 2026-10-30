#!/usr/bin/env python3
"""Prepare a new two-record baseline batch without starting inference.

This shared additive helper binds each batch41–50 to its exact preceding receipt.
The frozen generation utility and scientific settings are unchanged.
The project NN lease is deliberately NOT asserted free during this preparation.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT = EXPECTED_BEFORE = EXPECTED_AFTER = BATCH_ID = PRIOR_BATCH = OUT = None
UTILITY_COMMIT='a76efecfa9293c4d83d6bcd04cdfb11eaf610630'

def configure(root,batch_number):
    global ROOT,EXPECTED_BEFORE,EXPECTED_AFTER,BATCH_ID,PRIOR_BATCH,OUT
    if type(batch_number) is not int or not 41 <= batch_number <= 50:
        raise ValueError('Only fixed batches41–50')
    ROOT=Path(root).resolve();EXPECTED_BEFORE=2*batch_number-2;EXPECTED_AFTER=EXPECTED_BEFORE+2
    BATCH_ID=f'batch{batch_number:02d}';PRIOR_BATCH=f'batch{batch_number-1:02d}'
    OUT=ROOT/f'results/continuation-20261026-{BATCH_ID}'

def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(obj,stream,ensure_ascii=False,indent=2);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if not __debug__:
        raise ValueError('Optimized Python disables operational assertions; refuse preparation')
    sys.path.insert(0,str(ROOT/'scripts'))
    import resume_large_campaign_development_v3 as resume
    import checkpoint_large_campaign_development_v3 as audit
    started=time.time()
    intent_path=ROOT/f'results/large-campaign-continuation-v3/intents/{BATCH_ID}.json'
    assert not (ROOT/resume.BATCHES/BATCH_ID).exists()
    assert not intent_path.exists()
    active=json.loads((ROOT/'results/large-campaign-continuation-v3/active-intent.json').read_text())
    assert active['batch_id']==PRIOR_BATCH and active['phase']=='completed_and_remote_verified'
    assert active['generation_outcome_known'] is True and active['completed_after']==EXPECTED_BEFORE
    baseline=json.loads((ROOT/'results/large-campaign-continuation-v3/status.json').read_bytes())
    native=json.loads((ROOT/'results/native-continuation-20261025/status.json').read_bytes())
    assert baseline['completed']==native['baseline_completed']==EXPECTED_BEFORE
    assert baseline['public_transport_latest_completed_covers']==80
    assert baseline['first80_transport_execution']==native['first80_transport_execution']
    assert native['first80_transport_execution']['status']=='completed_saved_verified'
    assert native['first80_transport_execution']['active'] is False
    assert native.get('active_model_process') is None
    operational_path='experiments/baseline81-100-operations-v1/manifest.json'
    operational=json.loads((ROOT/operational_path).read_bytes())
    operation_paths={
      'scripts/prepare_baseline81_100_v1.py',
      'scripts/package_baseline81_100_v1.py',
      'scripts/close_baseline81_100_v1.py',
      'tests/test_baseline81_100_operations_v1.py',
      'docs/baseline81_100_operations_v1_ru.md'}
    assert operational['schema']=='baseline81-100-operations-v1'
    assert operational['allowed_batch_numbers']==list(range(41,51))
    assert operational['scientific_utility_freeze_commit']==UTILITY_COMMIT
    assert set(operational['files_sha256'])==operation_paths
    for name,expected in operational['files_sha256'].items():
        assert digest(ROOT/name)==expected,'Operational source changed: '+name
    utility=resume.verify_utility(ROOT)
    frozen=resume.verify_frozen_environment(ROOT,resources=True)
    postrun_dependencies={
     'results/large-campaign-continuation-v3-audit/audit_bounded_batch.py':'5c53ace045bdee7e67ee03b3d880ef93b9b2fb63f41485cfd960a4099e94b035',
     'results/large-campaign-continuation-v3-audit/baseline-audit.json':'54c7c9a77a5cd6d8cc2269891c63464924489d34c0203d079be68805de42a123',
     'scripts/restore_native_continuation_artifact_v1.py':'612260031d56dec54a6bee71319ef78acd7a9b43705003b8ef1b80c4551d05e3',
     'scripts/reassemble_baseline_primary_chunks_v1.py':'dcea5c193b04620749c0c161202d27a1bc8efe0aa835b1029a91c2d6af17fe55'}
    for name,expected in postrun_dependencies.items():
        assert digest(ROOT/name)==expected, 'Missing/changed postrun dependency: '+name
    receipt_path=Path(f'results/large-campaign-continuation-v3/git-receipts/after-{PRIOR_BATCH}.json')
    receipt=json.loads((ROOT/receipt_path).read_text())
    with resume.exclusive(ROOT/resume.BATCHES/'resume-supervisor.lock'):
        with resume.evidence_locks(ROOT):
            before=resume.verify_lease(ROOT,receipt,EXPECTED_BEFORE)
    rows=[x for x in before['records'] if x['model_alias']=='tinyllama11']
    next_records=rows[EXPECTED_BEFORE:EXPECTED_AFTER]
    assert len(next_records)==2 and all(x['status']=='not_attempted' for x in next_records)
    # Derive the archive denominator from the actual preceding utility output.
    # The frozen supervisor captures both full run trees plus11 fixed batch files;
    # a successful next pair adds two attempt files and replaces no file paths.
    prior_root=ROOT/resume.BATCHES/PRIOR_BATCH
    prior_files={p.relative_to(prior_root).as_posix() for p in prior_root.rglob('*') if p.is_file()}
    prior_before=json.loads((prior_root/'before/files.json').read_text())
    prior_after=json.loads((prior_root/'after/files.json').read_text())
    assert prior_after==before['run_files_sha256']
    before_members={'before/'+p for p in prior_before}
    after_members={'after/'+p for p in prior_after}
    fixed_members=sorted(prior_files-before_members-after_members)
    assert len(fixed_members)==11
    expected_before_run=len(before['run_files_sha256'])
    expected_after_run=expected_before_run+2
    expected_batch=expected_before_run+expected_after_run+len(fixed_members)
    expected_primary=expected_batch+expected_after_run
    assert (expected_before_run,expected_after_run,expected_batch,expected_primary)==(EXPECTED_BEFORE+6,EXPECTED_AFTER+6,2*EXPECTED_AFTER+21,3*EXPECTED_AFTER+27)
    command=[frozen['python_relative'],'scripts/resume_large_campaign_development_v3.py',
     '--batch-id',BATCH_ID,'--expected-completed',str(EXPECTED_BEFORE),
     '--verified-git-receipt',str(receipt_path),'--utility-freeze-commit',UTILITY_COMMIT]
    identities={'experiments/large-campaign-resume-v3/manifest.json':digest(ROOT/resume.UTILITY),
     'experiments/large-campaign-development-v2/execution-manifest.json':digest(ROOT/audit.FIRST_MANIFEST),
     str(receipt_path):digest(ROOT/receipt_path),operational_path:digest(ROOT/operational_path),
     **operational['files_sha256'],**utility['source_sha256'],**frozen['frozen_files_sha256']}
    status={
     'schema':'baseline81-100-read-only-preparation-v1','batch_id':BATCH_ID,
     'status':'prepared_not_launched_requires_root_nn_lease_after_prior_stage_saved',
     'checked_at_unix':started,'completed_at_unix':time.time(),
     'current_completed':EXPECTED_BEFORE,'target_completed':EXPECTED_AFTER,'full_denominator':300,
     'current_run_files_verified':expected_before_run,
     'utility_source_hashes_verified':len(utility['source_sha256']),
     'frozen_source_hashes_verified':len(frozen['frozen_files_sha256']),
     'package_versions_verified':len(frozen['packages']),
     'all_model_resource_hashes_recomputed':True,'model_hash_scope':'All frozen public model/tokenizer bytes verified in the newly restored cloud working directory; frozen supervisor repeats this before inference','model_resource_files':frozen['model_files'],
     'python':frozen['python'],'platform':frozen['platform'],
     'baseline_execution_and_evidence_locks_obtained_then_released':True,
     'project_nn_lease_checked':False,'project_nn_lease_claimed':False,
     'parallel_stage_note':'Prior public Qwen pilot, native candidates v2/v3, mask-order-v1 and first80 transport are complete and saved in main. Legacy candidate v1 retains an evidence gap and must not be rerun. This preparation does not acquire the project-wide NN lease; root must confirm no concurrent heavy NN stage before launch.',
     'next_source_orders':[EXPECTED_BEFORE+1,EXPECTED_AFTER],
     'next_records':next_records,'command':command,
     'local_prior_receipt_commit':receipt['commit'],'remote_readback_in_this_preparation':False,
     'launch_conditions':['Root confirms no concurrent heavy NN stage and acquires exclusive project NN lease',
       'Root holds exclusive project NN lease before launching any baseline inference',
       'Root saves new immutable batch intent/preparation and reads them back exactly',
       f'Root confirms unchanged receipt/current{expected_before_run}run snapshot and no existing {BATCH_ID} outputs',
       'Frozen supervisor independently repeats model hashes and ledger locks'],
     'postrun_dependencies_sha256':postrun_dependencies,
     'verification_inputs_sha256':identities,'current_run_files_sha256':before['run_files_sha256'],
     'expected_primary_counts_if_success':{'before_run':expected_before_run,'after_run':expected_after_run,
       'immutable_batch':expected_batch,'total_primary':expected_primary,
       'fixed_batch_member_paths':fixed_members,
       'derived_from_prior_batch':PRIOR_BATCH,
       'derivation':f'frozen capture before{expected_before_run}+after{expected_after_run}+11fixed={expected_batch}batch; add{expected_after_run}currentrun={expected_primary}primary'},
     'operational_manifest_sha256':digest(ROOT/operational_path),
     'operational_files_sha256':operational['files_sha256'],
     'operational_scope':'Fixed batches41–50 only; logical archive ceiling16MiB, each Git part196608bytes; scientific utility unchanged.',
     'generation_calls':0,'model_loads':0,'tokenizer_calls':0,'git_writes':0,
     'active_intent_updated':False
    }
    put(OUT/'preflight.json',status)
    intent={'schema':'large-campaign-remote-intent-v1',
     'phase':'prepared_requires_git_readback_and_exclusive_nn_lease_before_launch',
     'batch_id':BATCH_ID,'expected_completed_before':EXPECTED_BEFORE,
     'target_completed_after':EXPECTED_AFTER,
     'planned_next_record_ids':[x['record_id'] for x in next_records],
     'utility_freeze_commit':UTILITY_COMMIT,'prior_remote_run_commit':receipt['commit'],
     'prior_receipt_path':str(receipt_path),'prior_receipt_sha256':digest(ROOT/receipt_path),
     'recorded_before_launch':True,'recorded_at_unix':time.time(),'max_new_requests':2,
     'automatic_retries':0,'generation_outcome_known':False,
     'root_launch_gate':'Only after exact immutable intent/source Git readback and root confirmation of exclusive project NN lease.',
     'safe_recovery':'Do not regenerate automatically after runtime loss; resolve actual outcomes first.'}
    put(intent_path,intent)
    print(json.dumps({k:status[k] for k in ['status','batch_id','current_completed','target_completed',
      'current_run_files_verified','next_source_orders','next_records','expected_primary_counts_if_success','command']},
       ensure_ascii=False,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--batch-number',type=int,choices=range(41,51),required=True)
    args=parser.parse_args();configure(args.root,args.batch_number);main()
