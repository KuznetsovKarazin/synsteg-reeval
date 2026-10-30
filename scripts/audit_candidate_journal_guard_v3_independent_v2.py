#!/usr/bin/env python3
"""Independent synthetic probe lifecycle/adversarial checks; no model imports."""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, json, os
from pathlib import Path
import shutil, tempfile, time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve(); output=args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    source_raw=source.read_bytes(); before=hashlib.sha256(source_raw).hexdigest()
    snapshot=output.with_suffix('.source.py')
    with snapshot.open('xb') as stream: stream.write(source_raw)
    spec=importlib.util.spec_from_file_location('guard_under_test',snapshot)
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    binding={'experiment_id':'author-candidate-local-native-v3','attempt_id':'synthetic-guard-review',
             'request_sha256':'1'*64,'resource_sha256':'2'*64,'protocol_sha256':'3'*64,'token_cap':2}
    results=[]
    def check(name,func):
        try:
            details=func()
            results.append({'name':name,'passed':True,'details':details})
        except BaseException as exc:
            results.append({'name':name,'passed':False,'exception_type':type(exc).__name__,'detail':str(exc)})
    def need(value,message):
        if not value: raise AssertionError(message)
    def raises(func):
        try:func()
        except (ValueError,OSError) as exc:return type(exc).__name__+': '+str(exc)
        raise AssertionError('expected rejection, accepted instead')
    @contextlib.contextmanager
    def fixture(prompt=True):
        with tempfile.TemporaryDirectory(prefix='independent-guard-v3-') as d:
            p=Path(d)/'bundle'; w=g.AttemptWriter(p,binding)
            try:
                if prompt:
                    ack=w.prepare_prompt([7,8]);w.commit_prompt(ack,2)
                yield w,p
            finally:w.close()
    def evidence(state='selected',token=10):
        return {'selection':{'state':state,**({'token_id':token,'piece_hex':'7b'} if state=='selected' else {})},
                'fixture':True,'not_a_native_or_semantic_proof':True}
    def probe(w,ev=None,counter=2):
        ev=evidence() if ev is None else ev
        ack=w.prepare_distribution();w.commit_distribution(ack,counter,sample_evidence=ev)
        return ev
    def step(w,raw=b'{',token=10,ev=None,counter=3):
        ev=probe(w) if ev is None else ev
        ack=w.prepare_step(token,raw,sample_evidence=ev);w.commit_step(ack,counter)
        return ev
    def terminal_refusal(state):
        with fixture() as (w,p):
            ev=probe(w,evidence(state)); proof=w.seal([],b'',stop_label=state)
            checked=g.verify_bundle(p,binding,expected_terminal_sha256=proof['terminal_sha256'])
            trace=g.verify_trace((p/'journal.jsonl').read_bytes(),binding)
            need(checked['committed_tokens']==0 and checked['response']==b'' and trace['probes']==1,'zero-step accounting')
            return {'steps':0,'probes':trace['probes'],'terminal_sha256':proof['terminal_sha256']}
    for state in ['empty_support','resource_refusal']:
        check('zero_step_'+state,lambda state=state:terminal_refusal(state))
    def ordinary_complete():
        with fixture() as (w,p):
            step(w);proof=w.seal([10],b'{',stop_label='complete')
            checked=g.verify_bundle(p,binding,expected_terminal_sha256=proof['terminal_sha256'])
            need(checked['committed_tokens']==1,'one committed step')
            return {'steps':1,'scope':'guard accepts complete stop label; native verifier must verify actual schema'}
    check('selected_probe_one_step_terminal',ordinary_complete)
    def two_steps():
        with fixture() as (w,p):
            step(w);ev=probe(w,evidence(token=11),counter=3);step(w,b'{}',11,ev,counter=4)
            proof=w.seal([10,11],b'{}',stop_label='token_cap')
            trace=g.verify_trace((p/'journal.jsonl').read_bytes(),binding)
            need(trace['ids']==[10,11] and trace['probes']==2,'two distinct probes consumed')
            return {'steps':2,'probes':2}
    check('two_steps_each_require_probe',two_steps)
    def rejected(action,setup=None,prompt=True):
        with fixture(prompt=prompt) as (w,p):
            if setup:setup(w)
            detail=raises(lambda:action(w))
            need(not (p/'terminal.json').exists(),'rejected operation published terminal')
            return detail
    negatives=[
      ('distribution_before_prompt',lambda w:w.prepare_distribution(),None,False),
      ('seal_without_prompt',lambda w:w.seal([],b'',stop_label='empty_support'),None,False),
      ('prompt_only_cannot_seal_refusal',lambda w:w.seal([],b'',stop_label='empty_support'),None,True),
      ('prompt_only_cannot_seal_complete',lambda w:w.seal([],b'',stop_label='complete'),None,True),
      ('pending_distribution_cannot_seal',lambda w:w.seal([],b'',stop_label='empty_support'),lambda w:w.prepare_distribution(),True),
      ('pending_distribution_cannot_step',lambda w:w.prepare_step(10,b'{',sample_evidence=evidence()),lambda w:w.prepare_distribution(),True),
      ('second_pending_distribution_forbidden',lambda w:w.prepare_distribution(),lambda w:w.prepare_distribution(),True),
      ('selected_probe_cannot_be_skipped',lambda w:w.prepare_distribution(),lambda w:probe(w),True),
      ('selected_probe_cannot_seal_empty',lambda w:w.seal([],b'',stop_label='empty_support'),lambda w:probe(w),True),
      ('selected_probe_cannot_seal_complete',lambda w:w.seal([],b'',stop_label='complete'),lambda w:probe(w),True),
      ('step_without_probe',lambda w:w.prepare_step(10,b'{',sample_evidence=evidence()),None,True),
      ('refused_probe_cannot_step',lambda w:w.prepare_step(10,b'{',sample_evidence=evidence('empty_support')),lambda w:probe(w,evidence('empty_support')),True),
      ('refused_probe_cannot_probe_again',lambda w:w.prepare_distribution(),lambda w:probe(w,evidence('empty_support')),True),
      ('refusal_stop_must_match',lambda w:w.seal([],b'',stop_label='resource_refusal'),lambda w:probe(w,evidence('empty_support')),True),
      ('mismatched_selected_token',lambda w:w.prepare_step(11,b'{',sample_evidence=evidence()),lambda w:probe(w),True),
      ('mismatched_probe_evidence',lambda w:w.prepare_step(10,b'{',sample_evidence={**evidence(),'fixture':False}),lambda w:probe(w),True),
      ('consumed_probe_cannot_reuse',lambda w:w.prepare_step(10,b'{{',sample_evidence=evidence()),lambda w:step(w),True),
      ('pending_step_cannot_seal',lambda w:w.seal([],b'',stop_label='complete'),lambda w:(probe(w),w.prepare_step(10,b'{',sample_evidence=evidence())),True),
      ('wrong_distribution_ack',lambda w:w.commit_distribution('f'*64,2,sample_evidence=evidence()),lambda w:w.prepare_distribution(),True),
      ('wrong_distribution_counter',lambda w:w.commit_distribution(w._state['pending'][1],3,sample_evidence=evidence()),lambda w:w.prepare_distribution(),True),
      ('boolean_distribution_counter',lambda w:w.commit_distribution(w._state['pending'][1],True,sample_evidence=evidence()),lambda w:w.prepare_distribution(),True),
      ('unknown_selection_cannot_commit',lambda w:w.commit_distribution(w._state['pending'][1],2,sample_evidence=evidence('unknown')),lambda w:w.prepare_distribution(),True),
      ('wrong_step_ack',lambda w:w.commit_step('f'*64,3),lambda w:(probe(w),w.prepare_step(10,b'{',sample_evidence=evidence())),True),
      ('wrong_step_counter',lambda w:w.commit_step(w._state['pending'][1],4),lambda w:(probe(w),w.prepare_step(10,b'{',sample_evidence=evidence())),True),
      ('consumed_step_cannot_fake_refusal',lambda w:w.seal([10],b'{',stop_label='empty_support'),lambda w:step(w),True),
    ]
    for name,action,setup,prompt in negatives:
        check(name,lambda action=action,setup=setup,prompt=prompt:rejected(action,setup,prompt))
    def final_refusal_after_step():
        with fixture() as (w,p):
            step(w);probe(w,evidence('empty_support'),counter=3)
            proof=w.seal([10],b'{',stop_label='empty_support')
            trace=g.verify_trace((p/'journal.jsonl').read_bytes(),binding)
            need(len(trace['ids'])==1 and trace['probes']==2,'terminal probe must remain distinct from STEP')
            return {'steps':1,'probes':2}
    check('terminal_refusal_probe_separate_from_steps',final_refusal_after_step)
    def pending_trace():
        with fixture() as (w,p):
            w.prepare_distribution();w.close();raw=(p/'journal.jsonl').read_bytes()
            rejected=raises(lambda:g.verify_trace(raw,binding))
            trace=g.verify_trace(raw,binding,require_terminal=False)
            need(trace['pending'][0]=='distribution_prepared' and not trace['terminal'],'pending unknown retained')
            return rejected
    check('pending_probe_preserved_but_never_complete',pending_trace)
    def cap_enforced():
        with fixture() as (w,p):
            step(w);ev=probe(w,evidence(token=11),counter=3);step(w,b'{}',11,ev,counter=4)
            return raises(w.prepare_distribution)
    check('no_probe_after_token_cap',cap_enforced)
    def mutation(kind):
        with fixture() as (w,p):
            probe(w,evidence('empty_support'));proof=w.seal([],b'',stop_label='empty_support')
            if kind=='truncate':
                f=p/'journal.jsonl';f.write_bytes(f.read_bytes()[:-1])
            elif kind=='replace_response':(p/'response.raw').write_bytes(b'forged')
            elif kind=='append_artifact':(p/'unexpected').write_bytes(b'x')
            elif kind=='wrong_terminal_pin':proof['terminal_sha256']='f'*64
            elif kind=='symlink_member':
                original=p/'response.raw';target=p.parent/'target';original.rename(target);original.symlink_to(target)
            elif kind=='symlink_bundle':
                alias=p.parent/'alias';alias.symlink_to(p,target_is_directory=True);p=alias
            elif kind=='change_terminal':
                f=p/'terminal.json';value=json.loads(f.read_bytes());value['journal_rows']+=1;f.write_bytes(g.canonical(value))
            return raises(lambda:g.verify_bundle(p,binding,expected_terminal_sha256=proof['terminal_sha256']))
    for kind in ['truncate','replace_response','append_artifact','wrong_terminal_pin','symlink_member','symlink_bundle','change_terminal']:
        check('sealed_bundle_rejects_'+kind,lambda kind=kind:mutation(kind))
    def no_overwrite():
        with fixture() as (w,p):
            before=(p/'journal.jsonl').read_bytes();detail=raises(lambda:g.AttemptWriter(p,binding))
            need((p/'journal.jsonl').read_bytes()==before,'duplicate writer overwrote evidence')
            return detail
    check('existing_attempt_refused_unchanged',no_overwrite)
    def poisoned_no_continue():
        with fixture() as (w,p):
            raises(lambda:w.prepare_step(10,b'{',sample_evidence=evidence()))
            return raises(w.prepare_distribution)
    check('writer_failure_permanently_poisoned',poisoned_no_continue)
    def active_writer_blocks_verification():
        with fixture() as (w,p):
            return raises(lambda:g.verify_bundle(p,binding,expected_terminal_sha256='f'*64))
    check('active_writer_blocks_verification',active_writer_blocks_verification)
    def rehashed_mutation(mutation):
        # Build a legal sequence once, then independently reconstruct every chain
        # hash after corrupting a semantic event. Rejection must not be merely a
        # stale SHA mismatch. Fixtures are synthetic, not native outcomes.
        with fixture() as (w,p):
            step(w);probe(w,evidence('empty_support'),counter=3)
            w.seal([10],b'{',stop_label='empty_support')
            events=[json.loads(row)['event'] for row in (p/'journal.jsonl').read_bytes().splitlines()]
            mutation(events)
            rows=[];previous='0'*64
            for number,event in enumerate(events):
                body={'sequence':number,'previous_sha256':previous,'event':event}
                # Independent compact canonical serialization and hashlib.
                raw=json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                previous=hashlib.sha256(raw).hexdigest()
                rows.append(json.dumps({**body,'event_sha256':previous},ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n')
            return raises(lambda:g.verify_trace(b''.join(rows),binding))
    def find(events,kind,index=0):return [e for e in events if e['kind']==kind][index]
    mutations=[
      ('wrong_probe_prefix_hash',lambda e:find(e,'distribution_prepared').update(raw_sha256='f'*64)),
      ('wrong_probe_prefix_ids',lambda e:find(e,'distribution_prepared').update(prefix_ids=[99])),
      ('boolean_probe_index',lambda e:find(e,'distribution_prepared').update(token_index=False)),
      ('wrong_probe_index',lambda e:find(e,'distribution_prepared').update(token_index=1)),
      ('extra_probe_field',lambda e:find(e,'distribution_prepared').update(extra='unexpected')),
      ('wrong_probe_ack_digest',lambda e:find(e,'distribution_committed').update(prepared_sha256='f'*64)),
      ('boolean_probe_counter',lambda e:find(e,'distribution_committed').update(evaluated_tokens=True)),
      ('unknown_probe_selection',lambda e:find(e,'distribution_committed')['sample_evidence']['selection'].update(state='unknown')),
      ('changed_step_evidence',lambda e:find(e,'step_prepared')['sample_evidence'].update(fixture=False)),
      ('mismatched_step_token',lambda e:find(e,'step_prepared').update(token_id=11)),
      ('boolean_step_token',lambda e:find(e,'step_prepared').update(token_id=True)),
      ('unknown_terminal_label',lambda e:find(e,'trace_complete').update(stop_label='unknown')),
      ('mismatched_refusal_label',lambda e:find(e,'trace_complete').update(stop_label='resource_refusal')),
      ('mismatched_terminal_ids',lambda e:find(e,'trace_complete').update(output_ids=[11])),
      ('boolean_terminal_bytecount',lambda e:find(e,'trace_complete').update(response_bytes=True)),
    ]
    for name,mutation in mutations:
        check('rehashed_trace_rejects_'+name,lambda mutation=mutation:rehashed_mutation(mutation))
    after=hashlib.sha256(source.read_bytes()).hexdigest()
    check('source_unchanged_during_run',lambda:need(before==after,'concurrent source change'))
    report={'schema':'candidate-journal-guard-v3-independent-review-v2','source':str(source),
      'source_sha256':before,'source_sha256_after':after,'source_snapshot':str(snapshot),
      'test_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'checks':results,'passed':sum(x['passed'] for x in results),'failed':sum(not x['passed'] for x in results),
      'total':len(results),'model_calls':0,'tokenizer_calls':0,'native_calls':0,
      'scope':'Synthetic journal mechanics only. Does not prove actual native execution, logits, token pieces, schema correctness or semantic usefulness.',
      'completed_at_unix':time.time()}
    raw=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'output':str(output),'source_sha256':before,'passed':report['passed'],'failed':report['failed'],'total':len(results)}))
    return int(report['failed']!=0)

if __name__=='__main__':raise SystemExit(main())
