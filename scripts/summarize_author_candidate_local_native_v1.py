#!/usr/bin/env python3
"""Read-only transcript summary for two candidate requests; no model/tokenizer.

No retry, response repair, embedding or decoding. ALL reserved attempts and
unknowns remain visible. Shape success is not semantic synonymy or full cover.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import json
import math
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
VERSION='author-candidate-local-native-v1'
BASE='results/author-candidate-local-native-v1/native-diagnostic-v1'
EXPECTED=['2a510a01e1b7ee830f393c97bce447889f28ff24e251f9c6ee82231259bc2247',
          'ec61d91995330841f842969e930c996004214e8822b6511c8f81383ae860d328']
TOTAL=1<<24


def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def strict_words(raw):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    x=json.loads(raw.decode('utf-8'),object_pairs_hook=unique,
                 parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
    if type(x) is not dict or set(x)!={'words'} or type(x['words']) is not list or not x['words']:
        raise ValueError('words shape')
    for row in x['words']:
        if type(row) is not dict or len(row)!=1:raise ValueError('base shape')
        base,candidates=next(iter(row.items()))
        if type(base) is not str or not base.strip() or type(candidates) is not list:
            raise ValueError('candidate shape')
        if any(type(s) is not str or not s.strip() for s in candidates):raise ValueError('candidate string')
    return x['words']


def json_complete(raw):
    try:json.loads(raw.decode('utf-8'))
    except RecursionError:return 'parser_resource_refusal'
    except (UnicodeDecodeError,ValueError,TypeError):return None
    return 'complete_json_value'


def reproduce_weights(rows):
    if len(rows)!=50 or len({r['token_id'] for r in rows})!=50:raise ValueError('50 unique rows required')
    decoded=[]
    for r in rows:
        if set(r)!={'token_id','logit_hex','weight'} or type(r['token_id']) is not int or not 0<=r['token_id']<151936:
            raise ValueError('logit row schema')
        raw=r['logit_hex']
        if type(raw) is not str or len(raw)!=8:raise ValueError('float32 encoding')
        value=struct.unpack('>f',bytes.fromhex(raw))[0]
        if not math.isfinite(value):raise ValueError('nonfinite logit')
        decoded.append((r['token_id'],raw,value))
    decoded.sort(key=lambda x:(-x[2],x[0]))
    masses=[math.exp((x[2]-decoded[0][2])/0.7) for x in decoded]
    denom=math.fsum(masses);scaled=[x/denom*TOTAL for x in masses]
    weights=[math.floor(x) for x in scaled];remaining=TOTAL-sum(weights)
    if not 0<=remaining<=50:raise ValueError('quantizer remainder')
    order=sorted(range(50),key=lambda i:(-(scaled[i]-weights[i]),decoded[i][0]))
    for i in order[:remaining]:weights[i]+=1
    return [{'token_id':x[0],'logit_hex':x[1],'weight':w} for x,w in zip(decoded,weights)]


def journal_check(path,request,attempt):
    previous='0'*64;ids=[];raw=b'';pending=None;prompt=None;prompt_committed=False
    known=None;unknown=False;torn=False;first_complete=None;events=0;prepared_steps=0
    seed=sha(canonical([VERSION,'fixed-public-diagnostic-seed',request,attempt]))
    lines=Path(path).read_bytes().splitlines(keepends=True)
    for sequence,line in enumerate(lines):
        if not line.endswith(b'\n'):
            if sequence!=len(lines)-1:raise ValueError('interior torn line')
            torn=True;break
        row=json.loads(line);own=row.pop('event_sha256')
        if row.get('sequence')!=sequence or row.get('previous_sha256')!=previous or sha(canonical(row))!=own:
            raise ValueError('journal hash chain')
        previous=own;events+=1;kind=row['kind']
        if kind=='prompt_prepared':
            if sequence!=0 or row['request_sha256']!=request:raise ValueError('prompt intent binding')
            prompt=row['prompt_ids']
        elif kind=='prompt_committed':
            if prompt is None or prompt_committed or row['request_sha256']!=request:raise ValueError('prompt commit ordering')
            prompt_committed=True
        elif kind=='step_prepared':
            if not prompt_committed or pending or known or unknown or row['token_index']!=len(ids) or row['prefix_ids']!=ids:
                raise ValueError('STEP ordering')
            if len(ids)>=512 or first_complete is not None:raise ValueError('STEP after predeclared stop')
            if row['request_sha256']!=request:raise ValueError('STEP request binding')
            if reproduce_weights(row['weights'])!=row['weights']:raise ValueError('quantized weights differ')
            ctx=[VERSION,request,attempt,len(ids),sha(canonical(ids))]
            draw=int.from_bytes(hmac.new(bytes.fromhex(seed),canonical(ctx),hashlib.sha256).digest()[:3],'big')
            if row['draw_uint24']!=draw or row['draw_context_sha256']!=sha(canonical(ctx)):
                raise ValueError('HMAC draw differs')
            residue=draw;selected=None
            for weight in row['weights']:
                if residue<weight['weight']:selected=weight['token_id'];break
                residue-=weight['weight']
            if row['token_id']!=selected:raise ValueError('selected token differs')
            intended=bytes.fromhex(row['intended_raw_hex'])
            if sha(intended)!=row['intended_raw_sha256']:raise ValueError('intended raw hash')
            pending=(own,row['token_id'],intended);prepared_steps+=1
        elif kind=='step_committed':
            if pending is None or unknown or row['prepared_sha256']!=pending[0] or row['token_id']!=pending[1] or row['token_index']!=len(ids):
                raise ValueError('STEP commit binding')
            if row['raw_sha256']!=sha(pending[2]):raise ValueError('STEP raw binding')
            ids.append(pending[1]);raw=pending[2];pending=None
            if first_complete is None:first_complete=json_complete(raw)
        elif kind=='known_response':
            if not prompt_committed or pending or known or unknown or row['request_sha256']!=request:
                raise ValueError('known response ordering')
            if row['raw_sha256']!=sha(raw) or row['committed_tokens']!=len(ids):raise ValueError('known response bytes')
            expected=first_complete or ('max_tokens' if len(ids)==512 else None)
            # Parser recursion thresholds can depend on call depth; retained resource
            # refusal is never accepted as a candidate success by this check.
            if row['stop']!=expected and row['stop']!='parser_resource_refusal':raise ValueError('stop differs')
            if row.get('outcome') not in {'accepted','shape_rejected'}:raise ValueError('known response outcome')
            known=row
        elif kind=='unknown':
            if row['request_sha256']!=request or unknown:raise ValueError('UNKNOWN ordering')
            unknown=True
        else:raise ValueError('unregistered journal event')
    return {'events':events,'prepared_steps':prepared_steps,'committed_steps':len(ids),
        'prompt_prepared':prompt is not None,'prompt_committed':prompt_committed,
        'prompt_token_count':None if prompt is None else len(prompt),'committed_ids':ids,
        'raw':raw,'raw_sha256':sha(raw),'pending_STEP':pending is not None,'UNKNOWN':unknown,
        'torn_last_line':torn,'known_response':known,'seed_hex':seed}


def summarize(root,base,output):
    root=Path(root);base_path=root/base;output=Path(output)
    if output.exists():raise FileExistsError('new summary output required')
    issues=[];inputs={}
    def read(path,optional=False):
        if not path.is_file():
            if not optional:issues.append({'path':str(path.relative_to(root)),'issue':'missing'})
            return None
        raw=path.read_bytes();inputs[str(path.relative_to(root))]={'bytes':len(raw),'sha256':sha(raw)}
        return json.loads(raw)
    campaign=read(base_path/'result.json',optional=True)
    intent=read(base_path/'intent.json',optional=True)
    if campaign is None:issues.append({'issue':'terminal campaign result missing; incomplete evidence, no retry'})
    if intent is None:issues.append({'issue':'campaign source-readback intent missing'})
    req=read(root/'experiments/author-candidate-local-native-v1/requests.json')
    if req is None:raise ValueError('frozen request descriptions required')
    if [x['request_sha256'] for x in req['requests']]!=EXPECTED:raise ValueError('fixed requests differ')
    logical=campaign.get('chunks',[]) if campaign else [{'chunk_order':i,'request_sha256':x,'status':'unknown_or_not_attempted','attempts':[]} for i,x in enumerate(EXPECTED)]
    if type(logical) is not list or len(logical)!=2:
        issues.append({'issue':'exact two logical rows missing'})
        logical=[]
    if campaign and (campaign.get('schema')!=VERSION+'-campaign-result' or campaign.get('requested_chunks')!=2 or campaign.get('max_attempts')!=6):
        issues.append({'issue':'campaign schema/denominator differs'})
    parents={};derived_status={};terminal_unknown=False
    for chunk,c in enumerate(logical):
        try:
            if type(c) is not dict or set(c)!={'chunk_order','request_sha256','status','attempts'}:
                raise ValueError('logical row schema')
            if type(c['chunk_order']) is not int or c['chunk_order']!=chunk or c['request_sha256']!=EXPECTED[chunk]:
                raise ValueError('logical row identity/order')
            records=c['attempts']
            if type(records) is not list or len(records)>3:raise ValueError('physical attempt bound')
            if terminal_unknown and records:raise ValueError('attempt after terminal UNKNOWN')
            for order,a in enumerate(records):
                identity={'schema':VERSION+'-attempt-result','chunk_order':chunk,'attempt_order':order,
                    'request_sha256':EXPECTED[chunk],
                    'seed_hex':sha(canonical([VERSION,'fixed-public-diagnostic-seed',EXPECTED[chunk],order]))}
                if type(a) is not dict or not all(type(a.get(k)) is type(v) and a.get(k)==v for k,v in identity.items()):
                    raise ValueError('parent attempt identity')
                if a.get('outcome') not in {'accepted','shape_rejected','unknown'}:raise ValueError('parent attempt outcome')
                if order<len(records)-1 and a['outcome']!='shape_rejected':raise ValueError('retry after accepted/UNKNOWN')
                parents[(chunk,order)]=a
            if not records:
                derived='not_attempted'
                if campaign and not terminal_unknown:raise ValueError('missing attempted logical request without prior UNKNOWN')
            elif records[-1]['outcome']=='shape_rejected':
                if len(records)!=3:raise ValueError('premature known shape exhaustion')
                derived='shape_attempts_exhausted'
            else:derived=records[-1]['outcome']
            derived_status[chunk]=derived
            if campaign and c['status']!=derived:raise ValueError('logical status not derived from verified attempt sequence')
            if derived=='unknown':terminal_unknown=True
        except (ValueError,TypeError,KeyError) as exc:issues.append({'logical_chunk':chunk,'issue':str(exc)})
    if campaign and campaign.get('status')!=('unknown_terminal' if terminal_unknown else 'complete_diagnostic'):
        issues.append({'issue':'global terminal status differs'})
    attempts=[]
    present_paths={p.name:p for p in base_path.glob('chunk*-attempt*') if p.is_dir()}
    for chunk,order in parents:present_paths.setdefault(f'chunk{chunk}-attempt{order}',base_path/f'chunk{chunk}-attempt{order}')
    paths=[present_paths[k] for k in sorted(present_paths)]
    for p in paths:
        import re
        m=re.fullmatch('chunk([01])-attempt([012])',p.name)
        if not m:
            issues.append({'path':str(p.relative_to(root)),'issue':'unregistered physical attempt directory'});continue
        chunk,order=map(int,m.groups());request=EXPECTED[chunk]
        row={'chunk_order':chunk,'attempt_order':order,'directory':str(p.relative_to(root)),
             'request_sha256':request,'outcome':'unknown','journal_verified':False}
        launch=read(p/'launch.json');result=read(p/'result.json',optional=True)
        try:
            if launch is None or launch['request_sha256']!=request or launch['attempt_order']!=order or launch['chunk_order']!=chunk:
                raise ValueError('launch binding')
            jpath=p/'journal.jsonl';data=jpath.read_bytes();inputs[str(jpath.relative_to(root))]={'bytes':len(data),'sha256':sha(data)}
            j=journal_check(jpath,request,order);row.update(journal_verified=True,
                committed_steps=j['committed_steps'],prepared_steps=j['prepared_steps'],
                prompt_committed=j['prompt_committed'],prompt_token_count=j['prompt_token_count'],
                raw_bytes=len(j['raw']),raw_sha256=j['raw_sha256'],pending_STEP=j['pending_STEP'],
                torn_last_line=j['torn_last_line'],journal_UNKNOWN=j['UNKNOWN'])
            if launch['seed_hex']!=j['seed_hex']:raise ValueError('launch seed binding')
            rp=p/'response.raw'
            if rp.exists():
                raw=rp.read_bytes();inputs[str(rp.relative_to(root))]={'bytes':len(raw),'sha256':sha(raw)}
                if raw!=j['raw']:raise ValueError('response.raw differs from committed journal')
            else:row['response_file_missing']=True
            words=None
            try:words=strict_words(j['raw']);row['strict_shape_valid']=True
            except (UnicodeDecodeError,ValueError,TypeError,RecursionError):row['strict_shape_valid']=False
            if result is not None:
                if (result['schema']!=VERSION+'-attempt-result' or result['request_sha256']!=request or
                    result['chunk_order']!=chunk or result['attempt_order']!=order or result['seed_hex']!=j['seed_hex']):
                    raise ValueError('result identity binding')
                if result['generated_ids']!=j['committed_ids'] or result['raw_sha256']!=j['raw_sha256'] or result['raw_bytes']!=len(j['raw']):
                    raise ValueError('result bytes/history')
                if result['committed_output_tokens']!=j['committed_steps'] or result['step_calls_committed']!=j['committed_steps']:
                    raise ValueError('result STEP counters')
                if result['outcome'] not in {'accepted','shape_rejected','unknown'}:raise ValueError('result outcome')
                if result['outcome'] in {'accepted','shape_rejected'}:
                    if (j['known_response'] is None or j['pending_STEP'] or j['UNKNOWN'] or j['torn_last_line']
                        or not j['prompt_committed'] or j['known_response']['outcome']!=result['outcome']):
                        raise ValueError('known result lacks complete terminal response journal')
                if result['outcome']=='accepted' and (words is None or result['accepted_words']!=words):raise ValueError('accepted shape differs')
                if result['outcome']=='shape_rejected' and words is not None:raise ValueError('shape refusal disagrees')
                row.update(outcome=result['outcome'],stop_reason=result['stop_reason'],
                    elapsed_seconds=result['elapsed_seconds'],step_calls_invoked=result['step_calls_invoked'],
                    distribution_calls_invoked=result['distribution_calls_invoked'],native_pid=result['native_pid'],
                    native_returncode_after_cleanup=result['native_returncode_after_cleanup'],
                    prompt_invoked=result['prompt_invoked'],error=result['error'],cleanup_error=result['cleanup_error'])
                parent=parents.get((chunk,order))
                if parent!=result:raise ValueError('parent campaign differs from attempt result')
            else:
                row['result_missing_UNKNOWN']=True
                issues.append({'path':str(p.relative_to(root)),'issue':'attempt result missing; retained UNKNOWN, numerator not validated'})
            if words is not None:
                row['candidate_shape_details']={'expected_base_words':req['requests'][chunk]['words'],
                    'returned_rows':len(words),'returned_base_words':[next(iter(x)) for x in words],
                    'candidate_counts':[len(next(iter(x.values()))) for x in words],
                    'requested_candidates_per_word':8,'semantic_equivalence_verified':False,
                    'scope':'Structural counts only; incomplete/wrong lexical rows do not trigger semantic retries.'}
        except (ValueError,TypeError,KeyError,OSError,RecursionError) as exc:
            row['verification_error']=type(exc).__name__+': '+str(exc)
            issues.append({'path':str(p.relative_to(root)),'issue':row['verification_error']})
        attempts.append(row)
    if campaign:
        parent_count=len(parents)
        if parent_count!=campaign['attempts_executed'] or len(attempts)!=parent_count:
            issues.append({'issue':'physical directories / parent attempt count differ','directories':len(attempts),'parent':parent_count})
    accepted=sum(v=='accepted' for v in derived_status.values())
    if campaign and campaign['accepted_chunks']!=accepted:issues.append({'issue':'accepted chunk numerator differs'})
    scalar={'schema':VERSION+'-saved-summary','status':'passed' if not issues else 'needs_review',
        'logical_requests':2,'physical_attempt_upper_bound':6,'physical_attempt_directories':len(attempts),
        'validated_shape_accepted_chunks':accepted if campaign and not issues else None,
        'claimed_shape_accepted_chunks':None if campaign is None else campaign.get('accepted_chunks'),
        'known_shape_rejected_physical_attempts':sum(a['outcome']=='shape_rejected' for a in attempts),
        'UNKNOWN_physical_attempts':sum(a['outcome']=='unknown' for a in attempts),
        'UNKNOWN_parent_attempts':sum(a['outcome']=='unknown' for a in parents.values()),
        'committed_output_tokens':sum(a.get('committed_steps',0) for a in attempts),
        'prompt_tokens_for_committed_PROMPTs':sum(a.get('prompt_token_count',0) or 0 for a in attempts if a.get('prompt_committed')),
        'summed_attempt_elapsed_seconds':sum(a.get('elapsed_seconds',0) for a in attempts),
        'elapsed_scope':'Sum of attempt timers, not full coordinator wall time.',
        'raw_response_bytes':sum(a.get('raw_bytes',0) for a in attempts),
        'source_freeze_commit':None if intent is None else intent.get('source_readback',{}).get('commit'),
        'full_cover_required_chunks':109,'diagnostic_chunks':2,'outside_diagnostic_chunks':107,
        'full_cover_candidate_resource_complete':False,'original_fixed_transport_outcomes':0,
        'provider_usage':None,'model_or_tokenizer_calls_during_summary':0,'paid_calls':0,'closed_test_opened':False,
        'claims':['Shape-only local provider diagnostic, not GPT4o API replication.',
                  'MaskedEOS, whole-prefixJSON-or512 stopping; no repair/fence stripping.',
                  'Raw logits replay verifies recorded transcript; no fresh model logits or tokenizer recomputed.',
                  'No semantic synonymy, full-cover capacity, transport success or steganographic security inferred.'],
        'issues':issues}
    output.mkdir(parents=True)
    save(output/'summary.json',scalar);save(output/'attempts.json',attempts);save(output/'logical-requests.json',logical)
    save(output/'inputs.json',{'files':inputs,'summarizer_sha256':sha(Path(__file__).read_bytes())})
    accepted_display='не определён' if scalar['validated_shape_accepted_chunks'] is None else str(scalar['validated_shape_accepted_chunks'])
    lines=['# Два локальных candidate-запроса: сохранённый результат','',
        f"Проверка сохранённых журналов: **{scalar['status']}**. Форма JSON принята для **{accepted_display}/2** запросов; физических попыток — {len(attempts)} из верхней границы6.",
        '', '| Группа | Попытка | Исход | Остановка | Подтверждённых STEP | Raw bytes |',
        '|---|---:|---|---|---:|---:|']
    for a in attempts:lines.append(f"| {a['chunk_order']} | {a['attempt_order']} | {a['outcome']} | {a.get('stop_reason','unknown')} | {a.get('committed_steps','unknown')} | {a.get('raw_bytes','unknown')} |")
    lines += ['',f"Сохранено {scalar['committed_output_tokens']} подтверждённых выходных токенов; сумма attempt timers — {scalar['summed_attempt_elapsed_seconds']:.3f}s, это не полный wall time. Provider/API usage не измерялось и остаётся null.",
        '', 'Это отдельный EOS-masked Qwen2.5-1.5B-InstructQ4_K_M provider, а не GPT-4o JSON API. Остановка по целому JSON-prefix или512tokens; fences/ошибки не исправлялись. Все отказы и UNKNOWN остаются в знаменателе.',
        '', 'Принятая JSON-форма не означает, что для каждого из пяти исходных слов получено восемь пригодных синонимов. Полные raw и структурные количества сохранены отдельно; смысл, POS, значения и имена не сертифицированы.',
        '', 'Полный первый cover требует109групп. Здесь запрошены только первые2;107 находятся вне diagnostic, полный candidate resource не завершён. Новых original/fixed encode/decode outcomes:0. Старые результаты не меняются.',
        '', 'Проверка повторяет арифметику сохранённых logits/HMAC и структуру journals, не модельный forward и не tokenization. Поэтому это проверка записи исполнения, а не межмашинная или независимая нейросетевая репликация.']
    if issues:lines += ['', '**Есть несогласованности; валидированный числитель оставлен null. См. summary.json issues.**']
    with (output/'report_ru.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    return scalar


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--base',default=BASE);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(a.root,a.base,a.output)
    print(json.dumps({k:result[k] for k in ['status','validated_shape_accepted_chunks','physical_attempt_directories','committed_output_tokens']},sort_keys=True))
    return 0 if result['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
