#!/usr/bin/env python3
"""Independent saved-evidence structural audit. No production code imports or codec calls."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
ID = 'prospective-development-transport-first40-delta-slot-review-v1'
MILESTONE = ROOT/'results/prospective-development-transport-first40-v1/milestones/tinyllama-first40'
OUT = ROOT/'results'/ID
LABELS = {'meaning_change','grammar_or_pos','style_or_collocation','ambiguous','no_obvious_added_defect'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def rows(path):
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def audit():
    checks=[]
    def ck(condition,name):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def obj(path):return json.loads(Path(path).read_text())
    def read(name):return rows(OUT/name)
    def index(seq,key):
        v={x[key]:x for x in seq};ck(len(v)==len(seq),'unique '+key);return v
    inventory=obj(OUT/'inventory.json')
    for item in inventory['frozen_inputs']:
        ck(digest((ROOT/item['path']).read_bytes())==item['sha256'],'frozen input '+item['path'])
    for name,h in inventory['artifact_sha256'].items():ck(digest((OUT/name).read_bytes())==h,'inventory artifact '+name)
    all_cells=rows(MILESTONE/'cells.jsonl.gz');record_rows=rows(MILESTONE/'records.jsonl.gz')
    jobs=index(rows(MILESTONE/'recipient/receiver_jobs.jsonl.gz'),'row_id')
    receivers=index(rows(MILESTONE/'recipient/receiver_results.jsonl.gz'),'row_id')
    ck(len(all_cells)==15300 and len(record_rows)==300,'full cumulative grid and records')
    ck(len(jobs)==len(receivers)==99,'cumulative99 jobs/results')
    all_by_id=index(all_cells,'row_id')
    ck(set(jobs)==set(receivers)=={x['row_id'] for x in all_cells if x['full_exact'] is True},'job/result/full_exact join')
    delta=[x for x in all_cells if x['model_alias']=='tinyllama11' and 31<=x['source_order']<=40]
    attempts=[x for x in delta if x['encoder_called'] is True]
    successes=[x for x in delta if x['full_exact'] is True]
    prior_success={x['row_id'] for x in all_cells if x['model_alias']=='tinyllama11' and x['source_order']<=30 and x['full_exact'] is True}
    ck(len(delta)==510 and len(attempts)==240 and len(successes)==24,'new510 cells240 encodes24 success')
    ck(Counter(x['execution_status'] for x in delta)==Counter({'full_exact':24,'insufficient_capacity':216,'not_run_resource_blocked':270}),'all delta status denominator')
    ck(len(prior_success)==75 and not prior_success.intersection(x['row_id'] for x in successes),'old75 excluded')
    selected=sorted([x for x in record_rows if x['model_alias']=='tinyllama11' and 31<=x['source_order']<=40],key=lambda x:x['source_order'])
    ck([x['source_order'] for x in selected]==list(range(31,41)) and all(x['status']=='completed' for x in selected),'all ten covers included')
    source=index(rows(ROOT/'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'),'record_id')
    covers=index(read('covers.jsonl'),'record_id');outcomes=index(read('outcomes.jsonl'),'row_id');unique=index(read('unique_slots.jsonl'),'slot_id')
    ck(set(covers)=={x['record_id'] for x in selected},'cover membership exact')
    ck(set(outcomes)=={x['row_id'] for x in successes},'outcome membership exact')
    for rec in selected:
        cover=covers[rec['record_id']];src=source[rec['record_id']];ap=ROOT/cover['attempt_snapshot_path'];attempt=obj(ap)
        ck(digest(ap.read_bytes())==rec['attempt_sha256']==cover['attempt_sha256'],'attempt identity '+str(rec['source_order']))
        ck(cover['cover_text']==rec['cover_text']==attempt['result']['cover_text'],'cover text '+str(rec['source_order']))
        ck(cover['cover_sha256']==digest(cover['cover_text'].encode()),'cover hash '+str(rec['source_order']))
        ck(cover['source_excerpt']==src['excerpt'] and cover['source_excerpt_sha256']==digest(src['excerpt'].encode())==src['excerpt_sha256'],'source excerpt '+str(rec['source_order']))
        ck(cover['title']==src['title'] and rec['prompt_sha256']==digest(src['prompt'].encode()),'prompt/title '+str(rec['source_order']))
        local=[x for x in delta if x['identity']==rec['identity']]
        ck(len(local)==51 and cover['all_cell_execution_status_counts']==dict(Counter(x['execution_status'] for x in local)),'51 cover cells '+str(rec['source_order']))
        ck(cover['successful_row_ids']==sorted(x['row_id'] for x in local if x['full_exact'] is True),'cover successful crosswalk '+str(rec['source_order']))
    recomputed={};occurrence_count=0
    ordered=sorted(successes,key=lambda x:(x['source_order'],x['parent_payload_length'],x['row_id']))
    for cell in ordered:
        rid=cell['row_id'];cover=covers[cell['record_id']];original=cover['cover_text'];stego=jobs[rid]['stego_text'];out=outcomes[rid];receiver=receivers[rid]
        ck(receiver['status']=='decoded' and receiver['error'] is None and cell['fresh_receiver_exact'] is True,'saved successful receiver '+rid)
        bits=receiver['decoded_bits'];ck(len(bits)==cell['decoded_bits']==cell['requested_payload_bits'] and set(bits)<={'0','1'},'decoded length '+rid)
        ck(digest(bits.encode('ascii'))==cell['decoded_sha256']==cell['payload_sha256'],'decoded hash '+rid)
        ck(digest(stego.encode())==cell['stego_sha256']==out['stego_sha256'] and out['stego_text']==stego and out['cell']==cell,'stego/cell identity '+rid)
        left=list(re.finditer(r'\S+',original));right=list(re.finditer(r'\S+',stego))
        ck(len(left)==len(right)==cell['cover_words']==cell['stego_words'],'token count '+rid)
        ck(re.sub(r'\S+','TOKEN',original)==re.sub(r'\S+','TOKEN',stego),'unchanged whitespace '+rid)
        changed=[]
        for i,(a,b) in enumerate(zip(left,right)):
            if a.group()==b.group():continue
            def core(match):
                positions=[n for n,c in enumerate(match.group()) if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ']
                ck(bool(positions),'ASCII core exists')
                lo,hi=positions[0],positions[-1]+1
                ck(all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in match.group()[lo:hi]),'single ASCII core')
                return lo,hi,match.group()[lo:hi]
            alo,ahi,aw=core(a);blo,bhi,bw=core(b)
            ck(a.group()[:alo]==b.group()[:blo] and a.group()[ahi:]==b.group()[bhi:],'punctuation unchanged')
            key=[cell['record_id'],i,aw,bw];sid=digest(json.dumps(key,ensure_ascii=True,separators=(',',':')).encode());changed.append(sid)
            def ctxt(text,tokens):
                start=tokens[max(0,i-10)].start();end=tokens[min(len(tokens)-1,i+10)].end();return {'start':start,'end':end,'quote':text[start:end]}
            value=recomputed.setdefault(sid,{'slot_id':sid,'deduplication_key':key,'source_order':cell['source_order'],'record_id':cell['record_id'],'token_index_zero_based':i,'original':aw,'replacement':bw,'original_token':a.group(),'replacement_token':b.group(),'cover_sha256':cell['cover_sha256'],'source_excerpt_sha256':cover['source_excerpt_sha256'],'original_span':{'start':a.start()+alo,'end':a.start()+ahi,'quote':aw},'original_context':ctxt(original,left),'occurrences':[]})
            value['occurrences'].append({'row_id':rid,'parent_payload_length':cell['parent_payload_length'],'stego_sha256':cell['stego_sha256'],'stego_span':{'start':b.start()+blo,'end':b.start()+bhi,'quote':bw},'stego_context':ctxt(stego,right)})
        ck(changed==out['changed_slot_ids'] and len(changed)==cell['changed_positions']==out['changed_positions'],'exact outcome slot crosswalk '+rid)
        occurrence_count+=len(changed)
    ck(recomputed==unique,'all recomputed slots/context spans/occurrences exact')
    ck(occurrence_count==380 and len(recomputed)==180,'380 occurrences180 unique slots')
    ck(inventory['changed_slot_occurrences']==380 and inventory['unique_changed_slots']==180,'inventory total truth')
    annotation={}
    for part,orders in [('a',{31,32,33}),('b',{34,35,36}),('c',{38,39})]:
        partrows=read('review-part-'+part+'.jsonl');ck({x['slot_id'] for x in partrows}=={x['slot_id'] for x in unique.values() if x['source_order'] in orders},'part '+part+' exact ownership')
        for x in partrows:
            sid=x['slot_id'];ck(sid not in annotation,'annotation unique '+sid);annotation[sid]=x;slot=unique[sid]
            ck(all(x[k]==slot[k] for k in ['source_order','token_index_zero_based','original','replacement']),'annotation token binding '+sid)
            labels=x['labels'];ck(bool(labels) and len(labels)==len(set(labels)) and set(labels)<=LABELS,'label enum '+sid)
            ck('no_obvious_added_defect' not in labels or labels==['no_obvious_added_defect'],'exclusive no-obvious label '+sid)
            ids=[z['row_id'] for z in slot['occurrences']]
            ck(len(x['reviewed_occurrence_row_ids'])==len(ids) and set(x['reviewed_occurrence_row_ids'])==set(ids),'all occurrence review declared '+sid)
            ck(len(x['issue_row_ids'])==len(set(x['issue_row_ids'])) and set(x['issue_row_ids'])<=set(ids),'issue subset '+sid)
            ck(bool(x['issue_row_ids'])==(labels!=['no_obvious_added_defect']),'issue/no-obvious consistency '+sid)
            ck(x['review_status']=='read_all_associated_contexts' and len(x['reason_ru'])>=20,'manual rationale presence '+sid)
    ck(set(annotation)==set(unique),'complete manual180 coverage')
    annotated=index(read('annotated_unique_slots.jsonl'),'slot_id');ck(set(annotated)==set(unique),'compiled180 slot membership')
    for sid,v in annotated.items():ck(v=={**unique[sid],'annotation':annotation[sid]},'compiled annotation exact '+sid)
    reviews=index(read('outcome_reviews.jsonl'),'row_id');ck(set(reviews)==set(outcomes),'compiled24 outcomes')
    label_unique=Counter();label_occurrences=Counter()
    for sid,a in annotation.items():
        label_unique.update(a['labels']);n=len(unique[sid]['occurrences']) if a['labels']==['no_obvious_added_defect'] else len(a['issue_row_ids'])
        for label in a['labels']:label_occurrences[label]+=n
    for rid,v in reviews.items():
        ids=outcomes[rid]['changed_slot_ids'];issues=[i for i in ids if rid in annotation[i]['issue_row_ids']]
        ck(v['all_changed_slot_ids']==ids and v['observed_issue_or_ambiguity_slot_ids']==issues,'compiled outcome linkage '+rid)
        ck(v['observed_labels_present']==sorted({l for i in issues for l in annotation[i]['labels']}),'compiled outcome labels '+rid)
        ck(v['review_complete_for_changed_slots'] is True and v['validated_quality_pass_fail'] is None,'unvalidated quality marker '+rid)
    cover_reviews=index(read('cover_reviews.jsonl'),'record_id');ck(set(cover_reviews)==set(covers),'all10 cover review rows')
    for rid,v in cover_reviews.items():
        local=[x for x in unique.values() if x['record_id']==rid];oc=[x for x in outcomes.values() if x['record_id']==rid]
        ck(v['unique_changed_slots']==v['reviewed_unique_changed_slots']==len(local),'cover slot count '+rid)
        ck(v['successful_stego_outcomes']==len(oc) and v['changed_slot_occurrences']==sum(len(x['occurrences']) for x in local),'cover outcome occurrence totals '+rid)
        ck(v['validated_quality_pass_fail'] is None,'cover no validated metric '+rid)
    ck([x['source_order'] for x in cover_reviews.values() if not x['successful_stego_outcomes']]==[37,40],'zero-success37/40 retained')
    attempt_reviews=index(read('all_executed_attempts.jsonl'),'row_id');ck(set(attempt_reviews)=={x['row_id'] for x in attempts},'all240 executed attempts retained')
    for cell in attempts:
        v=attempt_reviews[cell['row_id']]
        ck(all(v[k]==cell[k] for k in ['record_id','source_order','arm','grid','parent_payload_length','requested_payload_bits','execution_status','full_exact','changed_positions']),'attempt projection '+cell['row_id'])
        ck(v['substitution_review_status']==('all_changed_slots_reviewed' if cell['full_exact'] is True else 'not_applicable_no_successful_stego'),'attempt review applicability '+cell['row_id'])
    summary=obj(OUT/'summary.json')
    for key,n in [('completed_covers',10),('actual_encoder_attempts',240),('insufficient_capacity',216),('full_exact_outcomes',24),('reviewed_full_exact_outcomes',24),('changed_slot_occurrences',380),('reviewed_changed_slot_occurrences',380),('unique_changed_slots',180),('reviewed_unique_changed_slots',180)]:ck(summary[key]==n,'summary '+key)
    ck(summary['observed_label_counts_unique_slots']==dict(label_unique) and summary['observed_label_counts_occurrences']==dict(label_occurrences),'summary descriptive label counts')
    ck(summary['unreviewed_slots']==[] and summary['human_gold'] is False and summary['blind_review'] is False and summary['numeric_factuality_or_security_metric'] is None,'summary honest scope')
    ck(summary['outcomes_with_at_least_one_recorded_issue_or_ambiguity']==sum(bool(x['observed_issue_or_ambiguity_slot_ids']) for x in reviews.values()),'summary issue outcome count')
    for x in summary['review_parts']:ck(digest((ROOT/x['path']).read_bytes())==x['sha256'],'part hash '+x['path'])
    ck(summary['inventory_sha256']==digest((OUT/'inventory.json').read_bytes()),'summary inventory hash')
    for path,h in summary['code_sha256'].items():ck(digest((ROOT/path).read_bytes())==h,'code hash '+path)
    return {'schema':'first40-delta-slot-review-independent-audit-v1','status':'passed','structural_checks':len(checks),'failed_checks':[],'scope':'Saved byte identities, all delta substitutions, offsets, unique keys, annotation coverage and aggregate crosswalks; semantic correctness of human/agent labels is not independently certified.','new_covers':10,'new_encoder_attempts':240,'new_full_exact_outcomes':24,'new_insufficient_capacity':216,'changed_occurrences':380,'unique_slots':180,'old_full_exact_not_reannotated':75,'no_successful_stego_cover_orders':[37,40],'generation_calls':0,'encoder_calls':0,'decoder_calls':0,'production_module_imports':0,'semantic_labels_validated':False,'human_gold':False,'source_sha256':digest(Path(__file__).read_bytes()),'artifact_sha256':{name:digest((OUT/name).read_bytes()) for name in ['all_executed_attempts.jsonl','annotated_unique_slots.jsonl','cover_reviews.jsonl','covers.jsonl','inventory.json','outcome_reviews.jsonl','outcomes.jsonl','report_ru.md','review-part-a.jsonl','review-part-b.jsonl','review-part-c.jsonl','summary.json','unique_slots.jsonl']}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-existing',action='store_true');args=parser.parse_args()
    target=OUT/'independent-audit.json'
    try:result=audit()
    except Exception as exc:
        result={'schema':'first40-delta-slot-review-independent-audit-v1','status':'failed_requires_review','error_type':type(exc).__name__,'detail':str(exc),'source_sha256':digest(Path(__file__).read_bytes()),'generation_calls':0,'encoder_calls':0,'decoder_calls':0}
        failure=OUT/'independent-audit-failure.json'
        with failure.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result));return 2
    raw=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
    if args.verify_existing:assert target.read_bytes()==raw,'existing audit differs'
    else:
        with target.open('xb') as f:f.write(raw)
    print(json.dumps({'status':result['status'],'checks':result['structural_checks'],'sha256':digest(raw),'file':str(target)}));return 0

if __name__=='__main__':raise SystemExit(main())
