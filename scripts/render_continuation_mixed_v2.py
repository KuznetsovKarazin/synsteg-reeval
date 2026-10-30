#!/usr/bin/env python3
"""Render current status from an actual mixed300 snapshot, outside execution root.

No Git/model calls. Root supplies the actual remote readback and complete census;
this renderer verifies local bytes and avoids replacing historical subtrees.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

ID='continuation-20261028-mixed-v2'
KEY='current_checkpoint_20261028_mixed_v2'
SOURCE='4050291345c43776e9c93528b6afe292c062ba7a'
MODELS=('tinyllama11','qwen05','falcon3_1b')
STATES=('completed','reserved_outcome_unknown','failed','not_attempted')
BASELINE='results/large-campaign-continuation-v3/status.json'
NATIVE='results/native-continuation-20261025/status.json'
CURRENT='docs/continuation_current_ru.md'
CHECKPOINT=f'results/{ID}/status.json'
REPORT='docs/continuation_20261028_mixed_v2_ru.md'
HISTORY_SHA={
    BASELINE:'b9a0d80118fc735498cbac3a932fac7aab93fb311818e490849482e16a85cf8d',
    NATIVE:'7e065b56609752338126608d889a2ca891507353663b6303265f76582863cf99',
    CURRENT:'0bc82dbe978812ab8f22fe61f01d7dbece121c81365e22649cb45cecb45af76d'}


def require(ok,message):
    if not ok: raise ValueError(message)


def raw(path):
    path=Path(path)
    require(not any(x.is_symlink() for x in (path,*path.parents)),'Symlink evidence refused')
    require(path.is_file(),'Missing regular file: '+str(path))
    return path.read_bytes()


def sha(path): return hashlib.sha256(raw(path)).hexdigest()


def load(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON key')
            result[key]=value
        return result
    def bad(value): raise ValueError('Nonfinite JSON')
    return json.loads(raw(path),object_pairs_hook=pairs,parse_constant=bad)


def relative(value):
    path=Path(value)
    require(type(value) is str and not path.is_absolute() and '..' not in path.parts,'Unsafe repository path')
    return path


def evidence(root,spec):
    path=root/relative(spec['path'])
    require(sha(path)==spec['sha256'],'Evidence hash differs: '+spec['path'])
    return path,load(path)


def summarize(mixed):
    require(mixed.get('schema')=='mixed-development-full300-merger-v2-result','Wrong mixed300 schema')
    rows=mixed['records']
    require(len(rows)==300 and mixed['full_denominator']==300,'Full300 denominator')
    require(all(type(r['source_order']) is int and r['status'] in STATES for r in rows),'Invalid outcome/order')
    keyed={(r['model_alias'],r['source_order']):r for r in rows}
    require(set(keyed)=={(m,n) for m in MODELS for n in range(1,101)},'Full300 unique grid')
    counts=dict(Counter(r['status'] for r in rows))
    models={m:dict(Counter(r['status'] for r in rows if r['model_alias']==m)) for m in MODELS}
    require(counts==mixed['counts'] and models==mixed['model_counts'],'Published counts differ from records')
    for n in range(1,101):
        require(len({keyed[(m,n)]['record_id'] for m in MODELS})==1,'Unpaired original source identity')
    for n in range(1,91):
        require(keyed[('tinyllama11',n)]['status']==('reserved_outcome_unknown' if n==82 else 'completed'),
                'Historical TinyLlama outcomes changed')
    require(keyed[('tinyllama11',91)]['status']=='not_attempted','Blocked91 historical outcome changed')
    require(any(x.get('model_alias')=='tinyllama11' and x.get('source_order')==91
                and x.get('scheduling_status')=='blocked_reserved_intent_evidence_gap'
                for x in mixed['scheduling_overlays']),'Separate91 scheduling block missing')
    changed={(m,n) for m,n in keyed if (m!='tinyllama11' or n>=92) and keyed[(m,n)]['status']!='not_attempted'}
    items=mixed['new_shard_evidence']
    require(len(items)==len(changed) and {(x['model_alias'],x['source_order']) for x in items}==changed,
            'New outcome lacks exact shard evidence')
    require(all(x['status']==keyed[(x['model_alias'],x['source_order'])]['status'] for x in items),
            'Shard evidence outcome differs from counted record')
    return keyed,{s:counts.get(s,0) for s in STATES},{m:{s:models[m].get(s,0) for s in STATES} for m in MODELS}


def inspect_new(root,mixed,audits):
    audit_specs={(x['model_alias'],x['source_order']):x for x in audits}
    require(len(audit_specs)==len(audits)==len(mixed['new_shard_evidence']),'Audit census differs')
    tokens=Counter();stops=Counter();checked=[];required_pins={}
    for row in mixed['new_shard_evidence']:
        key=(row['model_alias'],row['source_order']);prefix=relative(row['shard_path']);directory=root/prefix
        receipt_path=directory/'git-preservation-receipt.json';receipt=load(receipt_path)
        require(sha(receipt_path)==row['receipt_sha256'] and receipt['readback_verified'] is True
                and receipt['commit']==row['primary_commit'] and re.fullmatch('[0-9a-f]{40}',receipt['commit']),
                'New shard receipt differs')
        all_paths=list(directory.rglob('*'))
        require(not any(p.is_symlink() for p in (directory,*directory.parents,*all_paths)),'Symlink shard evidence refused')
        actual={str(p.relative_to(directory)):sha(p) for p in sorted(all_paths) if p.is_file() and p.name!='git-preservation-receipt.json'}
        require(actual==receipt['files_sha256'],'New raw inventory differs from preserved receipt')
        audit_path,audit=evidence(root,audit_specs[key])
        require(audit['source_order']==key[1] and audit.get('model_alias',key[0])==key[0]
                and audit.get('status') in ('passed','failed') and audit.get('git_receipt_required') is True,
                'Audit does not bind the actual shard')
        bound=audit.get('input_sha256',{})
        require(any(name.endswith(str(prefix/'reservation.json')) and digest==actual['reservation.json']
                    for name,digest in bound.items()),'Audit reservation digest differs')
        if 'process.json' in actual:
            require(any(name.endswith(str(prefix/'process.json')) and digest==actual['process.json']
                        for name,digest in bound.items()),'Audit process digest differs')
        required_pins[str(receipt_path.relative_to(root))]=sha(receipt_path)
        required_pins[str(audit_path.relative_to(root))]=sha(audit_path)
        elapsed=None;output_tokens=None
        if 'process.json' in actual:
            process=load(directory/'process.json');elapsed=process.get('elapsed_seconds')
            require(elapsed is None or type(elapsed) in (int,float) and elapsed>=0,'Invalid saved process duration')
        if row['status']=='completed':
            require(audit['status']=='passed' and not audit.get('errors'),'Completed outcome requires passed integrity audit')
            attempts=list((directory/'run/attempts').glob('*.json'))
            require(len(attempts)==1 and sha(attempts[0])==row['attempt_sha256'],'Raw completed attempt differs')
            require(any(name.endswith(str(attempts[0].relative_to(root))) and digest==row['attempt_sha256']
                        for name,digest in bound.items()),'Audit completed attempt digest differs')
            attempt=load(attempts[0]);result=attempt['result']
            require(attempt['status']=='completed' and hashlib.sha256(result['cover_text'].encode()).hexdigest()==row['cover_sha256'],
                    'Completed cover differs')
            n=result['usage']['output_tokens'];require(type(n) is int and 1<=n<=1024,'Token count out of bounds')
            output_tokens=n;tokens[key[0]]+=n;stops[result['stop_reason']]+=1
        checked.append({'model_alias':key[0],'source_order':key[1],'status':row['status'],
                        'primary_commit':receipt['commit'],'receipt':str(receipt_path.relative_to(root)),
                        'audit':str(audit_path.relative_to(root)),'audit_status':audit['status'],
                        'elapsed_process_seconds_including_load':elapsed,'output_tokens':output_tokens})
    return tokens,stops,checked,required_pins


def update_status(original,checkpoint,native=False):
    output=copy.deepcopy(original)
    counts=checkpoint['baseline']['counts'];models=checkpoint['baseline']['models']
    output.update(phase='mixed_development_snapshot_preserved_raw_outputs_not_quality',
                  current_model_counts=copy.deepcopy(models),
                  current_checkpoint_evidence_ref=CHECKPOINT,latest_stage_report=REPORT,
                  next_launch_ready=False,next_launch_gate='Fresh Git census and explicit per-shard activation required; never replay Tiny82/91 or lost ending-v2.',
                  updated_at_unix=checkpoint['recorded_at_unix'])
    output[KEY]=copy.deepcopy(checkpoint)
    output['current_historical_pointer_scope']={
        'rule':'All pre-existing nested dict/list objects except the current model_counts alias are preserved historical snapshots; this new checkpoint and current_model_counts are authoritative.',
        'latest80_fields':['last_completed_batch','latest_primary_archive','latest_primary_manifest','latest_verified_receipt','latest_primary_git_commit','last_remote_verified_completed','latest_audit'],
        'latest80_scope':'These legacy pointers still describe first80 archival preservation; they do not describe the current mixed300 snapshot.',
        'historical_model_count_fields':['model_unknown_counts','current_checkpoint_20261028'],
        'public_transport_scope':'The frozen public transport benchmark still covers first80 TinyLlama covers; new raw generations do not update delivery or quality metrics.'}
    if native:
        output.update(baseline_completed=counts['completed'],baseline_outcome_unknown=counts['reserved_outcome_unknown'],
                      baseline_not_attempted=counts['not_attempted'],baseline_output_tokens=checkpoint['baseline']['known_output_tokens'],
                      active_model_process=False,active_baseline_batch=None,closed_test_accessed=False,paid_calls=0)
    else:
        output['model_counts']={model:models[model]['completed'] for model in MODELS}
        output.update(completed=counts['completed'],completed_generations=counts['completed'],outcome_unknown=counts['reserved_outcome_unknown'],
                      not_attempted=counts['not_attempted'],tinyllama_completed=models['tinyllama11']['completed'],
                      tinyllama_not_attempted=models['tinyllama11']['not_attempted'],tinyllama_outcome_unknown=models['tinyllama11']['reserved_outcome_unknown'],
                      cumulative_generated_output_tokens=checkpoint['baseline']['known_output_tokens'],active_generation=False,
                      model_process_active=False,active_batch=None,active_intent_path=None,closed_test_opened=False,paid_calls=0)
    for key,value in original.items():
        if isinstance(value,(dict,list)) and key!='model_counts':
            require(output[key]==value,'Historical nested object modified: '+key)
    return output


def narrative(checkpoint,short=False):
    b=checkpoint['baseline'];c=b['counts'];models=b['models']
    lines=['# SynSteg: текущий результат сравнительной серии — 28 октября 2026','',
           f"Сохранено **{c['completed']} completed / {c['reserved_outcome_unknown']} unknown / {c['failed']} failed / {c['not_attempted']} not_attempted из300 development-заданий**. Счётчики получены из фактического mixed300 и сохранённых первичных файлов.",'',
           '| Модель | Completed | Unknown | Failed | Not attempted | Всего |','|---|---:|---:|---:|---:|---:|']
    for key,title in [('tinyllama11','TinyLlama1.1B'),('qwen05','Qwen2.5-0.5B'),('falcon3_1b','Falcon3-1B')]:
        m=models[key];lines.append(f"| {title} | {m['completed']} | {m['reserved_outcome_unknown']} | {m['failed']} | {m['not_attempted']} |100 |")
    q=checkpoint['qwen_source1']
    lines+=['',f"TinyLlama82 остаётся неизвестным;91 остаётся историческим not_attempted с отдельным запретом запуска. Первый Qwen source1: **{q['status']}**. Это сырые результаты генерации и проверка целостности; завершённая генерация не означает factuality, качество словаря или передачу сообщения.",
            '',f"Проверено известных output tokens: **{b['known_output_tokens']}**; stopping reasons: {json.dumps(b['stop_reasons'],ensure_ascii=False,sort_keys=True)}. Историческая база89 унаследована из сохранённого checkpoint; новый merger не пересчитывал все старые raw заново.",
            '',f"Текущие источники: [mixed300](../{b['mixed300_path']}), [checkpoint](../{CHECKPOINT}), [отчёт](continuation_20261028_mixed_v2_ru.md). Старые latest-primary/receipt/audit указатели в машинных статусах сохраняют область first80; текущие счётчики находятся в `{KEY}` и `current_model_counts`.",
            '',"Ending-v2: первичные native-результаты не сохранены; подтверждённая доставка, фактические endpoints и качество остаются **null**. Старый ID и восемь key/nonce пар запрещены к повтору. Отдельный прежний public-context4 сохраняет4/4 передачи по16B, но все8/8 текстов имеют отмеченные неподтверждённые сведения; доказательства безопасности и практической пригодности нет.",
            '',"Public transport по first80 не расширен:1920 условий,198 exact и1722 capacity refusals; primary128/256/512 —0/960. Диагностика словаря83–90 остаётся отдельным результатом: V1 294bits, V2 8bits, потеря97,2789%, V2 нулевая ёмкость3/8. Новые covers не превращают эти старые измерения в оценку полного набора.",
            '',"Claude:3 исторических smoke responses; полного сравнительного запуска нет. Исходный план4800 local +300 Claude не выполнен. Исторические серии Qwen60+2 исключены из текущих300. Новых платных вызовов —0; закрытый test не открыт.",'',
            'Генерация словарей native-v4 пока дала1/2 JSON accepted и40 строк-кандидатов; пригодный словарь и полезная ёмкость не установлены. Успех формата не заменяет проверку значений и source fidelity.','',
            'Дальше: продолжить оставшиеся Qwen и Falcon на тех же100 источниках с отдельными runtime/context gates; оценивать фактическую опору covers и потери ёмкости на полном знаменателе. Для метода исследовать source-budget-aware генерацию и receiver-invariant допустимые варианты. Seed/key может выбирать варианты, но не создаёт смысловую эквивалентность и ёмкость сам по себе.']
    future=checkpoint['future_qwen_sequential']
    timings=[x for x in checkpoint['new_shards'] if (x['model_alias'],x['source_order']) in
             [('tinyllama11',92),('tinyllama11',93),('qwen05',1)] and x.get('elapsed_process_seconds_including_load') is not None]
    if timings:
        measured='; '.join(f"{x['model_alias']} #{x['source_order']}: {x['elapsed_process_seconds_including_load']:.1f}с / {x['output_tokens']} output tokens" for x in timings)
        lines+=['',f"Сохранённые длительности процессов, включая загрузку: {measured}. Научные настройки сохранены: CPU, float32,6 threads, предел1024 output tokens. Эти измерения показывают фактическую стоимость отдельных заданий; срок полного большого прогона по ним не обещается."]
    if future:
        lines+=['',f"Отдельное продолжение Qwen2–100 подготовлено как **выключенная будущая работа**: [протокол](../{future['protocol']['path']}); выполнения этих заданий этот протокол не подтверждает."]
    if not short:
        lines+=['',f"Новые проверенные shard: {len(checkpoint['new_shards'])}; их raw receipts и независимые integrity audits перечислены в checkpoint. Срез Git: `{checkpoint['snapshot_commit']}`. Унаследованные вложенные статусы и все прежние научные результаты сохранены без изменения."]
    return '\n'.join(lines)+'\n'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    with path.open('x',encoding='utf-8') as stream: stream.write(payload)


def render(history,root,manifest_path,out):
    require(not out.resolve().is_relative_to(root.resolve()),'Render outside frozen execution checkout')
    for name,expected in HISTORY_SHA.items(): require(sha(history/name)==expected,'Historical Git source changed: '+name)
    old=load(history/BASELINE);native=load(history/NATIVE);old_checkpoint=old['current_checkpoint_20261028']
    manifest=load(manifest_path)
    require(manifest['schema']==ID+'-render-input' and manifest['entry_main']==SOURCE,'Render source identity')
    require(type(manifest['recorded_at_unix']) in (int,float) and manifest['recorded_at_unix']>0,'Actual snapshot timestamp required')
    require(manifest['paid_calls_this_stage']==0 and manifest['closed_test_opened'] is False
            and manifest['active_project_nn_processes']==0,'Require completed free development snapshot')
    mixed_path,mixed=evidence(root,manifest['mixed300']);keyed,counts,models=summarize(mixed)
    tokens,stops,new_shards,pins=inspect_new(root,mixed,manifest['audits'])
    pins[str(mixed_path.relative_to(root))]=sha(mixed_path)
    readback_path,readback=evidence(root,manifest['actual_git_readback'])
    require(readback['readback_verified'] is True and re.fullmatch('[0-9a-f]{40}',readback['commit'])
            and readback['remote_census_complete'] is True,'Actual snapshot Git readback/census required')
    future=manifest.get('future_qwen_sequential')
    if future:
        for field in ['runner','protocol','review']:
            require(sha(root/relative(future[field]['path']))==future[field]['sha256'],'Future source bytes differ')
            pins[future[field]['path']]=future[field]['sha256']
        future_protocol=load(root/future['protocol']['path']);future_review=load(root/future['review']['path'])
        require(future_protocol['execution_enabled'] is False,'Future Qwen protocol must remain disabled')
        require(future_review['status']=='passed_disabled_source_and_protocol',
                'Future Qwen source review must pass before describing prepared work')
        require(future_review['wrapper_sha256']==future['runner']['sha256']
                and future_review['proposed_protocol_sha256']==future['protocol']['sha256']
                and future_protocol['helper_sha256']==future['runner']['sha256'],
                'Future Qwen reviewed source/protocol binding differs')
    require(all(readback['files_sha256'].get(name)==digest for name,digest in pins.items()),'Snapshot readback missing actual receipt/audit/mixed/future-source bytes')
    old_baseline=old_checkpoint['baseline'];tokens['tinyllama11']+=old_baseline['known_output_tokens'];stops.update(old_baseline['stop_reasons'])
    checkpoint={'schema':ID+'-status','date':'2026-10-28','entry_main':SOURCE,'recorded_at_unix':manifest['recorded_at_unix'],
                'snapshot_commit':readback['commit'],'stage':'mixed_development_snapshot_preserved_raw_outputs_not_quality',
                'baseline':{'denominator':300,'counts':counts,'models':models,'known_output_tokens':sum(tokens.values()),
                            'known_output_tokens_by_model':{m:tokens[m] for m in MODELS},'stop_reasons':dict(stops),
                            'mixed300_path':str(mixed_path.relative_to(root)),'mixed300_sha256':sha(mixed_path),
                            'unknown_tiny_source_orders':[82],'blocked_intents_separate_from_counts':[91],
                            'raw_generation_is_not_quality_or_transport':True,'inherited_base_raw_reaudited':False},
                'qwen_source1':{'status':keyed[('qwen05',1)]['status'],'included_only_if_actual_saved_shard':True},
                'new_shards':new_shards,'ending_v2':copy.deepcopy(old_checkpoint['ending_v2']),
                'previous_verified_context4':copy.deepcopy(old_checkpoint['previous_verified_context4']),
                'claude':copy.deepcopy(old_checkpoint['claude']),'large_plan':copy.deepcopy(old_checkpoint['large_plan']),
                'advanced_dictionary':copy.deepcopy(old_checkpoint['advanced_dictionary']),
                'capacity_diagnostic':copy.deepcopy(old_checkpoint['capacity_diagnostic']),
                'future_qwen_sequential':future,'paid_calls_this_stage':0,'closed_test_opened':False,
                'active_generation':False,'model_calls_by_renderer':0,'tokenizer_calls_by_renderer':0,
                'actual_snapshot_readback':manifest['actual_git_readback'],
                'historical_scope':'All old nested objects retained; new snapshot supersedes current counters only.'}
    require(checkpoint['ending_v2']['authenticated_exact_deliveries'] is None and checkpoint['ending_v2']['replay_permitted'] is False,
            'Lost ending-v2 cannot become a result')
    baseline=update_status(old,checkpoint);native_status=update_status(native,checkpoint,True)
    preserved={BASELINE:[k for k,v in old.items() if isinstance(v,(dict,list)) and k!='model_counts'],NATIVE:[k for k,v in native.items() if isinstance(v,(dict,list)) and k!='model_counts']}
    write(out/BASELINE,baseline);write(out/NATIVE,native_status);write(out/CHECKPOINT,checkpoint)
    write(out/CURRENT,narrative(checkpoint,True));write(out/REPORT,narrative(checkpoint,False))
    validation={'schema':ID+'-render-validation','status':'passed','historical_subtrees_preserved':preserved,
                'historical_subtree_count':sum(map(len,preserved.values())),'actual_counts':counts,
                'input_manifest_sha256':sha(manifest_path),'model_calls':0,'tokenizer_calls':0}
    validation_path=f'results/{ID}/render-validation.json';write(out/validation_path,validation)
    saved_input=f'results/{ID}/render-input.json'
    write(out/saved_input,raw(manifest_path).decode('utf-8'))
    saved_readback=manifest['actual_git_readback']['path']
    write(out/saved_readback,raw(readback_path).decode('utf-8'))
    names=[BASELINE,NATIVE,CHECKPOINT,CURRENT,REPORT,validation_path,saved_input,saved_readback]
    plan={'files':[{'path':name,'local_path':str(out/name),'sha256':sha(out/name),'bytes':len(raw(out/name))} for name in names]}
    write(out/'save-plan.json',plan)
    return {'counts':counts,'models':models,'preserved_subtrees':validation['historical_subtree_count'],'save_plan':str(out/'save-plan.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(render(args.history_root,args.root,args.manifest,args.output),ensure_ascii=False))


if __name__=='__main__': main()
