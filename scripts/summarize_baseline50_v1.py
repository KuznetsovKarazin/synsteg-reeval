#!/usr/bin/env python3
"""Collect saved baseline50 evidence; no inference, tokenizer, or transport calls."""
from pathlib import Path
from collections import Counter
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/baseline50-summary-v1'
REPORT=ROOT/'docs/baseline50_continuation_ru.md'
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 if OUT.exists() or REPORT.exists():raise ValueError('Existing summary/report; refuse replacement')
 inputs={}
 def read(path):
  b=(ROOT/path).read_bytes();inputs[str(path)]=sha(b);return json.loads(b)
 source_path=Path('results/recovery-development-v1/linux_snapshot/development-prompts.jsonl')
 b=(ROOT/source_path).read_bytes();inputs[str(source_path)]=sha(b)
 src=[json.loads(line) for line in b.decode().splitlines()]
 assert len(src)==100 and all(x['split']=='development' for x in src)
 receipt=read(Path('results/large-campaign-continuation-v3/git-receipts/after-batch25.json'))
 assert receipt['readback_verified'] is True and receipt['expected_completed']==50
 for p,h in receipt['run_files_sha256'].items():
  b=(ROOT/p).read_bytes();assert sha(b)==h;inputs[p]=h
 assert len(receipt['run_files_sha256'])==56
 run=Path('runs/large-campaign-development-v2/models/tinyllama11')
 ledger=read(run/'local_ledger/ledger.json')
 rows=[]
 for i,x in enumerate(src[:50],1):
  ident=sha(json.dumps(['local','TinyLlama/TinyLlama-1.1B-Chat-v1.0',x['record_id']],sort_keys=True,ensure_ascii=True,separators=(',',':')).encode())
  a=read(run/'attempts'/f'{ident}.json')
  assert a['identity']==ident and a['status']=='completed' and ledger['attempts'][ident]['result']==a['result']
  result=a['result'];cover=result['cover_text']
  rows.append({'source_order':i,'record_id':x['record_id'],'title':x['title'],'excerpt_words':len(x['excerpt'].split()),'target_words':x['target_words'],'cover_words':len(cover.split()),'output_tokens':result['usage']['output_tokens'],'stop_reason':result['stop_reason'],'cover_sha256':sha(cover.encode())})
 assert len(ledger['attempts'])==50
 audits=[]
 for n in [23,24,25]:
  path=Path(f'results/large-campaign-continuation-v3-audit/batch{n}-postrun-audit'+('-recovered' if n==23 else '')+'.json')
  a=read(path);assert a['status']=='passed' and a['completed']==2*n and a['prior_completed_entries_preserved']==2*n-2
  audits.append({'batch':f'batch{n}','path':str(path),'completed':a['completed'],'wall_seconds':a['process_wall_seconds'],'peak_rss_kib':a['process_peak_rss_kib']})
 reviews=[];review_formats=[]
 for n in [23,24,25]:
  path=Path(f'results/large-campaign-continuation-v3-text-review/batch{n}.json')
  v=read(path);assert v['records_included']==2 and v['excluded_records']==[] and v['batch_source_orders']==[2*n-1,2*n]
  for x in v['records']:
   row=rows[x['source_order']-1]
   assert x['record_id']==row['record_id'] and x['cover_sha256']==row['cover_sha256']
   assert x['full_cover_and_excerpt_read'] is True
   review_formats.append({'source_order':x['source_order'],'format_summary':x['format_summary']})
  reviews.append({'batch':f'batch{n}','path':str(path),'summary':v['summary']})
 stops=dict(Counter(x['stop_reason'] for x in rows))
 total=sum(x['output_tokens'] for x in rows);new=rows[44:]
 assert sum(x['output_tokens'] for x in rows[:44])==29049
 summary={'schema':'baseline50-saved-evidence-summary-v1','campaign':'large-campaign-development-v2','completed_generations':50,'full_denominator':300,'not_attempted':250,'model_counts':{'tinyllama11':50,'qwen05':0,'falcon3_1b':0},'per_model_denominator':100,'cumulative_output_tokens':total,'stop_reason_counts':stops,'new_generations_since44':6,'new_output_tokens_since44':sum(x['output_tokens'] for x in new),'new_rows':new,'audits':audits,'descriptive_reviews':reviews,'descriptive_format_reviews':review_formats,'primary_commit':receipt['commit'],'new_model_calls_by_summary':0,'new_tokenizer_calls_by_summary':0,'new_transport_calls_by_summary':0,'historical_public_transport_reference_cover_count':40,'generation_success_is_not_factuality_or_steganographic_success':True}
 unsupported=sum(v['summary']['grouped_unsupported_observations'] for v in reviews)
 conflicts=sum(v['summary']['grouped_explicit_contradiction_observations'] for v in reviews)
 spans=sum(v['summary']['span_checks_passed'] for v in reviews)
 lines=['# Baseline50: сохранённые development-генерации','',
 '**Большой baseline достиг50/300: TinyLlama50/100, Qwen0.5B0/100, Falcon3-1B0/100.** Оставшиеся250 условий не запускались. Это число сохранённых генераций, не число качественных cover и не успешных стеганографических доставок.','',
 f'Всего {total} выходных токенов; end_turn={stops.get("end_turn",0)}, max_tokens={stops.get("max_tokens",0)}. Новые шесть текстов №45–50 получены ровно по одному разу в batch23–25, без повторов и отбора. Прежние44 сохранены. Замороженные модель,100development-подсказок, per-record seed, sampling и utility не менялись.','',
 '| № | Источник | Excerpt words | Цель | Cover words | Output tokens | Stop |','|---|---|---:|---:|---:|---:|---|']
 for x in new:lines.append(f'| {x["source_order"]} | {x["title"]} | {x["excerpt_words"]} | {x["target_words"]} | {x["cover_words"]} | {x["output_tokens"]} | {x["stop_reason"]} |')
 lines += ['', 'Длины — whitespace words из сохранённых строк. Цель в prompt не превращена задним числом в порог приёмки; укороченные/длинные тексты остаются в полном знаменателе.','',
 f'Все6 новых cover и исходных excerpts прочитаны в фиксированном порядке: {spans} точных span-проверок, {unsupported} сгруппированных наблюдений unsupported_by_excerpt и {conflicts} явных противоречий. Это описательная агентская разметка: не human gold, не исчерпывающая разметка атомарных утверждений, не внешняя проверка истинности и не частота ошибок. Неподтверждённое источником не объявляется ложным во внешнем мире.','',
 'Разметка отмечает завершённые окончания всех6 текстов, но №47 вместо требуемой статьи выдаёт план её написания (task_fulfilled=false). Его отсутствие новых фактических утверждений не считается качественным успехом. В №46 отмечено противоречие о ранге class, в №50 — о зимней полярной ночи. Эти конкретные ошибки сохранены, а тексты не заменялись.', '',
 'Независимый audit проверил каждую новую пару, сохранность прежних attempts/ledger entries,300строк полного знаменателя и текущий run. Batch23 первоначально остановил audit из-за удалённого при очистке scratch исторического baseline-audit.json; отказ сохранён. После точного восстановления зависимости изGit тот же auditor запущен в новой копии сохранённых входов, без повторной генерации. Batch24/25 проходят отдельно.','',
 '| Пакет | Результат | Primary files | Cumulative run files |','|---|---:|---:|---:|','| batch23 | 46/300 | 165 | 52 |','| batch24 | 48/300 | 171 | 54 |','| batch25 | 50/300 | 177 | 56 |','',
 f'Последний primary Git checkpoint: [{receipt["commit"]}](https://github.com/KuznetsovKarazin/synsteg-independent-reproduction/commit/{receipt["commit"]}). Receipt: results/large-campaign-continuation-v3/git-receipts/after-batch25.json. Архив: results/continuation-20261025-batch25/artifacts/full.zip; manifest и fresh-restore.json рядом. Все56 текущих run-файлов и архив прочитаны изGit координатором;177primaryfiles восстановлены и сверены поSHA256.','',
 'Public transport first40 остаётся отдельным прежним результатом:960outcomes,99exact,861insufficient; основные128/256/512bits —0/480. Baseline45–50 этим документом не кодируется. Новые transport-исходы для41–50 относятся к отдельному продолжению с переиспользованием прежних960исходов; они не входят в эту сводку. Авторский original/fixed candidate-model этап здесь не выполнен.','',
 'Claude ранее дал3 smoke-ответа; новых платных вызовов и повторения smoke нет. Исторический план4800+300Claude не объявляется выполненным. Closed test не открывался.','',
 'Следующий baseline — batch26, №51–52, expected-completed50. Перед запуском: новое immutable intent, фактический Git readback, receipt after-batch25 и исключительная NN-блокировка. Использовать неизменный scripts/resume_large_campaign_development_v3.py, utility freeze a76efecfa9293c4d83d6bcd04cdfb11eaf610630. Не повторять завершённые batch23–25.','',
 'Сводка воспроизводится scripts/summarize_baseline50_v1.py из сохранённых50attempts, источников, audit/review и receipt; NN, tokenizer и transport не запускаются. Машинные summary.json и inputs.json находятся в results/baseline50-summary-v1/. Текущее продолжение проекта см. [continuation_current_ru.md](continuation_current_ru.md).','']
 inputs[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__).read_bytes())
 OUT.mkdir()
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 (OUT/'inputs.json').write_text(json.dumps({'schema':'baseline50-summary-inputs-v1','files_sha256':inputs},indent=2)+'\n')
 REPORT.write_text('\n'.join(lines))
 print(json.dumps({'completed':50,'tokens':total,'stop_reason_counts':stops,'new_rows':new,'review_span_checks':spans,'unsupported_groups':unsupported,'conflict_groups':conflicts},ensure_ascii=False))
if __name__=='__main__':main()
