#!/usr/bin/env python3
"""Summarize all ten saved descriptive reviews; no model or codec calls.

This validates saved identities, exact quote spans and denominator completeness.
It does not independently validate the reviewers' semantic judgments. The output
is additive and refuses replacement; missing or incomplete batches stop the build.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / 'results/large-campaign-continuation-v3-text-review'
SOURCE = ROOT / 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
SOURCE_SHA = '3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def check_span(text, span):
    start, end = span['start'], span['end']
    assert type(start) is int and type(end) is int and 0 <= start < end <= len(text)
    assert text[start:end] == span['quote']
    assert text.count(span['quote']) == span['occurrences']


def build():
    out = REVIEWS / 'baseline51-60-summary-v1.json'
    report = ROOT / 'docs/baseline51_60_source_fidelity_ru.md'
    assert not out.exists() and not report.exists(), 'Refuse overwrite'
    source_raw = SOURCE.read_bytes()
    assert sha(source_raw) == SOURCE_SHA
    sources = [json.loads(line) for line in source_raw.decode().splitlines()]
    inputs = {}; records = []; conflicts = []; checks = 0; grouped = Counter(); formats = Counter()
    for number in range(26, 31):
        path = REVIEWS / f'batch{number}.json'
        raw = path.read_bytes(); review = json.loads(raw)
        inputs[path.relative_to(ROOT).as_posix()] = sha(raw)
        assert review['batch_id'] == f'batch{number}'
        orders = [2*number-1, 2*number]
        assert review['batch_source_orders'] == orders
        assert [r['source_order'] for r in review['records']] == orders
        assert review['excluded_records'] == [] and review['summary']['excluded_records'] == 0
        annotation = ROOT / review['method']['annotation_input']
        assert sha(annotation.read_bytes()) == review['method']['annotation_input_sha256']
        inputs[annotation.relative_to(ROOT).as_posix()] = sha(annotation.read_bytes())
        checkpoint = ROOT / review['method']['batch_checkpoint_path']
        assert sha(checkpoint.read_bytes()) == review['method']['batch_checkpoint_sha256']
        cp = json.loads(checkpoint.read_bytes())
        assert cp['status'] == 'verified' and cp['completed_generations'] == 2*number
        for r in review['records']:
            source = sources[r['source_order']-1]
            assert r['record_id'] == source['record_id'] and r['generation_status'] == 'completed'
            assert r['full_cover_and_excerpt_read'] is True
            attempt_raw = (ROOT / r['attempt_snapshot_path']).read_bytes()
            assert sha(attempt_raw) == r['attempt_file_sha256'] == cp['run_files_sha256'][r['attempt_path']]
            attempt = json.loads(attempt_raw); cover = attempt['result']['cover_text']
            assert sha(cover.encode()) == r['cover_sha256']
            assert sha(source['excerpt'].encode()) == r['source_excerpt_sha256']
            assert sha(source['prompt'].encode()) == r['prompt_sha256']
            assert r['source_excerpt'] == source['excerpt']
            assert len(cover.split()) == r['cover_whitespace_words']
            assert len(source['excerpt'].split()) == r['source_excerpt_words']
            labels = Counter()
            for field in ['observations', 'supported_observations', 'format_observations']:
                for ob in r[field]:
                    check_span(cover, ob['cover_span']); checks += 1
                    if 'source_span' in ob:
                        check_span(source['excerpt'], ob['source_span']); checks += 1
                    if 'prompt_span' in ob:
                        check_span(source['prompt'], ob['prompt_span']); checks += 1
                    if field == 'observations': labels[ob['label']] += 1
                    if field == 'format_observations': formats[ob['kind']] += 1
            grouped.update(labels)
            for ob in r['observations']:
                if ob['label'] == 'contradicts_excerpt':
                    conflicts.append({'source_order':r['source_order'], 'observation_id':ob['id'],
                        'cover_quote':ob['cover_span']['quote'], 'source_quote':ob['source_span']['quote'],
                        'reason_ru':ob['reason_ru'], 'review_path':path.relative_to(ROOT).as_posix()})
            records.append({
                'source_order':r['source_order'], 'record_id':r['record_id'], 'title':r['title'],
                'target_words':r['target_words'], 'source_excerpt_words':r['source_excerpt_words'],
                'cover_whitespace_words':r['cover_whitespace_words'],
                'output_tokens':r['format_summary']['output_tokens'],
                'stop_reason':r['format_summary']['stop_reason'],
                'output_limit_hit':r['format_summary']['output_limit_hit'],
                'ending_incomplete':r['format_summary']['ending_incomplete'],
                'headings_or_bullets_observed':r['format_summary']['headings_or_bullets_observed'],
                'task_failure_metaplan':r['format_summary'].get('task_failure_metaplan_instead_of_requested_feature',False),
                'grouped_observations':dict(labels),
                'format_note_ru':r['format_summary']['note_ru'],
                'selected_supported_observations':len(r['supported_observations']),
                'format_observations':len(r['format_observations']),
                'review_path':path.relative_to(ROOT).as_posix(),
                'review_record_index':r['source_order']-orders[0],
                'cover_sha256':r['cover_sha256'],
                'attempt_snapshot_path':r['attempt_snapshot_path']})
    assert [r['source_order'] for r in records] == list(range(51,61))
    assert len({r['record_id'] for r in records}) == 10
    result={
        'schema':'baseline51-60-descriptive-source-fidelity-summary-v1',
        'status':'complete_descriptive_review_with_exact_span_validation',
        'scope':{'planned_new_covers':10,'reviewed_covers':10,'excluded_covers':0,
                 'source_orders':list(range(51,61)), 'model_alias':'tinyllama11',
                 'source_development_denominator':100,'large_generation_plan_denominator':300,
                 'prior50_reviews_repeated':False,'closed_test_opened':False},
        'validation':{'exact_spans_checked':checks,'attempts_hash_checked':10,
                      'review_input_files_sha256':inputs,'source_sha256':SOURCE_SHA},
        'counts':{'output_tokens':sum(r['output_tokens'] for r in records),
                  'cover_whitespace_words':sum(r['cover_whitespace_words'] for r in records),
                  'stop_reasons':dict(Counter(r['stop_reason'] for r in records)),
                  'incomplete_endings':sum(r['ending_incomplete'] for r in records),
                  'metaplan_task_failures':sum(r['task_failure_metaplan'] for r in records),
                  'grouped_descriptive_labels':dict(grouped),'format_kinds':dict(formats)},
        'records':records,
        'all_explicit_source_conflicts':conflicts,
        'limitations':{'human_gold':False,'blinded_review':False,'external_world_factcheck':False,
                       'semantic_error_rate':None,'factuality_success_rate':None,
                       'grouped_counts_not_atomic_claims_or_error_rates':True,
                       'unsupported_does_not_mean_false_in_the_world':True,
                       'validation_confirms_saved_quotes_not_semantic_labels':True,
                       'all_model_tokenizer_codec_calls_during_review':0}}
    lines=['# Source-fidelity обзор новых baseline covers 51–60','',
      'Прочитаны все **10 новых covers** и их точные development excerpts/prompts. Ни один результат не исключён; прежние 50 reviews повторно не оценивались. Это описательный AI-assisted обзор, не независимая человеческая разметка, не внешний фактчек и не частота ошибок модели.','',
      '| № | Source title | Source words | Target | Cover words | Tokens | Stop |',
      '|---|---|---:|---:|---:|---:|---|']
    for r in records:
        lines.append(f"| {r['source_order']} | {r['title']} | {r['source_excerpt_words']} | {r['target_words']} | {r['cover_whitespace_words']} | {r['output_tokens']} | {r['stop_reason']} |")
    lines += ['',f"Проверено {checks} точных cover/source/prompt spans и SHA всех 10 попыток в immutable after snapshots. Все замечания, выбранные поддержанные факты, source excerpts и ссылки на полные covers сохранены в [машинной сводке](../results/large-campaign-continuation-v3-text-review/baseline51-60-summary-v1.json) и пяти batch reviews ниже. Слова подсчитаны по пробельным разделителям; target — цель запроса, а не введённый задним числом порог принятия.",'',
      'Неподтверждённое excerpt утверждение не объявляется автоматически ложным в мире. Explicit source conflicts отделены от unsupported additions и неоднозначностей. Группы замечаний могут охватывать несколько утверждений и имеют разную детализацию; суммирование служит учёту, а не статистической оценке точности. Выбранные source-supported spans также не являются полным доказательством качества текста.','']
    lines += [f"Группы замечаний: {dict(grouped)}. Обрывы ответа: {sum(r['ending_incomplete'] for r in records)}/10; остановки по лимиту: {sum(r['output_limit_hit'] for r in records)}/10; ответы, обозначенные как метаплан вместо выполнения задания: {sum(r['task_failure_metaplan'] for r in records)}/10. Это описательные результаты проверки этого фиксированного блока, без экстраполяции на модель или корпус.", '']
    if conflicts:
        lines += ['Все замечания категории `contradicts_excerpt` перечислены ниже; точные source/cover spans и их позиции доступны по ID. Эта категория относится к несоответствию данному excerpt, а не к внешней проверке факта.', '']
        for c in conflicts:
            lines += [f"- **{c['source_order']} / {c['observation_id']}**: «{c['cover_quote']}» — {c['reason_ru']}"]
        lines.append('')
    lines += ['Краткие итоги чтения формата по каждому cover (полные замечания о неподтверждённых деталях находятся в batch reviews):', '']
    for r in records:
        lines.append(f"- **{r['source_order']}**: {r['format_note_ru']}")
    lines.append('')
    for number in range(26,31): lines.append(f'- [Batch{number}: covers{2*number-1}–{2*number}](../results/large-campaign-continuation-v3-text-review/batch{number}.json).')
    lines += ['', 'Точный roundtrip stego, доступная ёмкость, factuality и качество выполнения задания — отдельные исходы. Этот обзор не выполнял encode/decode, не менял выборку, не исправлял cover и не запускал модель. Привязка всех 10 записей сохраняется, включая обрывы по лимиту и ответы в форме метаплана.', '',
      'Воспроизведение сводки: `python scripts/summarize_baseline51_60_text_reviews_v1.py` в новом восстановленном checkout без уже существующих outputs сводки. Скрипт проверяет текущие сохранённые hashes/spans и полноту десяти записей; он не повторяет семантическую оценку.', '']
    with out.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    with report.open('x') as stream:stream.write('\n'.join(lines))
    print(json.dumps({'summary':out.relative_to(ROOT).as_posix(),'summary_sha256':sha(out.read_bytes()),'report':report.relative_to(ROOT).as_posix(),'report_sha256':sha(report.read_bytes()),'covers':10,'exact_span_checks':checks,'model_calls':0},ensure_ascii=False))

if __name__=='__main__':build()
