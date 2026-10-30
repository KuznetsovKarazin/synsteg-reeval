#!/usr/bin/env python3
"""Descriptive enumeration of every byte of the fixed six terminal responses.

Not an independent native/sampling audit. No repair, generation, tokenizer or
semantic scoring. Read-only primary inputs; exclusive new report directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'results/author-candidate-local-native-v2/native-diagnostic-v2'
ARTIFACTS = 'results/author-candidate-local-native-v2-artifacts/native-diagnostic-v2'


def sha(raw): return hashlib.sha256(raw).hexdigest()

def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('regular report input required')
    return path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve(); primary = root / BASE
    campaign_raw = read(primary / 'result.json'); campaign = json.loads(campaign_raw)
    index = json.loads(read(root / ARTIFACTS / 'index.json'))
    if campaign['attempts_executed'] != 6 or campaign['requested_chunks'] != 2:
        raise ValueError('this descriptive report is bound to observed six-attempt diagnostic')
    rows = []
    for chunk in range(2):
        for attempt in range(3):
            name = f'chunk{chunk}-attempt{attempt}'; folder = primary / name
            raw_path = folder / 'sealed/response.raw'; raw = read(raw_path)
            observed = read(folder / 'observed-response.raw'); result_raw = read(folder / 'attempt-result.json')
            result = json.loads(result_raw); expected = index['files'][raw_path.relative_to(root).as_posix()]
            if (raw != observed or sha(raw) != result['raw_sha256'] or len(raw) != result['raw_bytes']
                    or expected != {'bytes': len(raw), 'sha256': sha(raw)}):
                raise ValueError('raw evidence copies or packaged identity differ')
            if raw not in (b'[\n', b'```'):
                raise ValueError('new observed content requires new descriptive assessment')
            rows.append({'attempt': name, 'chunk_order': chunk, 'attempt_order': attempt,
                'raw_path': raw_path.relative_to(root).as_posix(), 'raw_hex': raw.hex(),
                'raw_utf8': raw.decode('utf-8'), 'raw_bytes': len(raw), 'raw_sha256': sha(raw),
                'attempt_result_sha256': sha(result_raw), 'committed_output_tokens': result['committed_output_tokens'],
                'prompt_tokens_recorded': result['prompt_token_count'], 'generated_ids_recorded': result['generated_ids'],
                'recorded_outcome': result['outcome'], 'recorded_stop_reason': result['stop_reason'],
                'recorded_prefix': result['prefix'], 'elapsed_seconds_recorded': result['elapsed_seconds'],
                'observed_candidate_strings': 0, 'observed_complete_word_rows': 0,
                'descriptive_content': 'opening array bracket plus newline' if raw == b'[\n' else 'three backticks',
                'semantic_assessment': 'not assessable: no candidate content emitted', 'capacity_bits': None})
    summary = {'schema': 'author-candidate-local-native-v2-all-raw-descriptive-review',
        'scope': 'All six actual terminal raw responses; descriptive read only, not independent native audit.',
        'source_campaign_sha256': sha(campaign_raw), 'source_archive_sha256': index['archive']['sha256'],
        'campaign_status_recorded': campaign['status'], 'attempts_reviewed': len(rows), 'attempts_executed_recorded': 6,
        'fixed_chunks': 2, 'full_cover_chunks': 109, 'outside_diagnostic_chunks': 107,
        'format_accepted_chunks_recorded': campaign['format_accepted_chunks_observed'],
        'first_token_invalid_prefix_attempts': sum(row['recorded_stop_reason'] == 'invalid_prefix' and row['committed_output_tokens'] == 1 for row in rows),
        'array_opening_attempts': sum(row['raw_hex'] == '5b0a' for row in rows),
        'backtick_attempts': sum(row['raw_hex'] == '606060' for row in rows),
        'total_committed_output_tokens': sum(row['committed_output_tokens'] for row in rows),
        'total_raw_response_bytes': sum(row['raw_bytes'] for row in rows),
        'total_prompt_tokens_recorded': sum(row['prompt_tokens_recorded'] for row in rows),
        'sum_attempt_elapsed_seconds_recorded': sum(row['elapsed_seconds_recorded'] for row in rows),
        'candidate_strings_observed': 0, 'usable_candidate_rows': None, 'capacity_bits': None,
        'semantic_equivalence_verified': False, 'response_repair': False, 'salvage_or_gold_scoring': False,
        'model_calls_by_review': 0, 'tokenizer_calls_by_review': 0, 'paid_calls_by_review': 0,
        'limitations': ['Every response ended under the preregistered first-invalid-prefix stop; no later candidate content was generated.',
            'These failures measure compliance of the composite prompt/seed/sampler/strict-prefix condition, not synonym quality.',
            'No complete or repaired response was evaluated; no zero-bit capacity estimate inferred.',
            'Two diagnostic chunks are not the full109-chunk candidate resource or the closed test.',
            'Native execution/sampling evidence and actual Git readback require separate independent audit.'],
        'attempts': rows}
    lines = ['# Native-v2: описательный разбор всех шести ответов', '',
        'Проверены все **6/6 фактических raw-ответов** двух заранее фиксированных chunk. '
        'Каждая попытка остановилась после **первого подтверждённого токена** из-за недопустимого префикса. '
        'Принятых chunk: **0/2**. Это описание сохранённых bytes; независимый native-аудит сохраняется отдельно.', '',
        '| Попытка | Полные bytes в hex | Содержимое | Токены | Причина |',
        '|---|---|---|---:|---|']
    for row in rows:
        content = 'Открывающая `[` и перевод строки' if row['raw_hex'] == '5b0a' else 'Три обратные кавычки U+0060'
        lines.append(f"| {row['attempt']} | `{row['raw_hex']}` | {content} | {row['committed_output_tokens']} | Требуется JSON-объект верхнего уровня |")
    lines += ['', 'Итого: **6 output-токенов, 15 raw-байт**; три ответа начинались массивом, три — markdown fence. '
        'Ни одно слово-кандидат или полная строка словаря не были выданы. 512-токенный лимит не достигался. '
        'Внутреннее время попыток в сумме — ' + f"{summary['sum_attempt_elapsed_seconds_recorded']:.3f}" + ' с; это сумма записанных таймеров попыток, не полный wall time.', '',
        'Ответы не исправлялись: fence не удалялся, массив не преобразовывался в объект, генерация после отказа не продолжалась. '
        'Полные bytes приведены в таблице и JSON, а SHA256 каждой копии сверены с archive inventory и записанным результатом.', '',
        '**Семантическое качество и ёмкость не оценимы**: candidate content отсутствует. '
        'Нельзя подменять это оценкой «ноль полезных бит» или выводом о качестве синонимов. '
        'Результат относится к составному варианту prompt/seed/sampler/строгой ранней остановки, '
        'не к изолированному эффекту prompt и не к авторскому GPT-4o JSON API.', '',
        'Полный ресурс требует 109 chunk; 107 остаются вне этой диагностики. '
        'Новые модельные вызовы, tokenizer, salvage, gold scoring и closed test при разборе не использовались.', '']
    output = args.output; output.mkdir(parents=True, exist_ok=False)
    for name, raw in [('all-raw-review.json', (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()),
                      ('all-raw-review_ru.md', '\n'.join(lines).encode())]:
        with (output / name).open('xb') as f: f.write(raw)
    print(json.dumps({'attempts_reviewed': 6, 'all_raw_copies_match_packaged_inventory': True,
                      'candidate_strings_observed': 0, 'model_calls': 0}))


if __name__ == '__main__': main()
