import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
base = root / 'results/prospective-development-transport-v1-posthoc-slot-review-v1'
milestone = root / 'results/prospective-development-transport-v1/milestones/tinyllama-first30'
builder = root / 'experiments/prospective-development-transport-v1-posthoc-slot-review-v1/build_inventory.py'
source_path = root / 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
audit_path = base / 'inventory-independent-audit.json'
if audit_path.exists():
    raise FileExistsError('Exclusive audit output already exists')
checks = {}
failures = []
def check(name, condition, detail=None):
    entry = checks.setdefault(name, {'status': 'passed', 'evaluations': 0})
    entry['evaluations'] += 1
    if not condition:
        entry['status'] = 'failed'
        failures.append({'check': name, 'detail': detail})

def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()
def sha_text(value):
    return sha_bytes(value.encode('utf-8'))
def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
def rel(path):
    return str(path.relative_to(root))
def read_rows(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
def unique_map(rows, field, label):
    check(label, len(rows) == len({row[field] for row in rows}), len(rows))
    return {row[field]: row for row in rows}

# Deliberately independent segmentation: no regex and no imports from builder/codec.
def split_positions(text):
    words = []
    gaps = []
    cursor = 0
    while cursor < len(text):
        start_gap = cursor
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        gaps.append(text[start_gap:cursor])
        if cursor == len(text):
            return words, gaps
        start_word = cursor
        while cursor < len(text) and not text[cursor].isspace():
            cursor += 1
        words.append((start_word, cursor, text[start_word:cursor]))
    gaps.append('')
    return words, gaps

def word_core(token):
    ascii_positions = [i for i, ch in enumerate(token) if 'A' <= ch <= 'Z' or 'a' <= ch <= 'z']
    if not ascii_positions:
        raise ValueError(('No ASCII core in changed token', token))
    left, right = ascii_positions[0], ascii_positions[-1] + 1
    if any(not ('A' <= ch <= 'Z' or 'a' <= ch <= 'z') for ch in token[left:right]):
        raise ValueError(('More than one ASCII letter run in changed token', token))
    return left, right, token[:left], token[left:right], token[right:]

def window(text, positions, index):
    start = positions[max(0, index - 10)][0]
    end = positions[min(index + 10, len(positions) - 1)][1]
    return {'start': start, 'end': end, 'quote': text[start:end]}

def check_slice(text, slice_data, label):
    start, end, quote = slice_data['start'], slice_data['end'], slice_data['quote']
    check(label, isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text) and text[start:end] == quote, [start, end])

started = datetime.now(timezone.utc).isoformat()
input_paths = [milestone / name for name in ['cells.jsonl.gz', 'records.jsonl.gz', 'recipient/receiver_jobs.jsonl.gz', 'recipient/receiver_results.jsonl.gz', 'input_config.json', 'input_checkpoint.json', 'status.json']] + [source_path]
output_paths = [base / name for name in ['inventory.json', 'covers.jsonl', 'outcomes.jsonl', 'unique_slots.jsonl']]
code_before = sha_bytes(builder.read_bytes())
input_before = {rel(path): sha_bytes(path.read_bytes()) for path in input_paths}
output_before = {rel(path): sha_bytes(path.read_bytes()) for path in output_paths}
source_rows = read_rows(source_path)
records = read_rows(milestone / 'records.jsonl.gz')
cells = read_rows(milestone / 'cells.jsonl.gz')
jobs = read_rows(milestone / 'recipient/receiver_jobs.jsonl.gz')
receivers = read_rows(milestone / 'recipient/receiver_results.jsonl.gz')
covers = read_rows(base / 'covers.jsonl')
outcomes = read_rows(base / 'outcomes.jsonl')
slots = read_rows(base / 'unique_slots.jsonl')
inventory = json.loads((base / 'inventory.json').read_text())
checkpoint = json.loads((milestone / 'input_checkpoint.json').read_text())
config = json.loads((milestone / 'input_config.json').read_text())
status = json.loads((milestone / 'status.json').read_text())
source_by_id = unique_map(source_rows, 'record_id', 'source_record_ids_unique')
record_by_identity = unique_map(records, 'identity', 'record_identities_unique')
cell_by_id = unique_map(cells, 'row_id', 'all_cell_row_ids_unique')
job_by_id = unique_map(jobs, 'row_id', 'receiver_job_row_ids_unique')
receiver_by_id = unique_map(receivers, 'row_id', 'receiver_result_row_ids_unique')
cover_by_id = unique_map(covers, 'record_id', 'cover_record_ids_unique')
outcome_by_id = unique_map(outcomes, 'row_id', 'outcome_row_ids_unique')
slot_by_id = unique_map(slots, 'slot_id', 'slot_ids_unique')
checkpoint_by_id = unique_map(checkpoint['records'], 'identity', 'checkpoint_identities_unique')
check('source_count', len(source_rows) == 100, len(source_rows))
check('record_denominator', len(records) == 300, len(records))
check('cell_denominator', len(cells) == 15300, len(cells))
check('checkpoint_record_identity_set', set(checkpoint_by_id) == set(record_by_identity))
source_order_by_id = {row['record_id']: index + 1 for index, row in enumerate(source_rows)}
for source in source_rows:
    check('all_source_prompt_hashes', sha_text(source['prompt']) == source['prompt_sha256'], source['record_id'])
    check('all_source_excerpt_hashes', sha_text(source['excerpt']) == source['excerpt_sha256'], source['record_id'])
for record in records:
    source = source_by_id[record['record_id']]
    check('record_source_order', record['source_order'] == source_order_by_id[record['record_id']], record['identity'])
    for field in ['record_id', 'source_group_id', 'split', 'topic', 'genre', 'target_words', 'prompt_sha256']:
        check('record_source_lineage', record[field] == source[field], [record['identity'], field])
    old = checkpoint_by_id[record['identity']]
    for field in sorted(set(old) & set(record)):
        check('checkpoint_record_shared_fields', record[field] == old[field], [record['identity'], field])
cell_groups = defaultdict(list)
for cell in cells:
    record = record_by_identity[cell['identity']]
    cell_groups[cell['identity']].append(cell)
    for field in ['record_id', 'source_group_id', 'source_order', 'model_alias']:
        check('all_cells_record_lineage', cell[field] == record[field], [cell['row_id'], field])
    check('all_cells_cover_status_lineage', cell['cover_status'] == record['status'], cell['row_id'])
    check('all_cells_attempt_hash_lineage', cell['attempt_sha256'] == record['attempt_sha256'], cell['row_id'])
    check('full_exact_flag_status_equivalence', (cell['full_exact'] is True) == (cell['execution_status'] == 'full_exact'), cell['row_id'])
    check('encoder_called_status_equivalence', (cell['encoder_called'] is True) == (cell['execution_status'] in ['full_exact', 'insufficient_capacity']), cell['row_id'])
for identity, group in cell_groups.items():
    check('fifty_one_cells_per_record', len(group) == 51, identity)
check('cell_record_identity_set', set(cell_groups) == set(record_by_identity))
completed = sorted([row for row in records if row['status'] == 'completed'], key=lambda row: row['source_order'])
successes = sorted([row for row in cells if row['execution_status'] == 'full_exact'], key=lambda row: (row['source_order'], row['parent_payload_length'], row['row_id']))
success_ids = {row['row_id'] for row in successes}
check('completed_cover_count', len(completed) == 30, len(completed))
check('completed_source_orders', [row['source_order'] for row in completed] == list(range(1, 31)))
check('full_exact_count', len(successes) == 75, len(successes))
check('complete_receiver_outcome_row_id_crosswalk', success_ids == set(job_by_id) == set(receiver_by_id) == set(outcome_by_id))
check('all_completed_covers_in_inventory', set(cover_by_id) == {row['record_id'] for row in completed})
independent_covers = []
for record in completed:
    identity = record['identity']
    attempt_path = milestone / 'input_snapshot/runs/large-campaign-development-v2/models/tinyllama11/attempts' / (identity + '.json')
    input_paths.append(attempt_path)
    input_before[rel(attempt_path)] = sha_bytes(attempt_path.read_bytes())
    attempt = json.loads(attempt_path.read_text())
    cover_text = attempt['result']['cover_text']
    source = source_by_id[record['record_id']]
    check('attempt_file_hash', input_before[rel(attempt_path)] == record['attempt_sha256'], identity)
    check('attempt_internal_identity', attempt['identity'] == identity, identity)
    check('attempt_model_and_revision', attempt['model'] == record['model'] and attempt['result']['revision'] == record['revision'], identity)
    check('attempt_completed_status', attempt['status'] == record['status'] == 'completed', identity)
    check('attempt_cover_record_equality', cover_text == record['cover_text'], identity)
    check('attempt_record_metadata', all(attempt['record'][field] == record[field] for field in attempt['record']), identity)
    cover_tokens, _ = split_positions(cover_text)
    cp = checkpoint_by_id[identity]
    check('checkpoint_completed_cover_sha_and_words', cp['cover_sha256'] == sha_text(cover_text) and cp['whitespace_words'] == len(cover_tokens), identity)
    group = cell_groups[identity]
    expected = {'source_order': record['source_order'], 'record_id': record['record_id'], 'title': source['title'], 'cover_text': cover_text, 'cover_sha256': sha_text(cover_text), 'source_excerpt': source['excerpt'], 'source_excerpt_sha256': sha_text(source['excerpt']), 'attempt_snapshot_path': rel(attempt_path), 'attempt_sha256': sha_bytes(attempt_path.read_bytes()), 'all_cell_execution_status_counts': dict(sorted(Counter(row['execution_status'] for row in group).items())), 'successful_row_ids': sorted(row['row_id'] for row in group if row['execution_status'] == 'full_exact')}
    independent_covers.append(expected)
    check('exact_cover_artifact_all_fields', expected == cover_by_id[record['record_id']], identity)
check('cover_order_and_contents', independent_covers == covers)
independent_slots = {}
independent_outcomes = []
span_manifest = []
per_outcome = []
changes_total = 0
non_ascii_offset_occurrences = 0
for cell in successes:
    row_id = cell['row_id']
    original = cover_by_id[cell['record_id']]
    cover_text = original['cover_text']
    stego_text = job_by_id[row_id]['stego_text']
    result = receiver_by_id[row_id]
    left, left_gaps = split_positions(cover_text)
    right, right_gaps = split_positions(stego_text)
    check('successful_text_full_sha', sha_text(cover_text) == cell['cover_sha256'] and sha_text(stego_text) == cell['stego_sha256'], row_id)
    check('successful_word_count_alignment', len(left) == len(right) == cell['cover_words'] == cell['stego_words'], row_id)
    check('successful_exact_whitespace_alignment', left_gaps == right_gaps, row_id)
    check('successful_receiver_status', result['status'] == 'decoded' and result['error'] is None and cell['fresh_receiver_exact'] is True, row_id)
    bits = result['decoded_bits']
    check('receiver_binary_bitstring', isinstance(bits, str) and set(bits) <= {'0', '1'}, row_id)
    check('receiver_decoded_lengths', len(bits) == cell['decoded_bits'] == cell['requested_payload_bits'] == cell['delivered_payload_bits'], row_id)
    check('receiver_decoded_bitstring_sha_payload_match', sha_text(bits) == cell['decoded_sha256'] == cell['payload_sha256'], row_id)
    changed_ids = []
    evidence = []
    for index, (before, after) in enumerate(zip(left, right)):
        if before[2] == after[2]:
            continue
        p0, p1, prefix, original_word, suffix = word_core(before[2])
        q0, q1, replacement_prefix, replacement_word, replacement_suffix = word_core(after[2])
        check('changed_token_punctuation_alignment', (prefix, suffix) == (replacement_prefix, replacement_suffix), [row_id, index])
        check('changed_core_only_reconstructs_token', prefix + replacement_word + suffix == after[2], [row_id, index])
        key = [cell['record_id'], index, original_word, replacement_word]
        slot_id = sha_bytes(json.dumps(key, ensure_ascii=True, separators=(',', ':')).encode('utf-8'))
        original_span = {'start': before[0] + p0, 'end': before[0] + p1, 'quote': original_word}
        stego_span = {'start': after[0] + q0, 'end': after[0] + q1, 'quote': replacement_word}
        original_context = window(cover_text, left, index)
        stego_context = window(stego_text, right, index)
        check_slice(cover_text, original_span, 'exact_original_unicode_codepoint_span')
        check_slice(stego_text, stego_span, 'exact_stego_unicode_codepoint_span')
        check_slice(cover_text, original_context, 'exact_original_context_window')
        check_slice(stego_text, stego_context, 'exact_stego_context_window')
        fixed = {'slot_id': slot_id, 'deduplication_key': key, 'source_order': cell['source_order'], 'record_id': cell['record_id'], 'token_index_zero_based': index, 'original': original_word, 'replacement': replacement_word, 'original_token': before[2], 'replacement_token': after[2], 'cover_sha256': sha_text(cover_text), 'source_excerpt_sha256': original['source_excerpt_sha256'], 'original_span': original_span, 'original_context': original_context}
        occurrence = {'row_id': row_id, 'parent_payload_length': cell['parent_payload_length'], 'stego_sha256': sha_text(stego_text), 'stego_span': stego_span, 'stego_context': stego_context}
        if slot_id not in independent_slots:
            independent_slots[slot_id] = dict(fixed, occurrences=[])
        else:
            check('deduplicated_slot_fixed_fields_identical', fixed == {k: v for k, v in independent_slots[slot_id].items() if k != 'occurrences'}, slot_id)
        independent_slots[slot_id]['occurrences'].append(occurrence)
        changed_ids.append(slot_id)
        item = {'row_id': row_id, 'slot_id': slot_id, 'token_index_zero_based': index, 'original_start': original_span['start'], 'original_end': original_span['end'], 'stego_start': stego_span['start'], 'stego_end': stego_span['end'], 'original_span_sha256': sha_text(original_word), 'stego_span_sha256': sha_text(replacement_word), 'original_context_start': original_context['start'], 'original_context_end': original_context['end'], 'original_context_sha256': sha_text(original_context['quote']), 'stego_context_start': stego_context['start'], 'stego_context_end': stego_context['end'], 'stego_context_sha256': sha_text(stego_context['quote'])}
        evidence.append(item)
        span_manifest.append(item)
        non_ascii_offset_occurrences += len(cover_text[:original_span['start']].encode('utf-8')) != original_span['start'] or len(stego_text[:stego_span['start']].encode('utf-8')) != stego_span['start']
    check('changed_positions_no_omissions', len(changed_ids) == cell['changed_positions'], row_id)
    check('changed_slots_no_duplicates_per_outcome', len(changed_ids) == len(set(changed_ids)), row_id)
    changes_total += len(changed_ids)
    expected_outcome = {'row_id': row_id, 'source_order': cell['source_order'], 'record_id': cell['record_id'], 'arm': cell['arm'], 'grid': cell['grid'], 'parent_payload_length': cell['parent_payload_length'], 'requested_payload_bits': cell['requested_payload_bits'], 'execution_status': cell['execution_status'], 'receiver_status': result['status'], 'cover_sha256': cell['cover_sha256'], 'stego_sha256': cell['stego_sha256'], 'stego_text': stego_text, 'changed_positions': cell['changed_positions'], 'changed_slot_ids': changed_ids, 'cell': cell}
    independent_outcomes.append(expected_outcome)
    check('exact_outcome_artifact_all_fields', expected_outcome == outcome_by_id[row_id], row_id)
    per_outcome.append({'row_id': row_id, 'source_order': cell['source_order'], 'parent_payload_length': cell['parent_payload_length'], 'tokens': len(left), 'changed_positions': len(changed_ids), 'change_and_span_manifest_sha256': sha_bytes(canonical(evidence)), 'crosswalk_verified': expected_outcome == outcome_by_id[row_id]})
independent_slot_rows = sorted(independent_slots.values(), key=lambda row: (row['source_order'], row['token_index_zero_based'], row['original'], row['replacement']))
check('every_unique_slot_present_once', set(independent_slots) == set(slot_by_id))
for slot_id, expected in independent_slots.items():
    check('exact_unique_slot_and_all_occurrences', expected == slot_by_id[slot_id], slot_id)
check('outcomes_order_and_contents', independent_outcomes == outcomes)
check('slots_order_and_contents', independent_slot_rows == slots)
check('total_changed_occurrences', changes_total == 1252, changes_total)
check('total_unique_changed_slots', len(independent_slots) == 593, len(independent_slots))
check('occurrence_total_matches_inventory_rows', sum(len(row['occurrences']) for row in slots) == changes_total)
counts = {'planned_model_record_rows': len(records), 'completed_covers': len(completed), 'all_planned_cells': len(cells), 'actual_encoder_attempts': sum(cell['encoder_called'] is True for cell in cells), 'full_exact_outcomes': len(successes), 'insufficient_capacity': sum(cell['execution_status'] == 'insufficient_capacity' for cell in cells), 'changed_slot_occurrences': changes_total, 'unique_changed_slots': len(independent_slots)}
for field, actual in counts.items():
    check('inventory_count_' + field, actual == inventory[field], {'actual': actual, 'inventory': inventory[field]})
check('actual_encoder_attempt_count', counts['actual_encoder_attempts'] == 720)
check('insufficient_capacity_count', counts['insufficient_capacity'] == 645)
execution_counts = dict(sorted(Counter(cell['execution_status'] for cell in cells).items()))
check('inventory_status_counts', execution_counts == inventory['execution_status_counts_all_cells'])
check('inventory_successful_cover_orders', inventory['successful_cover_orders'] == [row['source_order'] for row in independent_covers if row['successful_row_ids']])
check('inventory_unsuccessful_cover_orders', inventory['cover_orders_without_successful_stego'] == [row['source_order'] for row in independent_covers if not row['successful_row_ids']])
for key, expected in [('completed_covers', 30), ('encoder_calls', 720), ('receiver_jobs', 75), ('full_denominator', 300), ('total_cells', 15300), ('unexpected_errors', 0)]:
    check('frozen_status_count_consistency', status[key] == expected, key)
check('frozen_status_no_test_access', status['closed_test_opened'] is False)
check('frozen_config_source_sha', config['development_candidate']['source_prompts_sha256'] == input_before[rel(source_path)])
check('inventory_execution_claims_are_zero', inventory['generation_calls'] == inventory['encoder_calls'] == inventory['receiver_calls'] == 0)
check('inventory_unicode_offset_convention', inventory['offset_convention'] == 'zero-based, end-exclusive Unicode code-point offsets')
manifest = unique_map(inventory['frozen_inputs'], 'path', 'frozen_manifest_paths_unique')
check('complete_frozen_input_manifest', set(manifest) == set(input_before))
for path, digest in input_before.items():
    check('frozen_input_manifest_sha_matches_bytes', manifest[path]['sha256'] == digest, path)
for path in output_paths:
    if path.name != 'inventory.json':
        check('artifact_sha_manifest_matches_bytes', inventory['artifact_sha256'][path.name] == output_before[rel(path)], path.name)
for filename, rows in [('covers.jsonl', independent_covers), ('outcomes.jsonl', independent_outcomes), ('unique_slots.jsonl', independent_slot_rows)]:
    expected_bytes = b''.join(canonical(row) + b'\n' for row in rows)
    check('independent_reconstruction_exact_canonical_bytes', expected_bytes == (base / filename).read_bytes(), filename)
imports = []
for node in ast.walk(ast.parse(builder.read_text())):
    if isinstance(node, ast.Import):
        imports.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.append(node.module)
check('builder_imports_stdlib_only', set(imports) <= {'__future__', 'argparse', 'collections', 'gzip', 'hashlib', 'json', 'pathlib', 're'}, imports)
# This is the only subprocess: documented read-only reconstruction of inventory.
verification = subprocess.run([sys.executable, str(builder), '--verify-existing'], cwd=root, text=True, capture_output=True, check=False)
check('builder_verify_existing_returncode', verification.returncode == 0, verification.stderr)
try:
    verification_json = json.loads(verification.stdout)
except json.JSONDecodeError:
    verification_json = None
    check('builder_verify_existing_json', False, verification.stdout)
if verification_json is not None:
    check('builder_verify_existing_mode', verification_json.get('mode') == 'verified')
    for filename, digest in verification_json.get('outputs', {}).items():
        check('builder_verify_existing_reported_sha', output_before[rel(base / filename)] == digest, filename)
for path in input_paths:
    check('frozen_input_bytes_unchanged_after_audit', input_before[rel(path)] == sha_bytes(path.read_bytes()), rel(path))
for path in output_paths:
    check('inventory_artifact_bytes_unchanged_after_audit', output_before[rel(path)] == sha_bytes(path.read_bytes()), rel(path))
check('builder_code_unchanged_after_audit', code_before == sha_bytes(builder.read_bytes()))
limitations = [
    'Структурная сверка сохранённых свидетельств, а не повторное выполнение генерации, кодирования или декодирования; исторические статусы и exact-показатели дополнительно согласованы с сохранёнными битовыми строками receiver, но транспорт заново не исполнялся.',
    'Никакие quality labels из review-part файлов не читались и не переоценивались; результаты не являются подтверждением языковой естественности, истинности содержания, скрытности или безопасности.',
    'Сам builder предполагает уникальность некоторых исходных идентификаторов и не проверяет битовые строки receiver, полный status/flag crosswalk или все внутренние attempt-поля; настоящий независимый проход проверил эти дополнительные свойства фактических файлов.',
    'Алгоритм инвентаря допускает один непрерывный ASCII-буквенный core изменённого whitespace-токена с неизменными prefix/suffix. Независимый посимвольный проход проверил все фактические отличия; обобщение на иной Unicode- или составной токенизатор не заявляется.',
    'Span/context в исходном инвентаре не имеют собственных SHA-полей. Проверены точные Unicode code-point срезы и границы окон, SHA полного cover/stego и артефактов; независимый digest манифестов срезов вычислен отдельно.',
    'SHA обеспечивает побайтовую согласованность с текущими frozen файлами и их записанными digest, но не является внешней аттестацией происхождения этих файлов.',
    'В режиме --verify-existing builder содержит OUT.mkdir(exist_ok=True). Каталог уже существовал; содержимое всех 38 входных файлов, четырёх инвентарных артефактов и builder сохранило исходные SHA. Единственный новый файл этого прохода — настоящий аудит.',
    'Пять covers без full_exact сохранены в общем знаменателе; для них нет stego-текстов для пословного сравнения. 645 insufficient_capacity сохранены как счётчик попыток, но не дают успешного stego outcome.',
    'Не открывались test-источники, сети, Git и модели. Самостоятельный checker не импортировал inventory builder, codec или receiver; builder запущен только с --verify-existing.'
]
report = {'schema': 'synsteg-posthoc-slot-inventory-independent-audit-v1', 'auditor': 'slot_review_a', 'status': 'passed' if not failures else 'failed', 'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'All frozen milestone rows and all inventory cover/outcome/unique-slot crosswalks; no linguistic relabeling.', 'independent_method': 'Character-by-character Unicode whitespace scanner; separate ASCII-core extraction; independently reconstructed covers/outcomes/slots and all occurrence/context boundaries without importing builder/codec/receiver.', 'code': {'audited_builder': {'path': rel(builder), 'sha256': code_before, 'imports': sorted(imports)}, 'independent_checker': {'source_field': 'independent_checker_source', 'sha256': sha_text(audit_source), 'execution': 'compile embedded source and exec in one Python process; no checker file written'}}, 'input_sha256': [{'path': path, 'sha256': digest} for path, digest in sorted(input_before.items())], 'output_sha256': [{'path': path, 'sha256': digest} for path, digest in sorted(output_before.items())], 'counts': counts, 'execution_status_counts_all_cells': execution_counts, 'receiver_validation': {'rows': len(receivers), 'decoded_sha_convention': 'SHA256 of UTF-8/ASCII 0/1 string; not packed payload bytes', 'decoded_bits_lengths': dict(sorted(Counter(len(row['decoded_bits']) for row in receivers).items()))}, 'independent_span_evidence': {'changed_occurrences': len(span_manifest), 'original_span_checks': changes_total, 'stego_span_checks': changes_total, 'original_context_checks': changes_total, 'stego_context_checks': changes_total, 'occurrences_with_unicode_codepoint_offset_different_from_utf8_byte_offset': non_ascii_offset_occurrences, 'canonical_manifest_sha256': sha_bytes(canonical(span_manifest)), 'canonical_serialization': 'UTF-8 JSON, ensure_ascii=False, sort_keys=True, separators=(comma,colon), no newline; ordered by source_order,parent_payload_length,row_id,token_index', 'manifest_reproducible_from_embedded_checker': True}, 'per_outcome': per_outcome, 'checks': checks, 'check_categories': len(checks), 'check_evaluations': sum(x['evaluations'] for x in checks.values()), 'failures': failures, 'builder_verify_existing': {'command': [sys.executable, rel(builder), '--verify-existing'], 'returncode': verification.returncode, 'stdout_json': verification_json, 'stderr': verification.stderr}, 'static_code_review': {'separate_reviewer': 'inventory_code_check', 'builder_does_not_import_or_invoke_codec_model_receiver': True, 'read_only_mode_caveat': 'OUT.mkdir(exist_ok=True) exists; output directory already present'}, 'limitations_ru': limitations, 'independent_checker_source': audit_source}
with audit_path.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'audit_path': rel(audit_path), 'sha256': sha_bytes(audit_path.read_bytes()), 'status': report['status'], 'counts': counts, 'checks': len(checks), 'evaluations': report['check_evaluations'], 'failures': failures, 'unicode_offset_cases': non_ascii_offset_occurrences}, ensure_ascii=False))
