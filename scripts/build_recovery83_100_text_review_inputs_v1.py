#!/usr/bin/env python3
"""Read only saved terminal shards; retain all18 review denominators and nulls.

No NN, tokenizer, codec, external fact checking or automated semantic judgment.
The Git receipt is a coordinator attestation, not an independent network fetch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

RECOVERY = 'baseline-single-record-recovery-v1'
ID = 'recovery83-100-text-review-inputs-v1'
MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
TERMINAL = {'completed_pending_remote_preservation', 'stopped_requires_review_no_retry',
            'integrity_review_required_no_retry'}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    if path.is_symlink():
        raise ValueError('Symlink evidence refused')
    return path.read_bytes()


def read_json(path):
    return json.loads(read_bytes(path))


def inventory(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink evidence refused')
        if path.is_file() and path.name != 'git-preservation-receipt.json':
            result[str(path.relative_to(directory))] = sha_bytes(read_bytes(path))
    return result


def spans(text):
    """Exact offsets of nonempty blocks separated by blank lines, not sentences."""
    result = []
    start = 0
    for separator in list(re.finditer(r'\n[ \t]*\n+', text)) + [None]:
        end = len(text) if separator is None else separator.start()
        raw = text[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            result.append({'start': start + left, 'end': start + right, 'text': raw[left:right]})
        start = len(text) if separator is None else separator.end()
    return result


def source_input(protocol_path, protocol, job):
    order = job['source_order']
    expected_path = f'prompts/order{order:03d}.jsonl'
    if job['prompt_path'] != expected_path:
        raise ValueError('Unexpected source path')
    data = read_bytes(protocol_path.parent / expected_path)
    if sha_bytes(data) != job['prompt_file_sha256']:
        raise ValueError('Input hash changed')
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError('Need one exact source row')
    row = records[0]
    if (row['record_id'] != job['record_id'] or row['split'] != 'development'
            or sha_bytes(row['prompt'].encode()) != job['original_prompt_sha256']
            or sha_bytes(row['excerpt'].encode()) != row['excerpt_sha256']):
        raise ValueError('Source metadata mismatch')
    return row


def observed_shard(run_root, job):
    directory = run_root / f"order{job['source_order']:03d}"
    empty = {'cover_text': None, 'cover_sha256': None, 'cover_words_whitespace': None,
             'generation_usage': None, 'stop_reason': None, 'hit_output_limit': None,
             'cover_blocks': [], 'candidate_number_spans': [], 'generation_status': None,
             'process_sha256': None, 'git_commit': None, 'quality_labels': None,
             'review_complete': False}
    if not directory.exists():
        return {**empty, 'availability': 'not_attempted'}
    if directory.is_symlink():
        raise ValueError('Symlink shard refused')
    process_path = directory / 'process.json'
    if not process_path.exists():
        return {**empty, 'availability': 'not_terminal_outputs_not_read'}
    process_bytes = read_bytes(process_path)
    process = json.loads(process_bytes)
    if (process.get('schema') != RECOVERY + '-process'
            or process.get('source_order') != job['source_order'] or process.get('status') not in TERMINAL):
        raise ValueError('Terminal process identity required')
    receipt_path = directory / 'git-preservation-receipt.json'
    if not receipt_path.exists():
        return {**empty, 'availability': 'terminal_pending_git_preservation_outputs_not_read',
                'process_sha256': sha_bytes(process_bytes)}
    receipt_bytes = read_bytes(receipt_path)
    receipt = json.loads(receipt_bytes)
    if (receipt.get('readback_verified') is not True
            or receipt.get('process_sha256') != sha_bytes(process_bytes)
            or not re.fullmatch('[0-9a-f]{40}', receipt.get('commit', ''))):
        raise ValueError('Require matching completed Git preservation receipt')
    # Only now may raw output/ledger bytes be read, including for inventory.
    before = inventory(directory)
    if before != receipt.get('files_sha256'):
        raise ValueError('Saved shard inventory changed')
    common = {**empty, 'process_sha256': sha_bytes(process_bytes), 'git_commit': receipt['commit'],
              'process_status': process['status'], 'preservation_receipt_sha256': sha_bytes(receipt_bytes),
              'availability': 'saved_terminal_without_completed_cover'}
    ledger_path = directory / 'run/local_ledger/ledger.json'
    attempt_path = directory / 'run/attempts' / (job['identity'] + '.json')
    if not ledger_path.exists() or not attempt_path.exists():
        return common
    ledger = read_json(ledger_path)
    if set(ledger['attempts']) != {job['identity']}:
        raise ValueError('Unexpected identities in saved shard')
    entry = ledger['attempts'][job['identity']]
    attempt = read_json(attempt_path)
    if (entry.get('status') != attempt.get('status') or entry.get('result') != attempt.get('result')
            or entry['metadata']['record']['record_id'] != job['record_id']
            or attempt['record']['record_id'] != job['record_id']):
        raise ValueError('Saved ledger and attempt disagree')
    common['generation_status'] = entry['status']
    common['attempt_sha256'] = sha_bytes(read_bytes(attempt_path))
    if entry['status'] != 'completed':
        return common
    result = entry['result']
    text = result.get('cover_text')
    if not isinstance(text, str) or not text or result.get('returned_model') != MODEL:
        raise ValueError('Completed cover identity invalid')
    common.update(availability='saved_completed_cover', cover_text=text,
                  cover_sha256=sha_bytes(text.encode()), cover_words_whitespace=len(text.split()),
                  generation_usage=result.get('usage'), stop_reason=result.get('stop_reason'),
                  hit_output_limit=result.get('hit_output_limit'), cover_blocks=spans(text),
                  candidate_number_spans=[{'start': match.start(), 'end': match.end(), 'text': match.group()}
                                          for match in re.finditer(r'\b\d+(?:[.,]\d+)*\b', text)])
    if inventory(directory) != before or read_bytes(receipt_path) != receipt_bytes:
        raise ValueError('Shard changed during extraction')
    return common


def build(protocol_path, run_root):
    protocol_path, run_root = Path(protocol_path), Path(run_root)
    protocol = read_json(protocol_path)
    if (protocol.get('experiment_id') != RECOVERY or protocol.get('new_source_orders') != list(range(83, 101))
            or protocol.get('permanently_unknown_source_orders') != [82]
            or [job['source_order'] for job in protocol.get('jobs', [])] != list(range(83, 101))):
        raise ValueError('Require complete fixed18 recovery protocol')
    records = []
    for job in protocol['jobs']:
        row = source_input(protocol_path, protocol, job)
        observed = observed_shard(run_root, job)
        records.append({'source_order': job['source_order'], 'record_id': row['record_id'],
                        'source_group_id': row['source_group_id'], 'source_title': row['title'],
                        'source_excerpt': row['excerpt'], 'source_excerpt_sha256': row['excerpt_sha256'],
                        'source_words_whitespace': len(row['excerpt'].split()),
                        'prompt': row['prompt'], 'prompt_sha256': sha_bytes(row['prompt'].encode()),
                        'target_words': row['target_words'], 'genre': row['genre'], 'topic': row['topic'],
                        'source_blocks': spans(row['excerpt']), **observed})
    counts = {name: sum(row['availability'] == name for row in records)
              for name in sorted({row['availability'] for row in records})}
    return {'schema': ID, 'protocol_sha256': sha_bytes(read_bytes(protocol_path)),
            'review_denominator': 18, 'tinyllama_denominator': 100, 'campaign_denominator': 300,
            'historical_unknown_source_orders_not_retried': [82], 'source_orders': list(range(83, 101)),
            'availability_counts': counts, 'records': records, 'model_calls': 0, 'tokenizer_calls': 0,
            'external_factchecking': False, 'semantic_error_rate': None, 'human_gold': False,
            'quality_labels_completed': 0, 'closed_test_accessed': False,
            'limitations': ['Metadata extraction, not semantic review or scientific generation audit.',
                            'Number and block spans are exact text navigation aids, not unsupported claims.',
                            'Coordinator Git receipt is checked locally; this script does not independently fetch Git.',
                            'Source unsupported is distinct from externally false; unavailable quality stays null.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.protocol, args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'review_denominator': result['review_denominator'], 'availability_counts': result['availability_counts']}))


if __name__ == '__main__':
    main()
