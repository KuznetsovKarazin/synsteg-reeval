#!/usr/bin/env python3
"""Read-only independent review of preserved PR28 baseline83..87 evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def review(root):
    checks, failures, shards = [], [], []
    def check(name, condition):
        checks.append(name)
        if not condition:
            failures.append(name)
    def load(path):
        return json.loads(path.read_bytes())
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for order in range(83, 88):
        directory = root / f'runs/baseline-single-record-recovery-v1/order{order:03d}'
        receipt = load(directory / 'git-preservation-receipt.json')
        inventory = {str(p.relative_to(directory)): digest(p)
                     for p in directory.rglob('*')
                     if p.is_file() and p.name != 'git-preservation-receipt.json'}
        check(f'{order}: complete 14-file inventory', len(inventory) == 14 and inventory == receipt['files_sha256'])
        check(f'{order}: process receipt identity', receipt['process_sha256'] == digest(directory / 'process.json'))
        ledger = load(directory / 'run/local_ledger/ledger.json')
        attempts = list((directory / 'run/attempts').glob('*.json'))
        check(f'{order}: one ledger and attempt', len(ledger['attempts']) == len(attempts) == 1)
        entry = next(iter(ledger['attempts'].values()))
        attempt = load(attempts[0])
        check(f'{order}: paired completed result', entry['status'] == attempt['status'] == 'completed'
              and entry['result'] == attempt['result'])
        check(f'{order}: no automatic retry', entry['automatic_retries'] == attempt['automatic_retries'] == 0)
        process = load(directory / 'process.json')
        check(f'{order}: normal terminal wrapper', process['status'] == 'completed_pending_remote_preservation'
              and process['exit_code'] == 0 and process['timed_out'] is False)
        audit = load(root / f'results/baseline-single-record-recovery-v1-root/order{order:03d}-audit.json')
        check(f'{order}: prior independent shard audit', audit['status'] == 'passed' and audit['checks'] == 28
              and audit['errors'] == [] and audit['model_calls'] == audit['tokenizer_calls'] == 0)
        shards.append({'source_order': order, 'output_tokens': entry['result']['usage']['output_tokens'],
                       'source_record': entry['metadata']['record']['record_id'],
                       'raw_primary': receipt['commit']})
    projection = load(root / 'results/baseline-single-record-recovery-v1-root/full300-after087.json')
    rows = projection['records']
    counts = dict(Counter(row['status'] for row in rows))
    check('all 300 unique model-record pairs', len(rows) == len({(r['model_alias'], r['record_id']) for r in rows}) == 300)
    check('full counts 86/1/213', counts == {'completed': 86, 'reserved_outcome_unknown': 1, 'not_attempted': 213})
    unknown = [row for row in rows if row['status'] == 'reserved_outcome_unknown']
    check('sole historical unknown82 retained', len(unknown) == 1 and unknown[0]['source_order'] == 82
          and unknown[0]['model_alias'] == 'tinyllama11' and unknown[0]['quality'] is None and unknown[0]['transport'] is None)
    check('other two models remain wholly unattempted', all(r['status'] == 'not_attempted' for r in rows if r['model_alias'] != 'tinyllama11'))
    check('new rows correspond to actual attempts', all(next(r for r in rows if r['model_alias'] == 'tinyllama11'
          and r['source_order'] == x['source_order'])['record_id'] == x['source_record'] for x in shards))
    check('new raw token sum', sum(x['output_tokens'] for x in shards) == 3145)
    summary = load(root / 'results/baseline-single-record-recovery-v1-root/summary-after087.json')
    check('summary conserves counts', summary['counts'] == counts and sum(summary['stop_reasons'].values()) == 86)
    check('summary declares exactly five new requests', summary['new_model_requests'] == 5 and summary['new_source_orders'] == list(range(83, 88)))
    closure = load(root / 'results/author-candidate-mask-first-v4-postrun/closure.json')
    check('dictionary v4 full two-case denominator', closure['diagnostic_denominator'] == 2
          and closure['format_accepted'] == closure['empty_schema_support'] == 1 and closure['outcome_unknown'] == 0)
    check('dictionary v4 negative quality retained', closure['candidate_strings'] == closure['punctuation_violations'] == 40
          and closure['new_synonymous_lexical_cores_observed'] == 0)
    check('dictionary capacity remains unmeasured', all(closure[k] is None for k in
          ['usable_dictionary_rows', 'capacity_bits', 'payload_delivery', 'receiver_eligibility_invariance']))
    return {'schema': 'synsteg-pr28-independent-initial-evidence-review-v1',
            'status': 'passed' if not failures else 'failed', 'checks': len(checks), 'check_names': checks,
            'failures': failures, 'baseline_shards': shards, 'counts': counts,
            'model_calls': 0, 'tokenizer_calls': 0, 'scientific_audit_reinvocations': 0,
            'limits': ['Read-only checks of already preserved evidence; not a new generation or transport evaluation.',
                       'Historical token total56944 is checked for internal reporting only; earlier81 outputs not independently reloaded here.',
                       'Public-context4 results were not yet preserved at this review snapshot and are outside this initial check.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'checks', 'failures']}))
    raise SystemExit(0 if result['status'] == 'passed' else 2)
