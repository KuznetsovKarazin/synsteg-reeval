#!/usr/bin/env python3
"""Check fixed review83 hashes, exact offsets and full18/null denominator."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(inputs, review):
    raw = Path(inputs).read_bytes()
    source = json.loads(raw)
    annotated = json.loads(Path(review).read_bytes())
    assert annotated['review_input_sha256'] == hashlib.sha256(raw).hexdigest()
    assert [r['source_order'] for r in annotated['records']] == list(range(83, 101))
    assert annotated['review_denominator'] == 18 and annotated['completed_reviews'] == 1
    assert all(r['quality_review'] is None for r in annotated['records'][1:])
    r = annotated['records'][0]['quality_review']
    text = source['records'][0]
    checks = {'review_input_sha': 1, 'full18_denominator': 1, 'null_other17': 17,
              'cover_spans': 0, 'source_spans': 0, 'literal_absences': 0, 'content_hashes': 0}
    assert r['record_id'] == text['record_id'] and r['source_order'] == 83
    assert r['complete_source_and_cover_read'] is True
    for field, content in [('cover_sha256', text['cover_text']), ('source_excerpt_sha256', text['source_excerpt'])]:
        assert r[field] == hashlib.sha256(content.encode()).hexdigest()
        checks['content_hashes'] += 1
    def span_check(value, content, counter):
        assert content[value['start']:value['end']] == value['text']
        assert 0 <= value['start'] < value['end'] <= len(content)
        checks[counter] += 1
    for observation in r['source_unsupported_observation_groups']:
        span_check(observation['cover_span'], text['cover_text'], 'cover_spans')
    for omission in r['omissions']:
        span_check(omission['source_span'], text['source_excerpt'], 'source_spans')
    for value in r['internal_inconsistency']['cover_spans']:
        span_check(value, text['cover_text'], 'cover_spans')
    for value in r['literal_absence_checks']:
        assert value['needle'].casefold() not in text['cover_text'].casefold()
        checks['literal_absences'] += 1
    assert r['format']['cover_words_whitespace'] == len(text['cover_text'].split())
    return {'schema': 'recovery83-source-review-validation-v1', 'status': 'passed', 'checks': checks,
            'review_sha256': hashlib.sha256(Path(review).read_bytes()).hexdigest(),
            'semantic_judgments_validated': False, 'new_model_calls': 0, 'live_output_reads': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = validate(args.inputs, args.review)
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
