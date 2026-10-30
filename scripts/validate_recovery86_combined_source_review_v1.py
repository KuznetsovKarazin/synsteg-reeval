#!/usr/bin/env python3
"""Validate saved83/84 reuse and new85/86 exact evidence spans, not semantics."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(inputs, review, previous):
    raw, review_raw, old_raw = [Path(p).read_bytes() for p in (inputs, review, previous)]
    source, annotated, old = [json.loads(data) for data in (raw, review_raw, old_raw)]
    assert annotated['review_input_sha256'] == hashlib.sha256(raw).hexdigest()
    assert annotated['previous_review_sha256'] == hashlib.sha256(old_raw).hexdigest()
    assert [r['source_order'] for r in annotated['records']] == list(range(83, 101))
    assert all(annotated['records'][i]['quality_review'] == old['records'][i]['quality_review'] for i in range(2))
    assert all(r['quality_review'] is None for r in annotated['records'][4:])
    checks = {'input_sha': 1, 'previous_review_sha': 1, 'exact83_84_reuse': 2,
              'full18_denominator': 1, 'null_other14': 14, 'cover_spans': 0, 'source_spans': 0,
              'content_hashes': 0, 'record_metadata_checks': 0}
    def exact(value, content, counter):
        assert 0 <= value['start'] < value['end'] <= len(content)
        assert content[value['start']:value['end']] == value['text']
        checks[counter] += 1
    for index in [2, 3]:
        r, text = annotated['records'][index]['quality_review'], source['records'][index]
        assert r['source_order'] == text['source_order'] and r['record_id'] == text['record_id']
        assert r['format']['cover_words_whitespace'] == len(text['cover_text'].split())
        assert r['stop_reason'] == 'end_turn' and r['generation_usage']['output_tokens'] == (565 if index == 2 else 502)
        checks['record_metadata_checks'] += 1
        for field, content in [('cover_sha256', text['cover_text']), ('source_excerpt_sha256', text['source_excerpt'])]:
            assert r[field] == hashlib.sha256(content.encode()).hexdigest()
            checks['content_hashes'] += 1
        for observation in r['observations']:
            exact(observation['cover_span'], text['cover_text'], 'cover_spans')
            if observation['source_span']:
                exact(observation['source_span'], text['source_excerpt'], 'source_spans')
        for omission in r['omissions']:
            exact(omission['source_span'], text['source_excerpt'], 'source_spans')
        for value in r['format']['violations']:
            exact(value['cover_span'], text['cover_text'], 'cover_spans')
    return {'schema': 'recovery86-combined-source-review-validation-v1', 'status': 'passed', 'checks': checks,
            'review_sha256': hashlib.sha256(review_raw).hexdigest(),
            'semantic_judgments_validated': False, 'new_model_calls': 0, 'live_output_reads': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--previous-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = validate(args.inputs, args.review, args.previous_review)
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
