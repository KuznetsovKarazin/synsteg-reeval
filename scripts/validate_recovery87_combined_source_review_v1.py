#!/usr/bin/env python3
"""Check immutable83..86 reuse, exact87 spans and full18 denominator."""
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
    assert all(annotated['records'][i]['quality_review'] == old['records'][i]['quality_review'] for i in range(4))
    assert all(r['quality_review'] is None for r in annotated['records'][5:])
    checks = {'input_sha': 1, 'previous_review_sha': 1, 'exact83_86_reuse': 4,
              'full18_denominator': 1, 'null_other13': 13, 'cover_spans': 0, 'source_spans': 0,
              'content_hashes': 0, 'literal_absences': 0}
    def exact(value, content, counter):
        assert 0 <= value['start'] < value['end'] <= len(content)
        assert content[value['start']:value['end']] == value['text']
        checks[counter] += 1
    r, text = annotated['records'][4]['quality_review'], source['records'][4]
    assert r['source_order'] == 87 and r['record_id'] == text['record_id']
    assert r['format']['cover_words_whitespace'] == len(text['cover_text'].split()) == 310
    assert r['stop_reason'] == 'end_turn' and r['generation_usage']['output_tokens'] == 455
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
    for value in r['literal_absence_checks']:
        assert value['needle'] not in text['cover_text']
        checks['literal_absences'] += 1
    return {'schema': 'recovery87-combined-source-review-validation-v1', 'status': 'passed', 'checks': checks,
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
