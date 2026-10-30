#!/usr/bin/env python3
"""Validate exact source/text identity and literal spans; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path

def digest(b):
    return hashlib.sha256(b).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--selected-contexts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    review = json.loads(a.review.read_bytes())
    contexts = json.loads(a.selected_contexts.read_bytes())['rows']
    assert len(contexts) == 4
    assert [x['source_order'] for x in contexts] == [1, 4, 7, 10]
    cases = review['cases']
    expected_ids = [f'o{x:03d}-{arm}' for x in [1, 4, 7, 10] for arm in ['stego', 'ordinary']]
    assert [c['case_id'] for c in cases] == expected_ids
    assert len(cases) == review['denominator'] == 8
    count = 0
    text_count = 0
    span_count = 0
    for case in cases:
        source_row = next(x for x in contexts if x['source_order'] == case['source_order'])
        source = source_row['record']['excerpt']
        source_hash = digest(source.encode('utf-8'))
        assert source_hash == source_row['record']['excerpt_sha256'] == case['source_sha256']
        assert case['source_words_registered'] == source_row['record']['excerpt_words']
        assert case['target_words'] == source_row['record']['target_words']
        if not case['text_available']:
            assert case['quality'] is None
            assert case['text_sha256'] is None
            assert case['text_path'] is None
            count += 1
            continue
        path = (a.artifact_root / case['text_path']).resolve()
        assert path.is_relative_to(a.artifact_root.resolve())
        raw = path.read_bytes()
        text = raw.decode('utf-8', errors='strict')
        assert digest(raw) == case['text_sha256']
        assert len(raw) == case['utf8_bytes']
        assert len(text.split()) == case['words_by_whitespace']
        quality = case['quality']
        assert quality['unfinished_ending_assistant_judgment'] is None or isinstance(quality['unfinished_ending_assistant_judgment'], bool)
        assert quality['ending_reason_ru']
        assert quality['ends_with_terminal_punctuation'] == bool(text.rstrip() and text.rstrip()[-1] in '.!?')
        documents = {'source': (source, source_hash), 'text': (text, case['text_sha256'])}
        refs = []
        for group in quality['annotation_groups']:
            assert group['label'] in {'source_supported', 'source_unsupported', 'source_conflict', 'nonfactual_or_evaluative', 'unclear_or_incomplete'}
            assert group['text_spans']
            if group['label'] in {'source_supported', 'source_conflict'}:
                assert group['source_spans']
            refs.extend(group['text_spans'])
            refs.extend(group['source_spans'])
        for group in quality['source_omission_groups']:
            refs.extend(group['source_spans'])
        for item in quality['format_violations']:
            refs.extend(item['text_spans'])
        refs.append(quality['ending_exact_span'])
        for span in refs:
            doc, doc_hash = documents[span['document']]
            start, end = span['start'], span['end']
            assert 0 <= start < end <= len(doc)
            assert doc[start:end] == span['quote']
            assert span['document_sha256'] == doc_hash
            assert digest(span['quote'].encode('utf-8')) == span['quote_sha256']
            assert span['utf8_start'] == len(doc[:start].encode('utf-8'))
            assert span['utf8_end'] == len(doc[:end].encode('utf-8'))
            span_count += 1
        text_count += 1
        count += 1
    result = {'schema': 'framed-interval-public-context4-v1-quality-span-validation',
              'review_sha256': digest(a.review.read_bytes()),
              'selected_contexts_sha256': digest(a.selected_contexts.read_bytes()),
              'status': 'passed', 'cases_validated': count, 'denominator': 8,
              'texts_available': text_count, 'literal_spans_validated': span_count,
              'nn_calls': 0, 'scope': 'Identity, counts and exact literal spans only; semantic judgments are not proved by this validator.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))

if __name__ == '__main__':
    main()
