#!/usr/bin/env python3
"""Full-cover87 annotations and immutable reuse of saved83..86 judgments."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def span(text, quote):
    if text.count(quote) != 1:
        raise ValueError('Quote must occur exactly once')
    start = text.index(quote)
    return {'start': start, 'end': start + len(quote), 'text': quote}


def build(inputs, previous_review):
    raw, old_raw = Path(inputs).read_bytes(), Path(previous_review).read_bytes()
    snapshot, old = json.loads(raw), json.loads(old_raw)
    if (snapshot['source_orders'] != list(range(83, 101))
            or snapshot['availability_counts'] != {'not_attempted': 13, 'saved_completed_cover': 5}):
        raise ValueError('Need exact saved83..87 snapshot and13 unavailable')
    reused = [copy.deepcopy(old['records'][i]['quality_review']) for i in range(4)]
    for annotation, row in zip(reused, snapshot['records'][:4]):
        if annotation['cover_sha256'] != row['cover_sha256'] or annotation['source_excerpt_sha256'] != row['source_excerpt_sha256']:
            raise ValueError('Reused content changed')
    row = snapshot['records'][4]
    source, cover = row['source_excerpt'], row['cover_text']
    observations = []
    def add(ident, label, quote, explanation, source_quote=None):
        observations.append({'id': ident, 'label': label, 'cover_span': span(cover, quote),
                             'source_span': span(source, source_quote) if source_quote else None,
                             'explanation_ru': explanation})
    add('S1', 'source_supported_core', 'which is a monthly peer-reviewed, open access, scientific journal published by Wiley-VCH on behalf of Chemistry Europe.',
        'Сохранены frequency/monthly, peer review, open access, Wiley-VCH и Chemistry Europe.',
        'ChemistryOpen is a monthly peer-reviewed, open access, scientific journal covering all areas of chemistry and related fields. It is published by Wiley-VCH on behalf of Chemistry Europe.')
    add('S2', 'source_supported', 'The Journal Citation Reports lists ChemistryOpen as having a 2025 impact factor of 3.',
        'Число3,2025 и attribution Journal Citation Reports переданы из supplied source; это не проверка текущей внешней метрики журнала.',
        'According to the Journal Citation Reports, the journal has a 2025 impact factor of 3')
    add('S3', 'source_supported', 'ChemistryOpen is an open access journal that publishes peer-reviewed research across all areas of chemistry and related fields.',
        'Сохранён заявленный охват chemistry and related fields.',
        'ChemistryOpen is a monthly peer-reviewed, open access, scientific journal covering all areas of chemistry and related fields.')
    add('I1', 'term_interpretation_not_independent_factcheck', 'ChemistryOpen offers its readers access to high-quality, peer-reviewed content that is freely available to the public.',
        'Свободная доступность для читателя — понятная интерпретация open access. Она не доказывает отсутствие платы за публикацию у авторов; high-quality также не устанавливается одним peer review. Внешние условия журнала не проверялись.', 'open access')
    add('U1', 'source_unsupported_evaluative_overinterpretation', 'This indicates that the journal is considered one of the most impactful in the category "Chemistry, Multidisciplinary."',
        'Оценка one of the most impactful не дана в excerpt; вместо неё source сообщает rank124/250. Без заранее заданного порога это не объявляется строгим числовым противоречием, но является неподтверждённым усилением.',
        'ranking it 124th out of 250 journals in the category "Chemistry, Multidisciplinary"')
    add('U2', 'source_unsupported_author_cost_claim', 'This open access model has several advantages, including allowing scholars to access and disseminate their work without being restricted by publishing costs, and increasing the reach and impact of research.',
        'Из source open access не следуют отсутствие publishing-cost ограничений для dissemination и измеренный рост reach/impact; специальных данных о costs/effects нет.')
    add('U3', 'source_unsupported_specific_contributor_claim', 'The journal includes articles from chemists, biologists, engineers, physicists, and other professionals working in these fields. This provides a platform for interdisciplinary collaboration and exchange of ideas.',
        'Source coverage all chemistry and related fields не перечисляет фактические профессии авторов и не измеряет collaboration. Это правдоподобное расширение, но не подтверждённый excerpt факт.')
    add('U4', 'source_unsupported_no_author_fee_claim', 'It provides free access to its articles to the public and does not charge authors or their institutions for publication.',
        'Утверждение does not charge authors/institutions не содержится в source. Free reader access и отсутствие author publication fees — разные условия. Внешний актуальный APC/funding policy журнала здесь не проверялся.')
    add('U5', 'source_unsupported_evaluative_expansion', "ChemistryOpen's commitment to open access publishing is a model that other journals can learn from and emulate.",
        'Рекомендация другим журналам и образцовый статус не даны в excerpt; это оценочное добавление.')
    omission = {'id': 'O1', 'label': 'source_numeric_rank_omitted',
                'source_span': span(source, 'ranking it 124th out of 250 journals'), 'cover_span': None,
                'explanation_ru': 'Конкретное место124 из250 не перенесено, хотя category сохранена.'}
    instruction = span(cover, 'In English, an accessible and comprehensive encyclopedic explanation can be provided for the ChemistryOpen journal, aiming for 250 words.')
    absence = []
    for term in ['124th', '124']:
        if term in cover:
            raise ValueError('Expected rank marker absence changed')
        absence.append({'needle': term, 'case_sensitive': True, 'occurrences_in_cover': 0})
    review = {'source_order': 87, 'record_id': row['record_id'], 'complete_source_and_cover_read': True,
              'record_level_judgment': 'core_preserved_but_author_fee_claim_and_rank_overinterpretation_unsupported',
              'observations': observations, 'omissions': [omission], 'literal_absence_checks': absence,
              'format': {'violations': [{'label': 'writing_instruction_echo_and_unrequested_meta_commentary', 'cover_span': instruction}],
                         'headings_observed': False, 'bullets_observed': False,
                         'cover_words_whitespace': row['cover_words_whitespace'], 'target_words': row['target_words'],
                         'length_ratio': row['cover_words_whitespace'] / row['target_words'],
                         'length_judgment': 'Descriptive only; aiming for250 is not an exact word-count rule.'},
              'source_excerpt_sha256': row['source_excerpt_sha256'], 'cover_sha256': row['cover_sha256'],
              'attempt_sha256': row['attempt_sha256'], 'primary_commit': row['git_commit'],
              'generation_usage': row['generation_usage'], 'stop_reason': row['stop_reason'],
              'quality_error_rate': None, 'external_factchecking': False, 'human_gold': False}
    reviews = reused + [review]
    records = [{'source_order': item['source_order'], 'record_id': item['record_id'],
                'availability_at_snapshot': item['availability'],
                'quality_review': reviews[i] if i < 5 else None,
                'annotation_reused_without_reanalysis': i < 4}
               for i, item in enumerate(snapshot['records'])]
    return {'schema': 'recovery87-combined-source-review-v1', 'review_input_sha256': digest(raw),
            'previous_review_sha256': digest(old_raw),
            'previous_review_path': 'results/recovery83-100-text-review-v1-after086/review.json',
            'reused_source_orders': [83, 84, 85, 86], 'newly_reviewed_source_orders': [87],
            'review_denominator': 18, 'completed_reviews': 5, 'unavailable_quality_rows': 13,
            'tinyllama_denominator': 100, 'campaign_denominator': 300, 'historical_unknown_source_orders': [82],
            'records': records, 'observer_identity': '/root/baseline_recovery83',
            'independence': 'Observer wrote recovery/extraction helpers; not blinded, independent or human gold.',
            'external_factchecking': False, 'source_unsupported_is_externally_false': False,
            'semantic_error_rate': None, 'human_gold': False, 'model_calls': 0,
            'live_outputs_read': False, 'closed_test_accessed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--previous-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = build(args.inputs, args.previous_review)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
