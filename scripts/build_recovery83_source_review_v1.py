#!/usr/bin/env python3
"""Render the completed whole-text observer review83 from retained inputs.

Annotations are assistant judgments fixed below, not an automated fact checker.
"""
import argparse
import hashlib
import json
from pathlib import Path


def span(text, quote):
    if text.count(quote) != 1:
        raise ValueError('Annotation quote must occur exactly once')
    start = text.index(quote)
    return {'start': start, 'end': start + len(quote), 'text': quote}


def build(path):
    raw = Path(path).read_bytes()
    snapshot = json.loads(raw)
    if snapshot['source_orders'] != list(range(83, 101)):
        raise ValueError('Require all18 fixed rows')
    row = snapshot['records'][0]
    if row['source_order'] != 83 or row['availability'] != 'saved_completed_cover':
        raise ValueError('Require saved completed83')
    source, cover = row['source_excerpt'], row['cover_text']
    observations = []
    def add(ident, label, quote, explanation, source_quote=None):
        observations.append({'id': ident, 'label': label, 'cover_span': span(cover, quote),
                             'source_span': span(source, source_quote) if source_quote else None,
                             'source_basis': 'no_basis_in_supplied_excerpt' if source_quote is None else 'quoted_source',
                             'explanation_ru': explanation})
    add('U1', 'source_unsupported', 'Inside the Grand Hall of the National Museum of Korea, the bustling hubbub of tourists and staff bustles around the room.',
        'Музей Кореи, Grand Hall, посетители и сотрудники отсутствуют в excerpt о пьесе Henry V.')
    add('U2', 'source_unsupported', 'The walls are adorned with intricate murals depicting Korean history and culture. In the center of the hall, a massive statue of King Sejong the Great stands, tall and imposing, with his arms extended out to welcome visitors.',
        'Росписи, статуя Седжона и её поза описаны без основания в данном источнике.')
    add('U3', 'source_unsupported', 'Standing in front of the statue, you are greeted by a brightly lit exhibit titled "The Life of King Sejong the Great."',
        'Название и наличие экспозиции о Седжоне не даны в источнике; это другой предмет текста.')
    add('U4', 'source_unsupported', 'The exhibit showcases the life and accomplishments of Sejong, the seventeenth-century king who played a key role in transforming the Joseon Dynasty into the modern era. The exhibit is divided into three sections: his early life, his reign, and his legacy.',
        'Биография, XVII век, роль в истории Чосона и три раздела экспозиции не подтверждены excerpt.')
    add('U5', 'source_unsupported', "You learn about the difficult times Sejong faced growing up in a poor family, but his determination to succeed never wavered. You're introduced to the man himself through his letters and diaries, which provide a glimpse into his thought process and how he navigated the complexities of politics and society.",
        'Бедная семья, детские трудности, письма и дневники добавлены без исходной опоры.')
    add('U6', 'source_unsupported', "The second section of the exhibit focuses on Sejong's reign as the nineteenth-century Joseon dynasty's first king.",
        'XIX век и статус первого короля Чосона отсутствуют в excerpt. Внешняя истинность здесь не проверялась.')
    add('U7', 'source_unsupported', "You learn about his policies and reforms, such as the development of the postal system, the establishment of a constitutional monarchy, and the introduction of Confucianism to the court. You also learn about the challenges he faced, such as the Seven Years' War and the outbreak of war with Japan.",
        'Перечень реформ, конфуцианство и войны не основаны на supplied excerpt.')
    add('U8', 'source_unsupported', 'You learn about his contributions to education and science, such as the establishment of the National Institute of Education and the development of the first Korean alphabet. You also learn about his efforts to promote cultural exchange and dialogue between Korea and other parts of the world.',
        'Названный институт, алфавит и культурный обмен не подтверждены источником.')
    add('U9', 'unsupported_evaluation', "The Life of King Sejong the Great is a must-see exhibit for anyone interested in Korean history or the Joseon dynasty. The exhibit is engaging, informative, and beautifully presented, making it an experience you won't forget.",
        'Рекомендация и оценка качества предполагаемой экспозиции не имеют основания в excerpt; это оценочные, а не независимо проверенные фактические утверждения.')
    omissions = []
    for ident, quote, explanation in [
        ('O1', 'The Life of Henry the Fifth, often shortened to Henry V, is a history play by William Shakespeare', 'Не переданы предмет источника, название пьесы и авторство Шекспира.'),
        ('O2', 'believed to have been written circa 1599', 'Не передана ориентировочная дата написания и её оговорка.'),
        ('O3', 'It tells the story of King Henry V of England', 'Не передан сюжетный субъект: Генрих V Английский.'),
        ('O4', "focusing on events immediately before and after the Battle of Agincourt (1415) during the Hundred Years' War", 'Не передан исторический фокус пьесы и дата Азенкура.'),
        ('O5', 'In the First Quarto text, it was titled The Cronicle History of Henry the fift , and The Life of Henry the Fifth in the First Folio text.', 'Не переданы два ранних текстовых издания/названия.')]:
        omissions.append({'id': ident, 'label': 'source_fact_omitted', 'source_span': span(source, quote),
                          'cover_span': None, 'explanation_ru': explanation})
    absent = []
    for term in ['Henry V', 'Henry the Fifth', 'Shakespeare', '1599', 'Agincourt', '1415', 'Hundred Years', 'First Quarto', 'First Folio']:
        if term.casefold() in cover.casefold():
            raise ValueError('Expected literal absence changed')
        absent.append({'needle': term, 'case_sensitive': False, 'occurrences_in_cover': 0})
    internal = {'label': 'internal_chronology_inconsistency',
                'cover_spans': [span(cover, 'Sejong, the seventeenth-century king'),
                                span(cover, "Sejong's reign as the nineteenth-century Joseon dynasty's first king")],
                'explanation_ru': 'Один и тот же Седжон внутри cover описан как король XVII века и в связи с царствованием в XIX веке. Это внутренняя несогласованность датировок; внешняя биография Седжона не проверялась.'}
    review = {'source_order': 83, 'record_id': row['record_id'], 'complete_source_and_cover_read': True,
              'record_level_judgment': 'severe_topic_substitution_and_source_unsupported_content',
              'substantive_source_supported_facts_identified': [],
              'source_unsupported_observation_groups': observations, 'omissions': omissions,
              'literal_absence_checks': absent, 'internal_inconsistency': internal,
              'format': {'headings_or_bullets_observed': False, 'mentions_Wikipedia_or_instructions': False,
                         'cover_words_whitespace': row['cover_words_whitespace'], 'target_words': row['target_words'],
                         'length_ratio': row['cover_words_whitespace'] / row['target_words'],
                         'length_judgment': 'Descriptive only: target is phrased as aiming for500, not a hard exact-word requirement.'},
              'attempt_sha256': row['attempt_sha256'], 'cover_sha256': row['cover_sha256'],
              'source_excerpt_sha256': row['source_excerpt_sha256'],
              'primary_commit': row['git_commit'], 'generation_stop_reason': row['stop_reason'],
              'quality_error_rate': None, 'human_gold': False}
    records = []
    for item in snapshot['records']:
        records.append({'source_order': item['source_order'], 'record_id': item['record_id'],
                        'availability_at_snapshot': item['availability'],
                        'quality_review': review if item['source_order'] == 83 else None})
    return {'schema': 'recovery83-source-review-v1', 'review_input_sha256': hashlib.sha256(raw).hexdigest(),
            'review_denominator': 18, 'completed_reviews': 1, 'unavailable_quality_rows': 17,
            'tinyllama_denominator': 100, 'campaign_denominator': 300,
            'historical_unknown_source_orders': [82], 'records': records,
            'scope': 'Assistant whole-source/whole-cover descriptive review of saved83 only; allother17 rows retained null.',
            'observer_identity': '/root/baseline_recovery83',
            'independence': 'Observer authored recovery helper and extraction script; not an independent or blinded semantic assessor.',
            'source_unsupported_is_externally_false': False, 'external_factchecking': False,
            'human_gold': False, 'semantic_error_rate': None, 'model_calls': 0,
            'later_live_outputs_read': False, 'closed_test_accessed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = build(args.inputs)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
