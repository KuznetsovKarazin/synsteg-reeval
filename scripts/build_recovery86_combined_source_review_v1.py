#!/usr/bin/env python3
"""Whole-text85/86 annotations with exact reuse of saved83/84 judgments."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def span(text, quote):
    if text.count(quote) != 1:
        raise ValueError('Quote must occur exactly once: ' + quote[:60])
    start = text.index(quote)
    return {'start': start, 'end': start + len(quote), 'text': quote}


def annotate(row):
    source, cover = row['source_excerpt'], row['cover_text']
    observations, omissions, format_spans = [], [], []
    def add(ident, label, quote, explanation, source_quote=None):
        observations.append({'id': ident, 'label': label, 'cover_span': span(cover, quote),
                             'source_span': span(source, source_quote) if source_quote else None,
                             'source_basis': 'quoted_source' if source_quote else 'no_basis_in_supplied_excerpt',
                             'explanation_ru': explanation})
    def omit(ident, quote, explanation):
        omissions.append({'id': ident, 'label': 'source_detail_omitted',
                          'source_span': span(source, quote), 'cover_span': None, 'explanation_ru': explanation})
    if row['source_order'] == 85:
        add('S1', 'source_supported_core', 'have been passed down orally from the Jain tradition.',
            'Сохранён core устной передачи. Source уточняет initially; эта поддержка не распространяется на любую письменную работу во все периоды.',
            'which was initially transmitted orally')
        add('S2', 'source_supported', 'The earliest surviving literary materials in this tradition are the Jain Agamas, which are composed in the Prakrit (Middle Indo-Aryan) language, Ardhamagadhi.',
            'Сохранены oldest-surviving material, Jain Agamas и Ardhamagadhi/Prakrit Middle Indo-Aryan.',
            'The oldest surviving material is contained in the canonical Jain Agamas, which are written in Ardhamagadhi, a Prakrit (Middle Indo-Aryan) language.')
        add('U1', 'source_unsupported', 'This ancient religious tradition is considered to be one of the oldest and most comprehensive in Indian culture, with its roots dating back to ancient India.',
            'Source говорит vast and ancient literary tradition; сравнение one of the oldest/most comprehensive именно в Indian culture и географическое происхождение не заданы.')
        add('U2', 'source_unsupported', 'The Jain Agamas are believed to have been composed by various Jain sages over a period of centuries, and they are considered to be the primary sources of Jain literature.',
            'Назначение авторства sages, период centuries и primary-sources статус добавлены без source evidence.')
        add('U3', 'source_unsupported', 'These texts are known for their emphasis on the principles of nonviolence, selfless service, and the concept of karma, which are still widely practiced in contemporary Jainism.',
            'Доктрины, акценты текстов и современная распространённость практик отсутствуют в excerpt.')
        add('U4', 'source_unsupported', 'Some of the notable works of Jain literature include the Jaina Tirthankaras, a series of biographies of the Jaina saints, which were compiled in the late first and early second centuries CE. Another important body of literature is the Jaina Puranas, which are written in the Jain Ardhamagadhi language and contain accounts of the Jaina history, philosophy, and religion.',
            'Название/классификация произведений, конкретные века, язык и содержание другого корпуса не даны в источнике.')
        add('U5', 'source_unsupported', 'Jain literature also includes the Jaina Panchasara, which is a textbook of moral and religious instruction, and the Jaina Rigveda, which contains the Jaina hymns. The Jaina Saiva Siddhanta, a philosophical system based on the teachings of the Jaina saints, is another important body of literature in Jainism.',
            'Три названных корпуса и их содержание добавлены без source evidence. Существование/корректность этих названий внешними справочниками здесь не проверялись.')
        add('U6', 'source_unsupported', 'In addition to these texts, Jainism also has a rich tradition of poetry and music, which have been passed down through the generations. Jain music, known as Jaina Natya, is characterized by its simplicity and spiritual content, and it has been performed in various forms throughout history.',
            'Музыка, название Jaina Natya, её характеристики и история исполнения не представлены в excerpt.')
        add('U7', 'source_unsupported', 'Today, Jainism is a vibrant and growing religion, with millions of followers around the world. The teachings of the Jain saints are still held in high regard, and many followers practice selfless service and nonviolence in their daily lives.',
            'Современный рост, количество/география followers и текущие практики не подтверждены данным источником.')
        add('U8', 'source_unsupported_evaluative_expansion', 'In conclusion, Jain literature is a rich and diverse body of works that has played a significant role in shaping Jainism as an ancient and vibrant religious tradition. These texts continue to inspire and inform Jain communities around the world, and they demonstrate the enduring importance of oral traditions in the world of literature and culture.',
            'Влияние на формирование религии и современные мировые communities — оценочное расширение beyond excerpt, не внешний фактчек.')
        omit('O1', 'canonical Jain Agamas', 'Название Jain Agamas передано, но квалификация canonical явно не сохранена; это частичная потеря детали, а не потеря всего факта.')
        format_spans.append({'label': 'forbidden_heading', 'cover_span': span(cover, 'Encyclopedic Explanation of Jain Literature')})
        judgment = 'core_retained_with_unsupported_titles_dates_doctrines_and_demographics'
    elif row['source_order'] == 86:
        add('S1', 'source_supported_core', 'The Arecibo Telescope, built into a natural sinkhole at the Arecibo Observatory in Puerto Rico, is a massive, spherical radio telescope',
            'Сохранены sinkhole/observatory/Puerto Rico/spherical radio telescope. Adjective massive и present is не получают отдельного подтверждения; current operating status здесь не устанавливается.',
            'The Arecibo Telescope was a 305 m spherical reflector radio telescope built into a natural sinkhole at the Arecibo Observatory located near Arecibo, Puerto Rico.')
        add('S2', 'source_supported_core', "The telescope was the world's largest single-aperture telescope for 53 years, until it was surpassed by the Five-hundred-meter Aperture Spherical Telescope (FAST) in Guizhou, China.",
            'Сохранены рекорд53years и FAST/Guizhou/China; July2016 опущено.',
            "Completed in November 1963, the Arecibo Telescope was the world's largest single-aperture telescope for 53 years, until it was surpassed in July 2016 by the Five-hundred-meter Aperture Spherical Telescope (FAST) in Guizhou, China.")
        add('S3', 'source_supported', 'was completed in November 1963',
            'Месяц и год завершения переданы точно.', 'Completed in November 1963')
        add('U1', 'source_unsupported', "The Arecibo Telescope's construction was led by astronomers from the United States and Puerto Rico.",
            'Кто руководил construction и откуда были руководители, source не сообщает.')
        add('U2', 'source_unsupported', "Its primary purpose was to study the universe's largest and most massive celestial objects, such as stars, black holes, and supernovae. The Arecibo Telescope's unique design and location allowed it to receive and focus radio signals from different parts of the electromagnetic spectrum, enabling astronomers to study even the most remote and hidden areas of the sky.",
            'Primary purpose, перечисленные объекты и характеристики наблюдений beyond excerpt.')
        add('U3', 'source_unsupported_major_discovery_claim', "One of the telescope's most significant contributions to astronomy was its discovery of the Higgs boson, a particle that was predicted to exist but had not been observed until 2012.",
            'Source не сообщает об открытии Higgs boson, авторстве telescope или2012. Это крупное неподтверждённое данным источником добавление; внешний фактчек открытия в этой проверке не выполнялся.')
        add('U4', 'source_unsupported', 'The Higgs boson is one of the most fundamental particles in the standard model of particle physics, which describes the fundamental forces of nature. By detecting and studying the Higgs boson, astronomers were able to test the theory of particle physics and confirm its existence.',
            'Standard model, particle-physics выводы и роль astronomers не даны в excerpt; не выдаём внешние знания за проверку source fidelity.')
        add('U5', 'source_unsupported', "In addition to its scientific importance, the Arecibo Telescope played an important role in Puerto Rico's economy and community. The telescope's construction and operation required significant resources, including thousands of workers, contractors, and suppliers. The telescope's presence also brought jobs, tourism, and economic development to the area, particularly in the towns surrounding the observatory.",
            'Масштаб workforce thousands, contractor/supplier состав, tourism/jobs и экономический эффект не подтверждены source.')
        add('U6', 'source_unsupported_evaluative_expansion', 'Today, the Arecibo Telescope remains a treasured symbol of Puerto Rican engineering and astronomy. Its impact on astronomy and the community it served cannot be overstated, and its legacy continues to inspire and inform researchers and engineers today.',
            'Современная символическая роль, оценка cannot be overstated и воздействие на researchers/engineers — дополнительные оценки без основания в excerpt.')
        omit('O1', '305 m spherical reflector', 'Точная апертура/размер305m и qualifier reflector не переданы; spherical сохранено отдельно.')
        omit('O2', 'A cable-mounted, steerable receiver and several radar transmitters for emitting signals were mounted 150 m above the dish.', 'Не передана вся hardware-конструкция с receiver/radar transmitters и150m.')
        omit('O3', 'July 2016', 'Превосходство FAST сохранено, но точная дата July2016 пропущена.')
        format_spans.append({'label': 'forbidden_heading', 'cover_span': span(cover, 'A brief history of the Arecibo Telescope and its impact on astronomy')})
        format_spans.append({'label': 'unrequested_editorial_instruction', 'cover_span': span(cover, '[Add a brief summary or thumbnail image of the Arecibo Telescope, such as its design or location, for visual appeal.]')})
        judgment = 'source_core_retained_but_major_unsupported_discovery_claim_and_editorial_instruction'
    else:
        raise ValueError('Only fixed new85/86 annotations')
    return {'source_order': row['source_order'], 'record_id': row['record_id'],
            'complete_source_and_cover_read': True, 'record_level_judgment': judgment,
            'observations': observations, 'omissions': omissions,
            'format': {'violations': format_spans, 'bullets_observed': False,
                       'cover_words_whitespace': row['cover_words_whitespace'], 'target_words': row['target_words'],
                       'length_ratio': row['cover_words_whitespace'] / row['target_words'],
                       'length_judgment': 'Descriptive only: aiming for250 is not an exact word-count demand.'},
            'source_excerpt_sha256': row['source_excerpt_sha256'], 'cover_sha256': row['cover_sha256'],
            'attempt_sha256': row['attempt_sha256'], 'primary_commit': row['git_commit'],
            'generation_usage': row['generation_usage'], 'stop_reason': row['stop_reason'],
            'quality_error_rate': None, 'external_factchecking': False, 'human_gold': False}


def build(inputs, previous_review):
    raw, old_raw = Path(inputs).read_bytes(), Path(previous_review).read_bytes()
    snapshot, old = json.loads(raw), json.loads(old_raw)
    if (snapshot['source_orders'] != list(range(83, 101))
            or snapshot['availability_counts'] != {'not_attempted': 14, 'saved_completed_cover': 4}):
        raise ValueError('Need exact saved83..86 snapshot and14 unavailable')
    reused = [copy.deepcopy(old['records'][i]['quality_review']) for i in range(2)]
    for annotation, row in zip(reused, snapshot['records'][:2]):
        if (annotation['cover_sha256'] != row['cover_sha256']
                or annotation['source_excerpt_sha256'] != row['source_excerpt_sha256']):
            raise ValueError('Reused83/84 content changed')
    reviews = reused + [annotate(row) for row in snapshot['records'][2:4]]
    records = [{'source_order': row['source_order'], 'record_id': row['record_id'],
                'availability_at_snapshot': row['availability'],
                'quality_review': reviews[i] if i < 4 else None,
                'annotation_reused_without_reanalysis': i < 2}
               for i, row in enumerate(snapshot['records'])]
    return {'schema': 'recovery86-combined-source-review-v1', 'review_input_sha256': digest(raw),
            'previous_review_sha256': digest(old_raw),
            'previous_review_path': 'results/recovery83-100-text-review-v1-after084/review.json',
            'reused_source_orders': [83, 84], 'newly_reviewed_source_orders': [85, 86],
            'review_denominator': 18, 'completed_reviews': 4, 'unavailable_quality_rows': 14,
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
