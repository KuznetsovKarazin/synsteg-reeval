#!/usr/bin/env python3
"""Source-bound full-cover84 observer annotations; reuse83 without reanalysis."""
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
    row = snapshot['records'][1]
    reused = copy.deepcopy(old['records'][0]['quality_review'])
    if (snapshot['source_orders'] != list(range(83, 101)) or row['source_order'] != 84
            or row['availability'] != 'saved_completed_cover'
            or reused['cover_sha256'] != snapshot['records'][0]['cover_sha256']
            or reused['source_excerpt_sha256'] != snapshot['records'][0]['source_excerpt_sha256']):
        raise ValueError('Saved inputs or immutable reused83 differ')
    source, cover = row['source_excerpt'], row['cover_text']
    observations = []
    def add(ident, label, quote, explanation, source_quote=None):
        observations.append({'id': ident, 'label': label, 'cover_span': span(cover, quote),
                             'source_span': span(source, source_quote) if source_quote else None,
                             'source_basis': 'quoted_source' if source_quote else 'no_basis_in_supplied_excerpt',
                             'explanation_ru': explanation})
    add('S1', 'source_supported_core', 'chemical compounds can also undergo chemical reactions',
        'Сохранено базовое положение о химических реакциях соединений; поддержка относится только к выделенному core, не к последующим примерам.',
        'A compound can be transformed into a different substance by a chemical reaction')
    add('C1', 'strict_source_conflict', 'They are made up of different molecules that come together to create a unique chemical structure.',
        'Excerpt определяет вещество через множество одинаковых molecules/entities, а cover говорит о разных molecules. Это прямое расхождение определения с supplied source.',
        'A chemical compound is a chemical substance composed of many identical molecules (or molecular entities)')
    add('C2', 'source_conflicting_category_definition', 'The most common type of chemical compound is called a monomer. Monomers are molecules that consist of just one atom of a single element',
        'Cover относит к chemical compounds объекты, которые сам определяет через единственный атом одного элемента. Source требует атомов более чем одного элемента; частотность most common также не поддержана.',
        'containing atoms from more than one chemical element held together by chemical bonds')
    add('U1', 'source_unsupported', 'a key component of modern life that underpins all of our daily products',
        'Универсальность all daily products и прикладной охват в excerpt не заданы.')
    add('U2', 'source_unsupported', 'The first step in creating a chemical compound is to break down one molecule into smaller units. This is called chemical synthesis.',
        'Первый обязательный шаг и определение chemical synthesis добавлены без опоры на excerpt.')
    add('U3', 'source_unsupported', 'Monomers are the building blocks of all other compounds, including polymers and oils.',
        'Универсальное утверждение про все остальные compounds, polymers и oils отсутствует в источнике.')
    add('U4', 'source_unsupported', 'Monomers are combined to form polymers, which are large molecules that consist of many identical units. For example, a monomer called ethylene (also known as the "building block" of plastics) may combine with other monomers to form ethylene glycol (which is used to make polyethylene), polypropylene, and polyethylene glycol.',
        'Описание полимеризации и конкретный ряд ethylene/ethylene glycol/polyethylene/polypropylene/polyethylene glycol не дан в excerpt; внешняя корректность названных реакций здесь не проверялась.')
    add('U5', 'source_unsupported', 'For example, polyethylene, used in plastic bags and food containers, is made up of a series of small polymers connected by chemical bonds.',
        'Состав и области применения polyethylene добавлены без source evidence.')
    add('U6', 'source_unsupported', 'For example, carbon monoxide (CO) is a monomer that combines with oxygen (O2) to create carbon dioxide (CO2). This process is the primary source of energy used to power our homes and vehicles.',
        'Конкретная реакция, классификация CO и primary-source-of-energy claim не основаны на excerpt.')
    add('U7', 'source_unsupported', 'One example of a chemical reaction involves the breaking down of polyethylene (also known as polyethylene terephthalate, or PET) into smaller molecules. When PET is exposed to water, it undergoes a chemical reaction called hydrolysis. This involves the breaking down of the polymer into smaller molecules, which can then be recombined to form new PET.',
        'Пример PET/hydrolysis, заявленная синонимия названий и повторное образование PET отсутствуют в source. Неверность химических отождествлений внешним справочником здесь не устанавливалась.')
    add('U8', 'source_unsupported', 'Another example involves the combustion of fossil fuels, such as coal and oil. These fuels contain carbon monoxide, which combines with oxygen to form carbon dioxide (CO2). This process releases energy and provides heat for your home or vehicle.',
        'Характеристика топлива, состав и энергетический пример не поддержаны данным excerpt.')
    add('U9', 'source_unsupported', 'For example, the monomer ethylene (also known as propene) combines with another monomer, 1,2-propanediol (also known as ethanol), to form ethylene glycol (also known as ethanolamine). This combination produces a compound that is widely used in the production of polyester fibers, which are used in clothing and other consumer products.',
        'Весь набор химических названий, also-known-as отождествлений, реакция и использование polyester добавлены без source evidence; не выдаём внешний фактчек за выполненный.')
    add('U10', 'source_unsupported', 'For example, the monomer xylene (also known as xylopent-2-ene) can be combined with another monomer, ethylbenzene (also known as ethylbenzene-1,3-dione), to form ethylbenzene-1,3-dione (also known as xylene). This combination produces a compound that is widely used in the production of polyethylene.',
        'Синонимии, реакция и использование product для polyethylene не даны в excerpt.')
    add('U11', 'source_unsupported', 'Chemical synthesis can involve a variety of methods, including solid-state reactions, solution chemistry, and homogeneous and heterogeneous reactions.',
        'Методы synthesis перечислены без исходной опоры.')
    headings = ['The Ultimate Guide to Chemical Compounds: Understanding the Science Behind Your Daily Products',
                'The Science Behind Your Daily Products', 'The Science of Chemical Compounds',
                'The Science of Chemical Combinations', 'The Science of Chemical Synthesis']
    heading_spans = []
    for heading in headings:
        # The shorter first subsection title also occurs inside the main title.
        candidates = []
        offset = 0
        for line in cover.splitlines(keepends=True):
            if line.rstrip('\n') == heading:
                candidates.append({'start': offset, 'end': offset + len(heading), 'text': heading})
            offset += len(line)
        if len(candidates) != 1:
            raise ValueError('Heading must be an exact standalone line')
        heading_spans += candidates
    internal = {'label': 'internal_atom_count_and_element_inconsistency', 'cover_spans': [
        span(cover, 'Monomers are molecules that consist of just one atom of a single element'),
        span(cover, '(such as water, which consists of two hydrogen atoms and one oxygen atom)')],
        'explanation_ru': 'В одном предложении приведён пример с двумя атомами hydrogen и одним oxygen после определения just one atom/single element. Это внутренняя несогласованность числа атомов/элементов, устанавливаемая без внешнего справочника.'}
    omissions = [{'id': 'O1', 'label': 'source_fact_not_conveyed',
                  'source_span': span(source, 'A molecule consisting of atoms of only one element, called homonuclear molecule, is therefore not a compound.'),
                  'cover_span': None, 'explanation_ru': 'Явное исключение homonuclear molecules из compounds не передано; новая классификация в cover ему не соответствует.'}]
    review = {'source_order': 84, 'record_id': row['record_id'], 'complete_source_and_cover_read': True,
              'record_level_judgment': 'topic_retained_but_definition_conflicts_and_extensive_unsupported_expansion',
              'observations': observations, 'omissions': omissions, 'internal_inconsistency': internal,
              'format': {'forbidden_heading_spans': heading_spans, 'headings_count': len(heading_spans),
                         'bullets_observed': False, 'mentions_Wikipedia_or_instructions': False,
                         'cover_words_whitespace': row['cover_words_whitespace'], 'target_words': row['target_words'],
                         'length_ratio': row['cover_words_whitespace'] / row['target_words'],
                         'length_judgment': 'Descriptive; prompt aims for500 rather than an exact word count.'},
              'source_excerpt_sha256': row['source_excerpt_sha256'], 'cover_sha256': row['cover_sha256'],
              'attempt_sha256': row['attempt_sha256'], 'primary_commit': row['git_commit'],
              'generation_usage': row['generation_usage'], 'stop_reason': row['stop_reason'],
              'hit_output_limit': row['hit_output_limit'],
              'truncated_mid_sentence_claimed': False, 'quality_error_rate': None, 'human_gold': False}
    records = []
    for item in snapshot['records']:
        records.append({'source_order': item['source_order'], 'record_id': item['record_id'],
                        'availability_at_snapshot': item['availability'],
                        'quality_review': reused if item['source_order'] == 83 else review if item['source_order'] == 84 else None,
                        'annotation_reused_without_reanalysis': item['source_order'] == 83})
    return {'schema': 'recovery84-source-review-v1', 'review_input_sha256': digest(raw),
            'previous_review_sha256': digest(old_raw), 'previous_review_path': 'results/recovery83-100-text-review-v1-after083/review.json',
            'reused_source_orders': [83], 'newly_reviewed_source_orders': [84],
            'review_denominator': 18, 'completed_reviews': 2, 'unavailable_quality_rows': 16,
            'tinyllama_denominator': 100, 'campaign_denominator': 300, 'historical_unknown_source_orders': [82],
            'records': records, 'observer_identity': '/root/baseline_recovery83',
            'independence': 'Observer wrote recovery/extraction helpers; not blinded, independent or human gold.',
            'external_factchecking': False, 'source_unsupported_is_externally_false': False,
            'semantic_error_rate': None, 'human_gold': False, 'model_calls': 0,
            'later_live_outputs_read': False, 'closed_test_accessed': False}


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
