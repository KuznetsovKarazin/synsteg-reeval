#!/usr/bin/env python3
"""Materialize and verify a post-hoc assistant review of EVERY decoded stego.

The judgments below were authored from the whole cover, stego, and supplied
source. This script checks provenance, exact spans, all changes and denominators;
it does not automate semantic judgment or call a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'results/long-summary-development-qwen15b-gguf-v1'
CAP = STAGE + '/capacity-audit'
SOURCE = 'data/development-full-revisions-v3/source-windows.jsonl'
DICTIONARY = 'data/shared_dictionary_recovery_v1.json'
FREEZE = '02ac3ff7086a2da3ebfa95e810f0bfd96c2c87af'
GEO = 'wikipedia-panel-recovery-v1-geography-educational-500-development-01'
HISTORY = 'wikipedia-panel-recovery-v1-history-popular_science-500-development-01'
FLAGS = ('definite_meaning_change', 'definite_grammar_damage', 'ambiguous_or_register_risk')

# Each line explicitly covers a saved changed token, including harmless changes.
# Flags refer ONLY to incremental damage versus the generated cover.
GEO_COMMON = [
    (2, 'situated', 'located', (False, False, False), 'Both denote the same Houston location in this sentence; no new issue identified.'),
    (8, 'Coast,', 'Seacoast,', (False, False, True), 'Coast and seacoast are near synonyms, but changing the conventional source region label Texas Gulf Coast creates a naming/register risk.'),
    (27, 'entirely', 'totally', (False, False, False), 'The extent modifier is preserved; the pre-existing acres/square-miles error is not created by this replacement.'),
    (40, 'part', 'portion', (False, True, False), 'The idiom as part of permits bare part; as portion of leaves the singular count noun portion without a needed determiner.'),
    (45, 'Plain', 'Apparent', (True, False, False), 'The named geographical plain is replaced by an unrelated apparent/obvious sense; Gulf Coastal Apparent no longer identifies the stated landform.'),
    (52, 'category.', 'class.', (False, False, False), 'Class preserves the intended classification sense here; no new issue identified.'),
    (68, 'still', 'motionless', (True, True, False), 'Temporal continuation still is replaced by physical immobility; the adjective motionless cannot fill this position in can still be found.'),
    (86, 'elevation', 'raising', (True, False, False), 'The measured height elevation becomes an act/process of raising, changing the referent of the value that varies.'),
    (93, 'level', 'degree', (True, False, False), 'Sea level is the height datum; sea degree is not an equivalent datum.'),
    (154, 'city.', 'metropolis.', (False, False, False), 'Houston remains the referent and metropolis is a contextual large-city synonym; no new issue identified.'),
    (182, 'Center', 'Middle', (True, False, False), 'The named Texas Medical Center becomes Texas Medical Middle, corrupting the location name stated in the source.'),
    (190, 'city', 'metropolis', (False, False, False), 'The same Houston referent remains; no new issue identified.'),
    (192, 'located', 'situated', (False, False, False), 'The geographical-location relation is preserved.'),
    (204, 'border,', 'surround,', (True, False, False), 'Louisiana border denotes a state boundary; surround does not express that boundary in this distance statement.'),
]
GEO8 = sorted(GEO_COMMON + [
    (112, 'level.', 'degree.', (True, False, False), 'The height datum sea level is again replaced by non-equivalent sea degree; any already unsupported low elevation is a separate cover issue.'),
])
GEO16 = sorted(GEO_COMMON + [
    (150, 'large', 'big', (False, False, True), 'Owned in big parts is less idiomatic and more colloquial than large parts; the ownership statement already exists in the cover.'),
    (245, 'city', 'metropolis', (False, False, False), 'The Houston referent is unchanged; no new issue identified.'),
    (268, 'lack', 'deficiency', (False, False, True), 'Deficiency can suggest an inadequate amount rather than absence and has an awkward evaluative connotation for seismic activity; the original cover already overstates the source.'),
    (280, 'fault', 'mistake', (True, False, False), 'A geological fault is replaced by a mistake; historically active mistake lines no longer denotes the mapped geological structures.'),
    (285, 'long,', 'hanker,', (True, True, False), 'The length adjective in 149 miles long becomes the verb hanker, removing the measurement relation and breaking the phrase.'),
    (288, 'Long', 'Hanker', (True, False, False), 'The Long Point-Eureka Heights fault-system name is corrupted to Hanker Point-Eureka Heights.'),
    (291, 'fault', 'error', (True, False, False), 'A geological fault system becomes an error system, selecting the wrong sense of fault.'),
    (299, 'city.', 'metropolis.', (False, False, False), 'The same Houston referent remains; no new issue identified.'),
])
AMAZON8 = [
    (7, 'primarily', 'mainly', (False, False, False), 'The focus qualifier is preserved.'),
    (19, 'often', 'frequently', (False, False, False), 'The frequency qualifier is preserved.'),
    (34, 'world"', 'cosmos"', (True, False, False), 'The quoted economic/cultural scope is changed from the world to the cosmos/universe; the direct quotation is also altered.'),
    (99, 'Whole', 'Altogether', (True, False, False), 'Whole Foods is a company name; Altogether Foods does not preserve that named entity.'),
    (101, 'Market,', 'Marketplace,', (True, False, False), 'The Market component of the named Whole Foods Market is altered; a common-noun synonym does not preserve the exact company name.'),
    (114, 'framework,"', 'model,"', (False, False, True), 'Framework and model may convey a similar concept, but the attributed regret minimization framework phrase is directly quoted and its wording is changed.'),
    (122, 'suitable', 'suited', (False, False, False), 'Products suited for is a grammatical paraphrase of products suitable for here.'),
    (126, 'market,', 'marketplace,', (False, False, False), 'Online marketplace retains the relevant trading-market sense in this sentence.'),
    (144, 'print.', 'publish.', (True, True, False), 'The in-print availability state is replaced by the verb publish after in; availability in publish is malformed and loses the publication-status meaning.'),
    (179, 'further', 'farther', (False, False, True), 'Farther is less idiomatic for figurative development and can suggest physical distance; definite grammar damage is not asserted.'),
    (185, 'integrating', 'integration', (False, True, False), 'By integrating Amazon Go is replaced by by integration Amazon Go; the nominalization requires of or a different construction.'),
    (198, 'profit', 'benefit', (True, False, False), 'Profit margin is a financial measure; benefit margin is not the same technical term. The already unsupported massive/pre-bubble profit claim is not attributed to this replacement.'),
    (218, 'Phone', 'Telephone', (True, False, False), 'The named Fire Phone product becomes Fire Telephone; the source product name is not preserved.'),
    (223, 'integration,', 'integrating,', (False, True, False), 'The noun phrase the Twitch integration becomes the Twitch integrating, an ill-formed nominal construction in this list.'),
    (224, 'eventually', 'finally', (False, False, False), 'The eventual culmination sense is preserved; pre-existing chronology problems remain separate.'),
    (226, 'Whole', 'Altogether', (True, False, False), 'The second occurrence of Whole Foods is again corrupted to Altogether Foods.'),
    (232, 'part', 'portion', (False, True, False), 'The idiom as part of is replaced with bare as portion of; portion requires a determiner here.'),
]
MANUAL = {(GEO, 'base_v1', 'plain_length16__8'): GEO8,
          (GEO, 'base_v1', 'plain_length16__16'): GEO16,
          (HISTORY, 'base_v1', 'plain_length16__8'): AMAZON8}
BASELINE_NOTES = {
    GEO: [
        {'cover_quote': '579.4 acres', 'source_quotes': ['579.4 sqmi'],
         'reason': 'The area unit already disagrees with the supplied source in the cover, before encoding.'},
        {'cover_quote': 'Salt, which is denser than other sediments',
         'source_quotes': ['Since salt is more buoyant than other sediments'],
         'reason': 'The cover already reverses the supplied source buoyancy relation; this passage is unchanged by the reviewed substitutions.'},
    ],
    HISTORY: [
        {'cover_quote': 'In March 1995, Amazon opened', 'source_quotes': ['On July 16, 1995, Amazon opened'],
         'reason': 'The opening date discrepancy already exists in the cover.'},
        {'cover_quote': 'achieving a massive profit margin before experiencing challenges during the dot-com bubble collapse',
         'source_quotes': ['In 2001, the dot-com bubble burst', 'The company finally turned its first profit in the fourth quarter of 2001', 'though extremely modest'],
         'reason': 'The cover already reverses the first-profit chronology and calls the margin massive. Profit-to-benefit is a separate new terminology injury.'},
    ],
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def formatted(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def lines(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def unique(rows, key):
    result = {key(row): row for row in rows}
    if len(rows) != len(result):
        raise ValueError('Duplicate evidence identity')
    return result


def span(text, index, radius):
    tokens = list(re.finditer(r'\S+', text))
    if not 0 <= index < len(tokens):
        raise ValueError('Token index outside exact text')
    left, right = max(0, index - radius), min(len(tokens) - 1, index + radius)
    a, b = tokens[left].start(), tokens[right].end()
    return {'start': a, 'end': b, 'text': text[a:b], 'first_token_index': left, 'last_token_index': right}


def build(root):
    root = Path(root).resolve()
    paths = [CAP + '/' + n for n in ('records.jsonl', 'arms.jsonl', 'summary.json', 'execution.json')]
    paths += [STAGE + '/capacity-protocol.json', STAGE + '/plan.json', SOURCE, DICTIONARY]
    hashes = {p: sha((root / p).read_bytes()) for p in paths}
    execution = read(root / CAP / 'execution.json')
    protocol = read(root / STAGE / 'capacity-protocol.json')
    if execution['frozen_commit'] != FREEZE or execution['capacity_protocol_sha256'] != hashes[STAGE + '/capacity-protocol.json']:
        raise ValueError('Wrong frozen execution/protocol')
    if set(execution['output_sha256']) != {'records.jsonl', 'arms.jsonl', 'summary.json'}:
        raise ValueError('Incomplete capacity execution manifest')
    for name, digest in execution['output_sha256'].items():
        if hashes[CAP + '/' + name] != digest:
            raise ValueError('Changed capacity output')
    for name, digest in protocol['source_files'].items():
        if name in hashes and hashes[name] != digest:
            raise ValueError('Changed preregistered source')
    raw_dictionary = read(root / DICTIONARY)
    classes = unique(raw_dictionary['classes'], lambda row: row['id'])
    records = lines(root / CAP / 'records.jsonl')
    by_record = unique(records, lambda row: row['record_id'])
    arms = lines(root / CAP / 'arms.jsonl')
    by_arm = unique(arms, lambda a: (a['record_id'], a['method'], a['arm_id']))
    sources = unique(lines(root / SOURCE), lambda row: row['record_id'])
    plan = unique(read(root / STAGE / 'plan.json')['records'], lambda row: row['record_id'])
    if len(records) != 20 or len(arms) != 180 or set(by_record) != set(plan):
        raise ValueError('Full20-record/180-arm denominator required')
    reconstructed = []
    for row in records:
        for method, entry in row['methods'].items():
            for arm_id, arm in entry['arms'].items():
                metadata = {k: row[k] for k in ('audit_record_id', 'record_id', 'plan_index', 'source_group_id', 'planned_action', 'generation_status', 'availability', 'text_sha256')}
                reconstructed.append({**metadata, 'method': method, 'arm_id': arm_id, 'capacity_bits': entry['capacity_bits'], **arm})
    # Structured JSON sorts object keys; compare exact arm identities rather
    # than incidental dictionary iteration order after parsing.
    if unique(reconstructed, lambda a: (a['record_id'], a['method'], a['arm_id'])) != by_arm:
        raise ValueError('Flat180 arms do not exactly match saved20 structured records')
    decoded = {key: row for key, row in by_arm.items() if row['status'] == 'decoded'}
    if set(decoded) != set(MANUAL):
        raise ValueError('Review must cover every and only decoded arm; no manual subset')
    if sum(a['planned_action'] == 'generate_once' for a in arms) != 72:
        raise ValueError('All72 selected arms required')
    reviews = []
    for key, arm in decoded.items():
        rid, method, arm_id = key
        row, src, selected = by_record[rid], sources[rid], plan[rid]
        cover, stego, source = row['text'], arm['stego_text'], src['source_window']
        if (src['split'] != 'development' or row['split'] != 'development' or src['source_group_id'] != row['source_group_id']
                or sha(source.encode()) != selected['source_window_sha256']
                or sha(cover.encode()) != row['text_sha256'] or row['text_sha256'] != arm['text_sha256']
                or sha(stego.encode()) != arm['stego_sha256'] or not arm['mechanical_success']):
            raise ValueError('Source/cover/stego/identity hash mismatch')
        before, after = cover.split(), stego.split()
        if len(before) != len(after):
            raise ValueError('Unexpected token-count change')
        changes = [{'token_index': i, 'original_token': a, 'stego_token': b} for i, (a, b) in enumerate(zip(before, after)) if a != b]
        if changes != arm['token_changes'] or len(changes) != arm['changed_slots']:
            raise ValueError('Saved changes are not the full exact token diff')
        manual = MANUAL[key]
        if [(x['token_index'], x['original_token'], x['stego_token']) for x in changes] != [(i, a, b) for i, a, b, _, _ in manual]:
            raise ValueError('Manual review omitted or altered a changed token')
        slots = unique(row['methods'][method]['analysis']['slots'], lambda s: s['token_index'])
        change_reviews = []
        for i, old, new, flags, reason in manual:
            slot = slots[i]
            words = classes[slot['class_id']]['words']
            old_core = re.fullmatch(r'[^A-Za-z]*([A-Za-z]+)[^A-Za-z]*', old).group(1)
            new_core = re.fullmatch(r'[^A-Za-z]*([A-Za-z]+)[^A-Za-z]*', new).group(1)
            if (old_core.lower() not in words or new_core.lower() not in words
                    or cover[slot['start']:slot['end']] != slot['surface']):
                raise ValueError('Changed token/class/source offsets do not match dictionary evidence')
            change_reviews.append({'token_index': i, 'original_token': old, 'stego_token': new,
                'dictionary_class_id': slot['class_id'], 'complete_class_words': words,
                'cover_token_span': span(cover, i, 0), 'stego_token_span': span(stego, i, 0),
                'cover_context_span': span(cover, i, 4), 'stego_context_span': span(stego, i, 4),
                **dict(zip(FLAGS, flags)), 'reason': reason,
                'damage_scope': 'incremental_from_replacement_only', 'no_flag_is_certification': False})
        baseline = BASELINE_NOTES[rid]
        for note in baseline:
            if note['cover_quote'] not in cover or any(quote not in source for quote in note['source_quotes']):
                raise ValueError('Pre-existing-error quotation is not literal supplied-source evidence')
        aggregate = {flag: any(r[flag] for r in change_reviews) for flag in FLAGS}
        reviews.append({'review_id': rid + '::' + method + '::' + arm_id,
            'record_id': rid, 'source_group_id': row['source_group_id'], 'plan_index': row['plan_index'],
            'method': method, 'arm_id': arm_id, 'arm_sha256': sha(canonical(arm)),
            'source_window_sha256': sha(source.encode()), 'cover_sha256': row['text_sha256'], 'stego_sha256': arm['stego_sha256'],
            'dictionary_file_sha256': hashes[DICTIONARY], 'public_dictionary_sha256': protocol['public_dictionary_sha256'][method],
            'source_window': source, 'cover_text': cover, 'stego_text': stego,
            'whole_source_cover_and_stego_read': True, 'saved_token_changes': arm['token_changes'],
            'all_changed_tokens_reviewed': len(change_reviews), 'change_reviews': change_reviews,
            'incremental_ratings': aggregate,
            'preexisting_cover_issue_examples_not_exhaustive': baseline,
            'preexisting_cover_issues_excluded_from_incremental_ratings': True,
            'mechanically_decoded': True, 'quality_success_established': False})
    review_index = {(r['record_id'], r['method'], r['arm_id']): r for r in reviews}
    ledger = []
    for arm in arms:
        key = arm['record_id'], arm['method'], arm['arm_id']
        review = review_index.get(key)
        ledger.append({k: arm[k] for k in ('record_id', 'source_group_id', 'plan_index', 'method', 'arm_id', 'planned_action', 'status')} |
            {'review_status': 'all_changes_reviewed' if review else 'not_decoded_not_quality_rated',
             'review_id': review['review_id'] if review else None, 'ratings': review['incremental_ratings'] if review else None})
    if sum(r['ratings'] is None for r in ledger) != 177:
        raise ValueError('Every177 nondecoded arm must retain null ratings')
    counts = {r['review_id']: {**{flag: sum(c[flag] for c in r['change_reviews']) for flag in FLAGS},
        'no_new_issue_identified_not_certified': sum(not any(c[f] for f in FLAGS) for c in r['change_reviews'])} for r in reviews}
    if any(sha((root / p).read_bytes()) != h for p, h in hashes.items()):
        raise ValueError('Evidence changed during review reconstruction')
    return {'schema': 'synsteg-qwen15b-all-decoded-stego-review-v1', 'review_type': 'assistant_not_human',
        'scope': 'post_hoc_incremental_damage_audit_of_all_decoded_outputs',
        'frozen_generation_and_capacity_commit': FREEZE,
        'source_files_sha256': hashes, 'script_sha256': sha(Path(__file__).read_bytes()),
        'denominators': {'planned_records': 20, 'selected_generation_records': 8, 'all_arms': 180,
            'selected_arms': 72, 'source_gate_arms': 45, 'not_selected_arms': 63,
            'decoded_and_fully_reviewed_arms': 3, 'nondecoded_null_ratings': 177,
            'distinct_decoded_covers_and_source_groups': 2, 'changed_tokens_all_decoded': 54},
        'all180_mechanical_outcomes': dict(Counter(a['status'] for a in arms)),
        'all_decoded_incremental_flags': {flag: sum(r['incremental_ratings'][flag] for r in reviews) for flag in FLAGS},
        'per_arm_changed_token_flag_counts_descriptive_only': counts,
        'all180_arm_review_ledger': ledger, 'records': reviews,
        'methods': {'whole_text_read': True, 'every_saved_token_change_retained': True,
            'compared_against_provided_source_only': True, 'source_cover_A_or_B_reviews_read': False,
            'external_truth_validation': False, 'new_model_calls': 0, 'paid_calls': 0, 'closed_test_opened': False,
            'secondary_check': 'A second assistant read the same three texts separately. Aggregate findings and first-arm details were exchanged before this primary artifact was saved; no blinded independent replication claim.'},
        'limitations': ['These are assistant judgments, not human validation or external fact checking.',
            'The three decoded arms share two covers and source groups; counts are descriptive and not independent observations.',
            'No-flag changes are not certified safe. Pre-existing cover issues are described separately and are not newly caused by encoding.',
            'All decoded outputs are included; the177 nondecoded arms are unreviewed with null quality ratings, not quality failures or successes.',
            'The raw v1 reference is known unsafe. Mechanical decoding of a frame does not establish a successful steganographic message.'],
        'semantic_safety_certified': False, 'quality_success_claimed': False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args(argv)
    root = a.root.resolve(); output = a.output or root / STAGE / 'stego-review.json'
    value = build(root); data = formatted(value)
    if a.verify:
        if output.read_bytes() != data:
            raise ValueError('Saved review differs from exact reconstruction')
    else:
        with output.open('xb') as stream:
            stream.write(data)
    print(json.dumps({'verified' if a.verify else 'saved': str(output), 'sha256': sha(data),
        'denominators': value['denominators'], 'incremental_flags': value['all_decoded_incremental_flags'],
        'changed_token_counts': value['per_arm_changed_token_flag_counts_descriptive_only']}, indent=2))


if __name__ == '__main__':
    main()
