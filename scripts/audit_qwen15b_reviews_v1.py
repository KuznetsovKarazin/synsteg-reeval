#!/usr/bin/env python3
"""Prepare whole-source review materials, validate two reviews, aggregate all20.

This module performs no generation, semantic judgment, NLI or external fact check.
Exact-quote validation checks evidence identity, not whether a judgment is correct.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = 'results/long-summary-development-qwen15b-gguf-v1'
SOURCE = 'data/development-full-revisions-v3/source-windows.jsonl'
RUBRIC = 'results/long-summary-development-v3/review_rubric.json'
CATEGORIES = ('unsupported_claim', 'contradiction', 'language_or_entity_error', 'ambiguous_or_borderline')
CODE = ('scripts/audit_qwen15b_reviews_v1.py', 'tests/test_qwen15b_reviews_v1.py')
INPUTS = (SOURCE, RUBRIC, OUT + '/plan.json', OUT + '/protocol.json', OUT + '/prompts.jsonl',
          'scripts/audit_qwen15b_capacity_v1.py', OUT + '/capacity-protocol.json')
PENDING = {'not_started', 'server_starting', 'started', 'generation_request_started'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def formatted(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(formatted(value))


def make_plan(root):
    return {'schema': 'qwen15b-source-bound-review-plan-v1',
        'generation_freeze_commit': 'f61750c6219e13d61f63000cc2f40bbeb1225bd3',
        'prepared_before_generation': True, 'preparation_model_calls': 0, 'test_data_used': False,
        'input_hashes': {p: digest((root / p).read_bytes()) for p in INPUTS},
        'code_hashes': {p: digest((root / p).read_bytes()) for p in CODE},
        'rubric': RUBRIC, 'categories': list(CATEGORIES),
        'reviewers': ['A', 'B'], 'review_type': 'source_bound_assistant_diagnostic',
        'required_denominators': {'all20': 20, 'planned8': 8, 'source_gates': 5, 'not_selected': 7},
        'source_boundary': 'Entire exact supplied source window, not a short excerpt or external source',
        'output_boundary': 'Entire exact generated cover, not selected sentences or edited text',
        'reviewability': 'Only generated nonempty output with matching declared SHA256; malformed/error/empty/truncated/schema-uncertain outputs retained with null ratings',
        'generation_evidence_validator': 'Frozen capacity.verify + capacity.load_ledger validate all20, raw responses, attempt/cover hashes, native schema/truncation and completed summary; no capacity evaluation is invoked',
        'unavailable_policy': 'All four flags null, no findings, whole_source_read=false, whole_output_read=false; retain every ledger row and raw text if present',
        'review_schema': {'records': 'All20 identities in frozen order', 'hashes': 'Exact source/output SHA256 from materials',
            'reads': 'Both whole-source and whole-output declarations true for available output',
            'flags': 'Four booleans for available output, null for unavailable',
            'findings': 'category, exact nonempty output_quote, exact nonempty source_quote or null for absence, nonempty reason',
            'consistency': 'Every true flag needs at least one finding, each finding requires its category flag; contradiction needs source quote'},
        'aggregation': 'Preserve both reviews and category disagreements for all20; report union/intersection/disagreement counts with all20/planned8/available denominators',
        'independence': 'Two separate reviewer agents read source and cover without seeing the other review; independence is procedural and not proven by schema validation',
        'not_claimed': ['external truth', 'human gold labels', 'no flag as proof of fidelity',
                        'quote identity as proof of interpretation', 'population error rates',
                        'quality success from capacity/roundtrip or NLI', 'stego semantic safety'],
        'separate_stego_review': 'New errors introduced by substitutions require a separate whole-stego/cover/source review; this plan rates covers only',
        'required_before_execution': 'Root Git freeze and remote-byte verification before generation; materials/reviews only after all run ledger outcomes are terminal'}


def verify_plan(root):
    expected = make_plan(root)
    actual = read(root / OUT / 'review-plan.json')
    if actual != expected:
        raise ValueError('Frozen review plan, code or source inputs changed')


def validate_materials(materials):
    if materials.get('schema') != 'qwen15b-review-materials-v1':
        raise ValueError('Invalid materials schema')
    rows = materials.get('records')
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError('All20 review material rows required')
    if len({r['record_id'] for r in rows}) != 20 or [r['plan_index'] for r in rows] != list(range(1, 21)):
        raise ValueError('Unique original ordered identities required')
    if Counter(r['planned_action'] for r in rows) != {'generate_once': 8, 'abstain_source_gate': 5, 'not_selected_fixed_eight_budget': 7}:
        raise ValueError('Required8/5/7 selection denominator differs')
    for row in rows:
        source = row.get('source_text')
        output = row.get('output_text')
        if not isinstance(source, str) or not source.strip() or digest(source.encode()) != row.get('source_sha256'):
            raise ValueError('Invalid whole source/hash')
        if output is not None and not isinstance(output, str):
            raise ValueError('Output must be entire text or null')
        if row.get('output_sha256') != (digest(output.encode()) if output is not None else None):
            raise ValueError('Output hash mismatch')
        if type(row.get('available')) is not bool:
            raise ValueError('Explicit availability boolean required')
        expected_available = (row['planned_action'] == 'generate_once' and row['status'] == 'generated'
                              and isinstance(output, str) and bool(output.strip())
                              and row.get('declared_output_sha256', row['output_sha256']) == row['output_sha256']
                              and row.get('generation_schema_issues') == []
                              and row.get('generation_truncated') is False)
        if row['available'] != expected_available:
            raise ValueError('Availability must follow objective status/text/hash rule, never output quality')


def validate_review(review, materials):
    validate_materials(materials)
    if (review.get('schema') != 'qwen15b-source-bound-review-v1' or review.get('reviewer_id') not in ('A', 'B')
            or review.get('review_type') != 'source_bound_assistant_diagnostic'):
        raise ValueError('Wrong reviewer/schema/type')
    records = review.get('records')
    expected = materials['records']
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError('Every review must retain all20 rows')
    if [r.get('record_id') for r in records] != [m['record_id'] for m in expected]:
        raise ValueError('Review identities/order differ from full materials')
    findings_count = 0
    for row, material in zip(records, expected):
        if any(row.get(k) != material[k] for k in ('source_sha256', 'output_sha256')):
            raise ValueError('Review source/output hash mismatch')
        flags = row.get('flags')
        if not isinstance(flags, dict) or set(flags) != set(CATEGORIES):
            raise ValueError('Exactly four category flags required')
        findings = row.get('findings')
        if not isinstance(findings, list):
            raise ValueError('Findings list required')
        if not material['available']:
            if any(v is not None for v in flags.values()) or findings or row.get('whole_source_read') is not False or row.get('whole_output_read') is not False:
                raise ValueError('Unavailable output requires null ratings and no read/quality claim')
            continue
        if row.get('whole_source_read') is not True or row.get('whole_output_read') is not True:
            raise ValueError('Explicit whole-source and whole-output reading required')
        if any(type(v) is not bool for v in flags.values()):
            raise ValueError('Available category flags must be booleans')
        categories_found = set()
        for finding in findings:
            if not isinstance(finding, dict) or set(finding) != {'category', 'output_quote', 'source_quote', 'reason'}:
                raise ValueError('Finding schema differs')
            category = finding['category']
            if category not in CATEGORIES or flags[category] is not True:
                raise ValueError('Finding category and flag inconsistent')
            quote = finding['output_quote']
            if not isinstance(quote, str) or not quote.strip() or quote not in material['output_text']:
                raise ValueError('Output quote is not an exact nonempty substring')
            quote = finding['source_quote']
            if quote is not None and (not isinstance(quote, str) or not quote.strip() or quote not in material['source_text']):
                raise ValueError('Source quote is not an exact nonempty substring')
            if category == 'contradiction' and quote is None:
                raise ValueError('Contradiction requires exact source evidence')
            if not isinstance(finding['reason'], str) or not finding['reason'].strip():
                raise ValueError('Reason required; absence is explained, not fabricated as a quote')
            categories_found.add(category)
            findings_count += 1
        if categories_found != {k for k, value in flags.items() if value}:
            raise ValueError('Every true flag requires at least one finding')
    return {'reviewer_id': review['reviewer_id'], 'rows': 20,
            'available': sum(m['available'] for m in expected), 'findings': findings_count,
            'exact_quote_identity_verified': True, 'semantic_judgment_validated': False}


def aggregate_reviews(a, b, materials):
    validation = [validate_review(a, materials), validate_review(b, materials)]
    if {a['reviewer_id'], b['reviewer_id']} != {'A', 'B'}:
        raise ValueError('Two distinct A/B reviews required')
    by_reviewer = {a['reviewer_id']: a, b['reviewer_id']: b}
    a, b = by_reviewer['A'], by_reviewer['B']
    rows = []
    counts = {category: {'union': 0, 'intersection': 0, 'disagreements': 0} for category in CATEGORIES}
    for material, ra, rb in zip(materials['records'], a['records'], b['records']):
        disagreements = []
        if material['available']:
            for category in CATEGORIES:
                va, vb = ra['flags'][category], rb['flags'][category]
                counts[category]['union'] += int(va or vb)
                counts[category]['intersection'] += int(va and vb)
                counts[category]['disagreements'] += int(va != vb)
                if va != vb:
                    disagreements.append(category)
        rows.append({'record_id': material['record_id'], 'plan_index': material['plan_index'],
                     'planned_action': material['planned_action'], 'status': material['status'],
                     'available': material['available'], 'source_sha256': material['source_sha256'],
                     'output_sha256': material['output_sha256'], 'flags_A': ra['flags'], 'flags_B': rb['flags'],
                     'disagreements': disagreements, 'findings_A': ra['findings'], 'findings_B': rb['findings']})
    return {'schema': 'qwen15b-review-aggregate-v1', 'records': rows, 'counts': counts,
            'denominators': {'all20': 20, 'planned8': 8, 'available': sum(r['available'] for r in rows)},
            'validation': validation, 'materials_sha256': digest(formatted(materials)),
            'review_A_sha256': digest(formatted(a)), 'review_B_sha256': digest(formatted(b)),
            'no_flag_is_proof_of_fidelity': False, 'external_truth_or_human_validation': False,
            'quote_identity_proves_interpretation': False, 'stego_quality_assessed': False}


def build_materials(root):
    verify_plan(root)
    # Share one frozen evidence contract with mechanical analysis; this only
    # verifies data and never computes capacity or calls a model.
    import audit_qwen15b_capacity_v1 as capacity
    stage = root / OUT
    capacity.verify(root, stage)
    execution = read(stage / 'run/execution.json')
    strict_rows, generation_evidence = capacity.load_ledger(root, stage, execution['freeze_commit'])
    strict_by_id = {r['record_id']: r for r in strict_rows}
    frozen = read(root / OUT / 'plan.json')['records']
    ledger_path = root / OUT / 'run/ledger.json'
    ledger = read(ledger_path)
    sources = {r['record_id']: r for r in (json.loads(s) for s in (root / SOURCE).read_text(encoding='utf-8').splitlines())}
    if not isinstance(ledger, list) or len(ledger) != 20 or [r.get('record_id') for r in ledger] != [r['record_id'] for r in frozen]:
        raise ValueError('Complete ordered all20 generation ledger required')
    if any(r.get('status') in PENDING for r in ledger):
        raise ValueError('Finish or explicitly reconcile interrupted outcomes before preparing independent reviews')
    rows = []
    for planned, result in zip(frozen, ledger):
        if any(result.get(k) != planned[k] for k in ('planned_action', 'plan_index', 'source_window_sha256')):
            raise ValueError('Ledger identity/provenance changed')
        source = sources[planned['record_id']]['source_window']
        output_text = result.get('text') if isinstance(result.get('text'), str) else None
        output_hash = digest(output_text.encode()) if output_text is not None else None
        available = strict_by_id[planned['record_id']]['availability'] == 'available'
        rows.append({'record_id': planned['record_id'], 'plan_index': planned['plan_index'],
            'planned_action': planned['planned_action'], 'status': result.get('status'), 'available': available,
            'reviewability_reason': 'whole_output_available' if available else 'unavailable_or_malformed_generation_outcome',
            'source_text': source, 'source_sha256': digest(source.encode()), 'output_text': output_text,
            'output_sha256': output_hash, 'declared_output_sha256': result.get('text_sha256'),
            'generation_schema_issues': result.get('schema_issues'), 'generation_truncated': result.get('truncated'),
            'attribution': planned['attribution'], 'target_words': planned['target_words']})
    materials = {'schema': 'qwen15b-review-materials-v1', 'records': rows,
        'ledger_sha256': digest(ledger_path.read_bytes()), 'review_plan_sha256': digest((root / OUT / 'review-plan.json').read_bytes()),
        'generation_evidence_sha256': generation_evidence,
        'rubric': read(root / RUBRIC), 'denominator_override': {'all20': 20, 'planned8': 8},
        'instructions': 'Read each available whole source and whole output independently; do not see other reviewer labels. No flag is not proof of fidelity. Unsupported absence uses null source_quote and an explanation.'}
    validate_materials(materials)
    return materials


def prepare_materials(root, output):
    materials = build_materials(root)
    rows = materials['records']
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'materials.json', materials)
    for reviewer in ('A', 'B'):
        template = {'schema': 'qwen15b-source-bound-review-v1', 'reviewer_id': reviewer,
            'review_type': 'source_bound_assistant_diagnostic', 'records': [
                {'record_id': r['record_id'], 'source_sha256': r['source_sha256'], 'output_sha256': r['output_sha256'],
                 'whole_source_read': False, 'whole_output_read': False,
                 'flags': {k: None for k in CATEGORIES}, 'findings': []} for r in rows]}
        write_new(output / ('review_' + reviewer + '.template.json'), template)
    return {'status': 'materials_prepared', 'all20': 20, 'planned8': 8, 'available': sum(r['available'] for r in rows), 'reviews_performed': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'verify-plan', 'prepare-materials', 'validate', 'aggregate'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--materials', type=Path)
    parser.add_argument('--review-a', type=Path)
    parser.add_argument('--review-b', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == 'plan':
        write_new(root / OUT / 'review-plan.json', make_plan(root)); result = {'status': 'planned', 'reviews_performed': 0}
    elif args.stage == 'verify-plan':
        verify_plan(root); result = {'status': 'verified', 'reviews_performed': 0}
    elif args.stage == 'prepare-materials':
        result = prepare_materials(root, (args.output or root / OUT / 'reviews').resolve())
    else:
        verify_plan(root)
        if not args.materials or not args.review_a:
            parser.error('validate/aggregate require --materials and --review-a')
        materials, a = read(args.materials), read(args.review_a)
        if materials != build_materials(root):
            raise ValueError('Materials differ from frozen whole sources and current complete generation ledger')
        if args.stage == 'validate':
            result = validate_review(a, materials)
        else:
            if not args.review_b or not args.output:
                parser.error('aggregate requires --review-b and --output')
            result = aggregate_reviews(a, read(args.review_b), materials)
            write_new(args.output, result)
    print(json.dumps(result if args.stage != 'aggregate' else {k: result[k] for k in ('denominators', 'counts')}, indent=2))


if __name__ == '__main__':
    main()
