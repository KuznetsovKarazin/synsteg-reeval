#!/usr/bin/env python3
"""Compare all fixed NLI outcomes with unchanged assistant reference judgments.

These labels are diagnostic references, never human ground truth or safety
certificates. Selection/thresholds remain those frozen before classifier output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_nli_diagnostic_v1 as prep
import run_nli_diagnostic_v1 as runner

PRIOR = 'results/universal-context-development-v1'
STAGE = 'results/nli-diagnostic-v1'
REFERENCES = (PRIOR + '/reviews/reviewer_a.json', PRIOR + '/reviews/reviewer_b.json',
              PRIOR + '/reviews/comparison.json', PRIOR + '/accepted-slots.jsonl')
FLAGS = ('definite_meaning_change', 'definite_grammar_damage', 'ambiguous_or_register_risk')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_bytes().splitlines() if s.strip()]


def category(a, b):
    definite = [r['definite_meaning_change'] or r['definite_grammar_damage'] for r in (a, b)]
    if all(definite): return 'both_assistants_definite'
    if any(definite): return 'one_assistant_definite'
    if a['ambiguous_or_register_risk'] or b['ambiguous_or_register_risk']: return 'ambiguity_or_register_only'
    return 'no_flag_by_either_not_certified'


def prepare(root):
    root = Path(root).resolve()
    path = root / STAGE / 'assessment-plan.json'
    if path.exists(): raise FileExistsError('Do not overwrite assessment plan')
    plan = {'schema': 'synsteg-nli-assessment-plan-v1', 'reference_files': {p: sha((root / p).read_bytes()) for p in REFERENCES},
        'analysis_code_sha256': sha(Path(__file__).read_bytes()),
        'selection': 'Every52 nonidentity alternative from fixed134-pair diagnostic; no outcome filtering',
        'categories': ['both_assistants_definite', 'one_assistant_definite', 'ambiguity_or_register_only', 'no_flag_by_either_not_certified'],
        'diagnostic_counterexample': 'Accepted by fixed NLI criterion despite definite problem independently flagged by both assistants',
        'position_rule': 'Every nonidentity member must pass observed-original bidirectional/identity criterion; not allpairs or jointorbit and not receiverinvariant',
        'human_ground_truth': False, 'threshold_tuning_allowed': False, 'classifier_outputs_read_during_preparation': False}
    prep.write_exclusive(path, prep.formatted(plan))
    return {'status': 'assessment_frozen_before_scores', 'sha256': sha(prep.formatted(plan))}


def assess(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError('Do not overwrite NLI assessment')
    stage = root / STAGE
    plan = read(stage / 'assessment-plan.json')
    if (set(plan.get('reference_files', {})) != set(REFERENCES)
            or plan.get('analysis_code_sha256') != sha(Path(__file__).read_bytes())):
        raise ValueError('Changed assessment code or reference inventory')
    for p, h in plan['reference_files'].items():
        if sha((root / p).read_bytes()) != h: raise ValueError('Changed reference: ' + p)
    execution = read(stage / 'scores/execution.json')
    protocol = read(stage / 'protocol.json')
    if (sha((stage / 'protocol.json').read_bytes()) != execution['protocol_sha256']
            or set(protocol.get('code_files', {})) != set(prep.CODE_FILES)
            or set(protocol.get('frozen_files', {})) != {'pairs.jsonl', 'resource-manifest.json'}):
        raise ValueError('Scores belong to a changed or incomplete classifier protocol')
    for location, manifest in ((root, protocol['code_files']), (stage, protocol['frozen_files'])):
        for name, digest in manifest.items():
            path = (location / name).resolve()
            if not path.is_relative_to(location.resolve()) or sha(path.read_bytes()) != digest:
                raise ValueError('Changed classifier preparation evidence')
    for name, h in execution['output_sha256'].items():
        if name not in {'pairs.jsonl', 'summary.json', 'started.json'} or sha((stage / 'scores' / name).read_bytes()) != h:
            raise ValueError('Classifier output changed')
    if set(execution['output_sha256']) != {'pairs.jsonl', 'summary.json', 'started.json'}:
        raise ValueError('Incomplete classifier output manifest')
    pairs = jsonl(stage / 'pairs.jsonl')
    scores = jsonl(stage / 'scores/pairs.jsonl')
    recomputed = runner.summarize(pairs, scores)
    if recomputed != read(stage / 'scores/summary.json'):
        raise ValueError('Summary does not follow exact fixed scoring rules')
    materials = {r['review_id']: r for r in jsonl(root / PRIOR / 'accepted-slots.jsonl')}
    reference_hashes = read(root / PRIOR / 'reviews/comparison.json')['review_sha256']
    reviews = {}
    for label in ('A', 'B'):
        p = root / PRIOR / f'reviews/reviewer_{label.lower()}.json'
        if sha(p.read_bytes()) != reference_hashes[label]: raise ValueError('Raw review differs from verified comparison')
        reviews[label] = {r['review_id']: {a['word']: a for a in r['alternatives']} for r in read(p)['records']}
    records = []
    seen = set()
    for alternative in recomputed['alternatives']:
        rid, word = alternative['review_id'], alternative['alternative_word']
        if (rid, word) in seen or rid not in materials or word not in materials[rid]['words']:
            raise ValueError('Duplicate or unexpected alternative')
        seen.add((rid, word))
        flags = {label: {f: reviews[label][rid][word][f] for f in FLAGS} for label in reviews}
        if reviews['A'][rid][word]['unchanged'] or reviews['B'][rid][word]['unchanged']:
            raise ValueError('Identity must not enter replacement assessment')
        records.append({**alternative, 'reference_flags': flags, 'reference_category': category(flags['A'], flags['B'])})
    if len(records) != 52 or len({r['review_id'] for r in records}) != 30:
        raise ValueError('All52 alternatives and all30 positions required')
    categories = {name: {'planned_alternatives': sum(r['reference_category'] == name for r in records),
        'accepted': sum(r['reference_category'] == name and r['accepted'] for r in records),
        'outcomes': dict(Counter(r['status'] for r in records if r['reference_category'] == name))} for name in plan['categories']}
    position_results = []
    for rid, material in materials.items():
        group = [r for r in records if r['review_id'] == rid]
        position_results.append({'review_id': rid, 'cohort': material['cohort'],
            'all_observed_original_comparisons_accepted': all(r['accepted'] for r in group),
            'has_both_definite_reference': any(r['reference_category'] == 'both_assistants_definite' for r in group)})
    result = {'schema': 'synsteg-nli-diagnostic-assessment-v1', 'planned_alternatives': 52, 'planned_positions': 30,
        'score_summary_recomputed_exactly': True, 'reference_categories': categories,
        'accepted_both_definite_counterexamples': sum(r['accepted'] and r['reference_category'] == 'both_assistants_definite' for r in records),
        'position_all_member_passes': sum(r['all_observed_original_comparisons_accepted'] for r in position_results),
        'position_all_member_passes_with_definite_reference': sum(r['all_observed_original_comparisons_accepted'] and r['has_both_definite_reference'] for r in position_results),
        'human_validation': False, 'semantic_safety_certified': False, 'receiver_invariant_codec_gate': False,
        'threshold_tuning_performed': False, 'inputs_sha256': {str(p.relative_to(root)): sha(p.read_bytes()) for p in
            (stage / 'assessment-plan.json', stage / 'scores/pairs.jsonl', stage / 'scores/summary.json', stage / 'scores/execution.json')},
        'records': records, 'positions': position_results}
    prep.write_exclusive(output, prep.formatted(result))
    return {k: v for k, v in result.items() if k not in ('records', 'positions')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'assess'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path)
    a = p.parse_args(argv)
    result = prepare(a.root) if a.command == 'prepare' else assess(a.root, a.output or a.root / STAGE / 'assessment.json')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
