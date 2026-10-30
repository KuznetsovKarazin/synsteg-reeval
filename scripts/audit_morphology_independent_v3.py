#!/usr/bin/env python3
"""Independent static-source checks using the pinned package's own lookup API.

No corpus files, OOV rules, statistical models or candidate capacity are read.
The class oracle does not call the new builder's construct/build functions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WHEEL_SHA = '1cf470efa783733ca5654cc1a6ee1de774e59031e61bf6210467625e84a8cf1c'
WHEEL_SIZE = 769698
POS = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 'r': 'ADV'}
TAG_POS = {tag: pos for pos, tags in {'NOUN': ['NN', 'NNS'], 'ADJ': ['JJ', 'JJR', 'JJS'],
    'ADV': ['RB', 'RBR', 'RBS'], 'VERB': ['VB', 'VBP', 'VBD', 'VBN', 'VBG', 'VBZ']}.items() for tag in tags}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def analyses(table, word):
    return {(pos, lemma) for pos, lemmas in table.get(word, {}).items() for lemma in lemmas}


def oracle(parent, annotations, forward, backward):
    """Derive eligible full tag-spectrum classes directly from official tables."""
    ann = {a['id']: a for a in annotations}
    reverse = defaultdict(set)
    for lemma, tags in forward.items():
        for tag, words in tags.items():
            for word in words:
                reverse[word].add((TAG_POS[tag], lemma))
    provisional = []
    trial_keys = set()
    for group in parent:
        upos = POS[ann[group['id']]['pos']]
        by_lemma = {}
        profiles = set()
        lemma_okay = {}
        for lemma in group['words']:
            tags = forward.get(lemma, {})
            lemma_okay[lemma] = (bool(tags) and {TAG_POS[t] for t in tags} == {upos}
                                  and analyses(backward, lemma) == {(upos, lemma)})
            surfaces = defaultdict(set)
            for tag, forms in tags.items():
                for form in forms:
                    surfaces[form].add(tag)
            by_profile = defaultdict(set)
            for form, tags in surfaces.items():
                spectrum = tuple(sorted(tags))
                if all(TAG_POS[t] == upos for t in tags):
                    by_profile[spectrum].add(form)
                    profiles.add(spectrum)
            by_lemma[lemma] = by_profile
        for spectrum in profiles:
            cid = 'morph3-' + group['id'] + '-' + '-'.join(t.lower() for t in spectrum)
            trial_keys.add(cid)
            candidate = []
            valid = True
            for lemma in group['words']:
                forms = by_lemma[lemma].get(spectrum, set())
                if not lemma_okay[lemma] or len(forms) != 1:
                    valid = False
                    continue
                form = next(iter(forms))
                if (re.fullmatch('[a-z]{2,}', form, flags=re.ASCII) is None
                        or reverse[form] != {(upos, lemma)} or analyses(backward, form) != {(upos, lemma)}):
                    valid = False
                candidate.append(form)
            if valid and len(candidate) == len(set(candidate)) and len(candidate) >= 2:
                provisional.append({'id': cid, 'words': sorted(candidate)})
    frequencies = Counter(word for c in provisional for word in c['words'])
    classes = sorted((c for c in provisional if all(frequencies[word] == 1 for word in c['words'])), key=lambda c: c['id'])
    return classes, trial_keys


def run(root, wheel, output):
    root, wheel, output = Path(root).resolve(), Path(wheel).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Independent evidence never overwrites existing output')
    raw = wheel.read_bytes()
    if len(raw) != WHEEL_SIZE or sha(raw) != WHEEL_SHA:
        raise ValueError('Untrusted wheel')
    builder_path = root / 'scripts/build_morphology_dictionary_v3.py'
    spec = importlib.util.spec_from_file_location('morph_builder_under_review', builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    parsed_forward, parsed_backward = builder.parse_tables(builder.load_wheel(wheel))
    parent = load(root / 'data/shared_dictionary_recovery_v2.json')['classes']
    annotation = load(root / 'results/dictionary-recovery-v2/provenance.json')['selected_classes']
    target_words = sorted({w for c in parent for w in c['words']})
    with tempfile.TemporaryDirectory(prefix='morph-independent-api-') as directory:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(directory)  # Complete archive is fixed by SHA above.
        sys.path.insert(0, directory)
        import lemminflect
        from lemminflect.core.Inflections import Inflections
        from lemminflect.core.Lemmatizer import Lemmatizer
        def forbidden(*args, **kwargs):
            raise AssertionError('OOV/model path must not be invoked')
        Inflections.getAllInflectionsOOV = forbidden
        Lemmatizer.getAllLemmasOOV = forbidden
        inflector, lemmatizer = Inflections(), Lemmatizer()
        official_forward = deepcopy(inflector._getInflDict())
        official_backward = deepcopy(lemmatizer._getLemmaDict())
        for lemma, tags in inflector._getOverridesDict().items():
            official_forward.setdefault(lemma, {}).update(tags)
        for word, values in lemmatizer._getOverridesDict().items():
            official_backward.setdefault(word, {}).update(values)
        assert parsed_forward == official_forward, 'Global forward table differs from official static implementation'
        assert parsed_backward == official_backward, 'Global backward table differs from official static implementation'
        for word in target_words:
            assert parsed_forward.get(word, {}) == lemminflect.getAllInflections(word), word
            assert parsed_backward.get(word, {}) == lemminflect.getAllLemmas(word), word
        expected, expected_trials = oracle(parent, annotation, official_forward, official_backward)
    dictionary = load(root / 'data/shared_dictionary_morphology_v3.json')
    assert dictionary['classes'] == expected, 'Independent complete-profile class oracle differs'
    trials = [json.loads(s) for s in (root / 'results/morphology-recovery-v3/trials.jsonl').read_bytes().splitlines()]
    assert len(trials) == len(expected_trials) and {t['trial_id'] for t in trials} == expected_trials
    retained = {c['id'] for c in expected}
    assert {t['trial_id'] for t in trials if t['status'] == 'retained'} == retained
    ledger = load(root / 'results/morphology-recovery-v3/parent-ledger.json')
    assert len(ledger) == len(parent) and {r['parent_id'] for r in ledger} == {c['id'] for c in parent}
    actual_annotations = load(root / 'results/morphology-recovery-v3/annotations.json')['classes']
    assert len(actual_annotations) == len(expected) and {a['class_id'] for a in actual_annotations} == retained
    parent_by_id = {c['id']: c for c in parent}
    for a in actual_annotations:
        assert sorted(m['lemma'] for m in a['members']) == parent_by_id[a['parent_id']]['words']
        for member in a['members']:
            tags = sorted(t for t, forms in official_forward[member['lemma']].items() if member['word'] in forms)
            assert tags == a['tag_spectrum']
    named = ['scripts/build_morphology_dictionary_v3.py', 'data/shared_dictionary_recovery_v2.json',
        'results/dictionary-recovery-v2/provenance.json', 'data/shared_dictionary_morphology_v3.json',
        'results/morphology-recovery-v3/annotations.json', 'results/morphology-recovery-v3/trials.jsonl',
        'results/morphology-recovery-v3/parent-ledger.json']
    report = {'schema': 'morphology-static-independent-validation-v3', 'status': 'matched',
        'wheel_sha256': WHEEL_SHA, 'official_package_version': lemminflect.__version__,
        'parent_classes': len(parent), 'parent_lemmas': len(target_words), 'target_api_lookup_comparisons': 2 * len(target_words),
        'whole_forward_lemmas_compared': len(official_forward), 'whole_reverse_surfaces_compared': len(official_backward),
        'oracle_classes': len(expected), 'oracle_words': sum(len(c['words']) for c in expected),
        'profile_trials_checked': len(expected_trials), 'annotation_classes_checked': len(actual_annotations),
        'input_sha256': {p: sha((root / p).read_bytes()) for p in named},
        'method': 'Official pinned package static loaders/API plus independently written complete-profile class oracle; no construct/build call',
        'case_scope': 'Exact lowercase query; not a guarantee excluding uppercase proper-name homographs',
        'underlying_lexical_truth_validated': False, 'contextual_grammar_validated': False, 'semantic_safety_validated': False,
        'experimental_corpus_read': False, 'model_inference_calls': 0, 'paid_calls': 0, 'closed_test_opened': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write((json.dumps(report, sort_keys=True, indent=2) + '\n').encode())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.wheel, args.output or args.root / 'results/morphology-nli-independent-v1/source-validation.json'), indent=2))


if __name__ == '__main__':
    main()
