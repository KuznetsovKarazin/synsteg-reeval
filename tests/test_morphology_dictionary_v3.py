"""Source morphology invariants; no natural-corpus input or semantic claims."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('morph_builder', ROOT / 'scripts/build_morphology_dictionary_v3.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


def parent(words=('quick', 'rapid'), pos='a', cid='p'):
    classes = [{'id': cid, 'words': sorted(words)}]
    annotations = [{'id': cid, 'pos': pos, 'gloss': 'Synthetic fixture only', 'synset_type': pos,
        'offset': 1, 'members': [{'word': w, 'sense_key': w + '%synthetic', 'tag_count': 1} for w in sorted(words)]}]
    return classes, annotations


def reverse(forward):
    result = {}
    for lemma, tags in forward.items():
        for tag, forms in tags.items():
            for form in forms:
                pos = b.TAG_POS[tag]
                values = set(result.setdefault(form, {}).get(pos, ()))
                values.add(lemma)
                result[form][pos] = tuple(sorted(values))
    return result


def run(forward, words=('quick', 'rapid'), pos='a', backward=None):
    return b.construct(*parent(words, pos), forward, reverse(forward) if backward is None else backward)


def test_complete_verb_spectrum_merges_same_surface_tags():
    forward = {w: {'VB': (w,), 'VBP': (w,), 'VBD': (w + 'ed',), 'VBN': (w + 'ed',)} for w in ('walk', 'talk')}
    classes, annotations, parents, trials, counts = run(forward, ('talk', 'walk'), 'v')
    assert len(classes) == 2
    assert {tuple(c['tag_spectrum']) for c in annotations} == {('VB', 'VBP'), ('VBD', 'VBN')}
    assert len({w for c in classes for w in c['words']}) == 4
    assert counts['parent_classes'] == 1 and len(parents[0]['trial_ids']) == 2


def test_invariant_singular_plural_is_its_own_profile():
    forward = {w: {'NN': (w,), 'NNS': (w,)} for w in ('fish', 'sheep')}
    classes, annotations, *_ = run(forward, ('fish', 'sheep'), 'n')
    assert len(classes) == 1 and annotations[0]['tag_spectrum'] == ['NN', 'NNS']
    forward['fish']['NNS'] = ('fishes',)
    classes, *_ = run(forward, ('fish', 'sheep'), 'n')
    assert classes == []


def test_missing_abbreviation_rejects_entire_parent_not_single_member():
    forward = {'kilometer': {'NN': ('kilometer',), 'NNS': ('kilometers',)}}
    classes, _, parents, trials, _ = run(forward, ('kilometer', 'km'), 'n')
    assert classes == [] and len(parents) == 1
    assert all('missing_static_lemma' in t['reasons'] for t in trials)
    assert parents[0]['original_lemmas'] == ['kilometer', 'km']


def test_no_suffix_or_oov_guesses():
    forward = {w: {'NN': (w,)} for w in ('car', 'vehicle')}
    classes, annotations, parents, *_ = run(forward, ('car', 'vehicle'), 'n')
    assert len(classes) == 1 and annotations[0]['tag_spectrum'] == ['NN']
    assert parents[0]['tag_coverage']['NNS'] == []


def test_lemma_extra_pos_rejects_even_if_expected_pos_exists():
    forward = {'quick': {'JJ': ('quick',), 'RB': ('quick',)}, 'rapid': {'JJ': ('rapid',)}}
    classes, _, _, trials, _ = run(forward)
    assert not classes
    assert all('lemma_forward_pos_not_unique_expected' in t['reasons'] for t in trials)


def test_forward_inverse_checks_unrelated_lemmas_outside_parent():
    forward = {'ax': {'NN': ('ax',), 'NNS': ('axes',)}, 'tool': {'NN': ('tool',), 'NNS': ('tools',)},
               'axis': {'NN': ('axis',), 'NNS': ('axes',)}}
    back = reverse(forward)
    # Pretend the independent reverse lexicon misses axis; forward still catches it.
    back['axes']['NOUN'] = ('ax',)
    classes, _, _, trials, _ = run(forward, ('ax', 'tool'), 'n', back)
    assert len(classes) == 1
    failed = next(t for t in trials if t['tag_spectrum'] == ['NNS'])
    assert 'surface_forward_inverse_ambiguous_or_missing' in failed['reasons']


def test_reverse_lookup_extra_pos_catches_missing_forward_analysis():
    forward = {w: {'JJ': (w,)} for w in ('quick', 'rapid')}
    back = reverse(forward);back['quick']['NOUN'] = ('quick',)
    assert not run(forward, backward=back)[0]


def test_missing_reverse_never_falls_back_to_identity():
    forward = {w: {'JJ': (w,)} for w in ('quick', 'rapid')}
    back = reverse(forward);del back['quick']
    assert not run(forward, backward=back)[0]


def test_multiple_profile_forms_not_ranked_or_first_selected():
    forward = {'quick': {'JJ': ('quick', 'speedy')}, 'rapid': {'JJ': ('rapid',)}}
    classes, _, _, trials, _ = run(forward)
    assert not classes and 'multiple_profile_forms' in trials[0]['reasons']


@pytest.mark.parametrize('form', ['two words', 'Rapid', 'éclair', 'x', 'hyphen-word', 'a_b'])
def test_noncanonical_surfaces_reject_profile(form):
    forward = {'quick': {'JJ': ('quick',)}, 'rapid': {'JJ': (form,)}}
    back = reverse(forward);back['rapid'] = {'ADJ': ('rapid',)}
    classes, _, _, trials, _ = run(forward, backward=back)
    assert not classes and 'noncanonical_surface' in trials[0]['reasons']


def test_cross_parent_collisions_reject_all_not_first_winner():
    forward = {w: {'JJ': (w,)} for w in ('quick', 'rapid', 'speedy')}
    a, aa = parent(('quick', 'rapid'), cid='a')
    c, cc = parent(('quick', 'speedy'), cid='b')
    result = b.construct(a + c, aa + cc, forward, reverse(forward))
    assert not result[0]
    assert all(t['reasons'] == ['cross_class_surface_collision'] and len(t['collision_with']) == 1 for t in result[3])
    assert b.construct(c + a, cc + aa, forward, reverse(forward)) == result


def tables(infl='quick,adj,,\nrapid,adj,,\n', lemma='quick,adj,quick\nrapid,adj,rapid\n', io='', lo=''):
    return {b.INFLECTION: gzip.compress(infl.encode()), b.LEMMAS: gzip.compress(lemma.encode()),
        b.INFLECTION_OVERRIDES: io.encode(), b.LEMMA_OVERRIDES: lo.encode()}


def test_static_loader_overrides_happen_after_modal_table():
    forward, backward = b.parse_tables(tables(io='can,VBD,canned\nquick,RB,quick\n', lo='quick,ADV,quick\n'))
    assert forward['can'] == {'VB': ('can',), 'VBD': ('canned',)}
    assert forward['quick'] == {'JJ': ('quick',), 'RB': ('quick',)}
    assert backward['quick'] == {'ADJ': ('quick',), 'ADV': ('quick',)}
    assert not run(forward, backward=backward)[0]


def test_static_loader_no_vbd_vbn_or_adjective_adverb_fallback():
    forward, _ = b.parse_tables(tables(infl='walk,verb,walked,,walking,walks\nquick,adj,,\n'))
    assert 'VBN' not in forward['walk'] and 'RB' not in forward['quick']
    assert forward['walk']['VB'] == forward['walk']['VBP'] == ('walk',)


@pytest.mark.parametrize('payload', [
    {'infl': 'quick,adj,,\nquick,adj,,\n'},
    {'infl': 'quick,unknown,a\n'},
    {'lemma': 'quick,adj,quick\nquick,adj,quick\n'},
    {'io': 'quick,JJ,x\nquick,JJ,y\n'},
    {'io': 'quick,UNKNOWN,x\n'},
    {'infl': 'quick,adj,quicker/,quickest\n'},
])
def test_malformed_or_duplicate_source_records_rejected(payload):
    with pytest.raises(b.BuildError):
        b.parse_tables(tables(**payload))


def test_annotations_must_cover_whole_parent():
    classes, annotations = parent()
    annotations[0]['members'].pop()
    with pytest.raises(b.BuildError):
        b.construct(classes, annotations, {}, {})


def test_write_once_refuses_different_bytes(tmp_path):
    path = tmp_path / 'evidence.json'
    b.write_once(path, b'one\n');b.write_once(path, b'one\n')
    with pytest.raises(b.BuildError):
        b.write_once(path, b'two\n')
    assert path.read_bytes() == b'one\n'


def test_real_bundle_full_parent_denominator_and_no_model_or_corpus(tmp_path):
    members = b.load_bundle(ROOT / 'data/morphology-recovery-v3/source')
    dictionary, annotations, parents, trials, provenance = b.build(ROOT, members)
    assert len(parents) == 239 and sum(len(p['original_lemmas']) for p in parents) == 503
    assert sum(provenance['counts']['trial_status'].values()) == len(trials)
    assert provenance['experimental_text_read'] is False and provenance['model_calls'] == 0
    assert len(dictionary['classes']) == len(annotations['classes'])
    assert len({w for c in dictionary['classes'] for w in c['words']}) == sum(len(c['words']) for c in dictionary['classes'])
    # Two semantic counterexamples intentionally survive morphology.
    assert any(c['words'] == ['conduction', 'conductivity'] for c in dictionary['classes'])
    assert any(c['words'] == ['sportsman', 'sportswoman'] for c in dictionary['classes'])
    assert all('km' not in c['words'] for c in dictionary['classes'])
    # Bundle-only reconstruction preserves exact consumed source bytes.
    b.save_bundle(members, tmp_path / 'bundle')
    assert b.load_bundle(tmp_path / 'bundle') == members


def test_bundle_tamper_fails_before_build(tmp_path):
    members = b.load_bundle(ROOT / 'data/morphology-recovery-v3/source')
    b.save_bundle(members, tmp_path)
    (tmp_path / b.bundle_name(b.INFLECTION_OVERRIDES)).write_bytes(b'eA==\n')
    with pytest.raises(b.BuildError):
        b.load_bundle(tmp_path)
