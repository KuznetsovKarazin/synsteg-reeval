#!/usr/bin/env python3
"""Source-only morphology ablation over every frozen WordNet-v2 class.

Static LemmInflect lookups only: no OOV model, fallback tags, contextual NLP,
experimental text, or semantic certification. Frozen dictionaries are untouched.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'morph-recovery-v3'
WHEEL_SHA256 = '1cf470efa783733ca5654cc1a6ee1de774e59031e61bf6210467625e84a8cf1c'
WHEEL_BYTES = 769698
WHEEL_URL = 'https://files.pythonhosted.org/packages/9e/98/9fac8cc177474ddd8d1725d5cae052098662d5e4a6756633335f4393673b/lemminflect-0.2.3-py3-none-any.whl'
PARENT_SHA256 = '5dafba7b2179b40be560bacfd9ffa783ef98350d07e8d4ac7919c3a0865e6aa6'
PARENT_PROVENANCE_SHA256 = 'fd57abbb28c910513aa194b827cbe4daa923b0d0da7e5baf81f675679f07b36d'
SOURCE_MEMBERS = {
    'lemminflect/resources/infl_lu.csv.gz': (274600, '8c44bd5bec62488bef043f98b0cab022a431d6cb03954864ea70334530908ba7'),
    'lemminflect/resources/lemma_lu.csv.gz': (326767, 'fd6f807e796da07f2a430d073cefae17004f7147abda065bb48b129dc29a850f'),
    'lemminflect/resources/infl_overrides.csv': (1490, 'c6ccfd2f17d706cf259c6cf96cbb89b5a641a584c2456755808425821c7652e8'),
    'lemminflect/resources/lemma_overrides.csv': (1662, 'f9968d1cc7689350d738c85ee59490cc23efe8cf4ad5a7d5092f070b5fc46c9f'),
    'lemminflect-0.2.3.dist-info/LICENSE': (1078, '8e69f1c68fc9f23bd5065a6266f6febb5aaa83a43b41070276760807037abc85'),
}
INFLECTION = 'lemminflect/resources/infl_lu.csv.gz'
LEMMAS = 'lemminflect/resources/lemma_lu.csv.gz'
INFLECTION_OVERRIDES = 'lemminflect/resources/infl_overrides.csv'
LEMMA_OVERRIDES = 'lemminflect/resources/lemma_overrides.csv'
LICENSE = 'lemminflect-0.2.3.dist-info/LICENSE'
POS = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 'r': 'ADV'}
TAG_POS = {tag: pos for pos, tags in {
    'NOUN': ('NN', 'NNS'), 'VERB': ('VB', 'VBP', 'VBD', 'VBN', 'VBG', 'VBZ'),
    'ADJ': ('JJ', 'JJR', 'JJS'), 'ADV': ('RB', 'RBR', 'RBS')}.items() for tag in tags}
COLUMNS = {'noun': ('NNS',), 'adj': ('JJR', 'JJS'), 'adv': ('RBR', 'RBS'),
           'verb': ('VBD', 'VBN', 'VBG', 'VBZ')}
BASE_TAGS = {'noun': ('NN',), 'adj': ('JJ',), 'adv': ('RB',), 'verb': ('VB', 'VBP')}
WORD = re.compile(r'[a-z]{2,}\Z', re.ASCII)
NIH_TERMS_URL = 'https://lhncbc.nlm.nih.gov/LSG/Projects/lexicon/current/web/termsAndConditions.html'


class BuildError(ValueError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def formatted(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + '\n').encode('ascii')


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('ascii')


def rules():
    return {
        'version': VERSION,
        'parent': 'All239 frozen WordNet-v2 classes; no source/capacity selection and no per-observed-word blacklist',
        'static_lookup': 'LemmInflect0.2.3 raw tables, official synthesized base tags, hardcoded auxiliary/modal table, then per-tag overrides',
        'no_inference': 'No package import needed, OOV rules, pickle models, pennTagAlts, contextual POS tagger or frequency selection',
        'surface': 'Exact lowercase ASCII [a-z]{2,}; no casefolding of proper-noun source keys',
        'parent_lemma': 'Both whole forward tag POS set and complete lemma lookup of parent lemma must uniquely support its WordNet POS and exact spelling',
        'feature_profile': 'Complete set of exact tags assigned to a surface by that parent lemma across its full forward record',
        'profile_trials': 'Union of complete observed profiles of all parent members; parent ledger also lists every declared POS tag and whether a profile covers it',
        'one_form': 'Every original parent lemma must supply exactly one raw surface with the profile; multiple spellings fail closed, never choose first',
        'inverse_evidence': 'Both inverse of complete post-override forward resource and complete post-override lemma lookup must equal exactly {(own POS, own lemma)}; absent evidence rejects',
        'class_integrity': 'Retain every original parent lemma or reject the entire profile trial; at least two distinct surfaces',
        'collision_rule': 'Build all provisional classes simultaneously, then reject every class sharing any surface with another provisional class; never greedy',
        'source_interpretation': 'Finite resource tag-spectrum agreement, not true countability, contextual POS, sense agreement or semantic safety',
        'case_scope': 'Lowercase exact-query compatibility; uppercase/proper-name homographs are not asserted globally unambiguous',
        'synthetic_base_warning': 'Upstream converter injects NN/JJ/RB/VB+VBP; these are not independently attested grammatical certificates',
        'class_id': 'morph3-{parent WordNet class ID}-{complete sorted lowercased PTB tag spectrum}',
        'serialization': 'ASCII JSON, sorted keys, indent2, final LF; trials JSONL canonical sorted keys',
    }


def validate_members(members):
    if set(members) != set(SOURCE_MEMBERS):
        raise BuildError('Exact consumed source-member set required')
    for name, (size, digest) in SOURCE_MEMBERS.items():
        if len(members[name]) != size or sha(members[name]) != digest:
            raise BuildError('Pinned source member mismatch: ' + name)


def load_wheel(path):
    data = Path(path).read_bytes()
    if len(data) != WHEEL_BYTES or sha(data) != WHEEL_SHA256:
        raise BuildError('Pinned wheel identity mismatch')
    try:
        with zipfile.ZipFile(Path(path)) as archive:
            members = {name: archive.read(name) for name in SOURCE_MEMBERS}
    except (KeyError, zipfile.BadZipFile) as error:
        raise BuildError('Required wheel member unavailable') from error
    validate_members(members)
    return members


def bundle_name(name):
    return Path(name).name + '.base64'


def load_bundle(path):
    path = Path(path)
    try:
        metadata = json.loads((path / 'manifest.json').read_bytes())
        if metadata.get('wheel_sha256') != WHEEL_SHA256 or metadata.get('wheel_bytes') != WHEEL_BYTES:
            raise BuildError('Source bundle identifies a different wheel')
        members = {name: base64.b64decode((path / bundle_name(name)).read_bytes(), validate=False) for name in SOURCE_MEMBERS}
    except (OSError, ValueError) as error:
        raise BuildError('Malformed source bundle') from error
    validate_members(members)
    return members


def write_once(path, data):
    path = Path(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise BuildError('Refusing to overwrite differing output: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def save_bundle(members, path):
    validate_members(members)
    path = Path(path)
    content = {bundle_name(name): base64.encodebytes(data) for name, data in members.items()}
    metadata = {'schema': 'synsteg-lemminflect-static-source-v1', 'package': 'lemminflect', 'version': '0.2.3',
        'wheel_url': WHEEL_URL, 'wheel_sha256': WHEEL_SHA256, 'wheel_bytes': WHEEL_BYTES,
        'members': {name: {'bytes': len(data), 'sha256': sha(data), 'file': bundle_name(name)} for name, data in members.items()},
        'source_kind': 'Exact static lookup and override bytes extracted from pinned wheel; model pickle members excluded',
        'upstream': 'NIH SPECIALIST Lexicon-derived via LemmInflect; underlying original SPECIALIST release not identified by this wheel',
        'licenses': {'package': 'MIT, complete text in LICENSE.base64 and output LEMMINFLECT-LICENSE.txt',
                     'upstream_terms': NIH_TERMS_URL}, 'modification': 'No modifications to consumed member bytes; base64 transport only'}
    content['manifest.json'] = formatted(metadata)
    for name, data in content.items():
        if (path / name).exists() and (path / name).read_bytes() != data:
            raise BuildError('Existing source bundle differs: ' + name)
    for name, data in content.items():
        write_once(path / name, data)


def _text(members, name):
    data = members[name]
    if name.endswith('.gz'):
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError) as error:
            raise BuildError('Malformed gzip lookup') from error
    try:
        return data.decode('utf-8')
    except UnicodeError as error:
        raise BuildError('Malformed UTF-8 lookup') from error


def _overrides(text, allowed):
    result = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.strip().split(',')
        if len(parts) != 3 or any(not p for p in parts) or parts[1] not in allowed:
            raise BuildError('Malformed override record')
        word, pos, form = parts
        if (word, pos) in result:
            raise BuildError('Duplicate override record')
        result[(word, pos)] = (form,)
    return result


def parse_tables(members):
    """Replicate exact static lookup tables without importing package/model code."""
    forward, seen = {}, set()
    for line in _text(members, INFLECTION).splitlines():
        parts = line.strip().split(',')
        if len(parts) < 2 or not parts[0] or parts[1] not in COLUMNS:
            raise BuildError('Malformed inflection row')
        lemma, category, *fields = parts
        if (lemma, category) in seen or len(fields) > len(COLUMNS[category]):
            raise BuildError('Duplicate or overlong inflection row')
        seen.add((lemma, category))
        forms = {tag: tuple(field.split('/')) for tag, field in zip(COLUMNS[category], fields) if field}
        if any(any(not form for form in values) for values in forms.values()):
            raise BuildError('Empty form inside alternatives')
        forms.update({tag: (lemma,) for tag in BASE_TAGS[category]})
        forward.setdefault(lemma, {}).update(forms)
    # Official InflectionLUCodec.updateForAuxMod, before per-tag overrides.
    forward.update({
        'can': {'VB': ('can',), 'VBD': ('could',)}, 'may': {'VB': ('may',), 'VBD': ('might',)},
        'will': {'VB': ('will',), 'VBD': ('would',)}, 'shall': {'VB': ('shall',), 'VBD': ('should',)},
        'must': {'VB': ('must',), 'VBD': ('must',)}, 'ought': {'VB': ('ought',), 'VBD': ('ought',)},
        'dare': {'VB': ('dare',)},
        'be': {'VB': ('be',), 'VBD': ('was', 'were'), 'VBG': ('being',), 'VBN': ('been',),
               'VBP': ('am', 'are'), 'VBZ': ('is',)}})
    for (lemma, tag), forms in _overrides(_text(members, INFLECTION_OVERRIDES), set(TAG_POS)).items():
        forward.setdefault(lemma, {})[tag] = forms
    backward, seen = {}, set()
    for line in _text(members, LEMMAS).splitlines():
        parts = line.strip().split(',')
        if len(parts) != 3 or any(not p for p in parts):
            raise BuildError('Malformed lemma lookup row')
        surface, category, lemmas = parts
        pos = 'AUX' if category == 'modal' else category.upper()
        if pos not in {'NOUN', 'VERB', 'ADJ', 'ADV', 'AUX'} or (surface, pos) in seen:
            raise BuildError('Duplicate or unsupported lemma lookup row')
        values = tuple(lemmas.split('/'))
        if any(not v for v in values):
            raise BuildError('Empty reverse lemma')
        seen.add((surface, pos))
        backward.setdefault(surface, {})[pos] = values
    for (surface, pos), lemmas in _overrides(_text(members, LEMMA_OVERRIDES), {'NOUN', 'VERB', 'ADJ', 'ADV', 'AUX'}).items():
        backward.setdefault(surface, {})[pos] = lemmas
    return forward, backward


def inverse_forward(forward):
    inverse = defaultdict(set)
    for lemma, tags in forward.items():
        for tag, forms in tags.items():
            if tag not in TAG_POS:
                raise BuildError('Unsupported exact inflection tag')
            for form in forms:
                inverse[form].add((TAG_POS[tag], lemma))
    return inverse


def inverse_lemma(backward, word):
    return {(pos, lemma) for pos, lemmas in backward.get(word, {}).items() for lemma in lemmas}


def construct(parent_classes, annotation_classes, forward, backward):
    """Pure source construction, accepting synthetic dictionaries for tests."""
    annotation = {c['id']: c for c in annotation_classes}
    if len(annotation) != len(annotation_classes) or set(annotation) != {c['id'] for c in parent_classes}:
        raise BuildError('Parent class annotations incomplete or duplicate')
    reverse = inverse_forward(forward)
    trials, parents, provisional = [], [], []
    for parent in sorted(parent_classes, key=lambda c: c['id']):
        cid, words = parent['id'], parent['words']
        ann = annotation[cid]
        if ann['pos'] not in POS or sorted(m['word'] for m in ann['members']) != words:
            raise BuildError('Parent POS or lemma annotations mismatch')
        upos = POS[ann['pos']]
        declared_tags = sorted(t for t, p in TAG_POS.items() if p == upos)
        source_forms, lemma_reasons, profiles = {}, {}, set()
        for lemma in words:
            tags = forward.get(lemma, {})
            reasons = []
            if not tags:
                reasons.append('missing_static_lemma')
            elif {TAG_POS[t] for t in tags} != {upos}:
                reasons.append('lemma_forward_pos_not_unique_expected')
            if inverse_lemma(backward, lemma) != {(upos, lemma)}:
                reasons.append('lemma_reverse_not_unique_expected')
            by_surface = defaultdict(set)
            for tag, values in tags.items():
                for surface in values:
                    by_surface[surface].add(tag)
            by_profile = defaultdict(set)
            for surface, spectrum in by_surface.items():
                spectrum = tuple(sorted(spectrum))
                if all(TAG_POS[t] == upos for t in spectrum):
                    by_profile[spectrum].add(surface)
                    profiles.add(spectrum)
            source_forms[lemma] = by_profile
            lemma_reasons[lemma] = reasons
        parent_record = {'parent_id': cid, 'pos': ann['pos'], 'upos': upos, 'original_lemmas': words,
            'declared_tags': declared_tags,
            'tag_coverage': {tag: [list(p) for p in sorted(profiles) if tag in p] for tag in declared_tags},
            'lemma_reasons': lemma_reasons, 'trial_ids': [], 'output_class_ids': []}
        if not profiles:
            parents.append(parent_record)
            continue
        for spectrum in sorted(profiles):
            trial_id = 'morph3-' + cid + '-' + '-'.join(t.lower() for t in spectrum)
            members, reasons = [], []
            for lemma in words:
                forms = sorted(source_forms[lemma].get(spectrum, set()))
                why = list(lemma_reasons[lemma])
                if len(forms) != 1:
                    why.append('missing_profile_form' if not forms else 'multiple_profile_forms')
                if len(forms) == 1:
                    surface = forms[0]
                    if not WORD.fullmatch(surface):
                        why.append('noncanonical_surface')
                    if reverse.get(surface, set()) != {(upos, lemma)}:
                        why.append('surface_forward_inverse_ambiguous_or_missing')
                    if inverse_lemma(backward, surface) != {(upos, lemma)}:
                        why.append('surface_lemma_inverse_ambiguous_or_missing')
                members.append({'lemma': lemma, 'forms': forms, 'reasons': sorted(set(why)),
                    'forward_inverse': {s: [list(v) for v in sorted(reverse.get(s, set()))] for s in forms},
                    'lemma_inverse': {s: [list(v) for v in sorted(inverse_lemma(backward, s))] for s in forms}})
                reasons.extend(why)
            surfaces = sorted(m['forms'][0] for m in members if len(m['forms']) == 1)
            if len(surfaces) != len(set(surfaces)):
                reasons.append('parent_lemmas_share_surface')
            trial = {'trial_id': trial_id, 'parent_id': cid, 'pos': ann['pos'], 'upos': upos,
                'tag_spectrum': list(spectrum), 'members': members, 'reasons': sorted(set(reasons)),
                'status': 'rejected' if reasons else 'provisional', 'candidate_words': surfaces,
                'collision_with': []}
            parent_record['trial_ids'].append(trial_id)
            trials.append(trial)
            if not reasons:
                if len(surfaces) < 2:
                    raise BuildError('Parent class minimum violated')
                provisional.append(trial)
        parents.append(parent_record)
    by_surface = defaultdict(set)
    for trial in provisional:
        for surface in trial['candidate_words']:
            by_surface[surface].add(trial['trial_id'])
    for trial in provisional:
        other = set().union(*(by_surface[s] for s in trial['candidate_words'])) - {trial['trial_id']}
        if other:
            trial['status'] = 'rejected'
            trial['reasons'].append('cross_class_surface_collision')
            trial['collision_with'] = sorted(other)
        else:
            trial['status'] = 'retained'
    retained = [t for t in trials if t['status'] == 'retained']
    classes = [{'id': t['trial_id'], 'words': t['candidate_words']} for t in retained]
    classes.sort(key=lambda c: c['id'])
    flat = [w for c in classes for w in c['words']]
    if len(flat) != len(set(flat)):
        raise BuildError('Final surface classes overlap')
    annotations = []
    for trial in retained:
        ann = annotation[trial['parent_id']]
        sense = {m['word']: m for m in ann['members']}
        annotations.append({'class_id': trial['trial_id'], 'parent_id': trial['parent_id'],
            'pos': ann['pos'], 'upos': trial['upos'], 'tag_spectrum': trial['tag_spectrum'],
            'grammatical_scope': 'finite static resource tag spectrum, not countability/contextual tag certification',
            'wordnet_gloss': ann['gloss'], 'synset_type': ann['synset_type'], 'offset': ann['offset'],
            'members': [{'word': m['forms'][0], 'lemma': m['lemma'],
                         'sense_key': sense[m['lemma']]['sense_key'], 'parent_tag_count': sense[m['lemma']]['tag_count']}
                        for m in trial['members']]})
    annotations.sort(key=lambda c: c['class_id'])
    for record in parents:
        record['output_class_ids'] = [c['id'] for c in classes if any(t['trial_id'] == c['id'] and t['parent_id'] == record['parent_id'] for t in retained)]
        record['status'] = 'retained_some_profiles' if record['output_class_ids'] else 'no_retained_profile'
    counts = {'parent_classes': len(parents), 'parent_lemmas': sum(len(p['original_lemmas']) for p in parents),
        'parent_status': dict(Counter(p['status'] for p in parents)), 'trials': len(trials),
        'trial_status': dict(Counter(t['status'] for t in trials)),
        'rejection_reasons_nonexclusive': dict(Counter(r for t in trials for r in t['reasons'])),
        'classes': len(classes), 'words': len(flat), 'classes_by_pos': dict(Counter(c['pos'] for c in annotations)),
        'classes_by_tag_spectrum': dict(Counter('+'.join(c['tag_spectrum']) for c in annotations))}
    return classes, annotations, parents, trials, counts


def build(root, members):
    root = Path(root)
    parent_bytes = (root / 'data/shared_dictionary_recovery_v2.json').read_bytes()
    provenance_bytes = (root / 'results/dictionary-recovery-v2/provenance.json').read_bytes()
    if sha(parent_bytes) != PARENT_SHA256 or sha(provenance_bytes) != PARENT_PROVENANCE_SHA256:
        raise BuildError('Frozen parent bytes differ')
    validate_members(members)
    parent, source = json.loads(parent_bytes), json.loads(provenance_bytes)
    forward, backward = parse_tables(members)
    classes, annotations, parents, trials, counts = construct(parent['classes'], source['selected_classes'], forward, backward)
    if counts['parent_classes'] != 239 or counts['parent_lemmas'] != 503:
        raise BuildError('Parent denominator changed')
    dictionary = {'schema': 'synsteg-public-dictionary-v1', 'version': VERSION, 'classes': classes,
        'provenance': {'parent_dictionary_sha256': PARENT_SHA256, 'parent_provenance_sha256': PARENT_PROVENANCE_SHA256,
            'source_wheel_sha256': WHEEL_SHA256, 'rules_sha256': sha(formatted(rules())),
            'interpretation': 'Source-only morphology ablation; not a semantic or contextual grammatical certificate'},
        'licenses': {'wordnet': parent['license'], 'lemminflect': {'name': 'MIT', 'text': members[LICENSE].decode()},
                     'nih_specialist_terms_url': NIH_TERMS_URL}}
    data = {'schema': 'synsteg-morphology-dictionary-annotations-v3', 'version': VERSION,
        'dictionary_sha256': sha(formatted(dictionary)), 'classes': annotations}
    provenance = {'schema': 'synsteg-morphology-provenance-v3', 'version': VERSION, 'rules': rules(),
        'builder_sha256': sha(Path(__file__).read_bytes()), 'counts': counts,
        'source': {'wheel_url': WHEEL_URL, 'wheel_sha256': WHEEL_SHA256, 'wheel_bytes': WHEEL_BYTES,
            'members': {n: {'bytes': len(b), 'sha256': sha(b)} for n, b in sorted(members.items())},
            'original_specialist_release': 'not identified by the pinned wheel; do not infer a release date',
            'global_forward_lemmas': len(forward), 'global_reverse_surfaces': len(backward)},
        'parent': {'dictionary_sha256': PARENT_SHA256, 'provenance_sha256': PARENT_PROVENANCE_SHA256},
        'dictionary': {'file_sha256': sha(formatted(dictionary)), 'canonical_sha256': sha(canonical({k: dictionary[k] for k in ('schema', 'version', 'classes')}))},
        'annotations_sha256': sha(formatted(data)), 'experimental_text_read': False, 'model_calls': 0}
    return dictionary, data, parents, trials, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--wheel', type=Path)
    source.add_argument('--source-bundle', type=Path)
    parser.add_argument('--bundle-output', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dictionary', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    members = load_wheel(args.wheel) if args.wheel else load_bundle(args.source_bundle)
    dictionary, annotations, parents, trials, provenance = build(root, members)
    directory = args.output_dir or root / 'results/morphology-recovery-v3'
    target = args.dictionary or root / 'data/shared_dictionary_morphology_v3.json'
    if target.resolve() in {(root / f'data/shared_dictionary_recovery_{v}.json').resolve() for v in ('v1', 'v2')}:
        raise BuildError('Frozen dictionary path forbidden')
    outputs = {target: formatted(dictionary), directory / 'annotations.json': formatted(annotations),
        directory / 'parent-ledger.json': formatted(parents),
        directory / 'trials.jsonl': b''.join(canonical(t) + b'\n' for t in trials),
        directory / 'provenance.json': formatted(provenance),
        directory / 'LEMMINFLECT-LICENSE.txt': members[LICENSE],
        directory / 'WORDNET-LICENSE.txt': dictionary['licenses']['wordnet']['text'].encode('ascii')}
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise BuildError('Output paths overlap')
    for p, data in outputs.items():
        if p.exists() and p.read_bytes() != data:
            raise BuildError('Refusing to overwrite differing output: ' + str(p))
    if args.bundle_output:
        save_bundle(members, args.bundle_output)
    for p, data in outputs.items():
        write_once(p, data)
    print(json.dumps({'status': 'built', **provenance['counts'], 'dictionary': str(target)}, indent=2))


if __name__ == '__main__':
    main()
