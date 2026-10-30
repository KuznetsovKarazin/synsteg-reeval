#!/usr/bin/env python3
"""Prepare a fixed full-text NLI diagnostic; never load or run ONNX weights.

Only public resource download is permitted here. The 30 previously accepted
positions are a diagnostic cohort, not a fresh or independent efficacy sample.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'synsteg-nli-diagnostic-v1'
REPOSITORY = 'cross-encoder/nli-deberta-v3-xsmall'
REVISION = 'a150876415327c80daeff35ca6f68f5ed8cf5c24'
MODEL_FILE = 'onnx/model_quint8_avx2.onnx'
RESOURCE_DIR = 'resources/nli-deberta-v3-xsmall-v1'
PREPARED = 'results/nli-diagnostic-v1'
SOURCE = 'results/universal-context-development-v1/accepted-slots.jsonl'
SOURCE_PLAN = 'results/universal-context-development-v1/acceptance-review-plan.json'
SOURCE_MAIN = '026b5b94494d65461fb0f440f12ddfc99c8ee2fb'
LABELS = {'0': 'contradiction', '1': 'entailment', '2': 'neutral'}
PACKAGES = {'onnxruntime': '1.23.2', 'tokenizers': '0.22.1', 'numpy': '2.3.5'}
THRESHOLDS = {'entailment_min': 0.90, 'contradiction_max': 0.05}
RUNTIME_PROFILE = {'provider': 'CPUExecutionProvider', 'intra_op_num_threads': 1, 'inter_op_num_threads': 1,
    'execution_mode': 'ORT_SEQUENTIAL', 'graph_optimization_level': 'ORT_DISABLE_ALL',
    'use_deterministic_compute': True, 'telemetry_events_disabled': True, 'batch_size': 1, 'score_cache': False}
MAX_TOKENS = 512
EXPECTED_PAIRS = 134
EXPECTED_POSITIONS = 30
EXPECTED_ALTERNATIVES = 52
CODE_FILES = ('scripts/prepare_nli_diagnostic_v1.py', 'scripts/run_nli_diagnostic_v1.py')
# Small files are additionally verified against Git blob IDs at the pinned
# upstream commit. Model weights use the upstream LFS SHA256 and exact size.
FILE_PINS = {
    MODEL_FILE: {'bytes': 87377068, 'sha256': '21b14751a95520953bfcc607ceeb617de7cbeaeb6d60f4c8966716c743985337'},
    'tokenizer.json': {'bytes': 8656624, 'git_blob_sha1': '12cc2e86bad0e731393847b017b32a7405ebd4f8',
        'sha256': '5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964'},
    'config.json': {'bytes': 1053, 'sha256': '8d9f07bf7ba54a6fc3b1962483056f94c39dcf188db4cf61843e1c88f94b2342'},
    'tokenizer_config.json': {'bytes': 1346, 'sha256': 'f3eecd07c370ef0bf7dd3780d3cd68cf9c8b00c267e21a208ddcd8f82bfec1a6'},
    'special_tokens_map.json': {'bytes': 301, 'sha256': 'ed7c099c988dbb414b18a6980d20cb57b91b7cd119f6f6941eb364b0e892e712'},
    'added_tokens.json': {'bytes': 26, 'sha256': 'a4b6bfe668f2b3cf6f0cd535e98a0663d2d0d4a4a15f13075ad3597d33985a23'},
    'README.md': {'bytes': 2871, 'sha256': '4aeb5585a9eeb71b2d572249f9faa22ef3378c6abee3eeaffaae8c961068a2c6'},
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def formatted(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def read(path):
    return json.loads(Path(path).read_bytes())


def write_exclusive(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def package_versions():
    actual = {name: importlib.metadata.version(name) for name in PACKAGES}
    if actual != PACKAGES:
        raise ValueError('Diagnostic requires exact package versions: ' + repr(PACKAGES))
    return actual


def verify_resource_bytes(name, data):
    pin = FILE_PINS[name]
    if len(data) != pin['bytes']:
        raise ValueError('Public resource size mismatch: ' + name)
    if 'sha256' in pin and sha(data) != pin['sha256']:
        raise ValueError('Public resource SHA256 mismatch: ' + name)
    if 'git_blob_sha1' in pin:
        digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if digest != pin['git_blob_sha1']:
            raise ValueError('Pinned Git blob mismatch: ' + name)
    return {'bytes': len(data), 'sha256': sha(data)}


def verify_resources(resources, *, download=False):
    resources = Path(resources).resolve()
    verified = {}
    for name, pin in FILE_PINS.items():
        path = resources / name
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
        if not path.exists():
            if not download:
                raise FileNotFoundError('Pinned public resource missing: ' + str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.download')
            # An incomplete previous transfer is never silently accepted/resumed.
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as stream:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > pin['bytes']:
                        raise ValueError('Public download exceeded pinned size')
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            verify_resource_bytes(name, temporary.read_bytes())
            if path.exists():
                raise FileExistsError('Resource appeared during transfer')
            temporary.rename(path)
        verified[name] = {**verify_resource_bytes(name, path.read_bytes()), 'url': url, **pin}
    config = read(resources / 'config.json')
    if (config.get('id2label') != LABELS or config.get('label2id') != {v: int(k) for k, v in LABELS.items()}
            or config.get('max_position_embeddings') != MAX_TOKENS):
        raise ValueError('Pinned model labels or maximum positions differ')
    tokenizer_config = read(resources / 'tokenizer_config.json')
    if tokenizer_config.get('model_max_length') != MAX_TOKENS:
        raise ValueError('Tokenizer maximum positions differ')
    return {'schema': SCHEMA + '-public-resources', 'repository': REPOSITORY, 'revision': REVISION,
        'model_file': MODEL_FILE, 'files': verified, 'labels': LABELS, 'max_pair_tokens': MAX_TOKENS,
        'license': 'apache-2.0 as declared by original model author in pinned README.md',
        'license_url': 'https://www.apache.org/licenses/LICENSE-2.0',
        'model_card_url': f'https://huggingface.co/{REPOSITORY}/blob/{REVISION}/README.md',
        'model_training': 'SNLI and MultiNLI; no claim of synonym-equivalence calibration',
        'weights_download_does_not_mean_inference_ran': True}


def load_tokenizer(resources):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(resources) / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if tokenizer.padding is not None or tokenizer.truncation is not None:
        raise ValueError('Padding/truncation must remain disabled')
    return tokenizer


def tokenization(tokenizer, premise, hypothesis):
    encoding = tokenizer.encode(premise, hypothesis, add_special_tokens=True)
    value = {'input_ids': encoding.ids, 'attention_mask': encoding.attention_mask, 'token_type_ids': encoding.type_ids}
    if not encoding.ids or any(len(v) != len(encoding.ids) for v in value.values()):
        raise ValueError('Malformed full pair tokenization')
    return value


def load_materials(root):
    root = Path(root)
    raw = (root / SOURCE).read_bytes()
    plan_raw = (root / SOURCE_PLAN).read_bytes()
    plan = json.loads(plan_raw)
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if (sha(raw) != plan.get('materials_sha256') or len(rows) != EXPECTED_POSITIONS
            or plan.get('unique_positions') != EXPECTED_POSITIONS
            or len({r['review_id'] for r in rows}) != EXPECTED_POSITIONS
            or Counter(r['cohort'] for r in rows) != Counter({'source_excerpts': 25, 'historical_covers': 2, 'grounded_covers': 3})):
        raise ValueError('Complete frozen accepted-position diagnostic cohort required')
    for row in rows:
        if not isinstance(row['text'], str) or sha(row['text'].encode()) != row['text_sha256']:
            raise ValueError('Source text/hash mismatch')
    return rows, {SOURCE: sha(raw), SOURCE_PLAN: sha(plan_raw)}


def build_pairs(materials):
    pairs = []
    for item in materials:
        text = item['text']
        tokens = list(re.finditer(r'\S+', text))
        index = item['token_index']
        if type(index) is not int or not 0 <= index < len(tokens) or tokens[index].group() != item['original_token']:
            raise ValueError('Focus token index or literal changed')
        match = re.fullmatch(r'([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)', tokens[index].group())
        if match is None:
            raise ValueError('Focus is not a single ASCII word with affixes')
        original = match.group(2)
        words = item['words']
        if (not isinstance(words, list) or words != sorted(set(words)) or len(words) < 2
                or original not in words or any(not re.fullmatch(r'[a-z]+', word) for word in words)):
            raise ValueError('Complete lowercase public class required')
        left = tokens[index].start() + match.start(2)
        right = tokens[index].start() + match.end(2)
        alternatives = [(original, text, True)] + [(word, text[:left] + word + text[right:], False) for word in words if word != original]
        for word, variant, identity in alternatives:
            directions = [('identity', text, text)] if identity else [('original_to_variant', text, variant), ('variant_to_original', variant, text)]
            for direction, premise, hypothesis in directions:
                identity_fields = {'review_id': item['review_id'], 'alternative_word': word, 'direction': direction}
                pairs.append({**identity_fields, 'pair_id': sha(canonical(identity_fields)),
                    'cohort': item['cohort'], 'source_group_id': item['source_group_id'], 'is_identity': identity,
                    'original_word': original, 'premise': premise, 'hypothesis': hypothesis,
                    'premise_sha256': sha(premise.encode()), 'hypothesis_sha256': sha(hypothesis.encode()),
                    'source_text_sha256': item['text_sha256']})
    if len({p['pair_id'] for p in pairs}) != len(pairs):
        raise ValueError('Duplicate pair identity')
    return pairs


def validate_pair_plan(pairs):
    if (len(pairs) != EXPECTED_PAIRS or len({p['pair_id'] for p in pairs}) != EXPECTED_PAIRS
            or Counter(p['direction'] for p in pairs) != Counter({'identity': 30, 'original_to_variant': 52, 'variant_to_original': 52})
            or len({p['review_id'] for p in pairs}) != EXPECTED_POSITIONS):
        raise ValueError('Exactly134 planned pairs,30 identities,52 bidirectional alternatives required')
    groups = {}
    for pair in pairs:
        if pair['is_identity'] != (pair['direction'] == 'identity'):
            raise ValueError('Identity/direction disagreement')
        if sha(pair['premise'].encode()) != pair['premise_sha256'] or sha(pair['hypothesis'].encode()) != pair['hypothesis_sha256']:
            raise ValueError('Pair text hash mismatch')
        identity = {k: pair[k] for k in ('review_id', 'alternative_word', 'direction')}
        if sha(canonical(identity)) != pair['pair_id']:
            raise ValueError('Pair identifier mismatch')
        groups.setdefault((pair['review_id'], pair['alternative_word']), []).append(pair)
    for group in groups.values():
        if group[0]['is_identity']:
            if len(group) != 1 or group[0]['premise'] != group[0]['hypothesis']:
                raise ValueError('Malformed identity control')
        elif (len(group) != 2 or {p['direction'] for p in group} != {'original_to_variant', 'variant_to_original'}
                or group[0]['premise'] != group[1]['hypothesis'] or group[0]['hypothesis'] != group[1]['premise']):
            raise ValueError('Bidirectional pair omitted or altered')


def prepare(root, output, resources, *, download=False):
    root, output, resources = Path(root).resolve(), Path(output).resolve(), Path(resources).resolve()
    if output.exists():
        raise FileExistsError('Preparation refuses an existing output namespace')
    versions = package_versions()
    manifest = verify_resources(resources, download=download)
    materials, inventory = load_materials(root)
    pairs = build_pairs(materials)
    validate_pair_plan(pairs)
    tokenizer = load_tokenizer(resources)
    for pair in pairs:
        encoded = tokenization(tokenizer, pair['premise'], pair['hypothesis'])
        pair['pair_tokens'] = len(encoded['input_ids'])
        pair['tokenization_sha256'] = sha(canonical(encoded))
        pair['planned_status'] = 'score_once' if pair['pair_tokens'] <= MAX_TOKENS else 'abstain_pair_too_long'
    content = {'pairs.jsonl': b''.join(canonical(p) + b'\n' for p in pairs), 'resource-manifest.json': formatted(manifest)}
    protocol = {'schema': SCHEMA, 'source_main_commit': SOURCE_MAIN,
        'scope': 'post_observation_development_diagnostic_not_semantic_certification',
        'receiver_invariant_codec_gate': False,
        'joint_substitution_orbit_tested': False,
        'diagnostic_comparison': 'Observed original versus one changed position; not a full sentence-orbit equivalence relation',
        'selection': 'Every previous accepted slot and every nonidentity class alternative; no rating, length or model-score selection',
        'positions': EXPECTED_POSITIONS, 'alternatives': EXPECTED_ALTERNATIVES, 'identity_controls': EXPECTED_POSITIONS,
        'planned_pairs': EXPECTED_PAIRS, 'planned_status_counts': dict(Counter(p['planned_status'] for p in pairs)),
        'text_scope': 'Entire supplied text in both directions; no truncation, segmentation or window selection',
        'max_pair_tokens': MAX_TOKENS, 'thresholds': THRESHOLDS,
        'alternative_acceptance': 'Identity control and both directed alternative pairs must be scored and pass both fixed thresholds; any absence/error/oversize fails closed',
        'identity_acceptance': 'Same fixed per-pair thresholds as alternatives',
        'model_repository': REPOSITORY, 'model_revision': REVISION, 'model_file': MODEL_FILE,
        'labels': LABELS, 'packages': versions,
        'runtime_profile': RUNTIME_PROFILE,
        'runtime_reproducibility_scope': 'Exact declared versions and CPU profile; cross-hardware numeric equality is not proved',
        'source_files': inventory, 'code_files': {p: sha((root / p).read_bytes()) for p in CODE_FILES},
        'frozen_files': {p: sha(b) for p, b in content.items()},
        'preparation_created_onnx_session': False, 'preparation_classifier_calls': 0,
        'generation_calls': 0, 'paid_calls': 0, 'closed_test_opened': False,
        'quality_labels_read_during_preparation': False, 'thresholds_are_not_calibrated_semantic_probabilities': True}
    for p, digest in inventory.items():
        if sha((root / p).read_bytes()) != digest:
            raise ValueError('Input changed during preparation')
    output.mkdir(parents=True, exist_ok=False)
    for name, data in content.items():
        write_exclusive(output / name, data)
    write_exclusive(output / 'protocol.json', formatted(protocol))
    return {'status': 'prepared_no_inference', 'pairs': len(pairs), 'planned_status_counts': protocol['planned_status_counts'],
            'protocol_sha256': sha(formatted(protocol)), 'resource_bytes': sum(v['bytes'] for v in manifest['files'].values())}


def verify_preparation(root, prepared, resources):
    root, prepared, resources = Path(root).resolve(), Path(prepared).resolve(), Path(resources).resolve()
    protocol = read(prepared / 'protocol.json')
    if (protocol.get('schema') != SCHEMA or protocol.get('source_main_commit') != SOURCE_MAIN
            or protocol.get('thresholds') != THRESHOLDS or protocol.get('max_pair_tokens') != MAX_TOKENS
            or protocol.get('packages') != PACKAGES or protocol.get('labels') != LABELS
            or protocol.get('runtime_profile') != RUNTIME_PROFILE
            or protocol.get('model_revision') != REVISION or protocol.get('model_file') != MODEL_FILE
            or set(protocol.get('source_files', {})) != {SOURCE, SOURCE_PLAN}
            or set(protocol.get('code_files', {})) != set(CODE_FILES)
            or set(protocol.get('frozen_files', {})) != {'pairs.jsonl', 'resource-manifest.json'}):
        raise ValueError('Unsupported/incomplete frozen NLI protocol')
    for location, fields in ((root, protocol['source_files']), (root, protocol['code_files']), (prepared, protocol['frozen_files'])):
        for name, expected in fields.items():
            path = (location / name).resolve()
            if not path.is_relative_to(location.resolve()) or sha(path.read_bytes()) != expected:
                raise ValueError('Frozen file changed: ' + name)
    package_versions()
    if read(prepared / 'resource-manifest.json') != verify_resources(resources):
        raise ValueError('Pinned public resources changed')
    pairs = [json.loads(line) for line in (prepared / 'pairs.jsonl').read_bytes().splitlines() if line.strip()]
    validate_pair_plan(pairs)
    materials, _ = load_materials(root)
    rebuilt = build_pairs(materials)
    for original, pair in zip(rebuilt, pairs):
        if any(pair.get(k) != v for k, v in original.items()):
            raise ValueError('Pair differs from complete source-bound reconstruction')
    if Counter(p['planned_status'] for p in pairs) != Counter(protocol['planned_status_counts']):
        raise ValueError('Planned pair statuses differ')
    return protocol, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'resources'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resources', type=Path)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve(); output = args.output or root / PREPARED; resources = args.resources or root / RESOURCE_DIR
    if args.command == 'prepare':
        result = prepare(root, output, resources, download=args.download)
    elif args.command == 'resources':
        result = verify_resources(resources, download=args.download)
    else:
        protocol, pairs = verify_preparation(root, output, resources)
        result = {'status': 'verified_no_inference', 'pairs': len(pairs), 'schema': protocol['schema']}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
