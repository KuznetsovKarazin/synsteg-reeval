#!/usr/bin/env python3
"""Posthoc, no-generation capacity diagnostic on fixed saved recovery covers.

No encoder, decoder, tokenizer or model is called. This script does not change
the old transport results or turn frame feasibility into a delivery result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys

ID = 'recovery83-100-capacity-v1'
COMMIT = '64860ac6dc7e7a36170e960b5256e550d12c0c9e'
ORDERS = list(range(83, 101))
OBSERVED = list(range(83, 91))
FRAMES = {'plain_length16': 16, 'aes_siv': 144}
PAYLOAD_BITS = [16, 128, 256, 512]
PUBLIC_PROTOCOL = 'experiments/prospective-development-transport-v1/protocol.json'
SNAPSHOT = 'results/baseline-single-record-recovery-v1-root/full300-after090.json'
CORE = 'src/synsteg_audit/shared_dictionary.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def bound_inputs(root, manifest):
    require(manifest['commit'] == COMMIT, 'wrong immutable input commit')
    require(manifest['actual_git_readback'] is True, 'missing readback declaration')
    paths = [x['path'] for x in manifest['files']]
    require(len(paths) == 21 and len(set(paths)) == 21, 'incomplete/duplicate input inventory')
    expected = {CORE, PUBLIC_PROTOCOL, SNAPSHOT,
                'data/shared_dictionary_recovery_v1.json', 'data/shared_dictionary_recovery_v2.json'}
    expected.update(f'experiments/baseline-single-record-recovery-v1/inputs/prompts/order{n:03d}.jsonl' for n in OBSERVED)
    for order in OBSERVED:
        pattern = rf'runs/baseline-single-record-recovery-v1/order{order:03d}/run/attempts/[0-9a-f]{{64}}\.json'
        matches = [p for p in paths if re.fullmatch(pattern, p)]
        require(len(matches) == 1, 'expected attempt inventory mismatch')
        expected.add(matches[0])
    require(set(paths) == expected, 'expected input inventory mismatch')
    require(root.is_dir() and not root.is_symlink(), 'input root missing or linked')
    for item in manifest['files']:
        rel = Path(item['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe input path')
        target = root / rel
        parent = target.parent
        while parent != root:
            require(not parent.is_symlink(), 'linked input ancestor')
            parent = parent.parent
        require(target.is_file() and not target.is_symlink(), 'missing or linked input')
        data = target.read_bytes()
        require(len(data) == item['bytes'] and sha(data) == item['sha256'], 'input byte mismatch: ' + item['path'])
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        require(blob == item['git_blob_sha1'], 'Git blob mismatch: ' + item['path'])
    return set(paths)


def frame_rows(order, dictionary_id, capacity):
    return [dict(source_order=order, dictionary_id=dictionary_id, framing=frame,
                 payload_bits=payload, required_bits=payload + overhead,
                 capacity_bits=capacity,
                 capacity_feasible=None if capacity is None else capacity >= payload + overhead,
                 delivery_status='not_evaluated')
            for frame, overhead in FRAMES.items() for payload in PAYLOAD_BITS]


def validate_snapshot(snapshot):
    records = snapshot['records']
    require(len(records) == 300, 'full300 denominator changed')
    keys = [(r['model_alias'], r['source_order']) for r in records]
    require(len(set(keys)) == 300, 'duplicate model/source identity')
    require(set(keys) == {(m, n) for m in ['qwen05', 'tinyllama11', 'falcon3_1b'] for n in range(1, 101)}, 'full300 identity grid changed')
    require(dict(Counter(r['status'] for r in records)) == snapshot['counts'], 'snapshot counts disagree')
    selected = {r['source_order']: r for r in records if r['model_alias'] == 'tinyllama11' and r['source_order'] in ORDERS}
    require(len(selected) == 18, 'recovery denominator changed')
    require([n for n in ORDERS if selected[n]['status'] == 'completed'] == OBSERVED, 'available prefix changed; use a new diagnostic ID')
    require(all(selected[n]['status'] == 'not_attempted' for n in range(91, 101)), 'unexpected missing-cover status')
    return selected


def load_core(root):
    spec = importlib.util.spec_from_file_location('frozen_shared_dictionary_capacity_v1', root / CORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(root, manifest, plan):
    import cryptography
    require(plan['experiment_id'] == ID and plan['source_orders'] == ORDERS and plan['available_orders'] == OBSERVED, 'plan identity changed')
    require(plan['runtime'] == {'python': platform.python_version(), 'cryptography': cryptography.__version__}, 'runtime differs from recorded plan')
    paths = bound_inputs(root, manifest)
    inherited = load(root / PUBLIC_PROTOCOL)
    common = inherited['arms']['public_common']
    require(common['min_group_size'] == 1 and common['max_bits_per_slot'] == 2, 'frozen public widths changed')
    selected = validate_snapshot(load(root / SNAPSHOT))
    core = load_core(root)
    dictionaries = {d: core.PublicDictionary.from_dict(load(root / f'data/shared_dictionary_recovery_{d}.json')) for d in ['v1', 'v2']}
    pinned = {x['path']: x['sha256'] for x in inherited['resource_pins']}
    actual = {x['path']: x['sha256'] for x in manifest['files']}
    require(all(actual[p] == pinned[p] for p in [CORE, 'data/shared_dictionary_recovery_v1.json', 'data/shared_dictionary_recovery_v2.json']), 'frozen codec/dictionary pin differs')
    dictionary_rows, conditions, lengths = [], [], []
    for order in ORDERS:
        record = selected[order]
        if order in OBSERVED:
            prefix = f'runs/baseline-single-record-recovery-v1/order{order:03d}/run/attempts/'
            matches = [p for p in paths if p.startswith(prefix)]
            require(len(matches) == 1, 'attempt is not unique')
            attempt = load(root / matches[0])
            prompt_path = f'experiments/baseline-single-record-recovery-v1/inputs/prompts/order{order:03d}.jsonl'
            prompt = load(root / prompt_path)
            require(attempt['status'] == 'completed' and attempt['model'] == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'wrong attempt status/model')
            require(attempt['record']['record_id'] == prompt['record_id'] == record['record_id'], 'cover/source identity mismatch')
            require(attempt['record']['split'] == prompt['split'] == 'development', 'nondevelopment input')
            require(attempt['record']['prompt_sha256'] == prompt['prompt_sha256'] == sha(prompt['prompt'].encode()), 'prompt binding mismatch')
            require(prompt['excerpt_sha256'] == sha(prompt['excerpt'].encode()), 'excerpt binding mismatch')
            cover = attempt['result']['cover_text']
            require(isinstance(cover, str) and len(cover) > 0, 'invalid completed cover')
            cover_sha = sha(cover.encode())
            source_words = len(prompt['excerpt'].split())
            require(source_words == prompt['excerpt_words'], 'source word count mismatch')
            lengths.append(dict(source_order=order, record_id=record['record_id'], source_words=source_words,
                                target_words=prompt['target_words'], cover_words=len(cover.split()),
                                requested_expansion=prompt['target_words']/source_words,
                                actual_expansion=len(cover.split())/source_words,
                                cover_sha256=cover_sha, cover_source=matches[0], source_prompt=prompt_path))
        else:
            cover, cover_sha = None, None
            lengths.append(dict(source_order=order, record_id=record['record_id'], source_words=None,
                                target_words=None, cover_words=None, requested_expansion=None,
                                actual_expansion=None, cover_sha256=None, cover_source=None, source_prompt=None))
        for did, dictionary in dictionaries.items():
            protocol = core.Protocol(min_group_size=1, max_bits_per_slot=2, framing='plain_length16')
            if cover is None:
                capacity, slots, invariant_sha = None, None, None
            else:
                analysis = core.analyze_text(cover, dictionary, protocol)
                capacity, invariant_sha = analysis.capacity_bits, analysis.normalization_sha256
                slots = [dict(index=s.index, token_index=s.token_index, start=s.start, end=s.end,
                              surface=s.surface, class_id=s.class_id, case_style=s.case_style, width=s.width)
                         for s in analysis.slots]
                require(capacity == sum(s['width'] for s in slots), 'slot sum mismatch')
            dictionary_rows.append(dict(source_order=order, record_id=record['record_id'], dictionary_id=did,
                                        dictionary_sha256=dictionary.sha256, saved_cover_status=record['status'],
                                        cover_sha256=cover_sha, capacity_bits=capacity,
                                        eligible_slots=None if slots is None else len(slots), slots=slots,
                                        normalization_sha256=invariant_sha, source_fidelity='not_evaluated',
                                        capacity_status='measured' if cover is not None else 'cover_unavailable'))
            conditions.extend(frame_rows(order, did, capacity))
    aggregate = {}
    for did in dictionaries:
        values = [x['capacity_bits'] for x in dictionary_rows if x['dictionary_id'] == did and x['capacity_bits'] is not None]
        aggregate[did] = dict(measured_covers=len(values), unavailable_covers=18-len(values), raw_bits=sum(values),
                              zero_capacity_covers=sum(v == 0 for v in values), max_raw_bits=max(values),
                              min_raw_bits=min(values), mean_raw_bits=sum(values)/len(values))
    denominator = aggregate['v1']['raw_bits']
    total = dict(experiment_id=ID, schema='recovery-capacity-diagnostic-v1', input_commit=COMMIT,
                 scope='posthoc mechanical capacity; no encode/decode, delivery, quality or security claim',
                 model_calls=0, encoder_calls=0, decoder_calls=0, paid_calls=0, closed_test_opened=False,
                 source_orders=ORDERS, available_orders=OBSERVED, cover_denominator=18, observed_covers=8,
                 missing_covers=10, dictionary_rows_denominator=36, measured_dictionary_rows=16,
                 missing_dictionary_rows=20, frame_condition_denominator=288,
                 measured_frame_conditions=128, missing_frame_conditions=160,
                 model_alias='tinyllama11', full300_snapshot_counts=load(root/SNAPSHOT)['counts'],
                 parameters=dict(min_group_size=1, max_bits_per_slot=2, framings=FRAMES, payload_bits=PAYLOAD_BITS),
                 runtime=plan['runtime'], dictionary_aggregates=aggregate,
                 v2_raw_bit_loss_fraction=None if denominator == 0 else 1-aggregate['v2']['raw_bits']/denominator,
                 lengths=lengths, dictionary_rows=dictionary_rows, frame_conditions=conditions,
                 limitations=[
                     'Eight observed texts are reused recovery outputs, not new generation or independent model trials.',
                     'Order91 is unavailable in this saved snapshot; this diagnostic does not authorize replay of its saved intent.',
                     'Missing output has null capacity, not zero or refusal.',
                     'Capacity feasibility is necessary, not observed delivery or semantic safety.',
                     'A key can permute fixed public labels; it cannot enlarge these fixed eligible slots.',
                     'No conclusions about generated dictionaries or key-controlled model generation follow from this fixed-dictionary diagnostic.',
                     'Source/output whitespace word counts describe expansion, not a causal explanation of hallucinations.',
                     'Other baseline covers, original/fixed per-cover candidate arms and historical frozen transport outcomes are outside this new diagnostic.'
                 ])
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'output destination exists; refuse to overwrite')
    plan = load(args.plan)
    require(sha(args.manifest.read_bytes()) == plan['input_manifest_sha256'], 'manifest does not match saved plan')
    result = evaluate(args.input_root, load(args.manifest), plan)
    result['input_manifest_sha256'] = sha(args.manifest.read_bytes())
    result['plan_sha256'] = sha(args.plan.read_bytes())
    result['script_sha256'] = sha(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ['experiment_id','observed_covers','missing_covers','dictionary_aggregates','v2_raw_bit_loss_fraction']}))


if __name__ == '__main__':
    main()
