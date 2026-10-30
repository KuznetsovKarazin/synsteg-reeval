#!/usr/bin/env python3
"""Fixed public-dictionary capacity only; no encode/decode, models or network."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys

ID = 'public-dictionary-full-development-v3'
SOURCE = 'results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
GUARD = 'results/current-git-status-audit-20261028-v1/no-replay-guard.json'
CORE = 'src/synsteg_audit/shared_dictionary.py'
PUBLIC_PROTOCOL = 'experiments/prospective-development-transport-v1/protocol.json'
OLD_AUDIT = 'scripts/measure_recovery83_100_capacity_v1.py'
PINS = {
    SOURCE: '3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1',
    GUARD: 'ed7c17d68df5cc524ec8e441f3febb2a96e6d8108eda6f22f2ac154fae74db0c',
    CORE: '21dc3dcfb2b0cd6396424280fc2a47695481eb815b29ab2381cfdcf7f2d47d2f',
    PUBLIC_PROTOCOL: '1220950e061801e1aebeaf0c6773ef103618c2aa3c5689122b59daa3dde3f09b',
    OLD_AUDIT: 'fa964002743e79a6fce3f3a008d95eab138387b695ea095e529f675c9022ff0f',
    'data/shared_dictionary_recovery_v1.json': 'a6266ac3c120a081970e3b74024b8618f2d69c0c1ad98e812135f30d92343239',
    'data/shared_dictionary_recovery_v2.json': '5dafba7b2179b40be560bacfd9ffa783ef98350d07e8d4ac7919c3a0865e6aa6',
}
MODELS = {'tinyllama11': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
          'qwen05': 'Qwen/Qwen2.5-0.5B-Instruct', 'falcon3_1b': 'tiiuae/Falcon3-1B-Instruct'}
REVISIONS = {'tinyllama11':'fe8a4ea1ffedaf415f4da2f062534de366a451e6',
             'qwen05':'7ae557604adf67be50417f59c2c2f167def9a775',
             'falcon3_1b':'28ba2251970a01dd1edc7ba7dad2eb71216ccfdf'}
FRAMES = {'plain_length16': 16, 'aes_siv': 144}
PRIMARY = [128, 256, 512]
RUNTIME = {'python': '3.12.14', 'cryptography': '50.0.2'}
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
HEX40 = re.compile(r'[0-9a-f]{40}\Z')


def require(value, message):
    if not value: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode())


def parse(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    def bad(value): raise ValueError('nonfinite JSON constant')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    def finite(item):
        if isinstance(item, float): require(math.isfinite(item), 'nonfinite JSON number')
        elif isinstance(item, dict):
            for v in item.values(): finite(v)
        elif isinstance(item, list):
            for v in item: finite(v)
    finite(value); return value


def relative(path):
    require(isinstance(path, str) and path, 'missing relative path')
    value = Path(path)
    require(not value.is_absolute() and '..' not in value.parts and str(value) == path, 'unsafe path')
    return value


def inventory(root):
    for p in (root, *root.parents): require(not p.is_symlink(), 'linked evidence ancestor')
    require(root.is_dir(), 'missing input directory')
    result = {}
    for p in root.rglob('*'):
        require(not p.is_symlink(), 'linked input')
        if p.is_file(): result[str(p.relative_to(root))] = sha(p.read_bytes())
    return result


def frozen_plan(plan):
    require(plan['experiment_id'] == ID and plan['scope'] == 'posthoc_mechanical_capacity_only', 'wrong plan')
    require(plan['models'] == ['tinyllama11', 'qwen05'] and plan['source_orders'] == list(range(1, 101)), 'changed full100 model cohort')
    require(plan['primary_payload_bits'] == PRIMARY and plan['secondary_payload_bits'] == 16
            and plan['secondary_repeated_per_primary_parent'] is True, 'changed payload grid')
    require(plan['frames'] == FRAMES and plan['min_group_size'] == 1 and plan['max_bits_per_slot'] == 2, 'changed frozen capacity definitions')
    require(plan['resource_pins'] == PINS and plan['runtime'] == RUNTIME, 'changed resource/runtime pins')
    import cryptography
    require({'python': platform.python_version(), 'cryptography': cryptography.__version__} == RUNTIME, 'different runtime')


def validate_inputs(root, manifest, plan):
    require(manifest['schema'] == ID + '-input' and manifest['actual_git_readback'] is True, 'unverified input manifest')
    require(isinstance(manifest['snapshot_commit'], str) and HEX40.fullmatch(manifest['snapshot_commit']), 'missing snapshot commit')
    pins = manifest['files_sha256']
    for path, digest in pins.items():
        relative(path); require(isinstance(digest, str) and HEX64.fullmatch(digest), 'invalid digest')
    require(inventory(root) == pins, 'changed/missing/orphan input bytes')
    require(all(pins.get(path) == digest for path, digest in PINS.items()), 'frozen resource pin mismatch')
    sources = [parse(line) for line in (root/SOURCE).read_bytes().splitlines() if line.strip()]
    require(len(sources) == 100 and len({r['record_id'] for r in sources}) == 100, 'source denominator differs')
    for row in sources:
        require(row['split'] == 'development' and sha(row['prompt'].encode()) == row['prompt_sha256']
                and sha(row['excerpt'].encode()) == row['excerpt_sha256']
                and len(row['excerpt'].split()) == row['excerpt_words'] > 0, 'source binding differs')
    snapshot_path = str(relative(manifest['snapshot_path']))
    require(snapshot_path in pins, 'unbound full300 snapshot')
    snapshot = parse((root/snapshot_path).read_bytes())
    require(snapshot['schema'] == 'mixed-development-full300-merger-v2-result'
            and snapshot['snapshot_ref'] == manifest['snapshot_commit'] and snapshot['full_denominator'] == 300,
            'wrong full300 snapshot')
    records = snapshot['records']
    require(all(type(r['source_order']) is int for r in records),'snapshot source_order must be integer')
    keys = [(r['model_alias'], r['source_order']) for r in records]
    require(len(records) == len(set(keys)) == 300 and set(keys) == {(a,n) for a in MODELS for n in range(1,101)}, 'full300 identity grid differs')
    require(dict(Counter(r['status'] for r in records)) == snapshot['counts'], 'snapshot counts differ')
    grid = {(r['model_alias'], r['source_order']): r for r in records}
    for (alias, n), row in grid.items():
        require(row['record_id'] == sources[n-1]['record_id'], 'snapshot source binding differs')
        require(row['status'] in ['completed', 'not_attempted', 'failed', 'reserved_outcome_unknown'], 'unexpected cover status')
    require(grid['tinyllama11',82]['status'] == 'reserved_outcome_unknown'
            and grid['tinyllama11',91]['status'] == 'not_attempted', '82/91 history changed')
    guard = parse((root/GUARD).read_bytes())
    require(snapshot['scheduling_overlays'] == guard['blocked_baseline_intents'], 'scheduling overlay differs')
    evidence=snapshot['new_shard_evidence']
    evidence_keys=[(r['model_alias'],r['source_order']) for r in evidence]
    require(all(type(r['source_order']) is int for r in evidence) and len(set(evidence_keys))==len(evidence_keys),'duplicate/mistyped new-shard evidence')
    new_evidence=dict(zip(evidence_keys,evidence))
    indexed = manifest['cover_sources']
    require(isinstance(indexed, list) and len(indexed) == 100 * len(plan['models']), 'incomplete cover source index')
    indexed_keys = [(x['model_alias'],x['source_order']) for x in indexed]
    require(len(set(indexed_keys)) == len(indexed) and set(indexed_keys) == {(a,n) for a in plan['models'] for n in range(1,101)}, 'cover source identity grid differs')
    allowed, attempt_paths, mapped = set(PINS) | {snapshot_path}, set(), {}
    for item in indexed:
        require(set(item) == {'model_alias','source_order','record_id','status','attempt_path'}, 'cover source keys differ')
        alias, order = item['model_alias'], item['source_order']; src=sources[order-1]; row=grid[alias,order]
        require(type(order) is int and item['record_id'] == src['record_id'] and item['status'] == row['status'], 'cover index differs from snapshot')
        path = item['attempt_path']
        if row['status'] != 'completed':
            require(path is None, 'noncompleted cover cannot supply measured text'); mapped[alias,order]=None; continue
        relative(path); require(path in pins and path not in attempt_paths, 'missing/duplicate attempt source')
        attempt = parse((root/path).read_bytes()); metadata={k:src[k] for k in ['genre','prompt_sha256','record_id','source_group_id','split','target_words','topic']}
        require(attempt['status'] == 'completed' and attempt['provider'] == 'local' and attempt['model'] == MODELS[alias]
                and attempt['identity'] == canonical(['local',MODELS[alias],src['record_id']])
                and attempt['record'] == metadata and type(attempt.get('automatic_retries')) is int
                and attempt['automatic_retries']==0, 'attempt/source/model/retry binding differs')
        result=attempt['result']
        require(result.get('returned_model')==MODELS[alias] and result.get('revision')==REVISIONS[alias]
                and result.get('dtype')=='float32' and type(result.get('seed')) is int
                and result['seed']==int(canonical([20261024,MODELS[alias],src['record_id']])[:8],16),
                'returned model/revision/dtype/seed differs')
        cover = attempt['result']['cover_text']
        require(isinstance(cover,str) and bool(cover), 'completed cover absent')
        if (alias,order) in new_evidence or alias!='tinyllama11' or order>=92:
            ev=new_evidence.get((alias,order),{})
            require(ev.get('status')=='completed' and ev.get('attempt_sha256')==pins[path]
                    and ev.get('cover_sha256')==sha(cover.encode()),'new attempt/cover differs from final snapshot evidence')
        mapped[alias,order]=(cover,path); allowed.add(path); attempt_paths.add(path)
    require(set(pins) == allowed, 'unreferenced evidence input')
    return sources, grid, mapped, snapshot


def core_module(root):
    name = 'frozen_shared_dictionary_full_development_v3'
    spec=importlib.util.spec_from_file_location(name,root/CORE); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module; old=sys.dont_write_bytecode; sys.dont_write_bytecode=True
    try: spec.loader.exec_module(module)
    finally: sys.dont_write_bytecode=old
    return module


def frame_rows(alias, order, did, capacity, status):
    rows=[]
    for frame, overhead in FRAMES.items():
        for kind in ['primary','secondary16']:
            for parent in PRIMARY:
                payload=parent if kind == 'primary' else 16
                feasible=None if capacity is None else capacity >= payload+overhead
                rows.append(dict(model_alias=alias,source_order=order,dictionary_id=did,framing=frame,
                                 payload_kind=kind,parent_primary_bits=parent,payload_bits=payload,
                                 frame_overhead_bits=overhead,required_bits=payload+overhead,capacity_bits=capacity,
                                 capacity_feasible=feasible,
                                 feasibility_status=status if feasible is None else 'mechanically_feasible' if feasible else 'mechanically_insufficient_capacity',
                                 delivery_status='not_evaluated',semantic_usability='not_evaluated'))
    return rows


def evaluate(root, manifest, plan):
    frozen_plan(plan); sources, grid, covers, snapshot=validate_inputs(root,manifest,plan)
    core=core_module(root); inherited=parse((root/PUBLIC_PROTOCOL).read_bytes())
    require(inherited['arms']['public_common']['min_group_size'] == 1
            and inherited['arms']['public_common']['max_bits_per_slot'] == 2
            and inherited['payloads']['primary_lengths'] == PRIMARY, 'prior protocol differs')
    dictionaries={d:core.PublicDictionary.from_dict(parse((root/f'data/shared_dictionary_recovery_{d}.json').read_bytes())) for d in ['v1','v2']}
    for frame, overhead in FRAMES.items():
        require(core.Protocol(min_group_size=1,max_bits_per_slot=2,framing=frame).frame_overhead_bits == overhead,'prior frame overhead differs')
    dictionary_rows, conditions, lengths=[],[],[]
    for alias in plan['models']:
        for order in range(1,101):
            src=sources[order-1]; record=grid[alias,order]; saved=covers[alias,order]
            cover, path=saved if saved else (None,None); cover_sha=None if cover is None else sha(cover.encode())
            lengths.append(dict(model_alias=alias,source_order=order,record_id=src['record_id'],cover_status=record['status'],
                                scheduling_status='blocked_reserved_intent_evidence_gap' if alias=='tinyllama11' and order==91 else None,
                                source_words=src['excerpt_words'],target_words=src['target_words'],
                                cover_words=None if cover is None else len(cover.split()),requested_expansion=src['target_words']/src['excerpt_words'],
                                actual_expansion=None if cover is None else len(cover.split())/src['excerpt_words'],
                                cover_sha256=cover_sha,attempt_path=path))
            for did,dictionary in dictionaries.items():
                capacity,slots,invariant,error=None,None,None,None
                status='cover_unavailable'
                if cover is not None:
                    try:
                        analysis=core.analyze_text(cover,dictionary,core.Protocol(min_group_size=1,max_bits_per_slot=2,framing='plain_length16'))
                        capacity,invariant=analysis.capacity_bits,analysis.normalization_sha256
                        slots=[dict(index=s.index,token_index=s.token_index,start=s.start,end=s.end,surface=s.surface,
                                    class_id=s.class_id,case_style=s.case_style,width=s.width) for s in analysis.slots]
                        require(type(capacity) is int and capacity>=0 and capacity==sum(s['width'] for s in slots),'capacity/slot sum differs')
                        status='measured'
                    except Exception as exc:
                        capacity,slots,invariant=None,None,None; status='analysis_error'; error=type(exc).__name__+': '+str(exc)
                dictionary_rows.append(dict(model_alias=alias,source_order=order,record_id=src['record_id'],dictionary_id=did,
                                            dictionary_sha256=dictionary.sha256,cover_status=record['status'],cover_sha256=cover_sha,
                                            capacity_status=status,capacity_bits=capacity,eligible_slots=None if slots is None else len(slots),
                                            slots=slots,normalization_sha256=invariant,analysis_error=error,
                                            source_fidelity='not_evaluated',semantic_substitutability='not_evaluated'))
                conditions.extend(frame_rows(alias,order,did,capacity,status))
    models={}
    for alias in plan['models']:
        rows=[r for r in dictionary_rows if r['model_alias']==alias]; agg={}
        observed=sum(covers[alias,n] is not None for n in range(1,101))
        for did in dictionaries:
            selected=[r for r in rows if r['dictionary_id']==did]; values=[r['capacity_bits'] for r in selected if r['capacity_bits'] is not None]
            agg[did]=dict(cover_denominator=100,available_saved_covers=observed,measured_covers=len(values),
                          unavailable_covers=100-observed,analysis_errors=sum(r['capacity_status']=='analysis_error' for r in selected),
                          observed_raw_bits=sum(values),zero_capacity_covers=sum(v==0 for v in values),
                          min_raw_bits=min(values) if values else None,max_raw_bits=max(values) if values else None,
                          mean_raw_bits_among_measured=sum(values)/len(values) if values else None)
        paired=[(next(r['capacity_bits'] for r in rows if r['source_order']==n and r['dictionary_id']=='v1'),
                 next(r['capacity_bits'] for r in rows if r['source_order']==n and r['dictionary_id']=='v2')) for n in range(1,101)]
        paired=[p for p in paired if None not in p]; v1=sum(p[0] for p in paired);v2=sum(p[1] for p in paired)
        feasibility=[]
        for did in dictionaries:
            for frame in FRAMES:
                for kind in ['primary','secondary16']:
                    for parent in PRIMARY:
                        cells=[c for c in conditions if (c['model_alias'],c['dictionary_id'],c['framing'],c['payload_kind'],c['parent_primary_bits'])==(alias,did,frame,kind,parent)]
                        yes=sum(c['capacity_feasible'] is True for c in cells);no=sum(c['capacity_feasible'] is False for c in cells)
                        feasibility.append(dict(dictionary_id=did,framing=frame,payload_kind=kind,parent_primary_bits=parent,
                                                full_denominator=100,feasible_covers=yes,insufficient_capacity_covers=no,
                                                null_covers=100-yes-no,feasible_fraction_all100=yes/100,
                                                feasible_fraction_measured=yes/(yes+no) if yes+no else None))
        models[alias]=dict(cover_denominator=100,available_saved_covers=observed,unavailable_covers=100-observed,
                           cover_status_counts=dict(Counter(grid[alias,n]['status'] for n in range(1,101))),
                           dictionary_row_denominator=200,frame_condition_denominator=2400,dictionary_aggregates=agg,
                           paired_capacity_covers=len(paired),v2_raw_bit_loss_fraction=None if v1==0 else 1-v2/v1,
                           feasibility=feasibility)
    return dict(schema=ID+'-result',experiment_id=ID,snapshot_commit=manifest['snapshot_commit'],
                scope='Posthoc mechanical raw capacity and necessary framing thresholds only; no observed delivery or usable semantic/factual payload claim.',
                model_calls=0,tokenizer_calls=0,encoder_calls=0,decoder_calls=0,paid_calls=0,closed_test_opened=False,
                historical_completed_cover_bytes_reused=True,historical_terminal_receipts_reaudited=False,
                runtime=RUNTIME,models=models,lengths=lengths,dictionary_rows=dictionary_rows,frame_conditions=conditions,
                full300_counts_context_only=snapshot['counts'],scheduling_overlays=snapshot['scheduling_overlays'],
                limitations=['Missing cover capacity is null, never zero or an observed encoder refusal.',
                             'Secondary16 cells repeat the same capacity threshold for three primary parents; cells are not independent texts.',
                             'Feasible/all100 is an availability-aware count with nulls retained separately, not a missing-as-failure estimate.',
                             'No source-support, factual accuracy, contextual synonym safety, security or delivery is inferred.',
                             'A key cannot increase the fixed eligible slots of these frozen public dictionaries.',
                             'Qwen is a separate full100 denominator; no capacity totals or means are pooled across models.',
                             'Frozen historical transport and prior recovery-capacity results are not modified.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--expected-manifest-sha256',required=True);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--expected-plan-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raw=args.manifest.read_bytes();plan_raw=args.plan.read_bytes()
    require(sha(raw)==args.expected_manifest_sha256 and sha(plan_raw)==args.expected_plan_sha256,'external manifest/plan binding differs')
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.input_root.resolve()),'output must be new and outside inputs')
    require(all(not p.is_symlink() for p in (args.output,*args.output.parents)),'linked output/ancestor refused')
    result=evaluate(args.input_root,parse(raw),parse(plan_raw));result.update(input_manifest_sha256=sha(raw),plan_sha256=sha(plan_raw),script_sha256=sha(Path(__file__).read_bytes()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'experiment_id':ID,'models':{a:{k:v[k] for k in ['cover_denominator','available_saved_covers','unavailable_covers','v2_raw_bit_loss_fraction']} for a,v in result['models'].items()}}))


if __name__=='__main__':main()
