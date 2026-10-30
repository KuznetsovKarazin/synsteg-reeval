#!/usr/bin/env python3
"""Read-only post-run checks; never import a model or call an inference server.

This independently traverses raw generation evidence and recomputes all capacity
outcomes. It shares the frozen codec implementation, not the production evaluator
or its summary function. The report is new evidence, not a preregistered analysis.
"""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synsteg_audit import shared_dictionary as base
from synsteg_audit import universal_context as universal

STAGE = 'results/long-summary-development-qwen15b-gguf-v1'
FREEZE = '02ac3ff7086a2da3ebfa95e810f0bfd96c2c87af'
SELECTED = [2, 5, 6, 7, 8, 9, 10, 11]
METHODS = ['base_v1', 'base_v2', 'universal_morph_v3']
ARMS = [('plain_length16', 8), ('plain_length16', 16), ('aes_siv', 8)]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def normal(value):
    return json.loads(canonical(value))


def check(value, label):
    if not value:
        raise ValueError(label)


def fixture(label, group, length=32):
    return hashlib.sha256(('PUBLIC-UNIVERSAL-CONTEXT-' + label + '-v1\0' + group).encode()).digest()[:length]


def signature(analysis):
    value = asdict(analysis)
    value.pop('virtual_normalization_sha256')
    return sha(canonical(value))


def alias_check(text, dictionary, policy, analysis):
    """Independent span substitution; exhaustive individual, two joint aliases."""
    expected = signature(analysis)
    raw = base.analyze_text(text, dictionary, base.Protocol(1, 1))
    classes = {entry.class_id: entry.words for entry in dictionary.classes}
    def style(word, name):
        return word.upper() if name == 'upper' else word.capitalize() if name == 'title' else word
    individual = []
    for slot in raw.slots:
        for word in classes[slot.class_id]:
            replacement = style(word, slot.case_style)
            if replacement == slot.surface:
                continue
            changed = text[:slot.start] + replacement + text[slot.end:]
            observed = signature(universal.analyze_text(changed, dictionary, policy))
            check(observed == expected, 'Alias eligibility changed')
            individual.append({'token_index': slot.token_index, 'replacement': replacement,
                               'signature': observed, 'equal': True})
    joint = []
    for choice in (0, -1):
        changed = text
        for slot in reversed(raw.slots):
            changed = changed[:slot.start] + style(classes[slot.class_id][choice], slot.case_style) + changed[slot.end:]
        observed = signature(universal.analyze_text(changed, dictionary, policy))
        check(observed == expected, 'Joint alias eligibility changed')
        joint.append({'representative': 'first' if choice == 0 else 'last', 'signature': observed, 'equal': True})
    return expected, individual, joint


def receiver(root, stego, key, dictionary, protocol):
    """Fresh frozen raw receiver, with independently constructed request."""
    request = {'stego_text': stego, 'key_hex': key.hex(), 'public_dictionary': dictionary.to_dict(),
               'public_protocol': protocol.to_dict()}
    with tempfile.TemporaryDirectory(prefix='independent-qwen-receiver-') as directory:
        child = subprocess.run([sys.executable, '-I', str(root / 'scripts/audit_qwen15b_capacity_v1.py'), 'raw-receiver'],
            input=canonical(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=directory, env={'PATH': os.defpath, 'PYTHONIOENCODING': 'utf-8'}, timeout=60, check=False)
    check(child.returncode == 0, 'Receiver process failed')
    response = json.loads(child.stdout)
    check(response['input_keys'] == sorted(request), 'Receiver received extra/missing inputs')
    check(response['dictionary_sha256'] == dictionary.sha256 and response['protocol_sha256'] == protocol.sha256,
          'Receiver public resource mismatch')
    check(response['io_guard_self_checks'] == {'file': True, 'socket': True, 'subprocess': True}, 'Receiver I/O guard')
    return bytes.fromhex(response['payload_hex']), {'returncode': child.returncode, 'input_keys': response['input_keys'],
        'io_guard_self_checks': response['io_guard_self_checks'], 'stdout_sha256': sha(child.stdout), 'stderr_sha256': sha(child.stderr)}


def audit(root):
    stage = root / STAGE
    summary = read(stage / 'run/summary.json')
    check(summary.get('post_run_hashes_verified') is True and summary.get('finished_at'), 'Run is not complete')
    runfiles = sorted(p for p in (stage / 'run').rglob('*') if p.is_file())
    capacityfiles = sorted(p for p in (stage / 'capacity-audit').rglob('*') if p.is_file())
    snapshot = {str(p.relative_to(root)): file_sha(p) for p in runfiles + capacityfiles}
    protocol, capacityprotocol, resources = (read(stage / name) for name in ('protocol.json', 'capacity-protocol.json', 'resources.json'))
    manifests = [protocol['input_hashes'], protocol['code_hashes'], capacityprotocol['source_files'], capacityprotocol['code_files'],
                 {STAGE + '/' + k: v for k, v in protocol['files'].items()}]
    frozen = {key: value for manifest in manifests for key, value in manifest.items()}
    for name, expected in frozen.items():
        check(file_sha(root / name) == expected, 'Frozen input changed: ' + name)
    plan = read(stage / 'plan.json')['records']
    ledger = read(stage / 'run/ledger.json')
    prompts = lines(stage / 'prompts.jsonl')
    sourcewindows = lines(root / 'data/development-full-revisions-v3/source-windows.jsonl')
    oldprompts = {r['record_id']: r for r in lines(root / 'results/long-summary-development-v3/prompts.jsonl')}
    check(len(plan) == len(ledger) == len(sourcewindows) == 20, 'Full20 denominator')
    check([r['plan_index'] for r in plan] == list(range(1, 21)), 'Plan order')
    check(len({r['record_id'] for r in plan}) == len({r['source_group_id'] for r in plan}) == 20, 'Unique IDs')
    check([r['plan_index'] for r in plan if r['planned_action'] == 'generate_once'] == SELECTED, 'Fixed eight selection')
    check(Counter(r['planned_action'] for r in plan) == Counter(generate_once=8, abstain_source_gate=5,
          not_selected_fixed_eight_budget=7), 'All abstentions retained')
    check([r['plan_index'] for r in prompts] == SELECTED, 'Exactly eight prompts')
    check(sorted(p.name for p in (stage / 'run/attempts').iterdir()) == ['%02d' % i for i in SELECTED], 'Exactly eight attempt directories')
    execution = read(stage / 'run/execution.json')
    check(execution['freeze_commit'] == FREEZE, 'Common inference freeze')
    check(execution['protocol_sha256'] == file_sha(stage / 'protocol.json'), 'Generation protocol hash')
    runtime = read(stage / 'preflight/runtime-preflight.json')['runtime']
    for entry in runtime['files']:
        path = (root / entry['path']).resolve()
        check(path.is_relative_to(root) and path.stat().st_size == entry['bytes'] and file_sha(path) == entry['sha256'], 'Runtime file hash')
    check(execution['server_sha256'] == file_sha(root / runtime['server_path']), 'Server hash')
    per_record = []
    prompt_map = {p['record_id']: p for p in prompts}
    model_paths = set()
    for expected, row, window in zip(plan, ledger, sourcewindows):
        check(all(row.get(k) == v for k, v in expected.items()), 'Frozen ledger changed')
        check(window['record_id'] == row['record_id'] and window['source_group_id'] == row['source_group_id'], 'Source identity')
        check(sha(window['source_window'].encode()) == row['source_window_sha256'] and len(window['source_window'].split()) == row['source_words'], 'Exact source window')
        check(row['split'] == window['split'] == 'development', 'Development only')
        if row['planned_action'] != 'generate_once':
            check(row['status'] == row['planned_action'] and row['generation_request_attempted'] is False and row.get('text') is None, 'Ungenerated rows')
            continue
        directory = stage / 'run/attempts' / ('%02d' % row['plan_index'])
        attempt = read(directory / 'attempt.json')
        check(all(row.get(k) == v for k, v in attempt.items()), 'Attempt/ledger mismatch')
        messages = prompt_map[row['record_id']]['messages']
        check(messages[1]['content'] == oldprompts[row['record_id']]['prompt'], 'Unchanged inherited user prompt')
        check(oldprompts[row['record_id']]['excerpt'] == window['source_window'], 'Full source included')
        check(read(directory / 'apply-template.request.json') == {'messages': messages}, 'Template request')
        rendered = ''.join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' for m in messages) + '<|im_start|>assistant\n'
        check(read(directory / 'apply-template.response.json')['prompt'] == rendered == (directory / 'rendered-prompt.txt').read_text(), 'Exact full rendered prompt')
        check(read(directory / 'tokenize.request.json') == {'content': rendered, 'add_special': True, 'parse_special': True}, 'Tokenizer request')
        tokens = read(directory / 'tokenize.response.json')['tokens']
        check(isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens), 'Prompt token IDs')
        check(row['prompt_tokens'] == len(tokens) and len(tokens) + 1536 + 1 <= 8192, 'Untruncated context gate')
        wanted = {k: v for k, v in protocol['sampling'].items() if k != 'seed_rule'}
        wanted.update(prompt=tokens, n_predict=1536, seed=20261024 + row['plan_index'], cache_prompt=False,
                      return_tokens=True, stream=False, ignore_eos=False, stop=[], id_slot=0, timings_per_token=True)
        check(read(directory / 'completion.request.json') == wanted, 'Exact completion request')
        response = read(directory / 'completion.response.json')
        accepted = response['generation_settings']
        for key, value in wanted.items():
            if key in ('prompt', 'cache_prompt', 'return_tokens', 'id_slot'):
                continue
            actual = accepted.get(key)
            check(math.isclose(actual, value, rel_tol=0, abs_tol=5e-7) if type(value) is float and type(actual) in (int, float) else actual == value,
                  'Accepted setting mismatch: ' + key)
        check(response['prompt'] == rendered and response['id_slot'] == 0, 'Returned prompt and slot')
        check(response['truncated'] is False and response['stop'] is True and response['stop_type'] == 'eos', 'Native stopping evidence')
        check(response['tokens_evaluated'] == len(tokens) == response['timings']['prompt_n'] and response['timings']['cache_n'] == 0, 'No omitted or cached prompt tokens')
        output_tokens = response['tokens']
        check(all(type(t) is int and t >= 0 for t in output_tokens), 'Native output token IDs')
        check(len(output_tokens) == response['tokens_predicted'] == response['timings']['predicted_n'] <= 1536, 'Observed native output count equality')
        text = response['content']
        check((directory / 'cover.txt').read_bytes() == text.encode() and row['text'] == text and row['text_sha256'] == sha(text.encode()), 'Exact output bytes/hash')
        words = len(text.split())
        check(row['status'] == 'generated' and text.strip() and row['schema_issues'] == [] and row['generation_request_attempted'], 'Generated state')
        check(words == row['output_words'] and row['within_soft_word_band'] == (row['preferred_word_band'][0] <= words <= row['preferred_word_band'][1]), 'Length measurement')
        command = read(directory / 'server-command.json')
        model_paths.add(Path(command[2]))
        expected_flags = ['--host', '127.0.0.1', '--port', command[6], '--threads', '6', '--threads-batch', '6', '--parallel', '1', '--ctx-size', '8192',
            '--gpu-layers', '0', '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--flash-attn', 'off', '--no-context-shift', '--no-warmup']
        check(command[0] == str(root / runtime['server_path']) and command[1] == '-m' and command[3:] == expected_flags, 'Server CPU/context flags')
        check(response['model'] == command[2], 'Returned model path')
        log = (directory / 'server.log').read_text()
        check('n_threads = 6' in log and 'n_slots = 1, n_ctx_slot = 8192' in log, 'Native thread/context evidence')
        logged_tasks = len(re.findall(r'processing task, is_child = 0', log))
        logged_releases = len(re.findall(r'stop processing: .*truncated = 0', log))
        check(logged_tasks in (0, 1) and logged_releases in (0, 1), 'No duplicated logged native tasks')
        check(row['wall_seconds'] < 600, 'Deadline not exceeded')
        per_record.append({'plan_index': row['plan_index'], 'record_id': row['record_id'], 'prompt_tokens': len(tokens), 'output_tokens': len(output_tokens),
            'tokens_predicted': response['tokens_predicted'], 'native_last_token_id': output_tokens[-1], 'output_words': words, 'target_words': row['target_words'],
            'within_soft_word_band': row['within_soft_word_band'], 'stop_type': response['stop_type'], 'truncated': False,
            'output_sha256': row['text_sha256'], 'wall_seconds': row['wall_seconds'], 'returned_settings_match_with_float32_tolerance': True,
            'native_log_task_entries':logged_tasks, 'native_log_release_entries':logged_releases,
            'native_log_confirms_one_complete_task': logged_tasks == logged_releases == 1})
    check(len(model_paths) == 1, 'One pinned model')
    model_path = next(iter(model_paths))
    check(model_path.stat().st_size == resources['model']['bytes'] and file_sha(model_path) == resources['model']['sha256'] == execution['model_sha256'], 'Actual model bytes')
    check(summary['planned_source_groups'] == 20 and summary['selected_generation_denominator'] == 8 and summary['generation_requests_attempted'] == 8 and
          summary['within_soft_word_band'] == sum(r['within_soft_word_band'] for r in per_record) and summary['status_counts_all20'] == dict(Counter(r['status'] for r in ledger)), 'Generation summary')
    dictionary_paths = {'base_v1': 'data/shared_dictionary_recovery_v1.json', 'base_v2': 'data/shared_dictionary_recovery_v2.json', 'universal_morph_v3': 'data/shared_dictionary_morphology_v3.json'}
    dictionaries = {name: base.PublicDictionary.from_dict(read(root / path)) for name, path in dictionary_paths.items()}
    policy = universal.UniversalPolicy.from_dict(read(root / 'results/morphology-development-v3/policy.json'), dictionaries['universal_morph_v3'])
    check(policy.sha256 == capacityprotocol['policy_sha256'] and {n:d.sha256 for n,d in dictionaries.items()} == capacityprotocol['public_dictionary_sha256'], 'Public codec resource hashes')
    records, arms = (lines(stage / 'capacity-audit' / name) for name in ('records.jsonl', 'arms.jsonl'))
    check(len(records) == 20 and len(arms) == 180, 'Complete capacity denominators')
    flattened, outcome_counts, capacities, deliveries = [], Counter(), {name: [] for name in METHODS}, []
    alias_individual_count = alias_joint_count = 0
    for record, row in zip(records, ledger):
        check(all(record[k] == row[k] for k in ('record_id', 'plan_index', 'source_group_id', 'planned_action', 'split')), 'Capacity row identity')
        check(record['text'] == row.get('text') and record['text_sha256'] == row.get('text_sha256'), 'Capacity exact output')
        state = 'available' if row['planned_action'] == 'generate_once' else 'source_gate_abstention' if row['planned_action'] == 'abstain_source_gate' else row['planned_action']
        check(record['availability'] == state and set(record['methods']) == set(METHODS), 'All methods/availability')
        key = fixture('KEY', row['source_group_id'])
        for method in METHODS:
            entry, dictionary = record['methods'][method], dictionaries[method]
            check(entry['semantic_safety_certified'] is False and entry['known_unsafe_reference'] == (method == 'base_v1'), 'Semantic distinction')
            analysis = None
            if state == 'available':
                analysis = universal.analyze_text(record['text'], dictionary, policy) if method == 'universal_morph_v3' else base.analyze_text(record['text'], dictionary, base.Protocol(1, 1))
                check(normal(asdict(analysis)) == entry['analysis'] and analysis.capacity_bits == entry['capacity_bits'], 'Independent capacity/analysis')
                capacities[method].append(analysis.capacity_bits)
                if method == 'universal_morph_v3':
                    ref, individual, joint = alias_check(record['text'], dictionary, policy, analysis)
                    saved = entry['alias_audit']
                    check(saved['reference_signature'] == ref and saved['individual_checks'] == individual and saved['all_equal'] is True, 'Individual alias evidence')
                    check([v['signature'] for v in saved['simultaneous_checks']] == [v['signature'] for v in joint], 'Joint alias evidence')
                    alias_individual_count += len(individual); alias_joint_count += len(joint)
            else:
                check(entry['capacity_bits'] is None and entry['analysis'] is None, 'Missing is not zero capacity')
            check(entry['input_status'] == state, 'Method availability')
            check(set(entry['arms']) == {f'{f}__{b}' for f,b in ARMS}, 'All three frames')
            for framing, bits in ARMS:
                arm_id = f'{framing}__{bits}'; arm = entry['arms'][arm_id]
                required = bits + (16 if framing == 'plain_length16' else 144)
                check(arm['required_frame_bits'] == required and arm['payload_bits'] == bits and arm['framing'] == framing and arm['semantic_success'] is None, 'Complete frame overhead')
                expected_status = state if state != 'available' else 'insufficient_capacity' if analysis.capacity_bits < required else 'decoded'
                check(arm['status'] == expected_status and arm['mechanical_success'] == (expected_status == 'decoded') and arm['encoding_called'] == (expected_status == 'decoded'), 'Independent arm outcome')
                if expected_status == 'decoded':
                    check(method != 'universal_morph_v3', 'This observed audit has raw-only deliveries')
                    payload = fixture('PAYLOAD', row['source_group_id'], bits // 8)
                    codec_protocol = base.Protocol(1, 1, framing)
                    encoded = base.encode(record['text'], payload, key, dictionary, codec_protocol)
                    check(encoded.stego_text == arm['stego_text'] and sha(encoded.stego_text.encode()) == arm['stego_sha256'], 'Deterministic encoding replay')
                    decoded, evidence = receiver(root, encoded.stego_text, key, dictionary, codec_protocol)
                    check(decoded == payload and arm['payload_hex'] == arm['decoded_payload_hex'] == payload.hex() and arm['receiver'] == evidence, 'Fresh receiver recovery')
                    before, after = record['text'].split(), encoded.stego_text.split()
                    changes = [{'token_index':i, 'original_token':x, 'stego_token':y} for i,(x,y) in enumerate(zip(before,after)) if x != y]
                    check(len(before) == len(after) and changes == arm['token_changes'] and len(changes) == encoded.changed_slots == arm['changed_slots'], 'All token changes preserved')
                    slots = {s.token_index for s in analysis.slots}
                    check(all(before[i] == after[i] for i in range(len(before)) if i not in slots), 'Protected spellings')
                    check(base.analyze_text(encoded.stego_text, dictionary, codec_protocol).normalization_sha256 == analysis.normalization_sha256, 'Receiver normalization invariance')
                    deliveries.append({'plan_index': row['plan_index'], 'method': method, 'arm_id': arm_id, 'changed_slots': len(changes), 'receiver_input_keys': evidence['input_keys']})
                outcome_counts[expected_status] += 1
                flattened.append({**{k:record[k] for k in ('audit_record_id','record_id','plan_index','source_group_id','planned_action','generation_status','availability','text_sha256')},
                                  'method':method, 'arm_id':arm_id, 'capacity_bits':entry['capacity_bits'], **arm})
    check(sorted(map(canonical, flattened)) == sorted(map(canonical, arms)), 'All180 flattened evidence rows match')
    capsummary = read(stage / 'capacity-audit/summary.json')
    check(capsummary['all180_outcomes'] == dict(outcome_counts) and capsummary['selected72_outcomes'] == dict(Counter(a['status'] for a in arms if a['planned_action'] == 'generate_once')), 'All outcome totals')
    for name, values in capacities.items():
        stats = {'count':len(values), 'min':min(values), 'max':max(values), 'mean':statistics.mean(values), 'zero_capacity':values.count(0)}
        check(capsummary['methods'][name]['gross_capacity_bits'] == stats, 'Capacity summary')
        for framing,bits in ARMS:
            arm_id=f'{framing}__{bits}'; subset=[a for a in arms if a['method']==name and a['arm_id']==arm_id]
            wanted={'all20_outcomes':dict(Counter(a['status'] for a in subset)), 'selected8_outcomes':dict(Counter(a['status'] for a in subset if a['planned_action']=='generate_once')),
                    'mechanical_successes':sum(a['mechanical_success'] for a in subset)}
            check(capsummary['methods'][name]['arms'][arm_id] == wanted, 'Per-method/frame denominators')
    capexec = read(stage / 'capacity-audit/execution.json')
    check(capexec['frozen_commit'] == FREEZE and capexec['capacity_protocol_sha256'] == file_sha(stage / 'capacity-protocol.json'), 'Capacity freeze')
    for name,digest in capexec['output_sha256'].items():
        check(file_sha(stage / 'capacity-audit' / name) == digest, 'Capacity output hash')
    for name,digest in capexec['generation_input_sha256'].items():
        check(file_sha(root / name) == digest, 'Capacity generation input hash')
    for name, digest in {**snapshot, **frozen}.items():
        check(file_sha(root / name) == digest, 'Evidence changed during independent audit')
    return {'schema':'qwen15b-gguf-v1-independent-postrun-review', 'completed_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed_with_evidence_limitations', 'common_pre_inference_freeze':FREEZE, 'independent_verifier_sha256':file_sha(Path(__file__)),
        'scope':'Raw execution evidence and complete mechanical remeasurement; no source-bound semantic quality assessment',
        'generation':{'all20_retained':True,'exact8_selected':SELECTED,'all8_native_eos':True,'truncated_outputs':0,'word_band_count':sum(r['within_soft_word_band'] for r in per_record),
            'model_bytes_and_all_runtime_files_rehashed':True,'sampling_float_absolute_tolerance':5e-7,'per_record':per_record},
        'capacity':{'all180_independently_recomputed':True,'outcomes':dict(outcome_counts),'capacities_by_selected_order':capacities,
                    'fresh_receiver_deliveries':deliveries,'individual_alias_checks':alias_individual_count,'joint_alias_checks':alias_joint_count},
        'evidence_sha256':snapshot,'frozen_input_sha256':frozen,'frozen_files_and_results_unchanged_during_audit':True,
        'limitations':['Shares frozen codec library and receiver implementation; independent orchestration is not a second codec implementation.',
            'Native server.log for plan_index05 contains startup only; its full raw request/response exists. Seven other logs show one completed task each. No cause for the missing task log is established.',
            'Raw requests/responses support eight observed calls; no external network or process trace proves absence of every possible unlogged event.',
            'The receiver Python I/O guard is not an OS sandbox; code/dependencies are loaded before the guard.',
            'Native token counts equal returned token-array lengths for this run, including terminal ID151645; no universal native-token convention is assumed.',
            'Alias checks cover every individual alias and two simultaneous representatives, not the global Cartesian product.',
            'No semantic safety, secrecy, population inference, or factual quality conclusion is made; raw v1 is known unsafe.',
            'No new generation, network requests, or closed-test inspection were performed.'],
        'new_model_calls':0,'paid_calls':0,'closed_test_opened':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    output=args.output or root/STAGE/'independent-postrun-review.json'
    if output.exists():
        raise FileExistsError('Independent evidence is never overwritten')
    report=audit(root)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({'status':report['status'],'report':str(output),'sha256':file_sha(output),'capacity':report['capacity']},indent=2))


if __name__=='__main__':
    main()
