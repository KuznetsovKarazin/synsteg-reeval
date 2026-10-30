#!/usr/bin/env python3
"""Read author ZIP and immutable covers; estimate candidate-stage inputs only.

Never imports author modules, calls a tokenizer/model/API, or embeds/decodes a
message. Reads only named source files, public experiment payload records and a
fixed completed snapshot. Original inputs are not modified.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = '793c419e19b09a038af439fceb0c8727f2d809e2e855444d66839fd8132ef790'
PREFIX = 'synsteg_reproduction/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--checkpoint', default='results/large-campaign-development-v2/batches/batch04/after-checkpoint.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--payload-map-output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.payload_map_output.exists()
    archive_bytes = args.archive.read_bytes()
    assert sha(archive_bytes) == ARCHIVE_SHA
    source_names = ['steganography/secret_key.py', 'steganography/core.py', 'steganography/gpt.py',
                    'steganography/helper.py', 'utils/prompts.py', 'utils/constants.py',
                    'models/pool_arguments.py', 'experiments/model_series/run_series.py',
                    'statistics_collection.py', 'requirements.txt']
    anchors = {
        'steganography/secret_key.py': ['MIN_DISTINCT_SHARE =', 'input_message = prompts.ALL_SYNONYMS',
            'prompt = prompts.ALL_SYNONYMS', 'for attempt in range(3)',
            'synonyms, usage = get_openai_json_output', 'pool.imap(generate_synonyms',
            'if secret_key_generation_body.additional_bits:', 'secret_key.append({key: binarize_synonyms'],
        'steganography/core.py': ['user_container =', 'for attempt in range(MAX_CAPACITY_ATTEMPTS)',
            'helper.divide_chunks(container.split(), 5)', 'secret_key, usage = generate_secret_key_mp',
            'secret_key = align_container_and_secret_key', 'encoded_message, n_embedded = embed',
            'if user_container:', 'fix_token_container_size(replacement_token'],
        'steganography/gpt.py': ['@backoff.on_exception', 'client = openai.OpenAI()',
            'model=constants.OPENAI_MODEL_SYNONYMS', 'temperature=temperature',
            'messages=[', 'response_format={'],
        'experiments/model_series/run_series.py': ['reports/gpt_omni_reports: container',
            'CODE =', 'constants.OPENAI_MODEL_CONTAINER', 'kw.pop("temperature"',
            'options=dict(num_predict=4096'],
    }
    evidence = []
    with ZipFile(args.archive) as archive:
        names = archive.namelist()
        for name in source_names:
            data = archive.read(PREFIX + name)
            lines = data.decode().splitlines()
            excerpts = [{'line': i + 1, 'text': line} for i, line in enumerate(lines)
                        if any(anchor in line for anchor in anchors.get(name, []))]
            evidence.append({'member': PREFIX + name, 'sha256': sha(data), 'bytes': len(data),
                             'line_evidence': excerpts})
        prompts = ast.parse(archive.read(PREFIX + 'utils/prompts.py').decode())
        prompt_values = {n.targets[0].id: ast.literal_eval(n.value) for n in prompts.body
                         if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        constants = ast.parse(archive.read(PREFIX + 'utils/constants.py').decode())
        keep = {'OPENAI_MODEL_CONTAINER', 'OPENAI_MODEL_SYNONYMS', 'CONTAINER_BUFFER',
                'CONTAINER_TEMPERATURE', 'MAX_BITS_PER_WORD', 'MAX_ADDITIONAL_BITS_MULTIPLIER'}
        defaults = {n.targets[0].id: ast.literal_eval(n.value) for n in constants.body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in keep}
        pool = {n: [] for n in (128, 256, 512)}
        for member in sorted(n for n in names if n.startswith(PREFIX + 'reports/gpt6_luna_missing/') and n.endswith('.json')):
            data = archive.read(member)
            record = json.loads(data)
            bits = record['message']
            assert len(bits) in pool and set(bits) <= {'0', '1'}
            pool[len(bits)].append({'archive_member': member, 'archive_record_sha256': sha(data),
                'archived_uuid': record['uuid'], 'payload_bits': bits, 'payload_sha256': sha(bits.encode())})
        assert {n: len(rows) for n, rows in pool.items()} == {128: 100, 256: 100, 512: 100}
        historical_tree = [n for n in names if '683e8cb/' in n]
        git_objects = [n for n in names if '/.git/' in n]
    root = args.root.resolve()
    config_path = root / 'configs/large_campaign_resume_v2.json'
    cohort = json.loads(config_path.read_text())['development_candidate']
    assert len(cohort['selected_record_metadata']) == 100
    payload_rows = []
    for order, meta in enumerate(cohort['selected_record_metadata'], 1):
        for length in (128, 256, 512):
            original = pool[length][order - 1]
            payload_rows.append({'source_order': order, 'record_id': meta['record_id'],
                'primary_payload_length': length, **original,
                'secondary16_bits': original['payload_bits'][:16],
                'secondary16_sha256': sha(original['payload_bits'][:16].encode())})
    payload_map = {'schema': 'prospective-development-transport-payload-map-v1',
        'source_archive_sha256': ARCHIVE_SHA, 'source_config_sha256': sha(config_path.read_bytes()),
        'rule': 'For each128/256/512 stratum, sort original luna archive member names; map nth member to nth fixed development source, shared by all3models. Secondary16 is the first16 bits of that same primary payload.',
        'input_rows': 300, 'new_random_payload_generation': False,
        'payloads_are_public_reused_synthetic_fixtures_not_secret_communications': True, 'rows': payload_rows}
    cp_path = root / args.checkpoint
    assert cp_path.name == 'after-checkpoint.json' and cp_path.parent.name.startswith('batch')
    cp_bytes = cp_path.read_bytes()
    cp = json.loads(cp_bytes)
    assert cp['status'] == 'verified' and cp['planned_model_record_pairs'] == 300
    snapshot = cp_path.parent / 'after'
    complete = [r for r in cp['records'] if r['status'] == 'completed']
    rows = []
    for row in complete:
        relative = f"runs/large-campaign-development-v2/models/{row['model_alias']}/attempts/{row['identity']}.json"
        raw = (snapshot / relative).read_bytes()
        assert sha(raw) == cp['run_files_sha256'][relative]
        attempt = json.loads(raw)
        words = attempt['result']['cover_text'].split()
        assert len(words) == row['whitespace_words']
        chunks = [words[i:i + 5] for i in range(0, len(words), 5)]
        requests = {}
        for b in (3, 5):
            system = prompt_values['ALL_SYNONYMS_GENERATION_PROMPT'].replace('N_SYNONYMS', str(2 ** b))
            user = [prompt_values['ALL_SYNONYMS_GENERATION_INPUT'].format(context=chunk) for chunk in chunks]
            requests[str(b)] = {'requested_synonym_slots': len(words) * (2 ** b),
                'logical_chunk_calls_one_pass': len(chunks),
                'system_plus_user_utf8_bytes_one_pass': sum(len(system.encode()) + len(s.encode()) for s in user),
                'system_prompt_sha256': sha(system.encode()),
                'ordered_user_prompts_sha256': sha(json.dumps(user, ensure_ascii=False, separators=(',', ':')).encode())}
        rows.append({'record_id': row['record_id'], 'identity': row['identity'],
            'model_alias': row['model_alias'], 'attempt_sha256': sha(raw), 'words': len(words),
            'current_archive_chunks5': len(chunks), 'historical_comment_chunks10_hypothetical': math.ceil(len(words) / 10),
            'requests': requests})
    result = {'schema': 'author-candidate-stage-readonly-feasibility-v1',
        'status': 'original_and_fixed_blocked_candidate_model_stage_not_run',
        'archive_sha256': ARCHIVE_SHA, 'archive_bytes': len(archive_bytes), 'author_members': evidence,
        'declared_model_defaults_from_source': defaults,
        'candidate_builder': {'kind': 'LLM JSON generation', 'wordnet_or_POS_dictionary_builder': False,
            'context': 'each5whitespace-word chunk of supplied cover in current ZIP core',
            'N_SYNONYMS': '2**bits_per_word', 'temperature_argument': 0.7,
            'JSON_shape_attempts_per_chunk_at_most': 3,
            'transport_retries_bounded': False,
            'transport_retry_reason': 'backoff.on_exception has no max_tries/max_time in saved gpt.py',
            'additional_bits_for_matched_native_regime': 0,
            'provider_seed_or_revision_pinned_by_author_client': False},
        'version_conflict': {'current_ZIP_chunk_words': 5,
            'historical_run_series_comment_chunk_words': 10, 'historical_commit_reference': '683e8cb',
            'historical_commit_tree_members_in_ZIP': len(historical_tree), 'git_object_members_in_ZIP': len(git_objects),
            'exact_historical_builder_restored': False},
        'snapshot': {'path': args.checkpoint, 'sha256': sha(cp_bytes), 'completed_covers': len(rows),
            'full_denominator': 300, 'rows': rows,
            'total_words': sum(r['words'] for r in rows),
            'current_chunks5_one_pass': sum(r['current_archive_chunks5'] for r in rows),
            'historical_chunks10_hypothetical': sum(r['historical_comment_chunks10_hypothetical'] for r in rows),
            'up_to_JSON_shape_attempts_excluding_unbounded_transport_retries': 3 * sum(r['current_archive_chunks5'] for r in rows),
            'requested_synonym_slots_b3': sum(r['requests']['3']['requested_synonym_slots'] for r in rows),
            'requested_synonym_slots_b5': sum(r['requests']['5']['requested_synonym_slots'] for r in rows)},
        'payload_map_sha256': sha((json.dumps(payload_map, ensure_ascii=False, indent=2) + '\n').encode()),
        'model_calls': 0, 'tokenizer_calls': 0, 'encode_decode_calls_on_new_covers': 0,
        'API_calls': 0, 'closed_test_opened': False,
        'limitations': ['Chunk/request/string-slot counts are deterministic planning quantities, not observed inference token usage, wall time, quality or capacity.',
            'Existing archive tables belong to different covers and were shortened during embedding; they cannot supply missing new-cover candidates.',
            'A local provider adapter would be a declared new provider/backend condition, not the same author API or exact historical experiment.']}
    for path, value in [(args.output, result), (args.payload_map_output, payload_map)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    assert args.archive.read_bytes() == archive_bytes and cp_path.read_bytes() == cp_bytes
    print(json.dumps({'status': result['status'], 'covers': len(rows),
        'words': result['snapshot']['total_words'], 'chunks5': result['snapshot']['current_chunks5_one_pass'],
        'payload_map_sha256': sha(args.payload_map_output.read_bytes()), 'receipt_sha256': sha(args.output.read_bytes())}))


if __name__ == '__main__':
    main()
