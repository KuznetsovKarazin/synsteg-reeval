#!/usr/bin/env python3
"""Pure offline candidate-adapter tests and deterministic workload planning. No NN."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synsteg_audit.author_candidate_local_adapter_v1 import (
    ID, ARCHIVE_SHA256, MEMBERS, canonical, digest, load_author, make_plan,
    parse_response, resolve_replies, assemble_pool, save_exclusive)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('exclusive output required')
    checks = []
    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
        if not value:
            raise AssertionError(name)
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, TypeError, FileExistsError):
            check(name, True)
        else:
            check(name, False)
    source, prompts = load_author(args.archive)
    plan = make_plan(args.archive, 'Quick birds fly over quiet fields.', bits_per_word=3)
    check('exact_5word_chunking', [r['words'] for r in plan['requests']] ==
          [['Quick', 'birds', 'fly', 'over', 'quiet'], ['fields.']])
    check('exact_author_python_list_prompt', plan['requests'][0]['user'] ==
          "Text: ['Quick', 'birds', 'fly', 'over', 'quiet']")
    check('exact_author_system_b3_sha256', digest(plan['requests'][0]['system'].encode()) ==
          '54a9b7959b0b0861da6c91ae4836b629991828b4e297701d233382ee73d20539')
    b5 = make_plan(args.archive, 'word', bits_per_word=5)
    check('exact_author_system_b5_sha256', digest(b5['requests'][0]['system'].encode()) ==
          '2249d780037676a763ea5196bc743bd38f938cc06dd3ea2939799086dc23a608')
    rejects('bool_bits_rejected', lambda: make_plan(args.archive, 'word', bits_per_word=True))
    rejects('empty_cover_rejected', lambda: make_plan(args.archive, '  '))
    rejects('malformed_json_rejected', lambda: parse_response('{'))
    rejects('duplicate_key_rejected', lambda: parse_response('{"words":[],"words":[]}'))
    rejects('empty_words_rejected', lambda: parse_response('{"words":[]}'))
    rejects('extra_top_level_rejected', lambda: parse_response('{"words":[{"x":[]}],"note":1}'))
    rejects('flat_list_rejected', lambda: parse_response('{"words":["x"]}'))
    rejects('multi_base_entry_rejected', lambda: parse_response('{"words":[{"x":[],"y":[]}]}'))
    rejects('nonstring_candidate_rejected', lambda: parse_response('{"words":[{"x":[1]}]}'))
    rejects('nonfinite_rejected', lambda: parse_response('{"words":NaN}'))
    def words_reply(words):
        return {'state':'returned', 'raw_text':json.dumps({'words':[
            {word: ['variant'+str(n) for n in range(8)]} for word in words]}), 'usage': None}
    replies = [[words_reply(r['words'])] for r in plan['requests']]
    pool = assemble_pool(args.archive, plan, replies)
    check('complete_pre_payload_pool_without_payload', pool['status'] == 'complete_pre_payload_pool'
          and len(pool['aligned_pre_payload_pool']) == 6 and not pool['payload_seen'])
    check('raw_response_preserved_exact', pool['chunk_outcomes'][0]['attempts'][0]['raw_text'] == replies[0][0]['raw_text'])
    check('original_order_codes_preserved', pool['binarized_pre_payload_pool'][0]['Quick'] ==
          {format(n,'03b'):'variant'+str(n) for n in range(8)})
    check('author_punctuation_alignment_preserved', list(pool['aligned_pre_payload_pool'][-1]) == ['fields'])
    check('author_b3_needs_all8_distinct', source['binarize_synonyms']('x',[f'z{n}' for n in range(7)],3) == {'0':'x','1':'x'})
    check('author_b5_accepts29of32', len(source['binarize_synonyms']('x',[f'z{n}' for n in range(29)],5)) == 29)
    check('author_base_and_prefix_filter_preserved', source['binarize_synonyms']('x',['x','x y','x-y','z','z q'],1) == {'0':'x','1':'x'})
    deeply_nested_raw = '[' * 10000 + '0' + ']' * 10000
    deep_json_rejection = resolve_replies(plan['requests'][0],
        [{'state':'returned', 'raw_text':deeply_nested_raw}])
    check('deep_json_shape_failure_retains_raw',
          deep_json_rejection['outcome']=='shape_rejected' and
          deep_json_rejection['attempts'][0]['raw_text']==deeply_nested_raw and
          deep_json_rejection['attempts'][0]['rejection'].startswith('RecursionError:'))
    malformed_then_good = [[{'state':'returned','raw_text':'{'}, replies[0][0]], replies[1]]
    retry = assemble_pool(args.archive, plan, malformed_then_good)
    check('shape_failure_retained_before_acceptance', retry['status']=='complete_pre_payload_pool'
          and retry['chunk_outcomes'][0]['attempts'][0]['outcome']=='shape_rejected')
    exhausted = assemble_pool(args.archive, plan, [[{'state':'returned','raw_text':'{'}]*3, []])
    check('full_denominator_and_no_pool_on_failure', exhausted['requested_chunks']==2 and
          exhausted['chunk_outcomes'][0]['outcome']=='shape_attempts_exhausted' and
          exhausted['chunk_outcomes'][1]['outcome']=='not_attempted' and
          exhausted['aligned_pre_payload_pool'] is None)
    unknown = assemble_pool(args.archive,plan,[[{'state':'unknown','detail':'process ended before outcome'}], replies[1]])
    check('unknown_preserved_separately', unknown['chunk_outcomes'][0]['outcome']=='unknown'
          and unknown['status']=='candidate_resource_incomplete')
    rejects('unknown_never_retried',lambda: resolve_replies(plan['requests'][0],
            [{'state':'unknown'},replies[0][0]]))
    rejects('accepted_never_retried',lambda: resolve_replies(plan['requests'][0],
            [replies[0][0],replies[0][0]]))
    rejects('maximum_three_shape_attempts',lambda: resolve_replies(plan['requests'][0],
            [{'state':'returned','raw_text':'{'}]*4))
    altered=copy.deepcopy(plan);altered['payload']='101'
    rejects('payload_or_extra_plan_field_rejected',lambda: assemble_pool(args.archive,altered,replies))
    altered=copy.deepcopy(plan);altered['requests'][0]['user']='substitute'
    rejects('request_tampering_rejected',lambda: assemble_pool(args.archive,altered,replies))
    rejects('missing_denominator_rejected',lambda: assemble_pool(args.archive,plan,replies[:1]))
    check('plan_no_payload_dependency', canonical(make_plan(args.archive,plan['cover_text'])) == canonical(plan))
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'pool.json';save_exclusive(p,pool)
        rejects('existing_result_not_overwritten',lambda:save_exclusive(p,unknown))
        check('saved_pool_unchanged_after_refusal',json.loads(p.read_text())==pool)
        p=Path(d)/'wrong.zip';p.write_bytes(b'notauthor')
        rejects('archive_identity_before_source_execution',lambda:load_author(p))
    receipt_path=ROOT/'results/large-campaign-continuation-v3/git-receipts/after-batch22.json'
    receipt=json.loads(receipt_path.read_text())
    if receipt['expected_completed'] != 44:
        raise ValueError('fixed baseline44 snapshot required')
    rows=[]
    for path, sha in receipt['run_files_sha256'].items():
        if '/attempts/' not in path:
            continue
        raw=(ROOT/path).read_bytes()
        if digest(raw)!=sha:
            raise ValueError('immutable attempt hash mismatch')
        obj=json.loads(raw)
        text=obj['result']['cover_text']; words=text.split();chunks=(len(words)+4)//5
        rows.append({'path':path,'sha256':sha,'words':len(words),'chunks5':chunks,
                     'shape_attempt_upper_bound':3*chunks,
                     'requested_slots_b3':len(words)*8,'requested_slots_b5':len(words)*32})
    check('all44_prior_snapshot_attempts_verified',len(rows)==44)
    first='runs/large-campaign-development-v2/models/tinyllama11/attempts/ab6c0aa9dc8e261e92073379661280ddd3f185c9c8a6b03a419109c3cb2d34e9.json'
    obj=json.loads((ROOT/first).read_text())
    full_plan=make_plan(args.archive,obj['result']['cover_text'],bits_per_word=3)
    args.output.mkdir(parents=True)
    save_exclusive(args.output/'deep-json-rejection.json',deep_json_rejection)
    save_exclusive(args.output/'synthetic-pool.json',pool)
    save_exclusive(args.output/'synthetic-rejections.json',{'exhausted':exhausted,'unknown':unknown})
    save_exclusive(args.output/'first-cover-request-plan.json',full_plan)
    save_exclusive(args.output/'workload-baseline44.json',{'input_receipt':str(receipt_path.relative_to(ROOT)),
        'receipt_sha256':digest(receipt_path.read_bytes()),'source_commit':receipt['commit'],
        'completed_covers':44,'planned_covers':300,'rows':rows,
        'total_words':sum(r['words'] for r in rows),'logical_chunks5_one_pass':sum(r['chunks5'] for r in rows),
        'shape_attempt_upper_bound':sum(r['shape_attempt_upper_bound'] for r in rows),
        'measured_candidate_tokens':None,'measured_candidate_seconds':None,
        'candidate_model_calls':0,'encode_decode_calls':0})
    result={'schema':ID+'-offline-preflight','status':'passed','checks':checks,'passed':len(checks),
        'failed':0,'author_archive_sha256':ARCHIVE_SHA256,'author_source_sha256':MEMBERS,
        'model_calls':0,'tokenizer_calls':0,'API_calls':0,'paid_calls':0,'closed_test_opened':False,
        'actual_comparator_outcomes':0,'independent_review':False,
        'source_files':{p:digest((ROOT/p).read_bytes()) for p in [
            'src/synsteg_audit/author_candidate_local_adapter_v1.py',
            'scripts/preflight_author_candidate_local_adapter_v1.py']},
        'limitations':['Only synthetic response handling; no local backend attached or evaluated.',
            'Stricter full JSON shape gate is a declared new adapter condition.',
            'Existing GPT4o paid-provider and683e8cb historical-version blockers remain.']}
    save_exclusive(args.output/'offline-preflight.json',result)
    print(json.dumps({'status':'passed','checks':len(checks),'model_calls':0,'first_cover_chunks':len(full_plan['requests']),
                      'all44chunks':sum(r['chunks5'] for r in rows)}))

if __name__=='__main__': main()
