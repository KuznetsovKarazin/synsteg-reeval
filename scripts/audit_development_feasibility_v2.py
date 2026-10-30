#!/usr/bin/env python3
"""Deterministic, development-only source/cover length and capacity diagnostic.

No network, model generation, training, encoding or receiver calls occur.
The audit never conflates Wikipedia source excerpts with generated cover texts.
Factual annotations are supplied judgments, not inferred from lexical overlap.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from synsteg_audit import shared_dictionary as codec

SCHEMA='synsteg-development-feasibility-v2'
FIELDS=('record_id','source_group_id','split','topic','genre','target_words','prompt_sha256')
PROTOCOL=codec.Protocol(min_group_size=1,max_bits_per_slot=1,framing='plain_length16')
ARMS={'plain_8':24,'plain_16':32,'plain_32':48,'aes_siv_8':152,'aes_siv_32':176,'aes_siv_128':272}
KNOWN_STATUSES={'completed','failed_or_unknown_no_retry','reserved_outcome_unknown','accounting_bound_violation_stop'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()


def jsonl_bytes(rows):
    return b''.join((json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode() for row in rows)


def words(text):
    return len(text.split())


def distribution(values):
    values=sorted(values)
    if not values: return {'n':0,'min':None,'p10':None,'median':None,'mean':None,'p90':None,'max':None,'sum':0}
    def percentile(p):
        index=(len(values)-1)*p
        left=math.floor(index);right=math.ceil(index)
        return values[left]+(values[right]-values[left])*(index-left)
    return {'n':len(values),'min':values[0],'p10':percentile(.1),'median':statistics.median(values),
            'mean':statistics.mean(values),'p90':percentile(.9),'max':values[-1],'sum':sum(values)}


class InputSnapshot:
    def __init__(self): self.files={}
    def read(self,path):
        path=Path(path).resolve();raw=path.read_bytes();digest=sha(raw)
        if path in self.files and self.files[path]!=digest: raise ValueError(f'Input changed: {path}')
        self.files[path]=digest
        return raw
    def json(self,path): return json.loads(self.read(path))
    def verify_unchanged(self):
        for path,digest in self.files.items():
            if sha(path.read_bytes())!=digest: raise ValueError(f'Input changed during audit: {path}')
    def inventory(self):
        rows=[]
        for path,digest in sorted(self.files.items()):
            try: name=path.relative_to(ROOT).as_posix()
            except ValueError: name=path.name
            rows.append({'path':name,'sha256':digest})
        return rows


def read_development_prompts(path,snapshot,expected_count=None):
    raw=snapshot.read(path);rows=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if expected_count is not None and len(rows)!=expected_count: raise ValueError(f'Expected {expected_count} development records')
    if not rows: raise ValueError('Empty development prompt file')
    ids=set();groups=set()
    for row in rows:
        if not isinstance(row,dict) or row.get('split')!='development':
            raise ValueError('The entire input must be development-only; other splits are refused')
        for field in ('record_id','source_group_id','topic','genre','prompt','excerpt','title'):
            if not isinstance(row.get(field),str) or not row[field].strip(): raise ValueError(f'Missing {field}')
        if row['record_id'] in ids or row['source_group_id'] in groups: raise ValueError('Duplicate development identity')
        ids.add(row['record_id']);groups.add(row['source_group_id'])
        if type(row.get('target_words')) is not int or row['target_words'] not in (250,500):
            raise ValueError('Expected original 250/500 target metadata')
        if sha(row['prompt'].encode())!=row.get('prompt_sha256'): raise ValueError('Prompt hash mismatch')
        if sha(row['excerpt'].encode())!=row.get('excerpt_sha256'): raise ValueError('Excerpt hash mismatch')
        if row.get('excerpt_words')!=words(row['excerpt']): raise ValueError('Excerpt word count mismatch')
    return rows,sha(raw)


def load_dictionaries(entries,snapshot):
    result={};identity={}
    for label,path in entries:
        if label in result or not re.fullmatch(r'[a-z0-9_-]+',label): raise ValueError('Invalid or duplicate dictionary label')
        raw=snapshot.read(path);dictionary=codec.PublicDictionary.from_dict(json.loads(raw))
        result[label]=dictionary
        identity[label]={'file':Path(path).name,'file_sha256':sha(raw),'canonical_sha256':dictionary.sha256,
                         'version':dictionary.version,'classes':len(dictionary.classes),
                         'words':sum(len(c.words) for c in dictionary.classes)}
    if not result: raise ValueError('At least one explicit dictionary is required')
    return result,identity


def text_capacity(text,dictionaries):
    result={}
    for label,dictionary in sorted(dictionaries.items()):
        analysis=codec.analyze_text(text,dictionary,PROTOCOL)
        result[label]={'capacity_bits':analysis.capacity_bits,'slots':len(analysis.slots),
            'unique_classes':len({s.class_id for s in analysis.slots}),
            'capacity_bits_per_100_whitespace_words':100*analysis.capacity_bits/words(text) if words(text) else 0,
            'capacity_eligibility':{arm:analysis.capacity_bits>=required for arm,required in ARMS.items()},
            'slot_occurrences':[{'surface':s.surface,'class_id':s.class_id,'width_bits':s.width,
                                 'start':s.start,'end':s.end} for s in analysis.slots]}
    return result


def capacity_summary(rows,labels):
    # Each input row denotes one actually available text in a named population.
    result={}
    for label in sorted(labels):
        values=[row['capacity'][label]['capacity_bits'] for row in rows]
        result[label]={'text_denominator':len(rows),'capacity_bits':distribution(values),
            'zero_capacity_texts':sum(value==0 for value in values),
            'eligible_texts':{arm:sum(row['capacity'][label]['capacity_eligibility'][arm] for row in rows) for arm in ARMS},
            'successful_deliveries_measured':False}
    return result


def load_source_builder(snapshot):
    path=ROOT/'scripts/build_wikipedia_panel_recovery.py';snapshot.read(path)
    spec=importlib.util.spec_from_file_location('_frozen_wikipedia_builder_for_feasibility',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def expanded_source(row,panel_root,builder,snapshot):
    root=Path(panel_root).resolve();raw_path=(root/row['raw_response']).resolve()
    if not raw_path.is_relative_to(root/'raw'): raise ValueError('Unsafe raw source path')
    data=snapshot.json(raw_path)
    # Other pages in the batch are not cleaned, measured or evaluated.
    matches=[page for page in data['query']['pages'] if page.get('pageid')==row['page_id']]
    if len(matches)!=1: raise ValueError('Source page missing or duplicated in raw response')
    revision=matches[0]['revisions'][0]
    if revision['revid']!=row['revision_id']: raise ValueError('Source revision mismatch')
    wikitext=revision['slots']['main']['content']
    if sha(wikitext.encode())!=row['lead_wikitext_sha256']: raise ValueError('Lead wikitext hash mismatch')
    cleaned=builder.lead_text(wikitext)
    if sha(cleaned.encode())!=row['lead_plain_sha256']: raise ValueError('Frozen cleaner identity/output mismatch')
    original=builder.source_excerpt(cleaned)
    if original!=row['excerpt']: raise ValueError('Frozen excerpt does not reproduce from raw source')
    # Broaden only the existing word/sentence ceilings. The same conservative
    # parser still stops at omitted lists/templates. No new facts are invented.
    prefix=builder.source_excerpt(cleaned,max_words=100000,max_sentences=100000)
    return {'record_id':row['record_id'],'source_group_id':row['source_group_id'],'split':'development',
        'page_id':row['page_id'],'revision_id':row['revision_id'],'revision_url':row['revision_url'],
        'raw_response':row['raw_response'],'raw_response_sha256':sha(snapshot.read(raw_path)),
        'original_excerpt_sha256':row['excerpt_sha256'],'target_words':row['target_words'],
        'safe_cached_lead_prefix':prefix,'safe_cached_lead_prefix_sha256':sha(prefix.encode()),
        'safe_cached_lead_prefix_words':words(prefix),'original_excerpt_words':row['excerpt_words'],
        'longer_than_original_excerpt':words(prefix)>row['excerpt_words'],
        'at_least_original_target_words':words(prefix)>=row['target_words'],
        'at_least_125_percent_original_target_words':4*words(prefix)>=5*row['target_words'],
        'new_network_requests':0,'new_generation_condition_executed_by_this_audit':False,
        'status':'source-material candidate only; not a replacement for any frozen v1 prompt'}


def source_record(row,dictionaries):
    n=words(row['excerpt']);target=row['target_words']
    return {'population':'source_excerpt','record_id':row['record_id'],'source_group_id':row['source_group_id'],
            'split':'development','topic':row['topic'],'genre':row['genre'],'title':row['title'],
            'page_id':row['page_id'],'revision_id':row['revision_id'],'target_words':target,
            'excerpt':row['excerpt'],'excerpt_sha256':row['excerpt_sha256'],'excerpt_words':n,
            'prompt_sha256':row['prompt_sha256'],'prompt_words':words(row['prompt']),
            'requested_to_excerpt_word_ratio':target/n,'excerpt_to_requested_word_ratio':n/target,
            'excerpt_below_half_requested_words':2*n<target,
            'soft_faithful_word_band_proposal':[max(1,math.floor(.8*n)),math.ceil(1.2*n)],
            'capacity':text_capacity(row['excerpt'],dictionaries)}


def read_run(label,run_path,prompt_path,source_by_id,dictionaries,snapshot):
    if not re.fullmatch(r'[a-z0-9_-]+',label): raise ValueError('Invalid run label')
    run_path=Path(run_path);manifest=snapshot.json(run_path/'run.json')
    if manifest.get('schema')!='synsteg-budget-generation-run-v1' or manifest.get('split')!='development':
        raise ValueError('Only development generation runs are accepted')
    prompt_rows,prompt_hash=read_development_prompts(prompt_path,snapshot)
    if prompt_hash!=manifest.get('prompts_sha256'): raise ValueError('Run prompt-file hash mismatch')
    prompts={row['record_id']:row for row in prompt_rows}
    selected=manifest.get('selected_records')
    if not isinstance(selected,list) or manifest.get('selected_records_count')!=len(selected) or not selected:
        raise ValueError('Invalid planned run denominator')
    planned={};groups=set()
    for row in selected:
        rid=row.get('record_id')
        if rid in planned or row.get('source_group_id') in groups: raise ValueError('Duplicate planned source')
        if row.get('split')!='development' or rid not in source_by_id or rid not in prompts:
            raise ValueError('Unknown or non-development planned source')
        source=source_by_id[rid];prompt=prompts[rid]
        if any(row.get(key)!=prompt.get(key) for key in FIELDS): raise ValueError('Planned record differs from run prompt')
        if any(prompt.get(key)!=source.get(key) for key in ('record_id','source_group_id','excerpt_sha256','excerpt','title','target_words')):
            raise ValueError('Prompt variant changed source facts or pairing')
        planned[rid]=row;groups.add(row['source_group_id'])
    attempt_paths=sorted((run_path/'attempts').glob('*.json'));attempted=set();outcomes=Counter();covers=[]
    for path in attempt_paths:
        raw=snapshot.read(path);attempt=json.loads(raw);record=attempt.get('record',{});rid=record.get('record_id')
        if attempt.get('schema')!='synsteg-budget-generation-attempt-v1' or rid not in planned or rid in attempted:
            raise ValueError('Unexpected/duplicate generation attempt')
        if any(record.get(key)!=planned[rid].get(key) for key in FIELDS): raise ValueError('Attempt record differs from plan')
        if any(attempt.get(key)!=manifest.get(key) for key in ('model','provider')): raise ValueError('Attempt model/provider mismatch')
        if path.stem!=attempt.get('identity'): raise ValueError('Attempt filename/identity mismatch')
        status=attempt.get('status')
        if status not in KNOWN_STATUSES: raise ValueError('Unknown attempt status')
        attempted.add(rid);outcomes[status]+=1
        if status!='completed': continue
        result=attempt.get('result',{});text=result.get('cover_text')
        if not isinstance(text,str) or not text.strip(): raise ValueError('Completed attempt lacks nonempty cover')
        if result.get('returned_model',manifest['model'])!=manifest['model'] or result.get('revision',manifest.get('revision'))!=manifest.get('revision'):
            raise ValueError('Completed result model/revision differs from run manifest')
        source=source_by_id[rid];target=source['target_words'];n=words(text);excerpt_n=words(source['excerpt'])
        covers.append({'population':'generated_cover','run_label':label,'cover_id':label+'/'+attempt['identity'],
            'attempt_id':attempt['identity'],'attempt_file':path.as_posix(),'attempt_sha256':sha(raw),
            'request_fingerprint':attempt.get('request_fingerprint'),'record_id':rid,'source_group_id':source['source_group_id'],
            'split':'development','topic':source['topic'],'genre':source['genre'],'source_title':source['title'],
            'source_excerpt_sha256':source['excerpt_sha256'],'source_excerpt_words':excerpt_n,
            'original_target_words':target,'actual_prompt_sha256':record['prompt_sha256'],
            'cover_text':text,'cover_sha256':sha(text.encode()),'cover_words':n,
            'cover_paragraphs':len([p for p in re.split(r'\n\s*\n',text.strip()) if p.strip()]),
            'cover_to_source_word_ratio':n/excerpt_n,'cover_to_original_target_word_ratio':n/target,
            'within_original_target_plus_minus_10_percent':.9*target<=n<=1.1*target,
            'generation_metadata':{key:result.get(key) for key in ('stop_reason','hit_output_limit','usage','seed','elapsed_seconds','dtype','revision','returned_model')},
            'capacity':text_capacity(text,dictionaries)})
    if sorted((run_path/'attempts').glob('*.json'))!=attempt_paths: raise ValueError('Run attempt inventory changed during audit')
    summary={'run_label':label,'run_directory':run_path.as_posix(),'run_manifest_sha256':sha(snapshot.read(run_path/'run.json')),
             'source_prompts_sha256':prompt_hash,'model':manifest.get('model'),'revision':manifest.get('revision'),
             'settings':manifest.get('settings'),'planned_generation_records':len(planned),
             'attempted_records':len(attempted),'completed_observed_covers':len(covers),
             'not_attempted_records':len(planned)-len(attempted),'attempt_statuses':dict(sorted(outcomes.items())),
             'capacity_eligibility_denominator':'observed completed covers only; unattempted count retained separately',
             'cover_words':distribution([r['cover_words'] for r in covers]),
             'capacity':capacity_summary(covers,dictionaries)}
    return summary,covers


def validate_review(review,covers,sources):
    if review.get('schema')!='synsteg-development-cover-grounding-review-v2': raise ValueError('Invalid qualitative review schema')
    by_id={row['cover_id']:row for row in covers};seen=set();result=[]
    for item in review.get('covers',[]):
        cover_id=item.get('cover_id')
        if cover_id not in by_id or cover_id in seen: raise ValueError('Unknown or duplicate reviewed cover')
        seen.add(cover_id);cover=by_id[cover_id];source=sources[cover['record_id']]
        if item.get('cover_sha256')!=cover['cover_sha256']: raise ValueError('Review cover hash mismatch')
        if item.get('source_excerpt_sha256')!=source['excerpt_sha256']: raise ValueError('Review source hash mismatch')
        if any(type(item.get(flag)) is not bool for flag in ('unsupported_additions_identified','source_contradiction_identified')):
            raise ValueError('Qualitative cover flags must be explicit booleans')
        for example in item.get('examples',[]):
            quotes=example.get('output_quotes',[])
            if any(not q or q not in cover['cover_text'] for q in quotes): raise ValueError('Review quotation absent from cover')
            if any(not q or q not in source['excerpt'] for q in example.get('source_quotes',[])): raise ValueError('Review quotation absent from source')
        result.append(item)
    if seen!=set(by_id): raise ValueError('Qualitative review must cover every actual observed cover')
    return {'method':review['method'],'independent_source_groups':len({r['source_group_id'] for r in covers}),
            'reviewed_observed_covers':len(result),
            'covers_with_identified_unsupported_additions':sum(r['unsupported_additions_identified'] for r in result),
            'covers_with_identified_source_contradictions':sum(r['source_contradiction_identified'] for r in result),
            'general_model_hallucination_rate_estimate':False,
            'review_file_supplies_judgments_not_automatic_semantic_detection':True}


def length_summary(source_rows,expanded_rows,dictionaries):
    expanded={r['record_id']:r for r in expanded_rows}
    result={}
    for key,rows in [('all',source_rows),*[(str(target),[r for r in source_rows if r['target_words']==target]) for target in (250,500)]]:
        result[key]={'source_groups':len(rows),'excerpt_words':distribution([r['excerpt_words'] for r in rows]),
            'requested_to_excerpt_word_ratio':distribution([r['requested_to_excerpt_word_ratio'] for r in rows]),
            'excerpt_below_half_requested_words':sum(r['excerpt_below_half_requested_words'] for r in rows),
            'safe_cached_lead_prefix_words':distribution([expanded[r['record_id']]['safe_cached_lead_prefix_words'] for r in rows]),
            'cached_lead_at_least_original_target':sum(expanded[r['record_id']]['at_least_original_target_words'] for r in rows),
            'cached_lead_at_least_125_percent_target':sum(expanded[r['record_id']]['at_least_125_percent_original_target_words'] for r in rows),
            'capacity':capacity_summary(rows,dictionaries)}
    return result


def proposal(source_rows):
    anchors=source_rows[:2];selected=list(anchors);seen={r['source_group_id'] for r in selected}
    for topic in sorted({r['topic'] for r in source_rows}):
        matches=[r for r in source_rows if r['topic']==topic and r['source_group_id'] not in seen]
        if not matches: raise ValueError('No unused source for one topic')
        row=matches[0];selected.append(row);seen.add(row['source_group_id'])
    return {'schema':'synsteg-development-faithfulness-proposal-v2','execution_enabled':False,
        'status':'proposal only; root owns generation protocol and model execution',
        'post_development_observation':True,'test_data_used':False,'source_material_changed':False,
        'selection':'Two earlier anchors plus first unused source per sorted topic in frozen development-file order; no capacity/output selection',
        'source_groups':len(selected),'previously_observed_anchor_groups':len(anchors),
        'new_unused_development_groups':len(selected)-len(anchors),
        'planned_arms':['original_250_or_500_prompt','faithfulness_first_source_length_prompt'],
        'proposed_generation_count_per_model':2*len(selected),
        'pairing':'Same exact source JSON, genre, model revision, generation settings, cap and per-source seed across arms; keep every attempt',
        'intervention_scope':'Combined change of instruction priority and preferred output length, not an isolated causal test of a single instruction',
        'faithful_length_rule':'Soft preference floor(0.8*source_words)..ceil(1.2*source_words); a shorter complete faithful paraphrase is allowed; never pad',
        'faithful_instruction':'Every factual sentence must restate or paraphrase information explicitly present in the quoted source. Preserve names, spellings, transliterations, qualifications, and relationships. Do not add outside history, symbolism, examples, dates, organizations, causal explanations, or claimed recommendations. Cover the supplied facts once without padding or repeated restatements. Factual fidelity takes priority over length: return a shorter complete text when the source supports only that. Return only coherent English prose.',
        'outcomes':['Source-supported vs unsupported statements, direct source contradictions, source-fact omissions, repetitions',
                    'Actual words and paragraphs; stop reason/output cap; dictionary-specific capacity with every attempt retained'],
        'long_target_followup':'Before claiming a 250/500-word faithful study, acquire and freeze enough real source material for the same groups/revisions. Availability threshold is a design rule, not a guarantee of factual coverage. Report unavailable groups; do not replace them after observing model outcomes.',
        'selected_sources':[{'record_id':r['record_id'],'source_group_id':r['source_group_id'],'topic':r['topic'],
                            'excerpt_words':r['excerpt_words'],'original_target_words':r['target_words'],
                            'soft_faithful_word_band':r['soft_faithful_word_band_proposal'],
                            'stratum':'reused_anchor' if i<2 else 'unused_topic_group'} for i,r in enumerate(selected)]}


def render_report(summary):
    lines=['# Development feasibility v2','',
        'Read-only diagnostic. No new generation, paid request, encoding, training or test-set evaluation occurred. Source excerpts and generated covers remain separate populations.','',
        'Topic strata are operational Wikipedia search-query labels, not expert semantic labels. The strict v2 dictionary is a structural ambiguity ablation, not a validated fix for meaning preservation or contextual suitability. No source is relabeled or reselected in this audit.','',
        '| Original target | Source N | Source words min / median / max | Median target/source ratio | Cached leads ≥ target | Cached leads ≥ 1.25× target |',
        '|---|---:|---:|---:|---:|---:|']
    for target in ('250','500'):
        row=summary['source_excerpts']['by_target'][target];d=row['excerpt_words'];ratio=row['requested_to_excerpt_word_ratio']['median']
        lines.append(f"| {target} | {row['source_groups']} | {d['min']} / {d['median']} / {d['max']} | {ratio:.3f} | {row['cached_lead_at_least_original_target']} | {row['cached_lead_at_least_125_percent_target']} |")
    lines+=['','Word ratios describe expansion pressure; they are not semantic-support scores or proof that faithful expansion is impossible. The 1.25× material threshold is an explicit exploratory design heuristic.','',
        '| Dictionary | Source excerpts | Zero capacity | Mean / max capacity bits | Source excerpts eligible for 8 plain bits |',
        '|---|---:|---:|---:|---:|']
    for label,row in summary['source_excerpts']['capacity'].items():
        lines.append(f"| {label} | {row['text_denominator']} | {row['zero_capacity_texts']} | {row['capacity_bits']['mean']:.3f} / {row['capacity_bits']['max']} | {row['eligible_texts']['plain_8']} |")
    lines+=['','Capacity eligibility is mechanical only, not a successful delivery or semantic-quality result. Plain 8-bit payload needs 24 total bits; AES-SIV 8-bit payload needs 152.','',
        '| Run | Planned | Attempted | Observed covers | Not attempted | Observed words |', '|---|---:|---:|---:|---:|---:|']
    for row in summary['generation_runs']:
        d=row['cover_words'];lines.append(f"| {row['run_label']} | {row['planned_generation_records']} | {row['attempted_records']} | {row['completed_observed_covers']} | {row['not_attempted_records']} | {d['min']}–{d['max']} |")
    r=summary['qualitative_grounding_review']
    lines+=['',f"The {r['reviewed_observed_covers']} reviewed covers reuse {r['independent_source_groups']} independent source groups. Identified unsupported additions occur in {r['covers_with_identified_unsupported_additions']} covers; an explicit source contradiction occurs in {r['covers_with_identified_source_contradictions']}. These are descriptions of these cases, not a population hallucination-rate estimate.",'',
        'The proposed next comparison keeps source facts fixed and contrasts the original long-target prompt with a faithfulness-first prompt that permits a concise complete paraphrase. This changes both the preferred length and instruction priority. Source fidelity takes precedence over capacity; low-capacity covers must remain in the denominator.','',
        'Files: source-records.jsonl (all 100 excerpts), cover-records.jsonl (all actual observed covers), cached-lead-candidates.jsonl (same-revision material availability only), source-review.json (independent source-only judgments), proposal.json (disabled proposal), summary.json and input-inventory.json.','']
    return '\n'.join(lines).encode()


def write_outputs(output,expected):
    output=Path(output)
    # Validate ALL existing outputs first, before publishing any missing file.
    for name,data in expected.items():
        path=output/name
        if path.exists() and path.read_bytes()!=data: raise FileExistsError(f'Refusing changed output: {name}')
    output.mkdir(parents=True,exist_ok=True)
    for name,data in expected.items():
        path=output/name
        if path.exists(): continue
        temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        with temp.open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        try: os.link(temp,path)
        except FileExistsError:
            if path.read_bytes()!=data: raise FileExistsError(f'Refusing concurrent overwrite: {name}')
        finally: temp.unlink(missing_ok=True)


def audit(source_prompts,panel_root,dictionary_entries,run_entries,review_file,output):
    snapshot=InputSnapshot();snapshot.read(Path(__file__));snapshot.read(ROOT/'src/synsteg_audit/shared_dictionary.py')
    source_rows,source_file_hash=read_development_prompts(source_prompts,snapshot,expected_count=100)
    source_by_id={r['record_id']:r for r in source_rows}
    dictionaries,dictionary_identity=load_dictionaries(dictionary_entries,snapshot)
    builder=load_source_builder(snapshot)
    measured=[source_record(row,dictionaries) for row in source_rows]
    expanded=[expanded_source(row,panel_root,builder,snapshot) for row in source_rows]
    runs=[];covers=[];labels=set()
    for label,path,prompts in run_entries:
        if label in labels: raise ValueError('Duplicate run label')
        labels.add(label);run,observed=read_run(label,path,prompts,source_by_id,dictionaries,snapshot)
        runs.append(run);covers.extend(observed)
    if not covers: raise ValueError('No observed covers to audit')
    review=snapshot.json(review_file);review_summary=validate_review(review,covers,source_by_id)
    by_target=length_summary(measured,expanded,dictionaries)
    summary={'schema':SCHEMA,'status':'completed_read_only_development_diagnostic',
        'population_policy':'100 source excerpts and observed generated covers are analyzed separately; repeated prompt arms do not increase independent N',
        'word_count_rule':'len(text.split())','distribution_quantiles':'linear interpolation at (n-1)*p',
        'source_prompts_sha256':source_file_hash,'dictionary_identities':dictionary_identity,
        'capacity_protocol':PROTOCOL.to_dict(),'capacity_threshold_bits':ARMS,
        'capacity_is_not_semantic_safety_or_delivery_success':True,
        'topic_strata_status':'Operational Wikipedia search-query membership, not expert semantic topic labels',
        'strict_dictionary_interpretation':'Structural ambiguity ablation, not a semantic or contextual safety fix',
        'source_excerpts':{'denominator':len(measured),'capacity':capacity_summary(measured,dictionaries),'by_target':by_target},
        'generated_covers':{'observed_cover_denominator':len(covers),
                           'independent_source_group_denominator':len({r['source_group_id'] for r in covers}),
                           'capacity':capacity_summary(covers,dictionaries)},
        'generation_runs':runs,'qualitative_grounding_review':review_summary,
        'scope':{'development_source_groups':len(measured),'train_texts_evaluated':0,'validation_texts_evaluated':0,'test_texts_evaluated':0,
                 'new_model_generations':0,'new_network_requests':0,'paid_requests':0,'encoding_or_receiver_calls':0},
        'limitations':['Source length is not a count of independent supported factual claims.',
                       'Topic strata are operational search membership; false positives remain and no source is reselected here.',
                       'The conservative cached lead may stop at unresolved formatting; additional article sections were not downloaded.',
                       'Source-only judgment tests fidelity to supplied excerpts, not external truth of Wikipedia.',
                       'Strict lexical filtering reduces ambiguity but does not prove every remaining substitution is grammatical/contextually appropriate.',
                       'Four observed covers reuse two sources; no causal or population-level claims are warranted.']}
    snapshot.verify_unchanged()
    expected={'source-records.jsonl':jsonl_bytes(measured),'cover-records.jsonl':jsonl_bytes(covers),
              'cached-lead-candidates.jsonl':jsonl_bytes(expanded),'summary.json':json_bytes(summary),
              'input-inventory.json':json_bytes({'schema':SCHEMA,'files':snapshot.inventory()}),
              'proposal.json':json_bytes(proposal(measured)),'source-review.json':json_bytes(review),
              'README.md':render_report(summary)}
    write_outputs(output,expected)
    return {'schema':SCHEMA,'source_excerpts':len(measured),'observed_covers':len(covers),
            'independent_observed_source_groups':len({r['source_group_id'] for r in covers}),
            'output':str(output),'summary_sha256':sha(expected['summary.json'])}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-prompts',type=Path,required=True)
    parser.add_argument('--panel-root',type=Path,required=True)
    parser.add_argument('--dictionary',nargs=2,action='append',metavar=('LABEL','PATH'),required=True)
    parser.add_argument('--run',nargs=3,action='append',metavar=('LABEL','RUN_DIR','PROMPTS_FILE'),required=True)
    parser.add_argument('--review-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try: result=audit(args.source_prompts,args.panel_root,args.dictionary,args.run,args.review_file,args.output)
    except (OSError,ValueError,KeyError,UnicodeError) as error: parser.exit(2,f'Audit refused: {error}\n')
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__': raise SystemExit(main())
