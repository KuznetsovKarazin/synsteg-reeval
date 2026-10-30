"""Small offline scientific/accounting cases for the new descriptive audit."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('development_feasibility_v2',ROOT/'scripts/audit_development_feasibility_v2.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


def prompt(i,topic='physics',target=250):
    excerpt=' '.join(['bright']*40)+f' Subject{i}.'
    text='Paraphrase the source. '+excerpt
    return {'record_id':f'r{i}','source_group_id':f'g{i}','split':'development','topic':topic,'genre':'educational',
            'target_words':target,'prompt':text,'prompt_sha256':audit.sha(text.encode()),'excerpt':excerpt,
            'excerpt_sha256':audit.sha(excerpt.encode()),'excerpt_words':audit.words(excerpt),'title':f'Subject {i}',
            'page_id':i+100,'revision_id':i+200}


def dictionaries():
    return {'broad':audit.codec.PublicDictionary.from_mapping({'a':['bright','clear']},version='test-broad'),
            'sparse':audit.codec.PublicDictionary.from_mapping({'b':['calm','still']},version='test-sparse')}


def save_prompts(path,rows):
    raw=audit.jsonl_bytes(rows);path.write_bytes(raw);return audit.sha(raw)


def make_run(tmp_path,rows,statuses=('completed','completed'),label='run'):
    directory=tmp_path/label;directory.mkdir();(directory/'attempts').mkdir()
    prompts=tmp_path/(label+'-prompts.jsonl');digest=save_prompts(prompts,rows)
    selected=[{key:r[key] for key in audit.FIELDS} for r in rows]
    manifest={'schema':'synsteg-budget-generation-run-v1','split':'development','model':'fixture-model','provider':'local',
              'prompts_sha256':digest,'selected_records':selected,'selected_records_count':len(selected),
              'revision':'fixture-revision','settings':{'seed':3}}
    (directory/'run.json').write_bytes(audit.json_bytes(manifest))
    for i,status in enumerate(statuses):
        attempt={'schema':'synsteg-budget-generation-attempt-v1','identity':f'a{i}',
                 'model':'fixture-model','provider':'local','record':selected[i],'status':status,
                 'request_fingerprint':f'f{i}'}
        if status=='completed':
            attempt['result']={'cover_text':' '.join(['bright']*(23+i)), 'stop_reason':'end_turn','hit_output_limit':False}
        (directory/'attempts'/f'a{i}.json').write_bytes(audit.json_bytes(attempt))
    return directory,prompts


def test_source_loader_refuses_other_splits_and_changed_hashes(tmp_path):
    row=prompt(1);path=tmp_path/'prompts.jsonl';save_prompts(path,[row])
    got,_=audit.read_development_prompts(path,audit.InputSnapshot(),expected_count=1)
    assert got[0]['record_id']=='r1'
    row['split']='test';save_prompts(path,[row])
    with pytest.raises(ValueError,match='development-only'):
        audit.read_development_prompts(path,audit.InputSnapshot())
    row['split']='development';row['excerpt']+=' changed';save_prompts(path,[row])
    with pytest.raises(ValueError,match='Excerpt hash'):
        audit.read_development_prompts(path,audit.InputSnapshot())


def test_capacity_thresholds_and_dictionaries_are_kept_separate():
    ds=dictionaries()
    low=audit.text_capacity(' '.join(['bright']*23),ds)
    exact=audit.text_capacity(' '.join(['bright']*24),ds)
    aes=audit.text_capacity(' '.join(['bright']*152),ds)
    assert low['broad']['capacity_bits']==23 and not low['broad']['capacity_eligibility']['plain_8']
    assert exact['broad']['capacity_eligibility']['plain_8']
    assert not exact['broad']['capacity_eligibility']['aes_siv_8']
    assert aes['broad']['capacity_eligibility']['aes_siv_8']
    assert aes['sparse']['capacity_bits']==0
    summary=audit.capacity_summary([{'capacity':low},{'capacity':exact}],ds)
    assert summary['broad']['text_denominator']==2
    assert summary['broad']['eligible_texts']['plain_8']==1
    assert summary['sparse']['zero_capacity_texts']==2
    assert summary['broad']['successful_deliveries_measured'] is False


def test_length_pressure_is_descriptive_not_a_support_label():
    row=prompt(1,target=500);result=audit.source_record(row,dictionaries())
    assert result['requested_to_excerpt_word_ratio']==500/41
    assert result['excerpt_below_half_requested_words'] is True
    assert result['soft_faithful_word_band_proposal']==[32,50]
    assert 'factual_support_score' not in result
    assert audit.distribution([1,2,9])=={'n':3,'min':1,'p10':1.2,'median':2,'mean':4,'p90':7.6000000000000005,'max':9,'sum':12}


def test_run_plan_denominator_retains_unattempted_and_failed(tmp_path):
    rows=[prompt(i) for i in range(4)]
    directory,prompts=make_run(tmp_path,rows,statuses=('completed','failed_or_unknown_no_retry'))
    summary,covers=audit.read_run('original',directory,prompts,{r['record_id']:r for r in rows},dictionaries(),audit.InputSnapshot())
    assert summary['planned_generation_records']==4
    assert summary['attempted_records']==2
    assert summary['completed_observed_covers']==1
    assert summary['not_attempted_records']==2
    assert summary['attempt_statuses']=={'completed':1,'failed_or_unknown_no_retry':1}
    assert summary['capacity']['broad']['text_denominator']==1
    assert len(covers)==1 and covers[0]['cover_id']=='original/a0'


def test_duplicate_attempt_id_in_two_arms_is_not_merged(tmp_path):
    rows=[prompt(1),prompt(2)];source={r['record_id']:r for r in rows};covers=[]
    for label in ('original','length'):
        directory,prompts=make_run(tmp_path,rows,label=label)
        _,observed=audit.read_run(label,directory,prompts,source,dictionaries(),audit.InputSnapshot());covers+=observed
    assert len(covers)==4 and len({r['cover_id'] for r in covers})==4
    assert len({r['source_group_id'] for r in covers})==2
    assert len({r['attempt_id'] for r in covers})==2


def test_changed_source_fact_in_prompt_variant_is_refused(tmp_path):
    originals=[prompt(1),prompt(2)];rows=[dict(r) for r in originals]
    rows[0]['excerpt']+=' Changed facts.';rows[0]['excerpt_sha256']=audit.sha(rows[0]['excerpt'].encode());rows[0]['excerpt_words']=audit.words(rows[0]['excerpt'])
    directory,prompts=make_run(tmp_path,rows)
    with pytest.raises(ValueError,match='changed source facts'):
        audit.read_run('variant',directory,prompts,{r['record_id']:r for r in originals},dictionaries(),audit.InputSnapshot())


def test_review_requires_all_actual_covers_and_exact_evidence(tmp_path):
    rows=[prompt(1),prompt(2)];sources={r['record_id']:r for r in rows};directory,prompts=make_run(tmp_path,rows)
    _,covers=audit.read_run('original',directory,prompts,sources,dictionaries(),audit.InputSnapshot())
    items=[{'cover_id':r['cover_id'],'cover_sha256':r['cover_sha256'],'source_excerpt_sha256':r['source_excerpt_sha256'],
            'unsupported_additions_identified':False,'source_contradiction_identified':False,
            'examples':[{'output_quotes':['bright'],'source_quotes':['bright']}]} for r in covers]
    review={'schema':'synsteg-development-cover-grounding-review-v2','method':'fixture','covers':items}
    assert audit.validate_review(review,covers,sources)['reviewed_observed_covers']==2
    review['covers']=items[:1]
    with pytest.raises(ValueError,match='every actual'):
        audit.validate_review(review,covers,sources)
    review['covers']=items;items[0]['examples'][0]['output_quotes']=['fabricated absent phrase']
    with pytest.raises(ValueError,match='absent from cover'):
        audit.validate_review(review,covers,sources)


def test_source_expansion_checks_identity_and_only_uses_named_page(tmp_path):
    row=prompt(1);text=row['excerpt'];row.update(raw_response='raw/batch.json',revision_url='https://example.invalid/rev',
        lead_wikitext_sha256=audit.sha(text.encode()),lead_plain_sha256=audit.sha(text.encode()))
    class Builder:
        @staticmethod
        def lead_text(value): return value
        @staticmethod
        def source_excerpt(value,**kwargs): return value
    (tmp_path/'raw').mkdir();page={'pageid':row['page_id'],'revisions':[{'revid':row['revision_id'],'slots':{'main':{'content':text}}}]}
    # A non-development page contains deliberately unusable data. It must not
    # be cleaned or analyzed while selecting the exact requested source page.
    payload={'query':{'pages':[{'pageid':999,'revisions':'UNUSABLE_NONDEVELOPMENT'},page]}}
    path=tmp_path/'raw/batch.json';path.write_bytes(audit.json_bytes(payload))
    got=audit.expanded_source(row,tmp_path,Builder,audit.InputSnapshot())
    assert got['safe_cached_lead_prefix']==text and got['new_network_requests']==0
    assert got['new_generation_condition_executed_by_this_audit'] is False
    assert 'used_in_prior_generation' not in got
    page['revisions'][0]['revid']+=1;path.write_bytes(audit.json_bytes(payload))
    with pytest.raises(ValueError,match='revision mismatch'):
        audit.expanded_source(row,tmp_path,Builder,audit.InputSnapshot())


def test_snapshot_and_output_writes_fail_without_silent_replacement(tmp_path):
    source=tmp_path/'input';source.write_bytes(b'a');snapshot=audit.InputSnapshot();snapshot.read(source)
    source.write_bytes(b'b')
    with pytest.raises(ValueError,match='changed'):
        snapshot.verify_unchanged()
    output=tmp_path/'out';audit.write_outputs(output,{'one':b'old'})
    with pytest.raises(FileExistsError):
        audit.write_outputs(output,{'new':b'value','one':b'changed'})
    assert (output/'one').read_bytes()==b'old' and not (output/'new').exists()


def test_proposal_selection_never_reads_capacity_or_outcome():
    rows=[]
    for i,topic in enumerate(['astronomy','astronomy','astronomy','biology','physics']):
        row=audit.source_record(prompt(i,topic=topic),dictionaries());row.pop('capacity');rows.append(row)
    result=audit.proposal(rows)
    assert result['source_groups']==5 and result['proposed_generation_count_per_model']==10
    assert [r['stratum'] for r in result['selected_sources']]==['reused_anchor','reused_anchor','unused_topic_group','unused_topic_group','unused_topic_group']
    assert result['execution_enabled'] is False


def test_returned_model_mismatch_is_not_silently_attributed(tmp_path):
    rows=[prompt(1),prompt(2)];directory,prompts=make_run(tmp_path,rows)
    path=directory/'attempts/a0.json';attempt=json.loads(path.read_text());attempt['result']['returned_model']='different-model';path.write_bytes(audit.json_bytes(attempt))
    with pytest.raises(ValueError,match='model/revision'):
        audit.read_run('original',directory,prompts,{r['record_id']:r for r in rows},dictionaries(),audit.InputSnapshot())
