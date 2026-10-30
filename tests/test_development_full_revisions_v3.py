"""Exact revision identity, prose boundaries and all-source retention cases."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('development_full_revisions_v3',ROOT/'scripts/build_development_full_revisions_v3.py')
full=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(full)
BUILDER,BUILDER_HASH=full.source_builder()
PROSE='The research team carefully recorded the physical conditions of the site and described the observations in a detailed report.'


def identity(i=1):
    text=PROSE;digest=hashlib.sha1(text.encode()).hexdigest()
    row={'record_id':f'r{i}','source_group_id':f'g{i}','page_id':i,'revision_id':100+i,'revision_sha1':digest,
         'revision_timestamp':'2026-01-01T00:00:00Z','split':'development','title':f'Subject {i}','topic':'physics',
         'genre':'encyclopedic','parent_target_words':500,'target_words':53,'prompt':'A source-bound task.',
         'prompt_sha256':full.sha256(b'A source-bound task.'),'excerpt':text,'excerpt_sha256':full.sha256(text.encode()),
         'revision_url':f'https://en.wikipedia.org/w/index.php?oldid={100+i}',
         'history_url':f'https://en.wikipedia.org/w/index.php?curid={i}&action=history',
         'attribution':f'Wikipedia contributors, Subject {i}', 'license':'CC BY-SA 4.0',
         'license_url':'https://creativecommons.org/licenses/by-sa/4.0/'}
    revision={'revid':100+i,'timestamp':row['revision_timestamp'],'sha1':digest,
              'slots':{'main':{'contentmodel':'wikitext','content':text,'sha1':digest}}}
    return row,{'pageid':i,'title':row['title'],'revisions':[revision]},revision


def save_selection(path):
    rows=[identity(i)[0] for i in range(1,21)]
    path.write_bytes(full.jsonl_bytes(rows));return rows


def test_exact_utf8_revision_identity_including_full_sha1():
    row,page,revision=identity()
    text,proof=full.validate_revision(page,revision,row)
    assert text==PROSE and proof['full_wikitext_sha1']==row['revision_sha1']
    assert proof['full_wikitext_sha256']==full.sha256(PROSE.encode())
    # Current page titles may change; immutable page/revision identities govern.
    page['title']='Renamed page'
    assert full.validate_revision(page,revision,row)[1]['page_title_change_detected'] is True


@pytest.mark.parametrize('failure',['page','revision','timestamp','sha1','slot_sha1','hidden','model','changed_content'])
def test_wrong_or_hidden_full_content_is_refused(failure):
    row,page,revision=identity()
    if failure=='page':page['pageid']=999
    elif failure=='revision':revision['revid']=999
    elif failure=='timestamp':revision['timestamp']='2025-01-01T00:00:00Z'
    elif failure=='sha1':revision['sha1']='0'*40
    elif failure=='slot_sha1':revision['slots']['main']['sha1']='0'*40
    elif failure=='hidden':revision['slots']['main'].pop('content')
    elif failure=='model':revision['slots']['main']['contentmodel']='json'
    else:revision['slots']['main']['content']+=' Changed.'
    with pytest.raises(ValueError):full.validate_revision(page,revision,row)


def test_selection_requires_all_twenty_development_ids_and_parent_target(tmp_path):
    path=tmp_path/'prompts.jsonl';rows=save_selection(path)
    got,_=full.select_sources(path)
    assert len(got)==20 and got[0]['target_words']==53 and got[0]['parent_target_words']==500
    rows[0]['split']='test';path.write_bytes(full.jsonl_bytes(rows))
    with pytest.raises(ValueError,match='Non-development'):full.select_sources(path)
    rows[0]['split']='development';rows[0]['parent_target_words']=53;path.write_bytes(full.jsonl_bytes(rows))
    with pytest.raises(ValueError,match='parent_target_words'):full.select_sources(path)
    rows[0]['parent_target_words']=500;rows[1]['revision_id']=rows[0]['revision_id'];path.write_bytes(full.jsonl_bytes(rows))
    with pytest.raises(ValueError,match='unique revision_id'):full.select_sources(path)


def test_nested_excluded_sections_and_unicode_offsets():
    raw='Ω lead sentence.\n\n== References ==\n\nReference text.\n\n=== Detail ===\n\nStill reference material.\n\n== History {{anchor|past}} ==\n\nHistorical prose.\n'
    blocks,warnings=full.raw_blocks(raw)
    assert not warnings
    assert [b['excluded_section'] for b in blocks]==[False,True,True,False]
    assert blocks[-1]['section_path']==['History']
    assert blocks[-1]['section_path_raw']==['History {{anchor|past}}']
    for b in blocks:assert raw[b['raw_start_char']:b['raw_end_char']]==b['raw_fragment']
    assert blocks[0]['raw_end_char']==len('Ω lead sentence.')


def test_unknown_initial_and_nested_body_templates_are_not_metadata():
    for raw in ['{{unknown|Important source fact}} '+PROSE,'{{nowrap|{{unknown|Important source fact}}}} '+PROSE]:
        text,reason,counts=full.clean_block(raw,BUILDER)
        assert text is None and reason is not None
        assert counts['opaque_templates']>=1


def test_known_metadata_removal_is_logged_without_losing_prose():
    cleaned,reason,counts=full.clean_block('{{advert|date=2026}}\n'+PROSE,BUILDER)
    assert cleaned==PROSE and reason is None
    assert counts['metadata_template:advert']==1


def test_formula_or_list_boundary_never_splices_the_far_side():
    raw=PROSE+' A formula is written as <math>E=mc^2</math>, and an unsupported continuation would follow.'
    cleaned,reason,counts=full.clean_block(raw,BUILDER)
    assert cleaned==PROSE and reason is None
    assert counts['complete_prefix_stopped_before:opaque_template_formula_table_or_nonprose']==1
    cleaned,_,_=full.clean_block(PROSE+' Examples include:\n* First item.\n* Second item.\nLater independent text.',BUILDER)
    assert cleaned==PROSE


def test_reference_removal_does_not_eat_real_text():
    raw=PROSE+'<ref name="a"/> A second observation describes the seasonal changes measured at this location.<ref>citation</ref>'
    cleaned,reason,_=full.clean_block(raw,BUILDER)
    assert reason is None and 'A second observation describes' in cleaned and 'citation' not in cleaned


def test_punctuation_inside_wikilink_names_survives_frozen_helper():
    raw='The exhibition included work by [[William Pope.L]] and other artists whose contributions were documented in the public catalogue.'
    cleaned,reason,_=full.clean_block(raw,BUILDER)
    assert reason is None and 'William Pope.L' in cleaned and 'Pope. L' not in cleaned
    assert not any(marker in cleaned for marker in full.PROTECTED_PUNCT.values())


def test_context_gap_after_omitted_formula_is_flagged():
    raw=PROSE+' These effects are described by <math>x=y</math>, with three particular processes.\n\nThese processes have different attenuation coefficients that depend on the incident energy and the composition of the detector material.'
    blocks,warnings=full.extract_prose(raw,BUILDER)
    assert not warnings and len(blocks)==2
    assert blocks[0]['unit_type']=='complete_sentence_prefix_before_boundary'
    assert 'possible_antecedent_after_omitted_content' in blocks[1]['context_dependency_flags']
    assert blocks[1]['preceding_opaque_or_excluded_block_index']==0


def unit(index,n):
    return {'block_index':index,'status':'included','text':' '.join(['word']*n),'words':n,
            'section_path':['Lead'],'context_dependency_flags':[],'unit_type':'complete_prose_paragraph'}


def test_window_keeps_complete_units_and_records_overshoot():
    result=full.choose_window([unit(0,300),unit(1,350),unit(2,200)],250)
    assert result['selected_block_indices']==[0,1] and result['selected_window_words']==650
    assert result['soft_target_overshoot_words']==150 and result['heuristic_minimum_words']==313
    assert result['meets_125_percent_word_heuristic'] is True
    assert result['complete_sentence_units_only'] is True and result['contiguous_raw_article_excerpt'] is False
    short=full.choose_window([unit(0,100)],500)
    assert short['selected_window_words']==100 and not short['meets_125_percent_word_heuristic']
    with pytest.raises(ValueError):full.choose_window([unit(0,100)],53)


def test_twenty_missing_revisions_remain_twenty_reported_failures(tmp_path,monkeypatch):
    path=tmp_path/'prompts.jsonl';rows=save_selection(path);calls=[]
    class FakeAPI:
        def __init__(self,*args,**kwargs):self.used={}
        def query(self,params):calls.append(params);return {'query':{'pages':[]}},'fixture'
    monkeypatch.setattr(full,'source_builder',lambda:(SimpleNamespace(CachedAPI=FakeAPI),BUILDER_HASH))
    output=tmp_path/'out';manifest=full.build(path,output,offline=True)
    sources=[json.loads(s) for s in (output/'sources.jsonl').read_text().splitlines()]
    assert len(sources)==20 and [s['source_group_id'] for s in sources]==[r['source_group_id'] for r in rows]
    assert manifest['retrieval_statuses']=={'revision_unavailable':20}
    assert len(calls)==2 and all(len(c['revids'].split('|'))==10 for c in calls)
    assert all('rvsection' not in c and 'pageids' not in c for c in calls)
    assert not (output/'revisions').exists()


def test_unrequested_or_duplicate_revision_cannot_enter_collection(tmp_path,monkeypatch):
    path=tmp_path/'prompts.jsonl';rows=save_selection(path);row,page,revision=identity(1)
    page['revisions']=[revision,copy.deepcopy(revision)]
    class FakeAPI:
        def __init__(self,*args,**kwargs):self.used={}
        def query(self,params):return {'query':{'pages':[page]}},'fixture'
    monkeypatch.setattr(full,'source_builder',lambda:(SimpleNamespace(CachedAPI=FakeAPI),BUILDER_HASH))
    with pytest.raises(ValueError,match='duplicate revision'):
        full.build(path,tmp_path/'out',offline=True)
    assert not (tmp_path/'out/manifest.json').exists()


def test_output_publication_never_replaces_other_bytes(tmp_path):
    full.publish_expected(tmp_path,{'proof.json':b'old'})
    with pytest.raises(FileExistsError):full.publish_expected(tmp_path,{'other':b'new','proof.json':b'different'})
    assert (tmp_path/'proof.json').read_bytes()==b'old' and not (tmp_path/'other').exists()
