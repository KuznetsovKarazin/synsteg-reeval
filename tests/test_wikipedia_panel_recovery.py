"""Offline tests of a real-source pipeline; fixtures never become study sources."""
import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('wikipedia_panel_recovery',ROOT/'scripts/build_wikipedia_panel_recovery.py')
panel=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)


def row(page_id, text=None, topics=('physics',)):
    text=text or ' '.join(f'word{page_id}_{i}' for i in range(70))
    return {'page_id':page_id,'title':f'Unique subject {page_id}','excerpt':text,
            'excerpt_sha256':panel.digest(text.encode()),'candidate_topics':list(topics),
            'wikidata_item':f'Q{page_id}','source_group_id':f'g{page_id}'}


def test_nested_cleanup_and_complete_sentence_limit():
    raw="{{Infobox|thing={{nested|value}}}} '''Earth''' is a [[planet]] with [[Liquid|water]].<ref>citation</ref> [[File:X.png|thumb|The [[planet]]]] Another fact is here. Final long sentence has lots of words which cannot fit. ==Other== no"
    cleaned=panel.lead_text(raw)
    assert cleaned.startswith('Earth is a planet with water.')
    assert 'citation' not in cleaned and 'File:' not in cleaned
    assert panel.source_excerpt(cleaned,max_words=13)=='Earth is a planet with water. Another fact is here.'
    assert panel.source_excerpt('A sentence without its ending')==''
    with pytest.raises(ValueError,match='Unbalanced'):
        panel.lead_text('Text {{broken')


def test_duplicate_components_precede_assignment():
    text='The brown fox jumped across the large field while the farmer was carefully watching the little animals at the edge of the forest near a quiet stream.'
    a,b,c,d=row(1,text),row(2,text+' Extra detail.'),row(3,'Unrelated astronomy story about lunar craters.'),row(4,'Completely different wording.')
    d['wikidata_item']=c['wikidata_item']
    groups,edges=panel.cluster_candidates([a,b,c,d])
    assert sorted(g['component_size'] for g in groups)==[2,2]
    assert {tuple(g['component_page_ids']) for g in groups}=={(1,2),(3,4)}
    assert any(e[-1]=='five_token_jaccard' for e in edges)


def test_matching_does_not_reuse_shared_sources_and_is_deterministic():
    groups=[row(1,topics=('physics','chemistry')),row(2,topics=('physics',))]
    args={'topic_names':['physics','chemistry'],'genres':['educational'],'lengths':[250],'splits':[('development',1)]}
    first=panel.allocate(groups,**args)
    second=panel.allocate(list(reversed(groups)),**args)
    assert first==second
    assert len({r['source_group_id'] for r in first})==2
    assert next(r for r in first if r['topic']=='chemistry')['page_id']==1
    with pytest.raises(ValueError,match='Insufficient'):
        panel.allocate(groups[:1],topic_names=['physics'],genres=['educational'],lengths=[250],splits=[('test',2)])


def test_full_cells_and_hash_validation():
    groups=[]
    for t,topic in enumerate(panel.TOPICS):
        groups.extend(row(1000*t+i,topics=(topic,)) for i in range(160))
    records=panel.allocate(groups)
    assert panel.validate_records(records)=={'development':100,'train':400,'validation':100,'test':1000}
    assert len({r['record_id'] for r in records})==1600
    records[0]['prompt']+='changed'
    with pytest.raises(ValueError,match='Prompt hash'):
        panel.validate_records(records)


def test_raw_cache_replay_checks_hash_and_never_fetches_offline(tmp_path):
    response=mock.MagicMock()
    response.__enter__.return_value.read.return_value=b'{"query":{"pages":[]}}'
    with mock.patch.object(panel.urllib.request,'urlopen',return_value=response) as opened, mock.patch.object(panel.time,'sleep'):
        data,key=panel.CachedAPI(tmp_path).query({'pageids':'1'})
        assert opened.call_count==1
    with mock.patch.object(panel.urllib.request,'urlopen',side_effect=AssertionError('network')):
        assert panel.CachedAPI(tmp_path,offline=True).query({'pageids':'1'})==(data,key)
        with pytest.raises(FileNotFoundError):
            panel.CachedAPI(tmp_path,offline=True).query({'pageids':'2'})
    (tmp_path/f'{key}.json').write_text('{}')
    with pytest.raises(ValueError,match='Corrupt'):
        panel.CachedAPI(tmp_path,offline=True).query({'pageids':'1'})


def test_api_error_is_not_frozen(tmp_path):
    response=mock.MagicMock()
    response.__enter__.return_value.read.return_value=b'{"error":{"code":"maxlag"}}'
    with mock.patch.object(panel.urllib.request,'urlopen',return_value=response):
        with pytest.raises(RuntimeError,match='maxlag'):
            panel.CachedAPI(tmp_path).query({'pageids':'1'})
    assert list(tmp_path.iterdir())==[]


def test_revision_source_provenance_and_redirect_rejection():
    text=('A terrestrial habitat is an environment where living organisms occupy the land and interact with its physical conditions. '
          'Water availability, temperature, soil structure, and biological interactions affect the communities that occur there. '
          'Different regions support a wide range of organisms with distinct ways of obtaining energy and resources.')
    page={'pageid':100,'ns':0,'title':'Habitat','pageprops':{'wikibase_item':'Q123'},
          'revisions':[{'revid':200,'timestamp':'2026-01-01T00:00:00Z','sha1':'abc',
                        'slots':{'main':{'content':text,'contentmodel':'wikitext'}}}]}
    got,error=panel.candidate(page,['biology'],'rawkey')
    assert error is None and got['revision_id']==200
    assert got['revision_url'].endswith('oldid=200')
    assert got['raw_response']=='raw/rawkey.json'
    assert got['lead_wikitext_sha256']==panel.digest(text.encode())
    page['redirect']=True
    assert panel.candidate(page,['biology'],'rawkey')[0] is None


def test_publish_never_overwrites_frozen_artifact(tmp_path):
    path=tmp_path/'artifact'
    panel.publish(path,b'first')
    panel.publish(path,b'first')
    with pytest.raises(ValueError,match='overwrite'):
        panel.publish(path,b'second')
    assert path.read_bytes()==b'first'


def test_template_values_abbreviations_and_unknown_inline_rejection():
    cleaned=panel.lead_text("{{Infobox|x=7}} Geography occupies an area of {{convert|581730|km2|sqmi|abbr=on}}. It lies in the west. The next fact is a third sentence. A fourth sentence must be omitted.")
    excerpt=panel.source_excerpt(cleaned)
    assert '581730 km2' in excerpt
    assert panel.render_templates('Between {{convert|1|to|3|km}} away.')=='Between 1 to 3 km away.'
    assert panel.render_templates('Area {{val|150| u =km2}}.')=='Area 150 km2.'
    assert 'OMITTEDINLINETEMPLATE' in panel.render_templates('Height {{convert|5|ft|9|in}}.')
    assert 'fourth' not in excerpt and 'third sentence.' in excerpt
    assert panel.source_excerpt('Its development (e.g. the first cycle) is explained here. More text follows.').startswith('Its development (e.g. the first cycle) is explained here.')
    assert panel.source_excerpt(panel.lead_text('The substance is {{unknown|a reactive material}}. A second fact follows.'))==''
    assert panel.source_excerpt('This is unbalanced (e.g. a broken passage. Still broken.')==''


def test_duplicate_edges_are_sorted_for_hash_seed_independence():
    base=' '.join('token'+str(i) for i in range(120))
    groups=[row(i,base+' '+str(i)) for i in range(8)]
    _,edges=panel.cluster_candidates(groups)
    assert edges==sorted(edges)


def test_atomic_publish_rejects_concurrent_different_bytes(tmp_path):
    path=tmp_path/'artifact'
    path.write_bytes(b'other process')
    with mock.patch.object(Path,'exists',return_value=False):
        with pytest.raises(ValueError,match='concurrent overwrite'):
            panel.publish(path,b'our process')
    assert path.read_bytes()==b'other process'
    assert not list(tmp_path.glob('*.tmp'))


def test_self_closing_reference_never_consumes_following_prose():
    raw='The fourth derivative is called snap.<ref name="repeat"/> The fifth is called crackle.<ref>Supporting citation.</ref> The sixth is called pop.'
    cleaned=panel.lead_text(raw)
    assert 'The fifth is called crackle.' in cleaned
    assert 'Supporting citation' not in cleaned
    assert panel.render_templates("{{Nihongo|Kensō Soai|硤合 憲三|Soai Kensō}} is a chemist.")=='Kensō Soai is a chemist.'


def test_quotes_ellipsis_and_removed_lists_never_yield_fragments():
    text='First complete sentence. Second complete sentence. The author said: "Science continues. Both approaches ... matter." Fourth statement.'
    assert panel.source_excerpt(text)=='First complete sentence. Second complete sentence. The author said: "Science continues. Both approaches ... matter."'
    text='A full first sentence. Examples are:\n* item one\n* item two\nThe next paragraph follows.'
    assert panel.source_excerpt(panel.lead_text(text))=='A full first sentence.'
    assert panel.lead_text('AECOM (, ; formerly a firm) is a company.')=='AECOM (formerly a firm) is a company.'
    assert panel.lead_text('They study physics.The department teaches.')=='They study physics. The department teaches.'


def test_exclamation_in_channel_name_does_not_cut_parenthetical():
    text='First fact. Second fact. Channels include MTV, TeleHit (Mexico), Kerrang! (UK), and other regional channels.'
    assert panel.source_excerpt(text)==text


def test_exclamation_in_wikilink_title_followed_by_tv_is_not_terminal():
    raw='First complete fact. Second complete fact. Channels include [[Kerrang! TV]] (UK, defunct), [[RAC105TV]], and other broadcasters.'
    assert panel.source_excerpt(panel.lead_text(raw)).endswith('Kerrang! TV (UK, defunct), RAC105TV, and other broadcasters.')
