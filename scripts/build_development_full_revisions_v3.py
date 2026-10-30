#!/usr/bin/env python3
"""Freeze exact full Wikipedia revisions for twenty already selected dev sources.

No generation or capacity selection. All twenty identities and all failures are
retained. Windows are ordered prose prefixes AFTER declared exclusions, not
contiguous verbatim excerpts from the raw article. V1 resources stay read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
VERSION='development-full-revisions-v3'
EXCLUDED_SECTIONS={'references','notes','footnotes','citations','bibliography','sources','further reading',
                   'external links','see also','works cited','notes and references','references and notes',
                   'selected bibliography','literature cited'}
METADATA_TEMPLATES={'short description','use dmy dates','use mdy dates','use american english','use british english',
    'use canadian english','use indian english','use australian english','about','for','other uses','other people',
    'distinguish','redirect','redirect-distinguish','italic title','displaytitle','coord','multiple image','listen',
    'commons category','commons','wiktionary','wikiquote','wikisource','portal','portal bar','authority control',
    'good article','featured article','pp','pp-protected','pp-semi-indef','reflist','refbegin','refend',
    'anchor','advert','advertisement','one source','refimprove','unreferenced','more citations needed','cleanup',
    'multiple issues','citation style','notability','primary sources','original research','unbalanced','globalize',
    'update','technical','lead too short','lead too long','lead rewrite','copy edit'}
METADATA_PREFIXES=('infobox','automatic taxobox','taxobox','speciesbox','chembox','drugbox','use ','cs1 config')
OPAQUE_TAGS=('math','gallery','score','timeline','pre','syntaxhighlight','source','code','nowiki','table')
WORD=re.compile(r'[A-Za-z]+')
PROTECTED_PUNCT={'.':'\ue010','!':'\ue011','?':'\ue012'}
STRUCTURE=re.compile(r'<!--[\s\S]*?-->|^={2,6}[^\r\n]*?={2,6}[ \t]*$|</?(?:ref|math|gallery|score|timeline|pre|syntaxhighlight|source|code|nowiki|table)\b[^>]*>|\{\{|\}\}|\{\||\|\}|\[\[|\]\]|\r?\n[ \t]*\r?\n',re.M|re.I)


def sha256(data): return hashlib.sha256(data).hexdigest()

def json_bytes(obj): return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()

def jsonl_bytes(rows): return b''.join((json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode() for r in rows)

def word_count(text): return len(text.split())

def normalize_space(text): return ' '.join(text.split())

def repository_path(path):
    resolved=Path(path).resolve()
    try:return resolved.relative_to(ROOT).as_posix()
    except ValueError:return resolved.name


def source_builder():
    path=ROOT/'scripts/build_wikipedia_panel_recovery.py'
    spec=importlib.util.spec_from_file_location('_frozen_v1_helpers_for_full_revision_v3',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,sha256(path.read_bytes())


def select_sources(path):
    raw=Path(path).read_bytes();rows=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(rows)!=20: raise ValueError('Exactly twenty preselected development records are required')
    for field in ('record_id','source_group_id','page_id','revision_id'):
        if len({r.get(field) for r in rows})!=20: raise ValueError(f'Expected twenty unique {field} values')
    for row in rows:
        if row.get('split')!='development': raise ValueError('Non-development source refused')
        if type(row.get('page_id')) is not int or type(row.get('revision_id')) is not int: raise ValueError('Invalid source identity')
        if type(row.get('parent_target_words')) is not int or row.get('parent_target_words') not in (250,500): raise ValueError('parent_target_words must preserve original 250/500 target')
        if sha256(row['prompt'].encode())!=row.get('prompt_sha256'): raise ValueError('Selected prompt hash mismatch')
        if sha256(row['excerpt'].encode())!=row.get('excerpt_sha256'): raise ValueError('Selected excerpt hash mismatch')
        if not re.fullmatch(r'[0-9a-f]{40}',row.get('revision_sha1','')): raise ValueError('Frozen full revision SHA-1 required')
        if not isinstance(row.get('revision_timestamp'),str): raise ValueError('Frozen revision timestamp required')
    return rows,sha256(raw)


def validate_revision(page,revision,expected):
    if page.get('pageid')!=expected['page_id'] or revision.get('revid')!=expected['revision_id']:
        raise ValueError('Page/revision identity mismatch')
    if revision.get('timestamp')!=expected['revision_timestamp']: raise ValueError('Revision timestamp mismatch')
    if revision.get('sha1')!=expected['revision_sha1']: raise ValueError('Revision SHA-1 differs from frozen identity')
    slot=revision.get('slots',{}).get('main',{})
    if slot.get('contentmodel')!='wikitext' or not isinstance(slot.get('content'),str):
        raise ValueError('Main-slot wikitext unavailable, hidden, or different content model')
    text=slot['content'];raw=text.encode('utf-8');sha1=hashlib.sha1(raw).hexdigest()
    if sha1!=expected['revision_sha1']: raise ValueError('Exact UTF-8 full content SHA-1 mismatch')
    if slot.get('sha1',sha1)!=sha1: raise ValueError('Main-slot SHA-1 mismatch')
    return text,{'full_wikitext_utf8_bytes':len(raw),'full_wikitext_sha1':sha1,'full_wikitext_sha256':sha256(raw),
                 'api_page_title':page.get('title'),'frozen_source_title':expected['title'],
                 'page_title_change_detected':page.get('title')!=expected['title']}


def heading_name(raw):
    raw=re.sub(r'\{\{anchor\|[^{}]*\}\}','',raw,flags=re.I)
    text=re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]',r'\2',raw)
    text=re.sub(r'\[\[([^\]]+)\]\]',r'\1',text)
    return normalize_space(text.replace("'''",'').replace("''",''))


def raw_blocks(text):
    """Paragraphs/headings outside balanced structures; offsets are code points."""
    blocks=[];warnings=[];cursor=0;template=table=link=0;html=[];sections=[];excluded_level=None
    def emit(end):
        nonlocal cursor
        start=cursor
        while start<end and text[start].isspace(): start+=1
        while end>start and text[end-1].isspace(): end-=1
        if end>start:
            blocks.append({'raw_start_char':start,'raw_end_char':end,'raw_fragment':text[start:end],
                           'section_path':[item[1] for item in sections] or ['Lead'],
                           'section_path_raw':[item[2] for item in sections] or ['Lead'],
                           'excluded_section':excluded_level is not None})
    for match in STRUCTURE.finditer(text):
        token=match.group()
        if token.startswith('<!--'): continue
        if token.startswith('<'):
            tag_match=re.match(r'<(/?)([a-zA-Z]+)',token);closing,tag=tag_match.group(1),tag_match.group(2).lower()
            if token.rstrip().endswith('/>'): continue
            if closing:
                if html and html[-1]==tag: html.pop()
                elif tag in html:
                    warnings.append({'offset_char':match.start(),'kind':'mismatched_html_block_close','tag':tag})
                    html=html[:html.index(tag)]
                else: warnings.append({'offset_char':match.start(),'kind':'unmatched_html_block_close','tag':tag})
            else: html.append(tag)
            continue
        if html: continue
        if token=='{{': template+=1;continue
        if token=='}}':
            if template: template-=1
            else: warnings.append({'offset_char':match.start(),'kind':'unmatched_template_close'})
            continue
        if token=='{|': table+=1;continue
        if token=='|}':
            if table: table-=1
            else: warnings.append({'offset_char':match.start(),'kind':'unmatched_table_close'})
            continue
        if token=='[[': link+=1;continue
        if token==']]':
            if link: link-=1
            else: warnings.append({'offset_char':match.start(),'kind':'unmatched_link_close'})
            continue
        if template or table or link: continue
        if token.startswith('='):
            match_heading=re.fullmatch(r'(={2,6})(.*?)(={2,6})[ \t]*',token)
            if not match_heading or len(match_heading[1])!=len(match_heading[3]): continue
            emit(match.start());level=len(match_heading[1]);name=heading_name(match_heading[2])
            sections=[item for item in sections if item[0]<level]+[(level,name,match_heading[2].strip())]
            if excluded_level is not None and level<=excluded_level: excluded_level=None
            if name.casefold() in EXCLUDED_SECTIONS: excluded_level=level
            cursor=match.end()
        else:
            emit(match.start());cursor=match.end()
    emit(len(text))
    if template or table or link or html:
        warnings.append({'offset_char':len(text),'kind':'unclosed_structure','templates':template,'tables':table,'links':link,'html':html})
    return blocks,warnings


def replace_balanced(text,opening,closing,replacement):
    out=[];i=0;count=0
    while i<len(text):
        if not text.startswith(opening,i): out.append(text[i]);i+=1;continue
        depth=1;j=i+len(opening)
        while j<len(text) and depth:
            if text.startswith(opening,j): depth+=1;j+=len(opening)
            elif text.startswith(closing,j): depth-=1;j+=len(closing)
            else:j+=1
        if depth: raise ValueError('Unbalanced '+opening)
        out.append(replacement);count+=1;i=j
    return ''.join(out),count


def remove_media_links(text):
    out=[];i=0;count=0
    while i<len(text):
        if not text.startswith('[[',i):out.append(text[i]);i+=1;continue
        depth=1;j=i+2
        while j<len(text) and depth:
            if text.startswith('[[',j):depth+=1;j+=2
            elif text.startswith(']]',j):depth-=1;j+=2
            else:j+=1
        if depth:raise ValueError('Unbalanced media/wiki link')
        content=text[i+2:j-2]
        if re.match(r'\s*(?:File|Image|Category):',content,re.I):out.append('\n\n');count+=1
        else:out.append(text[i:j])
        i=j
    return ''.join(out),count


def safe_templates(text,builder,counts):
    """Unknown BODY templates never get the v1 'leading metadata' exemption."""
    out=[];i=0
    while i<len(text):
        if not text.startswith('{{',i):out.append(text[i]);i+=1;continue
        depth=1;j=i+2
        while j<len(text) and depth:
            if text.startswith('{{',j):depth+=1;j+=2
            elif text.startswith('}}',j):depth-=1;j+=2
            else:j+=1
        if depth:raise ValueError('Unbalanced template')
        body=text[i+2:j-2];name=body.partition('|')[0].strip().replace('_',' ').casefold()
        if name in METADATA_TEMPLATES or name.startswith(METADATA_PREFIXES):
            value='\n\n';counts['documented_metadata_templates_removed']+=1;counts['metadata_template:'+name]+=1
        else:
            # Recursing before the supported wrapper renderer prevents an inner
            # unknown template from being silently dropped as leading metadata.
            cooked=safe_templates(body,builder,counts)
            guard='BODYGUARD '
            value=builder.render_templates(guard+'{{'+cooked+'}}')[len(guard):]
            if 'OMITTEDINLINETEMPLATE' in value:
                counts['opaque_templates']+=1;value=' OPAQUEV3TEMPLATE '
            else:counts['supported_templates_rendered_or_citations_removed']+=1
        out.append(value);i=j
    return ''.join(out)


def protect_link_punctuation(text):
    if any(marker in text for marker in PROTECTED_PUNCT.values()):raise ValueError('Reserved punctuation sentinel in source')
    out=[];i=0;count=0
    while i<len(text):
        if not text.startswith('[[',i):out.append(text[i]);i+=1;continue
        depth=1;j=i+2
        while j<len(text) and depth:
            if text.startswith('[[',j):depth+=1;j+=2
            elif text.startswith(']]',j):depth-=1;j+=2
            else:j+=1
        if depth:raise ValueError('Unbalanced link in punctuation protection')
        block=text[i:j]
        for char,marker in PROTECTED_PUNCT.items():
            count+=block.count(char);block=block.replace(char,marker)
        out.append(block);i=j
    return ''.join(out),count


def clean_block(raw,builder):
    counts=Counter();text=re.sub(r'<!--[\s\S]*?-->',' ',raw)
    text,n=re.subn(r'<ref\b[^>]*/\s*>',' ',text,flags=re.I);counts['self_closing_references_removed']+=n
    text,n=re.subn(r'<ref\b[^>]*>[\s\S]*?</ref\s*>',' ',text,flags=re.I);counts['paired_references_removed']+=n
    for tag in OPAQUE_TAGS:
        text,n=re.subn(r'<'+tag+r'\b[^>]*>[\s\S]*?</'+tag+r'\s*>',' OPAQUEV3NONPROSE ',text,flags=re.I)
        counts['opaque_'+tag+'_blocks']+=n
    text,n=replace_balanced(text,'{|','|}',' OPAQUEV3TABLE ');counts['opaque_wikitables']+=n
    text,n=remove_media_links(text);counts['media_or_category_links_removed']+=n
    text=safe_templates(text,builder,counts)
    boundaries=[]
    for marker in ('OPAQUEV3','OMITTEDINLINETEMPLATE'):
        index=text.find(marker)
        if index>=0:boundaries.append((index,'opaque_template_formula_table_or_nonprose'))
    list_match=re.search(r'(?m)^\s*[*#;:|!]',text)
    if list_match:boundaries.append((list_match.start(),'list_or_indented_nonprose'))
    if boundaries:
        index,reason=min(boundaries)
        text=text[:index]
        counts['complete_prefix_stopped_before:'+reason]+=1
    # Keep inline sub/sup text attached (e.g. H2O) rather than introducing spaces.
    text=re.sub(r'</?(?:sub|sup)\b[^>]*>','',text,flags=re.I)
    text,n=protect_link_punctuation(text);counts['wikilink_punctuation_preserved']+=n
    cleaned=builder.lead_text(text)
    if not cleaned:return None,'no_prose_after_declared_exclusions',dict(counts)
    if any(marker in cleaned for marker in ('OMITTED','OPAQUEV3','{{','}}','[[',']]')):
        return None,'residual_opaque_markup',dict(counts)
    if word_count(cleaned)<15:return None,'fewer_than_15_whitespace_words',dict(counts)
    complete=builder.source_excerpt(cleaned,max_words=100000,max_sentences=100000)
    if not complete or word_count(complete)<15:return None,'no_complete_supported_prefix_of_15_words',dict(counts)
    if complete!=cleaned:counts['incomplete_trailing_fragment_omitted']+=1
    cleaned=complete
    for char,marker in PROTECTED_PUNCT.items():cleaned=cleaned.replace(marker,char)
    if re.match(r'^(?:is|are|was|were|has|have|had|and|or|also)\b',cleaned):
        return None,'leading_fragment',dict(counts)
    return cleaned,None,dict(counts)


def extract_prose(text,builder):
    raw,warnings=raw_blocks(text);blocks=[]
    for index,block in enumerate(raw):
        source=block.pop('raw_fragment')
        row={'block_index':index,'offset_unit':'Python Unicode code points in exact full.wiki content',
             **block,'raw_fragment_sha256':sha256(source.encode())}
        if block['excluded_section']:
            row.update(status='excluded',reason='declared_reference_or_navigation_section',text=None,words=0)
        else:
            try:cleaned,reason,counts=clean_block(source,builder)
            except ValueError as error:cleaned,reason,counts=None,'unsupported_markup:'+str(error),{}
            row.update(status='included' if cleaned is not None else 'excluded',reason=reason,
                       text=cleaned,words=word_count(cleaned) if cleaned else 0,transform_counts=counts)
            if cleaned is not None:
                row['unit_type']='complete_sentence_prefix_before_boundary' if any(k.startswith('complete_prefix_stopped_before:') or k=='incomplete_trailing_fragment_omitted' for k in counts) else 'complete_prose_paragraph'
                row['text_sha256']=sha256(cleaned.encode())
                row['context_dependency_flags']=sorted(set(re.findall(
                    r'\b(?:equation above|equation below|figure above|figure below|the former|the latter|as described above|as shown below)\b',cleaned,re.I)))
        blocks.append(row)
    last_gap={}
    for row in blocks:
        section=tuple(row['section_path'])
        if row['status']=='included':
            if section in last_gap and re.match(r'^(?:These|Those|This|Such|They|It|Both|The latter|The former)\b',row['text']):
                row['context_dependency_flags'].append('possible_antecedent_after_omitted_content')
                row['preceding_opaque_or_excluded_block_index']=last_gap[section]
            if row['unit_type']=='complete_sentence_prefix_before_boundary':last_gap[section]=row['block_index']
        elif row['reason'] not in ('declared_reference_or_navigation_section','no_prose_after_declared_exclusions'):
            last_gap[section]=row['block_index']
    return blocks,warnings


def choose_window(blocks,parent_target_words):
    if parent_target_words not in (250,500):raise ValueError('Use original parent target 250/500, not short v2 target')
    target=2*parent_target_words;minimum=math.ceil(1.25*parent_target_words)
    selected=[];total=0
    for block in blocks:
        if block['status']!='included':continue
        if total>=target:break
        selected.append(block);total+=block['words']
    text='\n\n'.join(block['text'] for block in selected)
    return {'selection_rule':'Ordered prefix of complete eligible units after declared exclusions; each unit is a full paragraph or complete-sentence prefix ending before its first opaque boundary; stop after first unit reaching 2x parent target',
            'original_long_target_words':parent_target_words,'source_window_soft_target_words':target,
            'heuristic_minimum_words':minimum,'selected_window_words':total,
            'soft_target_overshoot_words':max(0,total-target),'meets_125_percent_word_heuristic':total>=minimum,
            'stop_reason':'complete_unit_soft_target_reached' if total>=target else 'eligible_prose_exhausted',
            'selected_block_indices':[block['block_index'] for block in selected],
            'selected_section_paths':[block['section_path'] for block in selected],
            'source_window':text,'source_window_sha256':sha256(text.encode()),
            'complete_sentence_units_only':True,'selected_unit_types':[b['unit_type'] for b in selected],
            'contiguous_raw_article_excerpt':False,
            'context_dependency_flags':[{'block_index':b['block_index'],'flags':b['context_dependency_flags'],'preceding_opaque_or_excluded_block_index':b.get('preceding_opaque_or_excluded_block_index')} for b in selected if b['context_dependency_flags']]}


def publish_expected(output,expected):
    output=Path(output)
    for name,data in expected.items():
        p=output/name
        if p.exists() and p.read_bytes()!=data:raise FileExistsError(f'Refusing different frozen output: {name}')
    for name,data in expected.items():
        p=output/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():continue
        temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
        with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        try:os.link(temp,p)
        except FileExistsError:
            if p.read_bytes()!=data:raise FileExistsError(f'Refusing concurrent output replacement: {name}')
        finally:temp.unlink(missing_ok=True)


def build(source_path,output,offline=False):
    rows,prompt_hash=select_sources(source_path);output=Path(output);builder,cleaner_hash=source_builder()
    code_hash=sha256(Path(__file__).read_bytes())
    selection={'schema':VERSION,'source_prompts_file':repository_path(source_path),'source_prompts_sha256':prompt_hash,
               'records':[{k:r[k] for k in ('record_id','source_group_id','page_id','revision_id','revision_timestamp','revision_sha1','parent_target_words','title','topic','genre')} for r in rows],
               'builder_sha256':code_hash,'frozen_v1_cleaner_sha256':cleaner_hash,
               'query_batch_size':10,'source_replacement_allowed':False,'topic_relabeling_allowed':False,
               'excluded_sections':sorted(EXCLUDED_SECTIONS),'minimum_eligible_unit_words':15,
               'known_metadata_template_names':sorted(METADATA_TEMPLATES),'known_metadata_template_prefixes':list(METADATA_PREFIXES),
               'opaque_html_tags':list(OPAQUE_TAGS),'raw_block_offset_unit':'Python Unicode code points in exact full.wiki content',
               'window_rule':'complete eligible units in original order until >=2*parent_target_words; full paragraphs or complete-sentence prefixes ending before opaque boundaries; no truncation of last eligible unit',
               'word_sufficiency_rule':'at least ceil(1.25*parent_target_words), exploratory availability heuristic only',
               'live_template_expansion':False,'generation_enabled':False}
    publish_expected(output,{'selection.json':json_bytes(selection)})
    api=builder.CachedAPI(output/'raw',offline=offline,timeout=60)
    ids=sorted(r['revision_id'] for r in rows);expected_by_id={r['revision_id']:r for r in rows}
    revisions={};fetch_failures={}
    for start in range(0,len(ids),10):
        batch=ids[start:start+10]
        data,key=api.query({'revids':'|'.join(map(str,batch)),'prop':'revisions',
                           'rvprop':'ids|timestamp|sha1|content|contentmodel|slotsha1','rvslots':'main'})
        if 'continue' in data:raise ValueError('Unexpected revision response pagination; preserve cache and inspect')
        for page in data.get('query',{}).get('pages',[]):
            for revision in page.get('revisions',[]):
                revid=revision.get('revid')
                if revid not in batch or revid in revisions or revid in fetch_failures:raise ValueError('Unexpected or duplicate revision returned')
                try:
                    text,proof=validate_revision(page,revision,expected_by_id[revid])
                    revisions[revid]=(text,proof,key)
                except ValueError as error:fetch_failures[revid]={'status':'retrieval_identity_or_content_failure','reason':str(error),'raw_response':'raw/'+key+'.json'}
        for revid in batch:
            if revid not in revisions and revid not in fetch_failures:
                fetch_failures[revid]={'status':'revision_unavailable','reason':'Requested exact revision omitted or hidden; no latest-version fallback','raw_response':'raw/'+key+'.json'}
        print(f'Exact revisions accounted for: {len(revisions)+len(fetch_failures)}/20',flush=True)
    sources=[];passages=[];windows=[];expected={}
    for row in rows:
        identity={k:row[k] for k in ('record_id','source_group_id','page_id','revision_id','revision_timestamp','revision_sha1','title','topic','genre','parent_target_words')}
        identity.update(split='development',grounded_v2_target_words=row['target_words'],original_excerpt_sha256=row['excerpt_sha256'])
        if row['revision_id'] in fetch_failures:
            failure=fetch_failures[row['revision_id']]
            sources.append({**identity,**failure,'extraction_status':'not_attempted','window_status':'unavailable'})
            windows.append({**identity,'status':'retrieval_failure_retained','source_window':None,'selected_window_words':None,'meets_125_percent_word_heuristic':False})
            continue
        text,proof,key=revisions[row['revision_id']];filename=f"revisions/{row['page_id']}-{row['revision_id']}.wiki"
        expected[filename]=text.encode('utf-8')
        blocks,warnings=extract_prose(text,builder);window=choose_window(blocks,row['parent_target_words'])
        original_preserved=normalize_space(row['excerpt']) in normalize_space(window['source_window'])
        status='word_heuristic_met' if window['meets_125_percent_word_heuristic'] else 'insufficient_selected_prose_words'
        if not original_preserved:status='review_required_original_excerpt_not_preserved'
        if warnings:status='review_required_structure_warnings'
        if any('possible_antecedent_after_omitted_content' in b['flags'] for b in window['context_dependency_flags']):
            status='review_required_context_gap'
        windows.append({**identity,**window,'status':status,'original_excerpt_preserved_verbatim_modulo_whitespace':original_preserved,
                        'factual_adequacy_certified':False,'raw_response':'raw/'+key+'.json','full_wikitext_file':filename})
        sources.append({**identity,**proof,'status':'exact_revision_verified','raw_response':'raw/'+key+'.json','full_wikitext_file':filename,
                        'extraction_status':'completed_with_structure_warnings' if warnings else 'completed',
                        'structure_warnings':warnings,'window_status':status,'raw_blocks':len(blocks),
                        'included_prose_units':sum(b['status']=='included' for b in blocks),
                        'total_extracted_prose_words':sum(b['words'] for b in blocks),
                        'exclusion_counts':dict(sorted(Counter(b['reason'] for b in blocks if b['status']=='excluded').items()))})
        passages.extend({**identity,**block} for block in blocks)
    expected['sources.jsonl']=jsonl_bytes(sources);expected['passages.jsonl']=jsonl_bytes(passages);expected['source-windows.jsonl']=jsonl_bytes(windows)
    attribution={'schema':VERSION,'license':'CC BY-SA 4.0','license_url':'https://creativecommons.org/licenses/by-sa/4.0/',
                 'license_policy_url':'https://en.wikipedia.org/wiki/Wikipedia:Copyrights',
                 'changes':'Exact full revision wikitext retained. Derivative ordered prose prefix removes declared metadata, references, lists, tables, formulas and unsupported templates, preserves supported numeric/style wrappers, and selects complete eligible units: full paragraphs or complete-sentence prefixes before opaque boundaries. No live template expansion.',
                 'sources':[{k:r[k] for k in ('source_group_id','title','page_id','revision_id','revision_url','history_url','attribution','license','license_url')} for r in rows]}
    expected['attribution.json']=json_bytes(attribution)
    raw_inventory=sorted(api.used.values(),key=lambda x:x['path'])
    manifest={'schema':VERSION,'status':'materialized_development_source_candidates_only','source_count':20,'split':'development',
              'source_prompts_sha256':prompt_hash,'selection_sha256':sha256(json_bytes(selection)),
              'builder_sha256':code_hash,'frozen_v1_cleaner_sha256':cleaner_hash,
              'retrieval_statuses':dict(sorted(Counter(s['status'] for s in sources).items())),
              'window_statuses':dict(sorted(Counter(w['status'] for w in windows).items())),
              'sources_meeting_word_heuristic':sum(w['meets_125_percent_word_heuristic'] for w in windows),
              'factual_adequacy_certified':False,'source_replacements':0,'topic_relabelings':0,'new_model_generations':0,
              'train_validation_test_sources_used':0,'paid_requests':0,'exact_revision_api_batches':2,
              'raw_responses':raw_inventory,'raw_inventory_sha256':sha256(json_bytes(raw_inventory)),
              'files':[{'path':name,'bytes':len(data),'sha256':sha256(data)} for name,data in sorted(expected.items())],
              'limitations':['Search strata remain operational labels, not semantic gold labels.',
                             'Longer source material changes the generation condition; these are reused development groups, not a new independent sample.',
                             'Only declared supported prose is retained; insufficient selected material does not prove the complete article has too few facts.',
                             'Word sufficiency is a heuristic, not a count or validation of factual claims.',
                             'The window is an ordered prose prefix after exclusions, not a contiguous verbatim raw-article extract.',
                             'No live rendering of transcluded templates is used. Further factual/content review is needed before a long-summary experiment.']}
    expected['manifest.json']=json_bytes(manifest)
    if sha256(Path(source_path).read_bytes())!=prompt_hash:raise ValueError('Selected development prompts changed during build')
    if sha256(Path(__file__).read_bytes())!=code_hash:raise ValueError('Extraction code changed during build')
    if sha256((ROOT/'scripts/build_wikipedia_panel_recovery.py').read_bytes())!=cleaner_hash:raise ValueError('Frozen helper changed during build')
    publish_expected(output,expected)
    return manifest


def verify(output):
    output=Path(output).resolve();manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!=VERSION or manifest.get('source_count')!=20:raise ValueError('Invalid v3 manifest')
    if sha256((output/'selection.json').read_bytes())!=manifest['selection_sha256']:raise ValueError('Selection hash mismatch')
    for row in manifest['files']:
        path=(output/row['path']).resolve()
        if not path.is_relative_to(output):raise ValueError('Unsafe artifact path')
        raw=path.read_bytes()
        if sha256(raw)!=row['sha256'] or len(raw)!=row['bytes']:raise ValueError('Artifact hash/size mismatch: '+row['path'])
    for row in manifest['raw_responses']:
        for path_key,sha_key in [('path','sha256'),('meta_path','meta_sha256')]:
            path=(output/row[path_key]).resolve()
            if not path.is_relative_to(output):raise ValueError('Unsafe raw path')
            if sha256(path.read_bytes())!=row[sha_key]:raise ValueError('Raw API hash mismatch')
    if sha256(json_bytes(manifest['raw_responses']))!=manifest['raw_inventory_sha256']:raise ValueError('Raw inventory mismatch')
    sources=[json.loads(line) for line in (output/'sources.jsonl').read_text(encoding='utf-8').splitlines()]
    windows=[json.loads(line) for line in (output/'source-windows.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(sources)!=20 or len(windows)!=20 or len({r['source_group_id'] for r in sources})!=20:raise ValueError('Missing/replaced source denominator')
    if [r['source_group_id'] for r in sources]!=[r['source_group_id'] for r in windows]:raise ValueError('Window/source order mismatch')
    for row in sources:
        if row['split']!='development':raise ValueError('Closed-split source found')
        if row['status']=='exact_revision_verified':
            raw=(output/row['full_wikitext_file']).read_bytes()
            if hashlib.sha1(raw).hexdigest()!=row['revision_sha1']:raise ValueError('Full revision SHA-1 mismatch')
    for row in windows:
        if row['source_window'] is not None:
            if sha256(row['source_window'].encode())!=row['source_window_sha256']:raise ValueError('Window text hash mismatch')
            if word_count(row['source_window'])!=row['selected_window_words']:raise ValueError('Window word count mismatch')
    return {'ok':True,'source_count':20,'retrieval_statuses':manifest['retrieval_statuses'],
            'window_statuses':manifest['window_statuses'],'manifest_sha256':sha256((output/'manifest.json').read_bytes())}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-prompts',type=Path,default=ROOT/'results/grounded-development-v2/prompts.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'data'/VERSION)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args(argv)
    try:result=verify(args.output) if args.verify else build(args.source_prompts,args.output,offline=args.offline)
    except (OSError,ValueError,KeyError,UnicodeError) as error:parser.exit(2,f'Full-revision build refused: {error}\n')
    print(json.dumps({k:result[k] for k in ('ok','status','source_count','retrieval_statuses','window_statuses','manifest_sha256') if k in result},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
