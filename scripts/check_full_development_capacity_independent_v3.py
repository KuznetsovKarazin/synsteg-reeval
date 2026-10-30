#!/usr/bin/env python3
"""Independent character-based capacity oracle; no codec/model/encoder imports."""
import argparse,collections,hashlib,json,pathlib,re,unicodedata
CORE='src/synsteg_audit/shared_dictionary.py'
SOURCE='results/recovery-development-v1/linux_snapshot/development-prompts.jsonl'
EXPECTED_PINS={CORE:'21dc3dcfb2b0cd6396424280fc2a47695481eb815b29ab2381cfdcf7f2d47d2f',SOURCE:'3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1','data/shared_dictionary_recovery_v1.json':'a6266ac3c120a081970e3b74024b8618f2d69c0c1ad98e812135f30d92343239','data/shared_dictionary_recovery_v2.json':'5dafba7b2179b40be560bacfd9ffa783ef98350d07e8d4ac7919c3a0865e6aa6'}
FRAMES={'plain_length16':16,'aes_siv':144};PARENTS=[128,256,512];MODELS=['tinyllama11','qwen05']
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return sha(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode('ascii'))
def load(path):return json.loads(path.read_bytes())
def same(left,right):
 if type(left)is not type(right):return False
 if isinstance(right,dict):return set(left)==set(right)and all(same(left[k],v)for k,v in right.items())
 if isinstance(right,list):return len(left)==len(right)and all(same(x,y)for x,y in zip(left,right))
 return left==right
def matches(record,expected):return all(k in record and same(record[k],v)for k,v in expected.items())
def dictionary(raw):
 normalized={'schema':raw['schema'],'version':raw['version'],'classes':[{'id':c['id'],'words':sorted(c['words'])}for c in sorted(raw['classes'],key=lambda c:c['id'])]}
 lookup={word:(c['id'],len(c['words']))for c in raw['classes']for word in c['words']}
 need(len(lookup)==sum(len(c['words'])for c in raw['classes']),'duplicate dictionary words')
 return lookup,canonical(normalized)
def analyze(text,lookup):
 """Derive an ASCII core by character locations, not the codec token regex."""
 slots=[];normalized=[];cursor=0
 for token_number,match in enumerate(re.finditer(r'\S+',text)):
  token=match.group();indices=[i for i,c in enumerate(token)if ('A'<=c<='Z')or('a'<=c<='z')]
  if not indices:continue
  first,last=indices[0],indices[-1]
  if indices!=list(range(first,last+1)):continue
  affixes=token[:first]+token[last+1:]
  if any(c=='_' or unicodedata.category(c)[0]!='P'for c in affixes):continue
  word=token[first:last+1];entry=lookup.get(word.lower())
  if entry is None:continue
  if word==word.lower():style='lower'
  elif word==word.upper():style='upper'
  elif word==word[0].upper()+word[1:].lower():style='title'
  else:continue
  class_id,size=entry;width=0
  while 2**(width+1)<=size and width<2:width+=1
  if width==0:continue
  start,end=match.start()+first,match.start()+last+1
  normalized.extend([['literal',text[cursor:start]],['class',class_id,style]])
  slots.append({'index':len(slots),'token_index':token_number,'start':start,'end':end,'surface':word,'class_id':class_id,'case_style':style,'width':width});cursor=end
 normalized.append(['literal',text[cursor:]])
 return {'capacity_bits':sum(s['width']for s in slots),'eligible_slots':len(slots),'slots':slots,'normalization_sha256':canonical(normalized)}
def keyed(rows,fields,label):
 result={tuple(row[k]for k in fields):row for row in rows};need(len(result)==len(rows),'duplicate '+label);return result

def check(root,manifest,result):
 root=pathlib.Path(root);pins=manifest['files_sha256']
 for p,h in EXPECTED_PINS.items():need(pins.get(p)==h,'frozen pin differs '+p)
 actual={str(p.relative_to(root)):sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}
 need(actual==pins,'input inventory differs')
 sources=[json.loads(line)for line in(root/SOURCE).read_bytes().splitlines()if line.strip()];need(len(sources)==100,'source denominator')
 snapshot=load(root/manifest['snapshot_path']);need(snapshot['snapshot_ref']==manifest['snapshot_commit']==result['snapshot_commit'],'snapshot commit differs')
 source_index=keyed(manifest['cover_sources'],['model_alias','source_order'],'source index')
 expected_cells={(a,n)for a in MODELS for n in range(1,101)};need(set(source_index)==expected_cells,'source grid differs')
 drows=keyed(result['dictionary_rows'],['model_alias','source_order','dictionary_id'],'dictionary rows')
 need(set(drows)=={(a,n,d)for a,n in expected_cells for d in ['v1','v2']},'dictionary denominator differs')
 lengths=keyed(result['lengths'],['model_alias','source_order'],'length rows');need(set(lengths)==expected_cells,'length denominator differs')
 conditions=keyed(result['frame_conditions'],['model_alias','source_order','dictionary_id','framing','payload_kind','parent_primary_bits'],'framing cells')
 need(len(conditions)==4800,'frame denominator differs')
 dictionaries={d:dictionary(load(root/f'data/shared_dictionary_recovery_{d}.json'))for d in ['v1','v2']}
 need(set(result['models'])==set(MODELS),'unexpected pooled/model summaries')
 oracle={};seen_conditions=set();report={}
 for alias in MODELS:
  for order in range(1,101):
   item=source_index[alias,order];src=sources[order-1];path=item['attempt_path'];cover=None
   if item['status']=='completed':
    need(path is not None,'missing completed attempt');attempt=load(root/path);cover=attempt['result']['cover_text'];need(attempt['record']['record_id']==src['record_id'],'source ID differs')
   else:need(path is None,'noncompleted has text')
   length=lengths[alias,order]
   expected_length={'record_id':src['record_id'],'cover_status':item['status'],'source_words':len(src['excerpt'].split()),'target_words':src['target_words'],'cover_words':None if cover is None else len(cover.split()),'requested_expansion':src['target_words']/len(src['excerpt'].split()),'actual_expansion':None if cover is None else len(cover.split())/len(src['excerpt'].split()),'cover_sha256':None if cover is None else sha(cover.encode()),'attempt_path':path}
   need(matches(length,expected_length),'length/source mismatch '+str((alias,order)))
   for did,(lookup,dictsha)in dictionaries.items():
    derived={'capacity_bits':None,'eligible_slots':None,'slots':None,'normalization_sha256':None}if cover is None else analyze(cover,lookup)
    status='cover_unavailable'if cover is None else'measured';oracle[alias,order,did]=derived['capacity_bits'];row=drows[alias,order,did]
    expected={**derived,'record_id':src['record_id'],'dictionary_sha256':dictsha,'cover_status':item['status'],'cover_sha256':None if cover is None else sha(cover.encode()),'capacity_status':status,'analysis_error':None,'source_fidelity':'not_evaluated','semantic_substitutability':'not_evaluated'}
    need(matches(row,expected),'capacity/slot mismatch '+str((alias,order,did)))
    for frame,overhead in FRAMES.items():
     for kind in ['primary','secondary16']:
      for parent in PARENTS:
       key=(alias,order,did,frame,kind,parent);seen_conditions.add(key);need(key in conditions,'missing frame cell');cell=conditions[key];payload=parent if kind=='primary'else 16;capacity=derived['capacity_bits'];feasible=None if capacity is None else capacity>=payload+overhead
       expected={'payload_bits':payload,'frame_overhead_bits':overhead,'required_bits':payload+overhead,'capacity_bits':capacity,'capacity_feasible':feasible,'delivery_status':'not_evaluated','semantic_usability':'not_evaluated','feasibility_status':status if feasible is None else 'mechanically_feasible'if feasible else'mechanically_insufficient_capacity'}
       need(matches(cell,expected),'frame mismatch '+str(key))
  model=result['models'][alias];observed=sum(source_index[alias,n]['status']=='completed'for n in range(1,101));need(model['cover_denominator']==100 and model['available_saved_covers']==observed and model['unavailable_covers']==100-observed,'model cover counts')
  need(model['dictionary_row_denominator']==200 and model['frame_condition_denominator']==2400,'per model denominators')
  need(model['cover_status_counts']==dict(collections.Counter(source_index[alias,n]['status']for n in range(1,101))),'model status counts')
  totals={}
  for did in dictionaries:
   values=[oracle[alias,n,did]for n in range(1,101)if oracle[alias,n,did]is not None];totals[did]=sum(values)
   expected={'cover_denominator':100,'available_saved_covers':observed,'measured_covers':len(values),'unavailable_covers':100-observed,'analysis_errors':0,'observed_raw_bits':sum(values),'zero_capacity_covers':sum(x==0 for x in values),'min_raw_bits':min(values)if values else None,'max_raw_bits':max(values)if values else None,'mean_raw_bits_among_measured':sum(values)/len(values)if values else None}
   need(same(model['dictionary_aggregates'][did],expected),'aggregate differs '+alias+'/'+did)
  need(model['paired_capacity_covers']==observed,'paired denominator differs');need(model['v2_raw_bit_loss_fraction']==(None if totals['v1']==0 else 1-totals['v2']/totals['v1']),'capacity loss differs')
  summaries=keyed(model['feasibility'],['dictionary_id','framing','payload_kind','parent_primary_bits'],'model feasibility summaries');need(len(summaries)==24,'model feasibility summary denominator')
  for (did,frame,kind,parent),row in summaries.items():
   cells=[conditions[alias,n,did,frame,kind,parent]for n in range(1,101)];yes=sum(x['capacity_feasible']is True for x in cells);no=sum(x['capacity_feasible']is False for x in cells)
   expected={'full_denominator':100,'feasible_covers':yes,'insufficient_capacity_covers':no,'null_covers':100-yes-no,'feasible_fraction_all100':yes/100,'feasible_fraction_measured':yes/(yes+no)if yes+no else None};need(matches(row,expected),'feasibility aggregate differs')
  report[alias]={'cover_denominator':100,'available_saved_covers':observed,'unavailable_covers':100-observed,'observed_raw_bits':totals}
 need(set(conditions)==seen_conditions,'extra frame condition')
 need(result['full300_counts_context_only']==snapshot['counts'],'full300 context changed')
 need(all(result[k]==0 for k in ['model_calls','tokenizer_calls','encoder_calls','decoder_calls','paid_calls'])and result['closed_test_opened']is False,'forbidden execution claim')
 return {'schema':'public-dictionary-full-development-v3-independent-check','status':'pass','method':'Independent character-location slot oracle; no import of evaluated audit or frozen codec.','models':report,'dictionary_rows_checked':400,'frame_conditions_checked':4800,'model_calls':0,'encoder_calls':0,'decoder_calls':0,'limitations':['Mechanical capacity only; no delivery, factual-support, contextual substitutability or security conclusion.','Input readback assertions remain external; exact provided input bytes are checked.']}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',type=pathlib.Path,required=True);p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--result',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();need(not a.output.exists()and not a.output.resolve().is_relative_to(a.input_root.resolve()),'output must be new and outside inputs');out=check(a.input_root,load(a.manifest),load(a.result));out.update(input_manifest_sha256=sha(a.manifest.read_bytes()),result_sha256=sha(a.result.read_bytes()),checker_sha256=sha(pathlib.Path(__file__).read_bytes()));a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
