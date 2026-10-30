#!/usr/bin/env python3
"""Independent pure-component external contract/rejection mapping review."""
from __future__ import annotations
import copy, hashlib, hmac, importlib.util, json, pathlib, sys

REPO_ROOT=pathlib.Path(__file__).resolve().parents[1]
MODULE=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else REPO_ROOT/'src/synsteg_audit/author_candidate_schema_v3.py'
OUT=pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else REPO_ROOT/'results/author-candidate-schema-v3-external-review-v1/reproduced-review.json'
if OUT.exists() or OUT.is_symlink():
 raise FileExistsError('Refusing to overwrite existing audit output: '+str(OUT))
spec=importlib.util.spec_from_file_location('under_review',MODULE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def check(name, actual, expected):
 checks.append({'name':name,'pass':actual==expected,'actual':actual,'expected':expected})
def raising(fn):
 try:fn();return False
 except (ValueError,TypeError,UnicodeError):return True

def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def rehash(x):
 x['law_sha256']=hashlib.sha256(canon({k:v for k,v in x.items() if k!='law_sha256'})).hexdigest();return x

def oracle_draw(support,key,request,attempt,step):
 total=support['retained_mass'];cutoff=(2**256//total)*total
 for counter in range(128):
  msg=canon({'domain':'author-candidate-schema-v3/draw-v1','request_sha256':request,'attempt':attempt,'step':step,'law_sha256':support['law_sha256'],'counter':counter})
  block=int.from_bytes(hmac.digest(key,msg,'sha256'),'big')
  if block>=cutoff:continue
  ticket=block%total;rest=ticket
  for r in support['accepted']:
   if rest<r['mass']:return {'state':'selected','token_id':r['token_id'],'piece_hex':r['piece_hex'],'ticket':ticket,'counter':counter,'law_sha256':support['law_sha256']}
   rest-=r['mass']
 return {'state':'resource_refusal','reason':'HMAC draw block limit','law_sha256':support['law_sha256']}

w=('Popular',);key=bytes(range(32));request='a'*64
rows=[{'token_id':8,'piece':b'{','mass':2},{'token_id':3,'piece':b' {','mass':3},{'token_id':9,'piece':b'[','mass':7}]
s=m.constrain_support(b'',w,rows)
check('exact_mass_filter',[s['input_mass'],s['retained_mass'],[r['token_id'] for r in s['accepted']],[r['mass'] for r in s['accepted']]],[12,5,[3,8],[3,2]])
check('row_order_invariance',m.constrain_support(b'',w,list(reversed(rows))),s)
check('complete_has_no_draw',m.constrain_support(b'{"words":[{"Popular":[]}]}',w,rows)['state'],'complete')
check('empty_support',m.constrain_support(b'',w,[{'token_id':1,'piece':b'[','mass':1}])['state'],'empty_support')
check('empty_law_fraction_undefined_null',m.constrain_support(b'',w,[])['retained_fraction'],None)
check('one_oversized_piece_whole_step_refusal',m.constrain_support(b'',w,rows+[{'token_id':30,'piece':b' '*50,'mass':1}],max_bytes=20)['state'],'resource_refusal')
for field,values in {'token_id':[True,False,-1,2**31,1.0,'1',None],'mass':[True,False,0,-1,2**24+1,1.0,'1',None],'piece':[b'',bytearray(b'{'),'{',None]}.items():
 for i,value in enumerate(values):
  bad=copy.deepcopy(rows);bad[0][field]=value
  check(f'input_{field}_{i}',raising(lambda:m.constrain_support(b'',w,bad)),True)
check('duplicate_ids',raising(lambda:m.constrain_support(b'',w,rows+[rows[0].copy()])),True)
for name,args in [('max_bytes_true',{'max_bytes':True}),('max_support_true',{'max_support':True}),('max_bytes_zero',{'max_bytes':0}),('max_support_zero',{'max_support':0})]:
 check(name,raising(lambda:m.constrain_support(b'',w,rows,**args)),True)
for name,raw in [('root_partial_utf8',b'[\xe2'),('key_partial_utf8',b'{"wo\xe2'),('source_partial_utf8',b'{"words":[{"P\xc3'),('wrong_key_escape',b'{"\\u1'),('wrong_source_escape',b'{"words":[{"\\u1'),('candidate_control_escape',b'{"words":[{"Popular":["\\u000'),('candidate_low_surrogate_escape',b'{"words":[{"Popular":["\\uDC'),('low_surrogate_wrong_escape',b'{"words":[{"Popular":["\\uD800\\uE')]:
 check(name,m.classify_prefix(raw,w)['state'],'invalid_prefix')
for name,raw in [('candidate_partial_utf8',b'{"words":[{"Popular":["\xe2'),('viable_key_escape',b'{"\\u00'),('viable_surrogate_escape',b'{"words":[{"Popular":["\\uD8')]:
 check(name,m.classify_prefix(raw,w)['state'],'incomplete')
for attempt in range(3):
 for step in [0,1,17,511]:
  for raw in [b'',b' ']:
   law=m.constrain_support(raw,w,rows)
   check(f'draw_oracle_{attempt}_{step}_{len(raw)}',m.select_token(law,key,request,attempt,step),oracle_draw(law,key,request,attempt,step))
for field,values in [('attempt',[True,-1,3,1.0]),('step',[True,-1,512,1.0]),('key',[b'',b'x'*31,b'x'*33,'x'*32]),('request',['a'*63,'A'*64,'z'*64,None])]:
 for i,value in enumerate(values):
  args={'support':s,'key':key,'request_sha256':request,'attempt':0,'step':0};args[{'request':'request_sha256'}.get(field,field)]=value
  check(f'draw_context_{field}_{i}',raising(lambda:m.select_token(**args)),True)
for name,mutation in [
 ('schema',lambda x:x.update(schema='wrong')),
 ('prefix_sha',lambda x:x.update(prefix_sha256='not-a-digest')),
 ('words_sha',lambda x:x.update(words_sha256='not-a-digest')),
 ('base_law_sha',lambda x:x.update(base_law_sha256='not-a-digest')),
 ('input_mass_bool',lambda x:x.update(input_mass=True)),
 ('input_mass_below_retained',lambda x:x.update(input_mass=1)),
 ('fraction_numerator',lambda x:x['retained_fraction'].update(numerator=1)),
 ('fraction_denominator',lambda x:x['retained_fraction'].update(denominator=1)),
 ('rejected_duplicate_accepted',lambda x:x['rejected'].append({'token_id':3,'reason':'fake'})),
 ('unknown_field',lambda x:x.update(unknown=True))]:
 bad=copy.deepcopy(s);mutation(bad);rehash(bad)
 check('rehashed_metadata_'+name,raising(lambda:m.select_token(bad,key,request,0,0)),True)
 # An un-rehashed mutation must always fail digest binding.
 bad=copy.deepcopy(s);mutation(bad)
 check('tamper_digest_'+name,raising(lambda:m.select_token(bad,key,request,0,0)),True)


# Force the astronomically rare rejection branch without modifying production.
# These are synthetic HMAC outputs, not scientific samples or key search.
original_new=m.hmac.new
calls=[]
class FakeDigest:
 def __init__(self,value):self.value=value
 def digest(self):return self.value.to_bytes(32,'big')
def make_fake(values):
 iterator=iter(values)
 def fake(key,message,digestmod):
  calls.append(json.loads(message));return FakeDigest(next(iterator))
 return fake
try:
 m.hmac.new=make_fake([2**256-1,0])
 got=m.select_token(s,key,request,0,0)
 check('rejection_boundary_then_accept',[got['counter'],got['ticket'],got['token_id']],[1,0,3])
 check('rejection_counter_contexts',[x['counter'] for x in calls],[0,1])
 calls.clear();m.hmac.new=make_fake([2**256-1]*128)
 got=m.select_token(s,key,request,0,0)
 check('bounded_rejection_refusal',got['state'],'resource_refusal')
 check('bounded_rejection_count',len(calls),128)
 check('bounded_rejection_last_context',calls[-1]['counter'],127)
 calls.clear();m.hmac.new=make_fake([2**256-2])
 got=m.select_token(s,key,request,0,0)
 check('last_accepted_block_boundary',[got['counter'],got['ticket'],got['token_id']],[0,4,8])
finally:m.hmac.new=original_new

report={'schema':'candidate-schema-v3-external-contract-review-v1','production_sha256':hashlib.sha256(MODULE.read_bytes()).hexdigest(),'test_source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'model_calls':0,'paid_calls':0,'checks':len(checks),'passed':sum(c['pass'] for c in checks),'failed':sum(not c['pass'] for c in checks),'details':checks,'scope':'Pure parser/support/sampling contracts; no semantic quality, native integration, or authentic model/tokenizer evidence.'}
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('x',encoding='utf-8') as stream: stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2));print(json.dumps([c for c in checks if not c['pass']],indent=2))
sys.exit(bool(report['failed']))
