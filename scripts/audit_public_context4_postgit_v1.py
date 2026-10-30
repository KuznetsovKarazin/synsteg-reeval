#!/usr/bin/env python3
"""Independent read-only artifact/interval/AEAD verifier. Never imports project code.
Uses only saved Git artifacts, stdlib and cryptography; no model/tokenizer calls.
"""
import argparse,ast,hashlib,hmac,io,json,math,struct,sys,zipfile
from fractions import Fraction
from pathlib import Path
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
ID='framed-interval-public-context4-v1'; DOMAIN=b'framed-interval-public-v1\0'; ORDERS=[1,4,7,10]
EXP='experiments/'+ID; OUT='results/'+ID; PUB='experiments/framed-interval-public-v1'; PAYLOAD=bytes(range(16))
PRIMARY='2a3be59b6a808464837156753d97f7278880cbcd'; HELPER_SHA='b97aa4ac71f6a2c6971ae72460d3b65275eff4db1539b34c05e39fb5c32367af'
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def pairs(rows):
 d={}
 for k,v in rows:
  if k in d:raise ValueError('duplicate JSON key '+k)
  d[k]=v
 return d
def decode(raw):return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def load(p):return decode(Path(p).read_bytes())
def check(ok,message):
 if not ok:raise ValueError(message)
def bits(raw):return [int(x) for b in raw for x in f'{b:08b}']
def exact_bytes(seq):return bytes(int(''.join(map(str,seq[i:i+8])),2) for i in range(0,len(seq),8))
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
def keys(key,env):
 enc_tail=HKDF(algorithm=hashes.SHA256(),length=64,salt=bytes.fromhex(env['nonce_hex']),info=DOMAIN+b'transport-subkeys\0'+hashlib.sha256(canonical(env)).digest()).derive(key)
 ctx=hashlib.sha256(canonical({k:v for k,v in env.items() if k not in ('arm','nonce_hex','ordinary_seed_hex')})).digest()
 order=HKDF(algorithm=hashes.SHA256(),length=32,salt=ctx,info=DOMAIN+b'matched-order-subkey\0').derive(key)
 return enc_tail[:32],order,enc_tail[32:],ctx
def aad(env):
 raw=canonical(env);return DOMAIN+b'aead\0'+len(raw).to_bytes(4,'big')+raw
def prf(seed,env,label,count):
 ctx=hashlib.sha256(canonical(env)).digest()
 raw=b''.join(hmac.digest(seed,DOMAIN+label+b'\0'+ctx+i.to_bytes(8,'big'),'sha256') for i in range((count+255)//256))
 return bits(raw)[:count]
def initial_stream(key,env):
 enc,order,tail,ctx=keys(key,env)
 frame=ChaCha20Poly1305(enc).encrypt(bytes.fromhex(env['nonce_hex']),b'\x00\x10'+PAYLOAD,aad(env)) if env['arm']=='stego' else None
 stream=(bits(frame)+prf(tail,env,b'stego-tail',65536-272)) if frame else prf(bytes.fromhex(env['ordinary_seed_hex']),env,b'ordinary-input',65536)
 return stream,frame,(enc,order,tail,ctx)
def quantized(rows,temperature,total):
 parsed=sorted([(r['token_id'],r['logit_hex'],struct.unpack('>f',bytes.fromhex(r['logit_hex']))[0]) for r in rows],key=lambda x:(-x[2],x[0]))
 check(len({x[0] for x in parsed})==len(parsed) and all(math.isfinite(x[2]) for x in parsed),'logit identity')
 mass=[math.exp((x[2]-parsed[0][2])/temperature) for x in parsed];den=math.fsum(mass)
 scaled=[x/den*total for x in mass];weights=[math.floor(x) for x in scaled]
 rest=total-sum(weights);check(0<=rest<=len(weights),'quantizer residual')
 for i in sorted(range(len(weights)),key=lambda i:(-(scaled[i]-weights[i]),parsed[i][0]))[:rest]:weights[i]+=1
 return [{'token_id':p[0],'logit_hex':p[1],'weight':w} for p,w in zip(parsed,weights)]
def journal(path):
 ids=[];wire=b'';traces=[];previous='0'*64;pending=None;suffixes=[];lines=path.read_bytes().splitlines(keepends=True)
 for n,line in enumerate(lines):
  check(line.endswith(b'\n'),'torn journal record');r=decode(line);digest=r.pop('event_sha256')
  check(r['sequence']==n and r['previous_sha256']==previous and sha(canonical(r))==digest,'journal hash-chain');previous=digest
  if r['kind']=='prepared':
   check(pending is None and r['token_index']==len(ids) and r['wire_prefix_sha256']==sha(wire),'prepared binding')
   suffix=bytes.fromhex(r['suffix_hex']);t=r['trace'];check(suffix and t['prefix_ids']==ids and t['observed_token']==r['token_id'] and t['step']==len(ids) and t['prefix_sha256']==sha(wire),'prepared trace binding')
   pending=(digest,r['token_id'],suffix,t)
  elif r['kind']=='committed':
   check(pending is not None and r['prepared_sha256']==pending[0] and r['token_id']==pending[1] and r['token_index']==len(ids),'commit pair')
   wire+=pending[2];check(r['wire_after_sha256']==sha(wire),'wire hash');ids.append(pending[1]);suffixes.append(pending[2]);traces.append(pending[3]);pending=None
  else:raise ValueError('journal terminal unknown event '+str(r['kind']))
 check(pending is None,'uncommitted native step')
 return {'ids':ids,'wire':wire,'traces':traces,'last':previous,'events':len(lines),'suffixes':suffixes}
def replay(j,key,env,profile,spec):
 stream,frame,ks=initial_stream(key,env);_,order,_,orderctx=ks
 low=0;high=(1<<32)-1;pending=0;shift=0;code=int(''.join(map(str,stream[:32])),2);consumed=32;released=[];first=None;prefix=b''
 full=1<<32;half=full//2;quarter=full//4;total_candidates=removed=0
 for n,t in enumerate(j['traces']):
  check(t['interval_before']=={'low':low,'high':high,'pending':pending,'symbols':n,'shift_count':shift},'interval before')
  ev=t['base_law_evidence'];rawrows=ev['float32_logits_and_integer_weights'];params=profile['qwen_binding']['parameters']
  check(ev['parameters']==params and ev['base_weight_total']==16777216,'base-law parameters')
  check(quantized(rawrows,params['temperature'],params['probability_integer_total'])==rawrows,'saved float32 quantization differs')
  ad=t['admission'];weights=ad['conditioned_weights'];verdict=ad['verdicts'];by={r['token_id']:r['weight'] for r in rawrows}
  check(len(by)==len(rawrows) and sorted((v['token_id'],v['weight']) for v in verdict)==sorted(by.items()),'admission base-law coverage')
  accepted=sorted([[v['token_id'],v['weight']] for v in verdict if v['reason']=='accepted' and v['weight']>0])
  check(weights==accepted and ad['retained_weight_total']==sum(w for _,w in weights),'admission retained law')
  check(ad['removed_weight_total']==sum(by.values())-sum(w for _,w in weights),'admission removed mass')
  check(ad['codec_digest']==spec['resources']['codec_sha256'] and ad['prefix_ids']==j['ids'][:n] and ad['prefix_bytes_sha256']==sha(prefix),'admission prefix/resource')
  pref=canonical(j['ids'][:n])
  def score(row):
   tok=canonical(row[0]);return hmac.digest(order,DOMAIN+b'order\0'+orderctx+len(pref).to_bytes(4,'big')+pref+len(tok).to_bytes(4,'big')+tok,'sha256'),row[0]
  law=sorted(weights,key=score);check(t['ordered_integer_law']==law,'HMAC ordering')
  check(all(tok not in profile['special_ids'] for tok,_ in law),'special token in admitted law')
  width=high-low+1;total=sum(w for _,w in law);cursor=0;bins=[];selected=None
  for tok,w in law:
   left=low+width*cursor//total;cursor+=w;right=low+width*cursor//total-1
   bins.append({'token_id':tok,'low':left,'high':right,'width':right-left+1})
   if tok==j['ids'][n]:selected=(left,right,w)
  check(bins==t['integer_bins'] and selected is not None,'integer bins')
  lo,hi,w=selected;check(lo<=code<=hi,'sender inverse-arithmetic choice')
  check((t['selected_integer_width'],t['parent_integer_width'],t['selected_weight'],t['weight_total'])==(hi-lo+1,width,w,total),'selected interval widths')
  tv=Fraction(sum(abs(b['width']*total-wt*width) for b,(_,wt) in zip(bins,law)),2*width*total)
  check(t['bin_law_vs_weights_TV']=={'numerator':tv.numerator,'denominator':tv.denominator},'integer TV')
  emitted=[];steps=0
  while True:
   if hi<half:emitted.extend([0]+[1]*pending);pending=0;offset=0
   elif lo>=half:emitted.extend([1]+[0]*pending);pending=0;offset=half
   elif lo>=quarter and hi<3*quarter:pending+=1;offset=quarter
   else:break
   lo=2*(lo-offset);hi=2*(hi-offset)+1;code=2*(code-offset)+stream[consumed];consumed+=1;steps+=1
  low,high=lo,hi;shift+=steps
  check(t['emitted_bits']==emitted and t['interval_after']=={'low':low,'high':high,'pending':pending,'symbols':n+1,'shift_count':shift},'interval transition/emitted bits')
  released+=emitted;check(released==stream[:len(released)],'resolved stream prefix')
  if len(released)>=272 and first is None:first=n+1
  check(n+1<=768 and pending<=4096 and consumed<=65536,'fixed limits')
  prefix+=j['suffixes'][n];total_candidates+=len(rawrows);removed+=ad['removed_weight_total']
 check(first==len(j['ids']),'first 272-bit threshold/fullwire stop')
 return {'tokens':len(j['ids']),'resolved_bits':len(released),'first_threshold':first,'consumed_bits':consumed,'pending_bits':pending,'frame':exact_bytes(released[:272]),'expected_frame':frame,'total_candidates':total_candidates,'removed_weight_total':removed}
def clean_supervisor(p,budget):
 x=load(p);check(x['status']=='completed_child' and x['wall_seconds']==budget and x['elapsed_seconds']<=budget and x['child_exit_code']==0 and x['observed_exit_before_reap']==0 and not x['deadline_expired'] and x['session_verified_before_worker'] and x['group_shutdown_before_reap'] and not x['owned_group_present_after_cleanup'] and x['shutdown_error'] is None and x['retry_allowed'] is False,'supervisor not clean')
 return x

def main(a):
 root=Path(a.root).resolve();download=Path(a.download).resolve();support=Path(a.support).resolve();dest=Path(a.output).resolve();dest.mkdir(parents=True,exist_ok=False)
 report={'schema':ID+'-independent-postgit-v1','primary_commit':PRIMARY,'model_calls':0,'tokenizer_calls':0,'git_writes':0,'endpoint_denominator':12,'generation_denominator':8,'delivery_denominator':4,'endpoints':[],'costs':[],'deliveries':[],'blockers':[]}
 try:
  receipt=load(a.restoration);check(receipt['actual_git_commit']==PRIMARY,'restoration commit');art=download/'artifacts'/ID/'raw-v1'
  for r in load(support/'primary-index-identities.json'):
   p=(download/r['path']) if r['path'].startswith('artifacts/') else (root/r['path'])
   check(not r['error'] and blob(p.read_bytes())==r['git_blob_sha'],'independent Git blob mismatch '+r['path'])
  archives=[]
  for label in ('results','sender-nonces'):
   index=load(art/label/'index.json');parts=[]
   for row in index['parts']:
    data=(art/label/row['name']).read_bytes();check(len(data)==row['bytes'] and sha(data)==row['sha256'],'archive part identity');parts.append(data)
   archive=b''.join(parts);check(len(archive)==index['archive']['bytes'] and sha(archive)==index['archive']['sha256'],'archive identity')
   with zipfile.ZipFile(io.BytesIO(archive)) as z:
    check(set(z.namelist())==set(index['files']) and len(z.namelist())==len(index['files']),'archive membership')
    for name,row in index['files'].items():
     raw=z.read(name);check(len(raw)==row['bytes'] and sha(raw)==row['sha256'] and (root/name).read_bytes()==raw,'restored file identity '+name)
   actual={str(p.relative_to(root)) for p in (root/index['source_base']).rglob('*') if p.is_file()};check(actual==set(index['files']),'restored extra/missing files')
   directories={str(p.relative_to(root)) for p in (root/index['source_base']).rglob('*') if p.is_dir()}|{index['source_base']};check(directories==set(index['directories']),'restored directories')
   archives.append({'label':label,'files':len(index['files']),'bytes':sum(x['bytes'] for x in index['files'].values()),'archive_sha256':sha(archive),'archive_bytes':len(archive),'directories':len(directories)})
  report['archive_inventory']=archives
  closure=load(root/EXP/'source-closure.json');check(len(closure['files_sha256'])==30,'closure count')
  for name,digest in closure['files_sha256'].items():check(sha((root/name).read_bytes())==digest,'source closure '+name)
  check(sha((root/'scripts/run_public_context4_v1.py').read_bytes())==HELPER_SHA,'frozen helper')
  for stage in ('tokenize','execute'):
   ac=load(support/(stage+'_activation.json'));pr=load(support/(stage+'_proof.json'))
   check(blob(ac['content'].encode())==ac['git_blob_sha'] and blob(pr['content'].encode())==pr['git_blob_sha'],'Git readback content')
   act=decode(ac['content']);proof=decode(pr['content']);check(act['enabled'] and act['stage']==stage and act['automatic_retries']==0,'activation')
   check(act['closure_sha256']==sha((root/EXP/'source-closure.json').read_bytes()) and act['proposal_sha256']==sha((root/EXP/'protocol.proposed.json').read_bytes()),'activation hashes')
   check(proof['commit']==ac['ref'] and proof['stage']==stage and proof['all_files_exact'],'gate proof binding')
   check((root/OUT/(stage+'-source-readback.json')).read_bytes()==pr['content'].encode(),'Git gate proof local bytes')
   for name,digest in proof['files_sha256'].items():
    raw=ac['content'].encode() if name==ac['path'] else (root/name).read_bytes();check(sha(raw)==digest,'gate hash '+name)
  report['provenance']={'closure_files':30,'helper_sha256':HELPER_SHA,'independent_Git_readbacks':7,'gate_commit_tokenize':load(support/'tokenize_proof.json')['ref'],'gate_commit_execute':load(support/'execute_proof.json')['ref'],'source_freeze_ancestor_of_primary':{'base':'4c09bdc67fece34d7e08a03cd8960c738663cb00','status':'ahead','ahead_by':11,'scope':'independent Git compare observed; root source-readback supplied for all30 files'}}
  profile=load(root/PUB/'native-public-profile.json');spec=load(root/PUB/'native-public-spec.json');protocol=load(root/EXP/'protocol.proposed.json');texts=load(root/EXP/'selected-context-texts.json')['rows'];contexts=load(root/OUT/'tokenization/contexts.json')['contexts'];fixtures=load(root/EXP/'fixtures.json')['cases'];g=root/OUT/'generation';provisional=load(g/'result.json')
  check([r['source_order'] for r in texts]==ORDERS and len(contexts)==4 and len(fixtures)==8,'fixed contexts/fixtures')
  tokenization=[]
  for i,(t,c) in enumerate(zip(texts,contexts)):
   d=root/OUT/'tokenization'/f'c{i:02d}';tk=load(d/'tokenization.json');sup=clean_supervisor(d/'supervisor-result.json',60)
   check(c==load(d/'public-context.json') and c['context_id']==ID+f'-o{ORDERS[i]:03d}' and tk['prompt_ids']==c['prompt_ids'] and tk['context_sha256']==sha(canonical(c)),'tokenized context binding')
   check(tk['status']=='completed' and (tk['tokenizer_calls'],tk['prompt_calls'],tk['step_calls'])==(1,0,0),'tokenization counters')
   check(sha(t['chatml_text'].encode())==t['chatml_sha256'],'selected chatml bytes')
   tokenization.append({'order':ORDERS[i],'prompt_tokens':len(c['prompt_ids']),'elapsed_seconds':sup['elapsed_seconds'],'context_bytes':len(canonical(c))})
  report['tokenization']=tokenization
  inv=load(g/'receiver-install-inventory.json');expected_inv={}
  module=ast.parse((root/'scripts/run_public_context4_v1.py').read_text())
  # The frozen receiver source list uses only string constants and ENTRY.
  receiver_names=[]
  for node in module.body:
   if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RECEIVER_SOURCES' for x in node.targets):receiver_names=[x.value if isinstance(x,ast.Constant) else PUB+'/run_native.py' for x in node.value.elts]
  for name in receiver_names+[PUB+'/native-public-profile.json',PUB+'/native-public-spec.json']:expected_inv[name]=sha((root/name).read_bytes())
  for i,c in enumerate(contexts):expected_inv[f'public-contexts/c{i:02d}.json']=sha(canonical(c))
  for name,row in profile['resource_files'].items():expected_inv['resources/seeded-pair-llm-v1/'+name]=row['sha256']
  check(inv['files_sha256']==expected_inv and inv['fixture_present'] is False and inv['expected_plaintext_present'] is False and inv['source_proof_present'] is False,'receiver public-only inventory')
  expected_cases=[f'o{o:03d}-{arm}' for o in ORDERS for arm in ('stego','ordinary')];check([x['case_id'] for x in fixtures]==expected_cases,'case order')
  initial=load(g/'initial-denominator.json');check(initial['senders']==8 and initial['fresh_receivers']==4 and initial['payload_delivery_denominator']==4,'initial denominator')
  reservations=list(g.glob('o*/**/reservation.json'));check(len(reservations)==12,'endpoint reservations count');identities=[];nonce_tags=[];paired={}
  for f in fixtures:
   cid=f['case_id'];ctx=contexts[f['context_order']];base=g/cid;sender_wire=None
   for role in ('sender','receiver') if f['arm']=='stego' else ('sender',):
    row={'case_id':cid,'role':role,'audit_status':'pending'}
    try:
     d=base/role;inputs=base/(role+'-inputs');process=base/(role+'-process');w=load(d/'worker-result.json');core=w['core'];env=load(inputs/'envelope.json');key=(inputs/'key.bin').read_bytes()
     check(w['status']=='returned' and w['execution_kind']=='native' and w['model_process_attempted'] is True and w['retry_permitted'] is False and not any('error' in k for k in w),'worker terminal')
     check(w['role']==('generate' if role=='sender' else 'receive'),'worker role');row['terminal_status']=core['status'];sup=clean_supervisor(process/'supervisor-result.json',300)
     check(load(process/'reservation.json')=={'status':'reserved_outcome_unknown','role':w['role'],'automatic_retries':0},'endpoint reservation')
     argv=load(process/'command.json')['argv'];flags=dict(zip(argv[2::2],argv[3::2]));check(len(flags)*2+2==len(argv) and flags['--role']==w['role'] and flags['--timeout']=='300','endpoint argv')
     original_root=protocol['execution_root'];deployment=original_root if role=='sender' else original_root+'/runtime/'+ID+'/receiver-install'
     check(flags['--root']==deployment and argv[1]==deployment+'/'+PUB+'/run_native.py' and flags['--profile']==deployment+'/'+PUB+'/native-public-profile.json' and flags['--spec']==deployment+'/'+PUB+'/native-public-spec.json','endpoint public deployment')
     for opt,name in [('--context','public-context.json'),('--envelope','envelope.json'),('--key-file','key.bin')]:check(flags[opt]==original_root+'/'+OUT+'/generation/'+cid+'/'+role+'-inputs/'+name,'role input path')
     expected_input={'key.bin','envelope.json','public-context.json'}|({'wire.bin'} if role=='receiver' else ({'payload.bin'} if f['arm']=='stego' else set()))
     check({p.name for p in inputs.iterdir()}==expected_input,'role input allowlist')
     check(key.hex()==f['key_hex'] and len(key)==32 and env['nonce_hex']==f['nonce_hex'] and env['arm']==f['arm'] and env['ordinary_seed_hex']==f['ordinary_seed_hex'],'fixtures key/nonce/arm')
     check(env['context_sha256']==sha(canonical(ctx)) and env['spec_sha256']==sha(canonical(spec)) and load(inputs/'public-context.json')==ctx,'wire envelope/context binding')
     for filename,obj in [('public-profile.json',profile),('public-spec.json',spec),('public-context.json',ctx),('envelope.json',env)]:check(load(d/filename)==obj,'worker public input copy')
     pre=load(d/'native-preprompt-verification.json');res=load(d/'native-resource-verification.json');capture=load(d/'proc-identity-capture.json')
     check(pre['status']=='passed' and pre['ready']==profile['ready'] and pre['prompt_ids']==ctx['prompt_ids'] and pre['context_token_count']==len(ctx['prompt_ids']),'preprompt context/READY')
     check(pre['prompt_calls_so_far']==pre['step_calls_so_far']==pre['distribution_calls_so_far']==0 and pre['model_process_loaded'],'preprompt counters')
     check(res['verified_files']==41 and res['public_profile_sha256']==sha(canonical(profile)) and pre['public_profile_sha256']==sha(canonical(profile)),'saved resource check')
     resident=pre['resident'];check(resident['status']=='passed' and resident['identity']['pidfd_live_before_and_after_maps'],'resident evidence')
     ident=resident['identity'];check(ident['before']==ident['after'] and capture['child']==ident['before']['identity'] and capture['executable']==ident['before']['executable'],'stable native process identity')
     identities.append((capture['child']['pid_namespace'],capture['child']['proc_pid'],capture['child']['starttime_ticks']))
     expected_map={deployment+'/resources/seeded-pair-llm-v1/'+name:row['sha256'] for name,row in profile['resource_files'].items()}
     expected_map.update(profile['system_libraries_sha256'])
     for name,digest in resident['mapped_files_sha256'].items():check(expected_map.get(name)==digest,'resident resource identity')
     check(any(x.endswith('/qwen2.5-1.5b-instruct-q4_k_m.gguf') for x in resident['mapped_files_sha256']),'resident model absent')
     j=journal(d/'token-journal.jsonl');check(j['wire']==(d/'committed-wire.bin').read_bytes()==(d/'wire.raw').read_bytes(),'complete committed wire')
     check(j['traces']==load(d/'traces.json') and sha(canonical(j['traces']))==core['trace_sha256'] and core['committed_trace_matches_returned_prefix'] is True and core['noncommitted_trace_suffix']==[],'trace evidence')
     rec=load(d/'reconciliation.json');check(rec==load(d/'recovered-status.json') and rec['wire_available'] and rec['known_committed_tokens']==len(j['ids']) and rec['wire_sha256']==sha(j['wire']) and rec['committed_trace_sha256']==sha(canonical(j['traces'])) and rec['last_event_sha256']==j['last'] and not rec['unknown_native_step'] and not rec['torn_last_record'],'reconciliation evidence')
     pure=replay(j,key,env,profile,spec);check(core['first_threshold_token']==pure['first_threshold'] and core['resolved_bits']==pure['resolved_bits'],'threshold summary')
     if role=='sender':
      check(core['emitted_tokens']==pure['tokens'] and core['wire_bytes']==len(j['wire']) and core['wire_sha256']==sha(j['wire']) and core['consumed_bits']==pure['consumed_bits'] and core['pending_bits']==pure['pending_bits'],'sender summary')
      check(flags['--nonce-ledger']==original_root+'/runtime/'+ID+'/sender-nonces','sender ledger path')
      tag=hmac.digest(key,DOMAIN+b'nonce-ledger-reservation\0'+bytes.fromhex(env['nonce_hex']),'sha256').hex();nonce_tags.append(tag)
      ledger=load(root/'runtime'/ID/'sender-nonces'/(tag+'.json'));check(ledger==load(d/'nonce-reservation.json') and ledger['reservation_id']==tag and ledger['nonce_hex']==f['nonce_hex'] and ledger['state']=='burned_before_generation' and ledger['key_saved'] is False,'nonce ledger receipt')
      if f['arm']=='stego':check((inputs/'payload.bin').read_bytes()==PAYLOAD and core['frame_sha256']==sha(pure['frame']) and pure['frame']==pure['expected_frame'],'sender payload/frame')
      else:check('--payload-file' not in flags and core['frame_sha256'] is None and core['status']=='ordinary_threshold_generated','ordinary control status')
      sender_wire=j['wire'];paired[cid]=j
      costs={'case_id':cid,'arm':f['arm'],'wire_bytes':len(j['wire']),'tokens':pure['tokens'],'resolved_bits':pure['resolved_bits'],'context_tokens':len(ctx['prompt_ids']),'envelope_file_bytes':(inputs/'envelope.json').stat().st_size,'context_file_bytes':(inputs/'public-context.json').stat().st_size,'compact_envelope_bytes':len(canonical(env)),'compact_context_bytes':len(canonical(ctx)),'payload_bytes':16 if f['arm']=='stego' else 0,'sender_seconds':w['wall_seconds']}
      costs['message_bytes_with_shared_context_spec']=costs['wire_bytes']+costs['compact_envelope_bytes'];costs['first_context_message_bytes']=costs['message_bytes_with_shared_context_spec']+costs['compact_context_bytes']+len(canonical(spec))
      costs.update(payload_bits_per_token=128/pure['tokens'] if f['arm']=='stego' else None,payload_bits_per_wire_byte=128/len(j['wire']) if f['arm']=='stego' else None,payload_bits_per_transport_byte=128/costs['message_bytes_with_shared_context_spec'] if f['arm']=='stego' else None,wire_frame_bits_per_token=272/pure['tokens'])
      report['costs'].append(costs)
     else:
      check('--payload-file' not in flags and '--nonce-ledger' not in flags and flags['--wire-file'].endswith('/receiver-inputs/wire.bin'),'receiver command allowlist')
      check((inputs/'wire.bin').read_bytes()==sender_wire==j['wire'] and paired[cid]['ids']==j['ids'] and paired[cid]['traces']==j['traces'],'independent receiver fullwire and laws/traces')
      plain=ChaCha20Poly1305(keys(key,env)[0]).decrypt(bytes.fromhex(env['nonce_hex']),pure['frame'],aad(env));check(plain==b'\x00\x10'+PAYLOAD and (d/'received-payload.bin').read_bytes()==PAYLOAD and core['payload_hex']==PAYLOAD.hex(),'exact AEAD payload')
      check(core['status']=='authenticated_delivery' and core['authenticated_delivery'] is True and core['processed_tokens']==core['wire_tokens']==pure['tokens'],'receiver fullwire delivery')
      report['deliveries'].append({'case_id':cid,'authenticated_exact':True,'payload_bytes':16,'payload_hex':PAYLOAD.hex(),'tokens':pure['tokens'],'resolved_bits':pure['resolved_bits'],'frame_sha256':sha(pure['frame']),'wire_sha256':sha(j['wire']),'receiver_seconds':w['wall_seconds']})
     row.update(audit_status='passed',tokens=pure['tokens'],journal_events=j['events'],resolved_bits=pure['resolved_bits'],first_threshold_token=pure['first_threshold'],wire_bytes=len(j['wire']),pure_quantized_candidates=pure['total_candidates'],worker_seconds=w['wall_seconds'],supervisor_seconds=sup['elapsed_seconds'],clean_native_exit=True)
    except Exception as e:
     row.update(audit_status='failed',error_type=type(e).__name__,error=str(e));report['blockers'].append({'case_id':cid,'role':role,'error':str(e)})
    report['endpoints'].append(row)
  check(len(report['endpoints'])==12,'full endpoint audit rows');check(len(set(identities))==12,'distinct native process identities');check(len(set(nonce_tags))==8,'unique key/nonce reservations')
  ledger_files={p.name for p in (root/'runtime'/ID/'sender-nonces').iterdir()};check(ledger_files=={x+'.json' for x in nonce_tags},'ledger completeness')
  outer=load(root/OUT/'root-execute-process/process.json');check(outer['exit_code']==0 and outer['automatic_retries']==0,'outer outcome')
  report['outer_process']=outer;report['nonce_reservations']=len(nonce_tags);report['distinct_native_processes']=len(set(identities));report['receiver_install_files']=len(expected_inv)
  report['shared_setup']={'key_bytes':32,'payload_bytes':16,'frame_bytes':34,'frame_overhead_bytes':18,'nonce_bytes_already_in_envelope':12,'spec_file_bytes':(root/PUB/'native-public-spec.json').stat().st_size,'profile_file_bytes':(root/PUB/'native-public-profile.json').stat().st_size,'canonical_spec_bytes':len(canonical(spec)),'canonical_profile_bytes':len(canonical(profile)),'named_resource_files':len(profile['resource_files']),'named_resource_bytes':sum(x['bytes'] for x in profile['resource_files'].values()),'model_bytes':profile['resource_files']['qwen2.5-1.5b-instruct-q4_k_m.gguf']['bytes'],'named_sum_is_unique_inode_bytes':False}
  report['coordinator_summary_consistent']=(provisional['planned_endpoints']==12 and provisional['endpoints_reserved']==12 and provisional['payload_delivery_denominator']==4 and provisional['exact_deliveries']==len(report['deliveries']) and provisional['fatal'] is None)
  check(report['coordinator_summary_consistent'],'coordinator projection differs')
  stego=[x for x in report['costs'] if x['arm']=='stego'];report['aggregate_delivery']={'exact_deliveries':len(report['deliveries']),'denominator':4,'payload_bytes':sum(x['payload_bytes'] for x in report['deliveries']),'stego_tokens':sum(x['tokens'] for x in stego),'stego_wire_bytes':sum(x['wire_bytes'] for x in stego),'stego_wire_plus_envelope_bytes':sum(x['message_bytes_with_shared_context_spec'] for x in stego),'stego_first_context_bytes':sum(x['first_context_message_bytes'] for x in stego)}
 except Exception as e:report['blockers'].append({'stage':'campaign','error_type':type(e).__name__,'error':str(e)})
 report['status']='passed' if not report['blockers'] and len(report['deliveries'])==4 and len(report['endpoints'])==12 else 'requires_review'
 report['limitations']=['same host and runtime; no cross-host reproduction','one public fixture key and one16-byte payload, four development contexts','ordinary is same transform with independent public bits, not free LM sampling','public envelope discloses arm','model weights/native/tokenizer not loaded or independently rehashed by auditor; saved identity evidence only','canonical token eligibility not re-tokenized; recorded sender/receiver agreement and gate evidence checked','saved float32 logits re-quantized; model logits not recomputed','fidelity/naturalness/ending quality assigned to separate auditor','byte ratios exclude separately disclosed shared model/runtime/profile/key setup']
 put(dest/'audit.json',report);print(json.dumps({'status':report['status'],'endpoints':len(report['endpoints']),'exact':len(report['deliveries']),'blockers':report['blockers'],'output':str(dest)}));return 0 if report['status']=='passed' else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--download',required=True);p.add_argument('--restoration',required=True);p.add_argument('--support',required=True);p.add_argument('--output',required=True);sys.exit(main(p.parse_args()))
