"""Execute only supplied pure codec functions, without importing API clients."""
from pathlib import Path
import ast,sys,time,math,itertools,copy,json,hashlib,statistics,zlib,importlib.util
from _paths import parse_paths
SRC, OUT, SOURCE_ARCHIVE = parse_paths('Re-encode and decode all archived report series')
sys.path.insert(0,str(SRC))
from steganography import helper
ns={'helper':helper,'time':time,'math':math,'product':itertools.product,'SYNONYM_MAP':list}
def source_functions(path,names):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
source_functions(SRC/'steganography/secret_key.py',{'fix_token_container_size'})
source_functions(SRC/'steganography/core.py',{'embed','decode_message'})
spec=importlib.util.spec_from_file_location('codec2023',SRC/'experiments/ppl_opt/ds2023/codec2023.py')
oldcodec=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldcodec)
series={'luna':'reports/gpt6_luna_missing','qwen':'reports/qwen_missing_b3','gpt4o':'reports/gpt_omni_reports','gpt4o_opt32':'experiments/ppl_opt/series/opt_k32','gpt4o_opt8':'experiments/ppl_opt/series/opt_k8','gpt4o_opt4':'experiments/ppl_opt/series/opt_k4','ds2023':'reports/json_reports','ds2023_opt':'experiments/ppl_opt/series/ds2023_opt'}
rows=[];summary={}
for name,folder in series.items():
 for f in sorted((SRC/folder).glob('*.json')):
  d=json.loads(f.read_text());key=copy.deepcopy(d['secret_key'])
  if name.startswith('ds2023'):
   enc,used=oldcodec.encode(d['container'],key,d['message']);dec=oldcodec.decode(d['encoded_message'],d['secret_key'])
  else:
   enc,used=ns['embed'](d['container'],key,d['message']);dec,_=ns['decode_message'](d['encoded_message'],d['secret_key'],False)
  kb=json.dumps(d['secret_key'],ensure_ascii=False,separators=(',',':')).encode()
  rows.append(dict(series=name,uuid=d['uuid'],requested_bits=len(d['message']),used_bits=used,decoded_bits=len(dec),reencoded_exact=enc==d['encoded_message'],full_exact=dec==d['message'],prefix_exact=d['message'].startswith(dec),stored_decoded_match=dec==d.get('decoded_message'),bpw_source=len(d['message'])/len(d['secret_key']),bpw_stego=used/len(d['encoded_message'].split()),key_bytes=len(kb),key_zlib9_bytes=len(zlib.compress(kb,9)),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 rr=[r for r in rows if r['series']==name]
 summary[name]={'n':len(rr),'reencoded_exact':sum(r['reencoded_exact'] for r in rr),'full_exact':sum(r['full_exact'] for r in rr),'prefix_exact':sum(r['prefix_exact'] for r in rr),'stored_decoded_match':sum(r['stored_decoded_match'] for r in rr),'decoded_bits':sum(r['decoded_bits'] for r in rr),'requested_bits':sum(r['requested_bits'] for r in rr),'used_bits':sum(r['used_bits'] for r in rr)}
 if name in ('luna','qwen'):
  summary[name]['by_length']={str(L):{'n':sum(r['requested_bits']==L for r in rr),**{k:statistics.mean(r[k] for r in rr if r['requested_bits']==L) for k in ('bpw_source','bpw_stego')},'median_key_zlib9_bytes':statistics.median(r['key_zlib9_bytes'] for r in rr if r['requested_bits']==L)} for L in (128,256,512)}
(OUT/'rows.json').write_text(json.dumps(rows,indent=2));(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
