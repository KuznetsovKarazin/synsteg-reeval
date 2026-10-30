#!/usr/bin/env python3
"""Independent lossless package, actual usage and frozen-control reconciliation."""
import argparse,collections,hashlib,json,pathlib

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def parse(raw):
 def unique(pairs):
  out={}
  for k,v in pairs:need(k not in out,'duplicate JSON key');out[k]=v
  return out
 def reject(value):raise ValueError('nonfinite JSON')
 return json.loads(raw,object_pairs_hook=unique,parse_constant=reject)
def load(path):return parse(path.read_bytes())
def equal(left,right,message):need(canonical(left)==canonical(right),message)
def indexed(rows,keys):
 out={tuple(r[k]for k in keys):r for r in rows};need(len(out)==len(rows),'duplicate row identity');return out

def package_check(manifest_path,result):
 root=manifest_path.parent;manifest=load(manifest_path);need(manifest['schema']=='public-dictionary-full-development-v3-lossless-package','wrong package schema');expected_files={manifest_path.name};verified=[]
 def read(entry):
  rel=pathlib.Path(entry['path']);need(not rel.is_absolute()and '..'not in rel.parts and str(rel)==entry['path'],'unsafe package path');need(entry['path']not in expected_files,'duplicate package member');expected_files.add(entry['path']);p=root/rel
  for ancestor in(p,*p.parents):need(not ancestor.is_symlink(),'symlink package member/ancestor')
  raw=p.read_bytes();need(len(raw)==entry['bytes']and sha(raw)==entry['sha256'],'package bytes/hash mismatch '+entry['path']);verified.append({'path':entry['path'],'bytes':len(raw),'sha256':sha(raw)});return raw
 reconstructed=parse(read(manifest['metadata']));need(set(manifest['tables'])=={'dictionary_rows','frame_conditions','lengths'},'table inventory differs')
 table_counts={}
 for table,entry in manifest['tables'].items():
  need(table not in reconstructed,'table repeated in metadata');rows=[]
  for part in entry['parts']:
   raw=read(part);lines=raw.splitlines();need(all(x.strip()for x in lines),'blank JSONL row');values=[parse(line)for line in lines];need(len(values)==part['rows'],'part row count differs');rows.extend(values)
  need(len(rows)==entry['rows']and sha(canonical(rows))==entry['canonical_sha256'],'table canonical hash/count differs');reconstructed[table]=rows;table_counts[table]=len(rows)
 actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()};need(actual==expected_files,'missing/orphan package file');equal(reconstructed,result,'reconstructed full object differs from author result');need(sha(canonical(reconstructed))==manifest['canonical_result_sha256'],'full canonical result hash differs')
 return {'status':'pass','package_manifest_sha256':sha(manifest_path.read_bytes()),'canonical_result_sha256':sha(canonical(reconstructed)),'ordered_table_counts':table_counts,'verified_payload_files':verified,'all_fields_and_ordered_rows_identical':True}

def usage_check(input_root,manifest,usage,result):
 need(usage['input_manifest_sha256']==result['input_manifest_sha256'],'usage manifest binding differs');need(usage['snapshot_commit']==manifest['snapshot_commit']==result['snapshot_commit'],'usage snapshot differs');need(usage['plan_sha256']==result['plan_sha256'],'usage plan differs')
 actual_rows=[];models={}
 for alias in ['tinyllama11','qwen05']:
  source=[r for r in manifest['cover_sources']if r['model_alias']==alias];need(len(source)==100,'usage source denominator');observed=[]
  for src in sorted(source,key=lambda x:x['source_order']):
   row={k:src[k]for k in ['model_alias','source_order','record_id']};row.update(cover_status=src['status'],output_tokens=None,stop_reason=None,hit_output_limit=None,attempt_sha256=None)
   if src['status']=='completed':
    p=input_root/src['attempt_path'];raw=p.read_bytes();need(sha(raw)==manifest['files_sha256'][src['attempt_path']],'usage raw attempt changed');a=parse(raw);value=a['result']['usage']['output_tokens'];stop=a['result']['stop_reason'];limit=a['result']['hit_output_limit'];need(type(value)is int and 0<=value<=1024,'invalid output tokens');need(type(limit)is bool and stop in ['end_turn','max_tokens'],'invalid termination metadata')
    row.update(output_tokens=value,stop_reason=stop,hit_output_limit=limit,attempt_sha256=sha(raw));observed.append(row)
   else:need(src['attempt_path']is None,'unavailable usage has source')
   actual_rows.append(row)
  total=sum(r['output_tokens']for r in observed);models[alias]={'full_source_denominator':100,'observed_completed_attempts':len(observed),'completed_output_tokens_sum':total,'stop_reason_counts':dict(collections.Counter(r['stop_reason']for r in observed)),'unavailable_usage_rows':100-len(observed),'full100_output_token_total':total if len(observed)==100 else None}
 equal(actual_rows,usage['records'],'usage per-record rows differ');equal(models,usage['models'],'usage aggregate differs')
 return {'status':'pass','records_checked':len(actual_rows),'models':models,'missing_usage_remains_null':True,'counts_computed_from_raw_attempts_not_expected_constants':True}

def prior_check(first80_path,recovery_path,result):
 first=load(first80_path);recovery=load(recovery_path);current=indexed(result['dictionary_rows'],['model_alias','source_order','dictionary_id']);old_rows=[]
 for did,item in first['cumulative1_80']['raw_capacity'].items():
  need(set(item['by_source_order'])=={str(n)for n in range(1,81)},'prior first80 denominator')
  for order,value in item['by_source_order'].items():
   equal(current['tinyllama11',int(order),did]['capacity_bits'],value,'first80 raw capacity changed');old_rows.append((int(order),did))
 deep=0
 for row in recovery['dictionary_rows']:
  if row['source_order']not in range(83,91):continue
  now=current['tinyllama11',row['source_order'],row['dictionary_id']]
  for key in ['record_id','dictionary_sha256','cover_sha256','capacity_bits','eligible_slots','slots','normalization_sha256']:
   equal(now[key],row[key],'recovery historical field changed '+key)
  old_rows.append((row['source_order'],row['dictionary_id']));deep+=1
 need(len(old_rows)==len(set(old_rows))==176 and deep==16,'historical dictionary comparison counts')
 nowframes=indexed(result['frame_conditions'],['model_alias','source_order','dictionary_id','framing','payload_kind','parent_primary_bits']);old_count=0;linked_count=0
 for row in recovery['frame_conditions']:
  if row['source_order']not in range(83,91):continue
  parent_values=[128,256,512]if row['payload_bits']==16 else[row['payload_bits']];kind='secondary16'if row['payload_bits']==16 else'primary';old_count+=1
  for parent in parent_values:
   now=nowframes['tinyllama11',row['source_order'],row['dictionary_id'],row['framing'],kind,parent]
   for key in ['payload_bits','required_bits','capacity_bits','capacity_feasible','delivery_status']:equal(now[key],row[key],'historical framing changed '+key)
   linked_count+=1
 need(old_count==128 and linked_count==192,'historical framing comparison counts')
 return {'status':'pass','prior_inputs':[{'path':str(p),'sha256':sha(p.read_bytes())}for p in [first80_path,recovery_path]],'raw_capacity_rows_identical':len(old_rows),'full_slot_and_cover_rows_identical':deep,'prior_unique_framing_conditions_checked':old_count,'new_parent_linked_framing_conditions_matched':linked_count,'reencoding_calls':0}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['input-root','manifest','result','package-manifest','usage','prior-first80','prior-recovery','output']:p.add_argument('--'+name,type=pathlib.Path,required=True)
 a=p.parse_args();need(not a.output.exists(),'output exists');manifest=load(a.manifest);result=load(a.result);need(sha(a.manifest.read_bytes())==result['input_manifest_sha256'],'result input binding differs')
 out={'schema':'public-dictionary-full-development-v3-independent-package-usage-prior-check','status':'pass','input_manifest_sha256':sha(a.manifest.read_bytes()),'author_result_sha256':sha(a.result.read_bytes()),'package':package_check(a.package_manifest,result),'usage':usage_check(a.input_root,manifest,load(a.usage),result),'prior_controls':prior_check(a.prior_first80,a.prior_recovery,result),'generation_usage_check_sha256':sha(a.usage.read_bytes()),'checker_sha256':sha(pathlib.Path(__file__).read_bytes()),'model_calls':0,'encoder_calls':0,'decoder_calls':0,'scope':'Exact package reconstruction, raw saved usage sums and frozen historical-row consistency; no transport or semantic claims.'}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
 print(json.dumps({'status':'pass','package_tables':out['package']['ordered_table_counts'],'usage_models':out['usage']['models'],'prior_capacity_rows':out['prior_controls']['raw_capacity_rows_identical']}))
if __name__=='__main__':main()
