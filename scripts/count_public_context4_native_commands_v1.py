#!/usr/bin/env python3
"""Count frozen executed code paths from audited traces; not native telemetry."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
ID='framed-interval-public-context4-v1'
DEPS=['src/synsteg_audit/canonical_channel_v2.py','src/synsteg_audit/framed_interval_llm_v1_qwen.py','src/synsteg_audit/framed_interval_llm_v1.py','src/synsteg_audit/framed_interval_public_v1.py','src/synsteg_audit/framed_interval_public_v1_native.py','src/synsteg_audit/framed_interval_public_v1_journal.py','src/synsteg_audit/seeded_pair_llm.py','scripts/run_public_context4_v1.py']
def load(p):return json.loads(p.read_bytes())
def main(a):
 root=Path(a.root);audit=load(Path(a.audit));assert audit['status']=='passed'
 closure=load(root/'experiments'/ID/'source-closure.json')['files_sha256'];source={}
 for name in DEPS:
  source[name]=hashlib.sha256((root/name).read_bytes()).hexdigest();assert source[name]==closure[name]
 rows=[]
 for e in audit['endpoints']:
  d=root/'results'/ID/'generation'/e['case_id']/e['role'];traces=load(d/'traces.json');assert len(traces)==e['tokens']
  reasons=Counter(v['reason'] for t in traces for v in t['admission']['verdicts']);reached=reasons['accepted']+reasons['retokenization_mismatch']
  assert sum(reasons.values())==e['pure_quantized_candidates']
  n=len(traces);wire_decode=1 if e['role']=='receiver' else 0
  rows.append({'case_id':e['case_id'],'role':e['role'],'PROMPT':1,'STEP':n,'DIST':n,'TOKENIZE_source_derived':2+3*n+reached+wire_decode,'TOKENIZE_gate_constructors':2,'TOKENIZE_prefix_validations':2*n,'TOKENIZE_selected_reinspection':n,'TOKENIZE_candidate_inspections':reached,'TOKENIZE_receiver_fullwire':wire_decode,'candidate_verdicts':dict(reasons),'prompt_input_tokens':len(load(d/'public-context.json')['prompt_ids'])})
 summary={k:sum(r[k] for r in rows) for k in ['PROMPT','STEP','DIST','TOKENIZE_source_derived','prompt_input_tokens']}
 out={'schema':ID+'-source-journal-command-counts-v1','mode':'source_and_complete_audited_journal_reconstruction; not separately instrumented native command telemetry','extra_model_calls':0,'extra_tokenizer_calls':0,'source_sha256':source,'preparation':{'native_process_loads':4,'TOKENIZE':4,'PROMPT':0,'STEP':0,'DIST':0,'basis':'four saved tokenization.json plus frozen sole TOKENIZE path'},'endpoints':summary,'all_stages':{'PROMPT':summary['PROMPT'],'STEP':summary['STEP'],'DIST':summary['DIST'],'TOKENIZE_source_derived':summary['TOKENIZE_source_derived']+4,'native_process_loads':16},'rows':rows,'derivation':['Two CanonicalGate constructors in PublicNativeService.prepare each validate empty prefix and issue one encode/TOKENIZE.','Each successful committed token: admit.validate one TOKENIZE; take.validate one TOKENIZE; selected token _inspect one TOKENIZE.','Per candidate _inspect reaches native TOKENIZE exactly for verdict accepted or retokenization_mismatch. invalid_utf8 fails Python decode before callback; zero_weight/special/rewrites/empty return earlier.','Each receiver does one additional fullwire codec.encode; no sender prompt tokenization because public prompt IDs frozen.','One Endpoint.begin/PROMPT and one observe/STEP plus distribution/DIST for every fully committed token; no uncommitted suffix or error exists in audited12 endpoints.'],'limitation':'Counts infer execution from pinned source and complete successful journals; native protocol command logs/counters were not independently instrumented. No tokenizer run used for counting.'}
 with Path(a.output).open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'preparation':out['preparation'],'endpoints':summary,'all_stages':out['all_stages']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);main(p.parse_args())
