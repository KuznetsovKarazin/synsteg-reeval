#!/usr/bin/env python3
"""Read saved public-v1 pilot evidence and count every fixed case; no NN."""
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from synsteg_audit.framed_interval_public_v1 import canonical_json,metadata_costs
from synsteg_audit.framed_interval_llm_v1_journal import recover_journal
OUT=ROOT/"results/framed-interval-public-v1-native-metrics"
CAMPAIGN=ROOT/"results/framed-interval-public-v1/native-pilot-v1"
def sha(raw):return hashlib.sha256(raw).hexdigest()
def put(path,obj):
 with path.open("x") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write("\n")
def main():
 if OUT.exists():raise FileExistsError(OUT)
 status=json.loads((CAMPAIGN/"campaign-status.json").read_bytes())
 if status["state"]=="running":raise ValueError("campaign notterminal")
 ids=[f"station-m{i:02d}-{arm}" for i in range(2) for arm in ("stego","ordinary")]
 if [r["run_id"] for r in status["cases"]]!=ids:raise ValueError("fourcase denominator differs")
 e=ROOT/"experiments/framed-interval-public-v1"
 spec=json.loads((e/"native-public-spec.json").read_bytes());ctx=json.loads((e/"native-public-context-station.json").read_bytes())
 profile=json.loads((e/"native-public-profile.json").read_bytes())
 rows=[];inputs={};known_sender_steps=known_receiver_steps=0;unknown_slots=[]
 for case in status["cases"]:
  name=case["run_id"];arm=case["arm"];slot=case["sender"];endpoint=CAMPAIGN/name/"sender"
  core=(slot.get("worker") or {}).get("core") or {}
  wirepath=endpoint/"committed-wire.bin";wire=wirepath.read_bytes() if wirepath.exists() else None
  envpath=endpoint/"envelope.json";env=json.loads(envpath.read_bytes()) if envpath.exists() else None
  journals={}
  for role in ("sender","receiver"):
   p=CAMPAIGN/name/role/"token-journal.jsonl"
   if not p.exists():continue
   j=recover_journal(p);n=len(j["committed_token_ids"])
   if role=="sender":known_sender_steps+=n
   else:known_receiver_steps+=n
   journals[role]={"committed_steps":n,"unknown_native_step":j["unknown_native_step"],"torn_last_record":j["torn_last_record"]}
   if j["unknown_native_step"]:unknown_slots.append(name+"/"+role)
  text=None if wire is None else wire.decode("utf-8",errors="strict")
  receiver=case["receiver"]
  rows.append({"run_id":name,"arm":arm,"sender_worker_status":slot["worker_status"],"sender_core_status":core.get("status"),
   "sender_process_exit_code":slot.get("process",{}).get("exit_code"),"sender_core_emitted_tokens":core.get("emitted_tokens"),
   "resolved_bits":core.get("resolved_bits"),"first_threshold_token":core.get("first_threshold_token"),
   "wire_bytes":None if wire is None else len(wire),"wire_sha256":None if wire is None else sha(wire),
   "words_regex":None if text is None else len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b",text)),
   "transport":None if env is None or wire is None else metadata_costs(env,spec,ctx,wire_bytes=len(wire),payload_bytes=16 if arm=="stego" else 0),
   "pretty_envelope_bytes":envpath.stat().st_size if envpath.exists() else None,
   "receiver_worker_status":None if receiver is None else receiver.get("worker_status"),
   "receiver_candidate_exact_authenticated":None if receiver is None else receiver.get("authenticated_delivery",False),
   "journals":journals})
 for p in sorted(CAMPAIGN.rglob("*")):
  if p.is_file():inputs[p.relative_to(ROOT).as_posix()]=sha(p.read_bytes())
 for p in (e/"native-public-spec.json",e/"native-public-context-station.json",e/"native-public-profile.json"):
  inputs[p.relative_to(ROOT).as_posix()]=sha(p.read_bytes())
 stego=[r for r in rows if r["arm"]=="stego"];ordinary=[r for r in rows if r["arm"]=="ordinary"]
 known_all=all(r["sender_core_emitted_tokens"] is not None for r in rows) and not unknown_slots
 deliveries=status["authenticated_payload_deliveries"]
 wire_total=sum(r["wire_bytes"] or 0 for r in stego)
 envelope_total=sum((r["transport"] or {}).get("compact_envelope_bytes",0) for r in stego)
 setup={"compact_public_spec_bytes":len(canonical_json(spec)),"compact_public_context_bytes":len(canonical_json(ctx)),
  "compact_public_native_profile_bytes":len(canonical_json(profile)),"shared_key_once_bytes":32,
  "shared_declared_resource_files":len(profile["resource_files"]),
  "shared_declared_resource_file_bytes_logical_sum":sum(v["bytes"] for v in profile["resource_files"].values()),
  "model_file_bytes":profile["resource_files"]["qwen2.5-1.5b-instruct-q4_k_m.gguf"]["bytes"],
  "physical_disk_or_network_transfer_bytes_measured":False,"includes_runtime_archive_and_repeated_library_names":True,
  "source_code_and_system_python_distribution_bytes":None}
 summary={"schema":"framed-interval-public-v1-native-metrics","planned_generations":4,"planned_fresh_receivers":2,
  "distinct_payloads":2,"payload_bytes_each":16,"same_public_spec_context_and_key":True,
  "candidate_exact_authenticated_deliveries":deliveries,"candidate_delivery_denominator":2,
  "independent_audit_status":"notassertedbythiscollector","known_committed_sender_steps":known_sender_steps,
  "known_committed_receiver_steps":known_receiver_steps,"known_committed_steps":known_sender_steps+known_receiver_steps,
  "unknown_slots":unknown_slots,"all_sender_token_counts_known_and_no_unknown_journal":known_all,
  "stego_tokens_all_attempts":sum(r["sender_core_emitted_tokens"] or 0 for r in stego) if known_all else None,
  "ordinary_tokens_all_attempts":sum(r["sender_core_emitted_tokens"] or 0 for r in ordinary) if known_all else None,
  "stego_wire_bytes_known":wire_total,"compact_stego_envelopes_bytes_known":envelope_total,
  "stego_wire_plus_compact_envelopes_bytes_known":wire_total+envelope_total,"shared_setup":setup,
  "campaign_wall_seconds":status["wall_seconds"],"native_calls_in_this_collection":0,
  "lower_level_graph_forward_calls":None,"word_counter_definition":"Unicode regex word groups with internal apostrophes/hyphens"}
 if known_all and summary["stego_tokens_all_attempts"]:
  summary["candidate_delivered_useful_bits_per_stego_token_all_attempts"]=deliveries*16*8/summary["stego_tokens_all_attempts"]
 else:summary["candidate_delivered_useful_bits_per_stego_token_all_attempts"]=None
 OUT.mkdir(parents=True);put(OUT/"summary.json",summary);put(OUT/"rows.json",rows);put(OUT/"input-sha256.json",inputs)
 print(json.dumps(summary,sort_keys=True))
if __name__=="__main__":main()
