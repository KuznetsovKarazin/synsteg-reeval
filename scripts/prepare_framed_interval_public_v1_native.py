#!/usr/bin/env python3
"""Derive payload-independent native public resources from saved evidence; no NN."""
import hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from synsteg_audit.framed_interval_public_v1 import make_public_spec,make_public_context,make_envelope,public_digest
from synsteg_audit.framed_interval_public_v1_native import validate_profile,resource_descriptor,verify_files

OUT=ROOT/"experiments/framed-interval-public-v1"
PREP=ROOT/"results/framed-interval-public-v1-preparation"
INPUTS={
 "experiments/framed-interval-llm-v1/activation-after-preflight-proc-v1.json":"1978302778134592a7835c63be363932cc8b9c353500873b9e7c7aaadc57911b",
 "experiments/seeded_pair_llm_v1/execution_manifest.json":"c877bfdd27a64abd126d6d22580277b3163159bc35a515679a5423e412f453e6",
 "results/framed-interval-llm-v1-tokenizer-preflight/native-proc-namespace-v2/receipt.json":"dc0fcc3320e28cf73526d25dc1f49221e33a0d9cc24e6bbb26bed14e6a83f114",
}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def put(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write("\n")
def main():
 values={}
 for p,d in INPUTS.items():
  raw=(ROOT/p).read_bytes()
  if sha(raw)!=d:raise ValueError("saved input differs: "+p)
  values[p]=json.loads(raw)
 old=values[next(iter(INPUTS))]
 execution=values["experiments/seeded_pair_llm_v1/execution_manifest.json"]
 receipt=values["results/framed-interval-llm-v1-tokenizer-preflight/native-proc-namespace-v2/receipt.json"]
 files={p.removeprefix("resources/seeded-pair-llm-v1/"):v for p,v in execution["channel_resources"]["files"].items()}
 profile={"schema":"framed-interval-public-v1-qwen-native-profile",
  "resource_files":files,"runtime":old["runtime"],"system_libraries_sha256":old["native_system_libraries_sha256"],
  "ready":old["ready"],"qwen_binding":old["qwen_binding_manifest"],"special_ids":receipt["inventory"]["special_ids"]}
 validate_profile(profile);check=verify_files(ROOT,profile)
 d=resource_descriptor(profile)
 declaration=SimpleNamespace(resource_id=d["resource_id"],resource_digest=d["resource_sha256"],
  provider_digest=d["provider_sha256"],context_tokens=d["context_tokens"],
  gate=SimpleNamespace(codec=SimpleNamespace(resource_digest=d["codec_sha256"])))
 spec=make_public_spec(declaration)
 context=make_public_context(tuple(receipt["prompts"]["station"]["token_ids"]),context_id="station-record-public-v1")
 key=hashlib.sha256(b"framed-interval-public-v1:fixed-pilot-key").digest()
 runs=[]
 for i,payload in enumerate((bytes(range(16)),bytes(range(16,32)))):
  for arm in ("stego","ordinary"):
   run_id=f"station-m{i:02d}-{arm}"
   nonce=hashlib.sha256(("framed-interval-public-v1:nonce:"+run_id).encode()).digest()[:12]
   runs.append({"run_id":run_id,"arm":arm,"key_hex":key.hex(),
    "payload_hex":payload.hex() if arm=="stego" else None,
    "envelope":make_envelope(spec,context,nonce,arm=arm)})
 fixtures={"schema":"framed-interval-public-v1-native-four-case-fixtures","planned_generations":4,
  "planned_fresh_receivers":2,"attempts_per_endpoint":1,"retry_permitted":False,
  "payloads_are_public_test_fixtures":True,"all_same_public_spec_and_context":True,
  "max_endpoint_seconds":1800,"runs":runs}
 paths={"profile":OUT/"native-public-profile.json","spec":OUT/"native-public-spec.json",
  "context":OUT/"native-public-context-station.json","sender_fixtures":OUT/"native-pilot-fixtures-v1.json"}
 for k,obj in (("profile",profile),("spec",spec),("context",context),("sender_fixtures",fixtures)):put(paths[k],obj)
 report={"schema":"framed-interval-public-v1-native-public-extraction","status":"prepared_no_native_calls",
  "input_sha256":INPUTS,"output_sha256":{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths.values()},
  "public_profile_is_independent_of":{"payload":True,"key":True,"nonce":True,"fixture_hash":True,"git_commit":True,
    "activation_hash":True,"freeze_or_readback_hash":True},"derivation_inputs_are_not_receiver_inputs":True,
  "same_static_spec_sha256":public_digest(spec),"same_public_context_sha256":public_digest(context),
  "public_native_descriptor":d,"resource_verification":check,"planned_generations":4,
  "planned_fresh_receivers":2,"distinct_payloads":2,"native_calls":0,"tokenizer_calls":0,"model_loads":0,
  "launch_ready":False,"launch_gate":"Independent source audit, complete Git freeze/readback and explicit root execution intent.",
  "created_at_unix":time.time()}
 put(PREP/"native-public-extraction.json",report)
 print(json.dumps({"status":report["status"],"public_profile_sha256":public_digest(profile),
  "static_spec_sha256":public_digest(spec),"context_sha256":public_digest(context),"native_calls":0},sort_keys=True))
if __name__=="__main__":main()
