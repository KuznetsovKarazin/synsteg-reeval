#!/usr/bin/env python3
"""Four preregistered sender calls/two isolated receiver calls; no automatic retry.

Only this experiment coordinator reads fixtures and source-freeze evidence.
The standalone receiver is installed with public source/resources only.
"""
import argparse,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from synsteg_audit.framed_interval_public_v1 import VERSION,canonical_json,public_digest,THRESHOLD_BITS,TOKEN_CAP,validate_envelope
from synsteg_audit.framed_interval_public_v1_native import RESOURCE_ROOT,validate_profile,verify_files
from synsteg_audit.framed_interval_public_v1_orchestration import mkdir_durable,write_once,reconcile
from synsteg_audit.framed_interval_llm_v1_journal import atomic_json,recover_journal,sha

EXP="experiments/framed-interval-public-v1"
SOURCE_FILES=(
 "src/synsteg_audit/__init__.py","src/synsteg_audit/canonical_channel_v2.py",
 "src/synsteg_audit/framed_interval_llm_v1.py","src/synsteg_audit/framed_interval_llm_v1_journal.py",
 "src/synsteg_audit/framed_interval_llm_v1_qwen.py","src/synsteg_audit/framed_interval_public_v1.py",
 "src/synsteg_audit/framed_interval_public_v1_journal.py","src/synsteg_audit/framed_interval_public_v1_native.py",
 "src/synsteg_audit/framed_interval_public_v1_orchestration.py","src/synsteg_audit/integer_interval_channel_v1.py",
 "src/synsteg_audit/proc_namespace_identity_v1.py","src/synsteg_audit/proc_namespace_identity_v2.py",
 "src/synsteg_audit/seeded_pair_llm.py",EXP+"/run_native.py")
PUBLIC_FILES=(EXP+"/native-public-profile.json",EXP+"/native-public-spec.json",EXP+"/native-public-context-station.json")
REQUIRED_FREEZE=(*SOURCE_FILES,*PUBLIC_FILES,"src/synsteg_audit/framed_interval_public_v1_nonce.py",
 EXP+"/native-pilot-fixtures-v1.json",EXP+"/native-execution-intent-v1.json",
 "docs/framed_interval_public_v1_native_execution_ru.md","scripts/run_framed_interval_public_v1_native_pilot.py")
OUTPUT="results/framed-interval-public-v1/native-pilot-v1"
RECEIVER_INSTALL="runtime/framed-interval-public-v1/receiver-install-v1"


def load(path):return json.loads(Path(path).read_bytes())


def verify_proof(root,path):
 proof=load(path)
 if (proof.get("schema")!=VERSION+"-native-source-readback" or proof.get("all_files_exact") is not True
   or re.fullmatch("[a-f0-9]{40}",proof.get("commit","")) is None
   or type(proof.get("files_sha256")) is not dict or not set(REQUIRED_FREEZE).issubset(proof["files_sha256"])):
  raise ValueError("complete actual coordinator source readback required")
 for name,digest in proof["files_sha256"].items():
  if sha((root/name).read_bytes())!=digest:raise ValueError("source changed after Git readback: "+name)
 return proof


def install_receiver(root,dest,profile):
 """Copies only source/public files and hardlinks known publicresources; nofixture."""
 mkdir_durable(dest,exclusive=True)
 inventory={}
 for name in (*SOURCE_FILES,*PUBLIC_FILES):
  data=(root/name).read_bytes();target=dest/name
  mkdir_durable(target.parent);write_once(target,data)
  inventory[name]={"bytes":len(data),"sha256":sha(data),"kind":"public_source_or_spec"}
 for name,item in profile["resource_files"].items():
  relative=RESOURCE_ROOT+"/"+name;target=dest/relative;mkdir_durable(target.parent)
  os.link(root/relative,target,follow_symlinks=False)
  inventory[relative]={**item,"kind":"shared_public_resource_hardlink"}
 actual={p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
 if actual!=set(inventory):raise ValueError("isolated receiver install inventory differs")
 return {"schema":VERSION+"-receiver-install","files":inventory,"file_count":len(inventory),
  "sender_fixture_present":False,"expected_plaintext_present":False,"activation_present":False,
  "freeze_or_readback_present":False,"nonce_ledger_present":False,"independent_install_materialized":True,
  "shared_public_resource_bytes":sum(x["bytes"] for x in profile["resource_files"].values()),
  "scope":"same host; separate filesystem install with publicresources hardlinked; notcrossmachine"}


def spawn(command,cwd,log_dir,timeout):
 mkdir_durable(log_dir);started=time.monotonic();p=None
 env=dict(os.environ);env["PYTHONPATH"]=str(cwd/"src");env["PYTHONDONTWRITEBYTECODE"]="1"
 try:
  with (log_dir/"stdout.log").open("xb") as stdout,(log_dir/"stderr.log").open("xb") as stderr:
   p=subprocess.Popen(command,cwd=cwd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
   try:
    code=p.wait(timeout=timeout)
    result={"status":"exited","exit_code":code,"pid":p.pid}
   except subprocess.TimeoutExpired:
    result={"status":"timeout","pid":p.pid}
   finally:
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    p.wait(timeout=3)
  result["exit_code"]=p.returncode
 except BaseException as exc:
  result={"status":"spawn_or_wait_error","error_type":type(exc).__name__,"error":str(exc)[:1024]}
  if p is not None:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   p.wait(timeout=3);result["exit_code"]=p.returncode
 result["wall_seconds"]=time.monotonic()-started
 atomic_json(log_dir/"process.json",result)
 return result


def collect(directory,process,role,wire=None):
 outcome={"process":process,"worker_status":"missing_terminal_record","authenticated_delivery":False}
 p=directory/"worker-result.json"
 if p.exists():
  worker=load(p);outcome["worker"]=worker;outcome["worker_status"]=worker.get("status")
  if worker.get("schema")!=VERSION+"-native-worker-result" or worker.get("role")!=role or worker.get("execution_kind")!="native":
   raise ValueError("worker identity differs")
  core=worker.get("core") or {}
  success=(role=="receive" and core.get("status")=="authenticated_delivery" and core.get("authenticated_delivery") is True)
  clean=(worker.get("status")=="returned" and process.get("status")=="exited" and process.get("exit_code")==0
   and worker.get("model_process_attempted") is True and not any("error" in k for k in worker))
  outcome["authenticated_delivery"]=success and clean
 try:outcome["reconciliation"]=reconcile(directory)
 except Exception as exc:
  outcome["artifact_error"]=str(exc)[:512];outcome["authenticated_delivery"]=False
 if outcome["authenticated_delivery"]:
  core=outcome["worker"]["core"];rec=outcome["reconciliation"]
  saved=recover_journal(directory/"token-journal.jsonl")
  traces=saved["committed_traces"];resolved=0;crossing=None
  for i,row in enumerate(traces,1):
   bits=row["emitted_bits"]
   if type(bits) is not list or any(type(b) is not int or b not in (0,1) for b in bits):raise ValueError("invalidbits")
   resolved+=len(bits)
   if crossing is None and resolved>=THRESHOLD_BITS:crossing=i
  count=core.get("processed_tokens")
  valid=(type(count) is int and 1<=count<=TOKEN_CAP and core.get("wire_tokens")==count
   and crossing==count==core.get("first_threshold_token") and resolved==core.get("resolved_bits")
   and rec.get("wire_available") and rec.get("known_committed_tokens")==count
   and not rec.get("unknown_native_step") and not rec.get("torn_last_record")
   and core.get("committed_trace_matches_returned_prefix") is True and core.get("noncommitted_trace_suffix")==[]
   and rec.get("committed_trace_sha256")==core.get("trace_sha256") and wire is not None
   and rec.get("wire_sha256")==sha(wire) and saved["committed_wire"]==wire)
  if not valid:
   outcome["authenticated_delivery"]=False;outcome["delivery_evidence_error"]="fullwire/firstcrossing/journal mismatch"
 return outcome



def validate_fixtures(fixtures,spec,context):
 fields={"schema","planned_generations","planned_fresh_receivers","attempts_per_endpoint","retry_permitted",
  "payloads_are_public_test_fixtures","all_same_public_spec_and_context","max_endpoint_seconds","runs"}
 if (type(fixtures) is not dict or set(fixtures)!=fields
  or fixtures["schema"]!=VERSION+"-native-four-case-fixtures"
  or type(fixtures["planned_generations"]) is not int or fixtures["planned_generations"]!=4
  or type(fixtures["planned_fresh_receivers"]) is not int or fixtures["planned_fresh_receivers"]!=2
  or type(fixtures["attempts_per_endpoint"]) is not int or fixtures["attempts_per_endpoint"]!=1
  or fixtures["retry_permitted"] is not False or fixtures["payloads_are_public_test_fixtures"] is not True
  or fixtures["all_same_public_spec_and_context"] is not True
  or type(fixtures["max_endpoint_seconds"]) is not int or fixtures["max_endpoint_seconds"]!=1800
  or type(fixtures["runs"]) is not list or len(fixtures["runs"])!=4):
  raise ValueError("strict frozen4sender/2receiver fixture contract required")
 ids=[f"station-m{i:02d}-{arm}" for i in range(2) for arm in ("stego","ordinary")]
 keys,nonces,payloads=set(),set(),set()
 for row,ident in zip(fixtures["runs"],ids):
  if type(row) is not dict or set(row)!={"run_id","arm","key_hex","payload_hex","envelope"}:
   raise ValueError("strict fixture row required")
  arm=ident.rsplit("-",1)[-1]
  if row["run_id"]!=ident or row["arm"]!=arm:raise ValueError("fixedcase/arm differs")
  key=row["key_hex"]
  if type(key) is not str or re.fullmatch("[a-f0-9]{64}",key) is None:raise ValueError("exact32byte fixturekey required")
  keys.add(key)
  env=row["envelope"];validate_envelope(env)
  if env["arm"]!=arm or env["spec_sha256"]!=public_digest(spec) or env["context_sha256"]!=public_digest(context):
   raise ValueError("samepublicspec/context andmatchingarm required")
  pair=(key,env["nonce_hex"])
  if pair in nonces:raise ValueError("fixture nonce reuse forbidden")
  nonces.add(pair)
  payload=row["payload_hex"]
  if arm=="ordinary":
   if payload is not None:raise ValueError("ordinaryfixturepayload mustbeNone")
  else:
   if type(payload) is not str or re.fullmatch("[a-f0-9]{32}",payload) is None:raise ValueError("exact16byte fixturepayload required")
   payloads.add(payload)
 if len(keys)!=1 or len(payloads)!=2 or len(nonces)!=4:raise ValueError("onekey/twodistinctpayloads/fournonces required")
 return fixtures


def run(root,proof_path):
 started=time.monotonic();root=Path(root).resolve();proof=verify_proof(root,proof_path)
 intent=load(root/EXP/"native-execution-intent-v1.json")
 if (intent.get("schema")!=VERSION+"-native-execution-intent" or intent.get("planned_generations")!=4
  or intent.get("planned_fresh_receivers")!=2 or intent.get("outcomes_known") is not False
  or intent.get("attempts_per_endpoint")!=1 or intent.get("automatic_retries")!=0
  or intent.get("source_readback_required") is not True or intent.get("paid_calls_allowed") is not False
  or intent.get("closed_test_allowed") is not False or intent.get("output")!=OUTPUT):
  raise ValueError("fixed explicitnative executionintent required")
 profile,spec,context=[load(root/p) for p in PUBLIC_FILES]
 validate_profile(profile);verify_files(root,profile)
 fixtures=validate_fixtures(load(root/EXP/"native-pilot-fixtures-v1.json"),spec,context)
 if (intent["static_public_spec_sha256"]!=public_digest(spec) or intent["public_context_sha256"]!=public_digest(context)
  or intent["public_profile_sha256"]!=public_digest(profile)
  or intent["fixture_sha256"]!=sha((root/EXP/"native-pilot-fixtures-v1.json").read_bytes())):
  raise ValueError("intentpublicresources/fixtures differ")
 output=root/OUTPUT;mkdir_durable(output,exclusive=True)
 install=root/RECEIVER_INSTALL
 inventory=install_receiver(root,install,profile)
 atomic_json(output/"receiver-install-inventory.json",inventory)
 atomic_json(output/"campaign-source-proof.json",proof)
 manifest={"schema":VERSION+"-four-case-campaign","planned_generations":4,"planned_fresh_receivers":2,
  "fixture_file_sha256":sha((root/EXP/"native-pilot-fixtures-v1.json").read_bytes()),
  "public_spec_sha256":public_digest(spec),"public_context_sha256":public_digest(context),
  "public_profile_sha256":public_digest(profile),"retry_permitted":False,
  "receiver_install":RECEIVER_INSTALL,"same_spec_and_context_for_both_payloads":True}
 atomic_json(output/"campaign-manifest.json",manifest)
 ledger={**manifest,"schema":VERSION+"-native-campaign-status","state":"running",
  "cases":[{"run_id":x["run_id"],"arm":x["arm"],"sender":{"worker_status":"not_started"},
   "receiver":{"worker_status":"not_started","authenticated_delivery":False} if x["arm"]=="stego" else None} for x in fixtures["runs"]]}
 atomic_json(output/"campaign-status.json",ledger)
 try:
  nonce_ledger=root/"runtime/framed-interval-public-v1/sender-nonces"
  for run,case in zip(fixtures["runs"],ledger["cases"]):
   case_root=output/run["run_id"];mkdir_durable(case_root)
   for role in ("generate","receive"):
    if role=="receive" and run["arm"]!="stego":continue
    slot="sender" if role=="generate" else "receiver"
    directory=case_root/slot;inputs=case_root/(slot+"-inputs");mkdir_durable(inputs)
    deployment=root if role=="generate" else install
    write_once(inputs/"key.bin",bytes.fromhex(run["key_hex"]))
    atomic_json(inputs/"envelope.json",run["envelope"])
    command=[sys.executable,str(deployment/EXP/"run_native.py"),"--root",str(deployment),
     "--profile",str(deployment/PUBLIC_FILES[0]),"--spec",str(deployment/PUBLIC_FILES[1]),
     "--context",str(deployment/PUBLIC_FILES[2]),"--envelope",str(inputs/"envelope.json"),
     "--key-file",str(inputs/"key.bin"),"--output",str(directory),"--role",role,"--timeout","1800"]
    wire=None
    if role=="generate":
     command.extend(["--nonce-ledger",str(nonce_ledger)])
     if run["payload_hex"] is not None:
      write_once(inputs/"payload.bin",bytes.fromhex(run["payload_hex"]));command.extend(["--payload-file",str(inputs/"payload.bin")])
    else:
     prior=case["sender"];rec=prior.get("reconciliation",{})
     sender=case_root/"sender"
     if not rec.get("wire_available") or rec.get("unknown_native_step") or prior.get("artifact_error"):
      case[slot]={"worker_status":"dependency_blocked","reason":"senderwire unavailable or unknown",
       "authenticated_delivery":False,"retry_permitted":False}
      atomic_json(output/"campaign-status.json",ledger,replace=True);continue
     wire=(sender/"committed-wire.bin").read_bytes()
     write_once(inputs/"wire.bin",wire);command.extend(["--wire-file",str(inputs/"wire.bin")])
    case[slot]={"worker_status":"running","authenticated_delivery":False}
    atomic_json(output/"campaign-status.json",ledger,replace=True)
    process=spawn(command,deployment,case_root/(slot+"-process"),1830)
    case[slot]=collect(directory,process,role,wire)
    if role=="receive" and case[slot]["authenticated_delivery"]:
     exact=case[slot]["worker"]["core"].get("payload_hex")==run["payload_hex"]
     case[slot]["exact_expected_payload"]=exact
     case[slot]["authenticated_delivery"]=exact
    atomic_json(output/"campaign-status.json",ledger,replace=True)
  ledger["state"]="finished_all_planned_outcomes"
 except BaseException as exc:
  ledger["state"]="aborted_no_retry";ledger["error_type"]=type(exc).__name__;ledger["error"]=str(exc)[:1024]
  for case in ledger["cases"]:
   for slot in ("sender","receiver"):
    if case[slot] and case[slot].get("worker_status") in {"not_started","running"}:
     previous=case[slot]["worker_status"]
     case[slot]={"worker_status":"unknown_after_abort" if previous=="running" else "not_attempted_after_abort",
      "authenticated_delivery":False,"retry_permitted":False}
 ledger["wall_seconds"]=time.monotonic()-started
 ledger["authenticated_payload_deliveries"]=sum(bool(x["receiver"] and x["receiver"].get("authenticated_delivery")) for x in ledger["cases"])
 ledger["authenticated_delivery_denominator"]=2
 # A transcript count is not an independent scientific audit.
 ledger["independent_audit_pending"]=True
 try:atomic_json(output/"campaign-status.json",ledger,replace=True)
 except Exception:atomic_json(output/"campaign-final-status.json",ledger)
 return ledger


if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--source-readback",type=Path,required=True)
 a=p.parse_args();result=run(ROOT,a.source_readback)
 print(json.dumps({"state":result["state"],"authenticated_payload_deliveries":result["authenticated_payload_deliveries"],
  "denominator":2,"independent_audit_pending":True}))
