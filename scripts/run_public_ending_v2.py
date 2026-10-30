#!/usr/bin/env python3
"""One-shot ending-v2 pilot. Frozen proposal remains disabled; root supplies Git-read activation."""
import argparse, ast, fcntl, hashlib, json, os, re, sys, time
from pathlib import Path

ID = "framed-interval-public-ending-v2"
EXP = "experiments/" + ID
OUT = "results/" + ID
SOURCE = "results/recovery-development-v1/linux_snapshot/development-prompts.jsonl"
SOURCE_SHA = "3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1"
PUBLIC = "experiments/framed-interval-public-v1"
PROFILE, SPEC = PUBLIC + "/native-public-profile.json", EXP + "/native-public-spec.json"
ENTRY = EXP + "/run_native.py"
CANONICAL_LOCK = "/workspace/scratch/af7adce5db85/recovery60/restored/results/project-nn.lock"
ORDERS = [13, 16, 19, 22]
PROPOSAL_SHA = "ff2d2fffa18f0f69f4f6775cc9f9d5ab0653809e7bd29c12f223caa2680d7fad"
SELECTED_TEXTS_SHA = "fcffc618e17b77d50b3fb63298ffc65391a962624e975d440a957930dc50c0ef"
HELPER = "scripts/run_public_ending_v2.py"
COLLECTOR = "src/synsteg_audit/framed_interval_public_ending_v2_collect.py"
RECEIVER_ROOTS = [ENTRY]
PYTHON_ROOTS = [HELPER, ENTRY, COLLECTOR,
 "src/synsteg_audit/framed_interval_public_ending_v2.py",
 "src/synsteg_audit/framed_interval_public_ending_v2_journal.py",
 "src/synsteg_audit/framed_interval_public_ending_v2_native.py",
 "src/synsteg_audit/framed_interval_public_ending_v2_orchestration.py",
 "src/synsteg_audit/author_candidate_mask_first_v4_native.py"]
PUBLIC_INPUTS = [PROFILE, SPEC, EXP+"/protocol.proposed.json", EXP+"/selected-context-texts.json", EXP+"/fixtures.json"]

def source_closure(root, roots):
 """Static transitive local imports, including lazy and relative imports.

 No imported module is executed. Standard/third-party modules are excluded;
 their pinned runtime identity remains part of the existing public profile.
 """
 root=Path(root); found=set(); pending=list(roots)
 def local(name):
  if name != "synsteg_audit" and not name.startswith("synsteg_audit."):return None
  stem="src/"+name.replace(".","/")
  for path in (stem+".py",stem+"/__init__.py"):
   if (root/path).is_file():return path
  raise ValueError("missing local dependency: "+name)
 while pending:
  path=pending.pop()
  if path in found:continue
  if Path(path).is_absolute() or ".." in Path(path).parts:raise ValueError("unsafe source path")
  tree=ast.parse((root/path).read_text(),filename=path);found.add(path)
  if path.startswith("src/synsteg_audit/") and path!="src/synsteg_audit/__init__.py":
   pending.append("src/synsteg_audit/__init__.py")
  package="synsteg_audit" if path.startswith("src/synsteg_audit/") else ""
  for node in ast.walk(tree):
   names=[]
   if isinstance(node,ast.Import):names=[a.name for a in node.names]
   elif isinstance(node,ast.ImportFrom):
    if node.level:
     if not package or node.level!=1:raise ValueError("unsupported relative import: "+path)
     base=package+("."+node.module if node.module else "")
    else:base=node.module or ""
    names=[base]
    # from package import submodule is also a dependency; exported symbols
    # have no corresponding module and are handled by their parent module.
    for alias in node.names:
     if base=="synsteg_audit" and alias.name!="*":
      candidate=base+"."+alias.name
      if (root/("src/"+candidate.replace(".","/")+".py")).is_file():names.append(candidate)
   for name in names:
    dependency=local(name)
    if dependency is not None and dependency not in found:pending.append(dependency)
 return sorted(found)

def required_source_paths(root):
 return set(source_closure(root,PYTHON_ROOTS)+PUBLIC_INPUTS)

def canonical(x):
 return json.dumps(x, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
def digest(x): return hashlib.sha256(x).hexdigest()
def load(p): return json.loads(Path(p).read_bytes())
def sync_dir(p):
 fd=os.open(Path(p),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def put(p, x):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("xb") as f: f.write(canonical(x)); f.flush(); os.fsync(f.fileno())
 sync_dir(p.parent)
def write(p, raw):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 sync_dir(p.parent)
def inventory(p):
 return {str(f.relative_to(p)):digest(f.read_bytes()) for f in sorted(Path(p).rglob("*")) if f.is_file()}

def fixtures():
 # Same PUBLIC fixture key as context4; this is not an API credential.
 key="67b90e29e8f1e922a93fe844bc1e1a056508b1a22b3e5a9fee8da1f97a036d1c"
 seed=digest(canonical([ID,"ordinary-control"]))
 return [{"context_order":i,"case_id":f"o{order:03d}-{arm}","arm":arm,"key_hex":key,
          "payload_hex":bytes(range(16)).hex() if arm=="stego" else None,
          "nonce_hex":digest(canonical([ID,"nonce",order,arm]))[:24],"ordinary_seed_hex":seed}
         for i,order in enumerate(ORDERS) for arm in ("stego","ordinary")]

def prepared(root):
 if digest((root/EXP/"protocol.proposed.json").read_bytes())!=PROPOSAL_SHA:
  raise ValueError("fixed disabled proposal bytes differ")
 if digest((root/EXP/"selected-context-texts.json").read_bytes())!=SELECTED_TEXTS_SHA:
  raise ValueError("fixed selected development source records differ")
 if load(root/EXP/"fixtures.json")!={"schema":ID+"-public-test-fixtures","cases":fixtures()}:
  raise ValueError("fixed key/payload/nonces/ordinary seed differ")
 return load(root/EXP/"protocol.proposed.json")

def imports(root):
 sys.path.insert(0,str(root/"src"))
 from synsteg_audit import framed_interval_public_ending_v2 as core
 from synsteg_audit import framed_interval_public_ending_v2_native as native
 from synsteg_audit import framed_interval_public_ending_v2_orchestration as worker
 from synsteg_audit.author_candidate_mask_first_v4_native import _supervise
 if Path(sys.modules["synsteg_audit.framed_interval_public_ending_v2"].__file__).resolve()!=root/"src/synsteg_audit/framed_interval_public_ending_v2.py":
  raise ValueError("public worker must be imported from frozen checkout")
 return core,native,worker,_supervise

def collect(root,directory,observed,role,wire,expected_payload=None):
 from synsteg_audit import framed_interval_public_ending_v2_collect as collector
 if Path(collector.__file__).resolve()!=root/COLLECTOR:raise ValueError("collector source differs")
 return collector.collect(directory,{"status":"exited","exit_code":0,"new_supervisor":observed},role,wire,
                          expected_payload=expected_payload)

def gate(root,stage,activation,proof):
 protocol=prepared(root)
 if (protocol["execution_enabled"] is not False or protocol["execution_root"]!=str(root)
     or protocol["source_orders"]!=ORDERS or protocol["canonical_nn_lock"]!=CANONICAL_LOCK
     or protocol["endpoint_seconds_each"]!=300 or protocol["tokenization_seconds_each"]!=60
     or protocol["automatic_retries"]!=0 or protocol["logical_endpoints"]!=12):
  raise ValueError("frozen protocol differs")
 closure=load(root/EXP/"source-closure.json")
 if set(closure)!={"schema","files_sha256"} or closure["schema"]!=ID+"-source-closure" or set(closure["files_sha256"])!=required_source_paths(root):
  raise ValueError("complete exact source closure required")
 if Path(__file__).resolve()!=root/HELPER:raise ValueError("helper must execute from frozen checkout")
 a,p=load(activation),load(proof)
 expected={**closure["files_sha256"],str(Path(activation).resolve().relative_to(root)):digest(Path(activation).read_bytes()),
           EXP+"/source-closure.json":digest((root/EXP/"source-closure.json").read_bytes())}
 if stage=="execute":
  contexts=root/OUT/"tokenization/contexts.json"
  expected.update({OUT+"/tokenization/"+p:h for p,h in inventory(contexts.parent).items()})
  aggregate=load(contexts)
  if aggregate.get("count")!=4 or len(aggregate.get("contexts",[]))!=4:raise ValueError("four successful tokenizations required")
  for i,context in enumerate(aggregate["contexts"]):
   d=contexts.parent/f"c{i:02d}"
   t=load(d/"tokenization.json");sup=load(d/"supervisor-result.json")
   if (t.get("status")!="completed" or t.get("tokenizer_calls")!=1 or t.get("prompt_calls")!=0 or t.get("step_calls")!=0
       or sup.get("status")!="completed_child" or load(d/"public-context.json")!=context or t.get("prompt_ids")!=context["prompt_ids"]):
    raise ValueError("tokenization evidence incomplete")
 if (a!={"schema":ID+"-activation","stage":stage,"enabled":True,"proposal_sha256":digest((root/EXP/"protocol.proposed.json").read_bytes()),
         "closure_sha256":digest((root/EXP/"source-closure.json").read_bytes()),"automatic_retries":0}
     or p.get("schema")!=ID+"-readback" or p.get("stage")!=stage or p.get("all_files_exact") is not True
     or re.fullmatch("[0-9a-f]{40}",p.get("commit","")) is None or p.get("files_sha256")!=expected):
  raise ValueError("actual Git activation and full source readback required")
 for path,sha in expected.items():
  if digest((root/path).read_bytes())!=sha:raise ValueError("source changed: "+path)
 texts=load(root/EXP/"selected-context-texts.json")["rows"]
 if len(texts)!=4 or [r["source_order"] for r in texts]!=ORDERS:raise ValueError("all four fixed contexts required")
 for i,r in enumerate(texts):
  if (r["context_order"]!=i or r["context_id"]!=ID+f"-o{ORDERS[i]:03d}"
      or digest(r["chatml_text"].encode())!=r["chatml_sha256"] or r["record"]["split"]!="development"):
   raise ValueError("context text changed")
  if stage=="execute" and aggregate["contexts"][i].get("context_id")!=r["context_id"]:
   raise ValueError("tokenized context source mapping differs")
 if load(root/EXP/"fixtures.json")["cases"]!=fixtures():raise ValueError("one fixed key/payload/nonces required")
 return protocol,texts,p

def tokenize_one(root,profile,item,directory,deadline):
 core,native,worker,_=imports(root)
 check=native.verify_files(root,profile);put(directory/"resource-check.json",check)
 base=root/native.RESOURCE_ROOT;bridge=None
 try:
  bridge=native.PublicOwnedBridge(base/native.BRIDGE,base/native.MODEL,base/native.CPU,directory/"native.stderr.log",deadline,
    expected_sha256=profile["resource_files"][native.BRIDGE]["sha256"],profile_sha256=core.public_digest(profile))
  ready={"vocabulary_size":bridge.vocabulary_size,"masked_tokens":bridge.masked_tokens,"context_tokens":bridge.context_tokens}
  if ready!=profile["ready"]:raise ValueError("actual READY differs from profile")
  put(directory/"resident.json",native.verify_resident(root,profile,bridge))
  before=getattr(bridge,"evaluated",None)
  ids=bridge.tokenize(item["chatml_text"],True,True)
  if getattr(bridge,"evaluated",None)!=before:raise ValueError("tokenization evaluated model")
  if (not ids or len(ids)+768>2048 or any(type(i) is not int or not 0<=i<bridge.vocabulary_size for i in ids)):
   raise ValueError("public context exceeds remaining768 budget or vocabulary")
  ctx=core.make_public_context(ids,context_id=item["context_id"])
  put(directory/"public-context.json",ctx)
  put(directory/"tokenization.json",{"status":"completed","tokenizer_calls":1,"prompt_calls":0,"step_calls":0,
                                    "prompt_ids":ids,"context_sha256":core.public_digest(ctx),
                                    "evaluated_attribute_before":before,"evaluated_attribute_after":getattr(bridge,"evaluated",None),
                                    "no_PROMPT_STEP_scope":"sole source-pinned TOKENIZE method; evaluated attribute absent before first PROMPT"})
 finally:
  if bridge is not None:bridge.close()

def tokenization(root,texts):
 core,native,worker,supervise=imports(root);profile=load(root/PROFILE);out=root/OUT/"tokenization"
 out.mkdir(parents=True,exist_ok=False);put(out/"initial-denominator.json",{"contexts":4,"physical_tokenizations_max":4})
 contexts=[]
 for item in texts:
  d=out/f"c{item['context_order']:02d}";d.mkdir();put(d/"reservation.json",{"status":"reserved_outcome_unknown","automatic_retries":0})
  observed=supervise(d,lambda deadline:tokenize_one(root,profile,item,d,deadline),seconds=60)
  if observed["status"]!="completed_child" or not (d/"tokenization.json").exists():
   put(out/"stopped.json",{"status":"unknown_or_failure_no_retry","context_order":item["context_order"]});return False
  contexts.append(load(d/"public-context.json"))
 put(out/"contexts.json",{"schema":ID+"-public-contexts","contexts":contexts,"count":4});return True

def install_receiver(root,dest,contexts,profile):
 from shutil import copymode
 dest.mkdir(parents=True,exist_ok=False)
 for name in source_closure(root,RECEIVER_ROOTS)+[PROFILE,SPEC]:
  write(dest/name,(root/name).read_bytes())
 for i,context in enumerate(contexts):put(dest/f"public-contexts/c{i:02d}.json",context)
 for name in profile["resource_files"]:
  src=root/"resources/seeded-pair-llm-v1"/name;target=dest/"resources/seeded-pair-llm-v1"/name
  target.parent.mkdir(parents=True,exist_ok=True);os.link(src,target,follow_symlinks=False)
 return {"files_sha256":inventory(dest),"expected_plaintext_present":False,"fixture_present":False,
         "source_proof_present":False,"scope":"fresh public-only install; hardlinks samehost, no OS sandbox"}

def exec_worker(command,cwd,log):
 os.environ["PYTHONPATH"]=str(cwd/"src");os.environ["PYTHONDONTWRITEBYTECODE"]="1"
 os.chdir(cwd)
 for fd,name in ((1,"stdout.log"),(2,"stderr.log")):
  out=os.open(log/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.dup2(out,fd);os.close(out)
 os.execv(command[0],command)

def execute(root,texts):
 core,native,worker,supervise=imports(root);profile,spec=load(root/PROFILE),load(root/SPEC)
 contexts=load(root/OUT/"tokenization/contexts.json")["contexts"]
 if len(contexts)!=4:raise ValueError("four public contexts required")
 for item,context in zip(texts,contexts):
  core.validate_public_context(context,spec)
  if context["context_id"]!=item["context_id"]:raise ValueError("public context binding differs")
 out=root/OUT/"generation";out.mkdir(parents=True,exist_ok=False)
 install=root/"runtime"/ID/"receiver-install"
 cases=[{"case_id":r["case_id"],"arm":r["arm"],"sender":"not_attempted","receiver":"not_attempted" if r["arm"]=="stego" else None,
         "authenticated_delivery":None,"exact_payload":None} for r in fixtures()]
 put(out/"initial-denominator.json",{"cases":cases,"senders":8,"fresh_receivers":4,"payload_delivery_denominator":4})
 endpoint_count=0;stopped=False;fatal=None
 try:
  inst=install_receiver(root,install,contexts,profile);put(out/"receiver-install-inventory.json",inst)
  for fixture,case in zip(fixtures(),cases):
   context=contexts[fixture["context_order"]]
   env=core.make_envelope(spec,context,bytes.fromhex(fixture["nonce_hex"]),arm=fixture["arm"],ordinary_seed=bytes.fromhex(fixture["ordinary_seed_hex"]))
   base=out/fixture["case_id"];base.mkdir();wire=None
   for role,slot in (("generate","sender"),("receive","receiver")):
    if slot=="receiver" and fixture["arm"]!="stego":continue
    if slot=="receiver" and wire is None:
     case[slot]="dependency_blocked";continue
    deployment=root if slot=="sender" else install
    inputs=base/(slot+"-inputs");inputs.mkdir()
    write(inputs/"key.bin",bytes.fromhex(fixture["key_hex"]));put(inputs/"envelope.json",env)
    context_path=inputs/"public-context.json";put(context_path,context)
    resultdir=base/slot;log=base/(slot+"-process");log.mkdir()
    put(log/"reservation.json",{"status":"reserved_outcome_unknown","role":role,"automatic_retries":0})
    command=[sys.executable,str(deployment/ENTRY),"--root",str(deployment),"--profile",str(deployment/PROFILE),
      "--spec",str(deployment/SPEC),"--context",str(context_path),"--envelope",str(inputs/"envelope.json"),
      "--key-file",str(inputs/"key.bin"),"--output",str(resultdir),"--role",role,"--timeout","300"]
    if slot=="sender":
     command+=["--nonce-ledger",str(root/"runtime"/ID/"sender-nonces")]
     if fixture["payload_hex"] is not None:
      write(inputs/"payload.bin",bytes.fromhex(fixture["payload_hex"]));command+=["--payload-file",str(inputs/"payload.bin")]
    else:write(inputs/"wire.bin",wire);command+=["--wire-file",str(inputs/"wire.bin")]
    put(log/"command.json",{"argv":command});case[slot]="reserved_outcome_unknown";endpoint_count+=1
    observed=supervise(log,lambda deadline:exec_worker(command,deployment,log),seconds=300)
    if observed["status"]!="completed_child" or not (resultdir/"worker-result.json").exists():
     stopped=True;put(out/f"after-{fixture['case_id']}-{slot}.json",{"cases":cases,"endpoints_reserved":endpoint_count});break
    result=load(resultdir/"worker-result.json");collected=collect(root,resultdir,observed,role,wire,bytes.fromhex(fixture["payload_hex"]) if slot=="receiver" else None)
    rec=collected.get("reconciliation",{});code=result.get("core") or {};case[slot+"_evidence"]=collected
    if (result.get("status")!="returned" or result.get("role")!=role or result.get("execution_kind")!="native"
        or result.get("model_process_attempted") is not True or rec.get("unknown_native_step") or rec.get("torn_last_record")
        or collected.get("artifact_error") or collected.get("delivery_evidence_error") or any("error" in k for k in result)):
     stopped=True;put(out/f"after-{fixture['case_id']}-{slot}.json",{"cases":cases,"endpoints_reserved":endpoint_count});break
    case[slot]=code.get("status","unknown")
    known_sender={"support_refusal","capacity_limit","wire_limit_exceeded","ending_pending_receiver","ordinary_ending_generated","ending_not_reached","insufficient_capacity","ordinary_cap_generated"}
    known_receiver={"ending_not_reached","capacity_limit","invalid_wire","incomplete_wire","insufficient_capacity","authentication_failed","invalid_frame","authenticated_delivery"}
    if case[slot] not in (known_sender if slot=="sender" else known_receiver):
     stopped=True;put(out/f"after-{fixture['case_id']}-{slot}.json",{"cases":cases,"endpoints_reserved":endpoint_count});break
    if slot=="sender" and rec.get("wire_available"):wire=(resultdir/"committed-wire.bin").read_bytes()
    if slot=="receiver":
     case["authenticated_delivery"]=collected.get("authenticated_delivery") is True
     case["exact_payload"]=case["authenticated_delivery"] and collected.get("exact_payload") is True
    put(out/f"after-{fixture['case_id']}-{slot}.json",{"cases":cases,"endpoints_reserved":endpoint_count})
   if stopped:break
 except BaseException as exc:
  stopped=True;fatal={"type":type(exc).__name__,"detail":str(exc)[:1024]}
 put(out/"result.json",{"schema":ID+"-result","cases":cases,"endpoints_reserved":endpoint_count,
   "planned_endpoints":12,"planned_senders":8,"planned_fresh_receivers":4,"quality_denominator":8,"payload_delivery_denominator":4,"exact_deliveries":sum(c["exact_payload"] is True for c in cases),
   "status":"stopped_requires_review_no_retry" if stopped else "completed_pending_independent_audit",
   "fatal":fatal,"sender_nonce_ledger":str(root/"runtime"/ID/"sender-nonces"),
   "preserve_also":["runtime/"+ID+"/sender-nonces","receiver-install-inventory.json"],
   "independent_audit_pending":True,"model_security_claim":False,"source_fidelity_measured":False})
 return not stopped

def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["validate-prepared","tokenize","execute"])
 p.add_argument("--root",type=Path,required=True)
 p.add_argument("--activation",type=Path);p.add_argument("--readback",type=Path)
 a=p.parse_args();root=a.root.resolve()
 if a.stage=="validate-prepared":print(json.dumps(prepared(root)));return
 if a.activation is None or a.readback is None:p.error("activation and actual Git readback are required")
 protocol,texts,proof=gate(root,a.stage,a.activation,a.readback)
 with open(CANONICAL_LOCK,"a+b") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   success=tokenization(root,texts) if a.stage=="tokenize" else execute(root,texts)
  except BaseException as exc:
   target=root/OUT/("tokenization" if a.stage=="tokenize" else "generation")
   target.mkdir(parents=True,exist_ok=True)
   put(target/"coordinator-error.json",{"schema":ID+"-coordinator-error","stage":a.stage,"status":"requires_review_no_retry",
       "error_type":type(exc).__name__,"error":str(exc)[:1024],"tokenizations_denominator":4,"endpoints_denominator":12})
   raise
  if not success:raise SystemExit(3)
if __name__=="__main__":main()
