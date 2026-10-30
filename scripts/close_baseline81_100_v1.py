#!/usr/bin/env python3
"""Close one fixed batch41–50 after root supplies actual Git readback evidence.

No model, tokenizer, encoder, decoder or scientific audit is run. The supplied
receipt records the root coordinator's remote verification, not an independent
remote proof. Previous immutable inputs/results are never rewritten. Only the
three current status/active-intent pointers advance after all gates pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

BATCH = BEFORE = AFTER = RUN_COUNT = PRIMARY_COUNT = None
OUT = BATCH_PATH = INTENT = RECEIPT = CLOSURE = AUDIT = None
REPO = "KuznetsovKarazin/synsteg-independent-reproduction"
RUN = Path("runs/large-campaign-development-v2/models/tinyllama11")
BASE_STATUS = Path("results/large-campaign-continuation-v3/status.json")
NATIVE_STATUS = Path("results/native-continuation-20261025/status.json")
ACTIVE = Path("results/large-campaign-continuation-v3/active-intent.json")

def configure(batch_number):
    global BATCH,BEFORE,AFTER,RUN_COUNT,PRIMARY_COUNT,OUT,BATCH_PATH,INTENT,RECEIPT,CLOSURE,AUDIT
    require(type(batch_number) is int and 41 <= batch_number <= 50, "Only fixed batches41–50")
    BEFORE=2*batch_number-2;AFTER=BEFORE+2;BATCH=f"batch{batch_number:02d}"
    RUN_COUNT=AFTER+6;PRIMARY_COUNT=3*AFTER+27
    OUT=Path(f"results/continuation-20261026-{BATCH}")
    BATCH_PATH=Path("results/large-campaign-development-v2/batches")/BATCH
    INTENT=Path(f"results/large-campaign-continuation-v3/intents/{BATCH}.json")
    RECEIPT=Path(f"results/large-campaign-continuation-v3/git-receipts/after-{BATCH}.json")
    CLOSURE=Path(f"results/large-campaign-continuation-v3/intent-closures/{BATCH}.json")
    AUDIT=Path(f"results/large-campaign-continuation-v3-audit/{BATCH}-postrun-audit.json")

def next_stage():
    if AFTER==100:
        return None, [], None, "TinyLlama100 complete. Qwen0.5B/Falcon3-1B require a separate model/runtime/launch freeze and full100-development intent; this TinyLlama utility must not launch them."
    next_batch=f"batch{AFTER//2+1:02d}"
    return next_batch,[AFTER+1,AFTER+2],AFTER,f"New {next_batch} immutable intent and actual Git readback; unchanged utility, expected{AFTER}, after-{BATCH} receipt and exclusive project NN lease. Do not rerun {BATCH}."


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(root, path):
    return json.loads((root / path).read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def replace_pointer(root, relative, original, value):
    target = root / relative
    require(target.read_bytes() == original, "Concurrent pointer edit: " + str(relative))
    temporary = target.with_name(target.name + "."+BATCH+"-next")
    with temporary.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    require(target.read_bytes() == original, "Concurrent pointer edit before replace: " + str(relative))
    os.replace(temporary, target)


def close(root, primary_commit, proof_path):
    root = root.resolve()
    require(re.fullmatch(r"[0-9a-f]{40}", primary_commit), "Require full primary Git commit")
    require(not proof_path.is_absolute() and ".." not in proof_path.parts,
            "Require repository-relative root readback receipt path")
    require(not (root / RECEIPT).exists() and not (root / CLOSURE).exists(),
            "Closure/receipt already exists; do not repeat")
    proof_raw = (root / proof_path).read_bytes()
    proof = json.loads(proof_raw)
    require(proof.get("schema") == f"baseline-{BATCH}-actual-git-readback-v1"
            and proof.get("commit") == primary_commit
            and proof.get("exact_git_readback") is True
            and proof.get("fresh_restore_verified") is True
            and type(proof.get("restored_primary_files")) is int
            and proof["restored_primary_files"] == PRIMARY_COUNT,
            "Actual Git readback/fresh-restore proof is absent or incomplete")
    archive = read(root, OUT / "artifacts/manifest.json")
    transport = read(root, OUT / "artifacts/transport-manifest.json")
    plan_path = OUT / "artifacts/primary-save-plan.json"
    plan = read(root, plan_path)
    expected_saved = {item["path"]: item["sha256"]
                      for item in plan["text_files"] + plan["binary_parts"]}
    expected_saved[str(plan_path)] = sha((root / plan_path).read_bytes())
    require(proof.get("verified_files_sha256") == expected_saved,
            "Root readback must bind every primary save-plan path and the plan itself")
    require(all(sha((root / p).read_bytes()) == digest for p, digest in expected_saved.items()),
            "Primary local bytes changed after actual Git readback")
    require(proof.get("archive_sha256") == archive["archive"]["sha256"]
            and archive["archive"] == transport["archive"]
            and len(archive["files"]) == PRIMARY_COUNT, "Archive identity/denominator differs")
    checkpoint = read(root, BATCH_PATH / "after-checkpoint.json")
    process = read(root, BATCH_PATH / "process.json")
    audit = read(root, AUDIT)
    require(checkpoint["status"] == "verified"
            and checkpoint["completed_generations"] == checkpoint["attempted"] == AFTER
            and checkpoint["planned_model_record_pairs"] == 300
            and process["status"] == "batch_completed_pending_remote_checkpoint"
            and process["exit_code"] == 0 and process["timed_out"] is False
            and process["new_attempts"] == 2 and process["prior_evidence_issues"] == [],
            "Require a successful known pair with complete denominator")
    run_hashes = checkpoint["run_files_sha256"]
    require(len(run_hashes) == RUN_COUNT and proof.get("run_files_sha256") == run_hashes,
            f"Require exact actual Git readback of all{RUN_COUNT} current run files")
    require(audit["status"] == "passed" and audit["completed"] == AFTER
            and audit["prior_completed_entries_preserved"] == BEFORE
            and audit["current_cumulative_run_matches_immutable_after_snapshot"] is True,
            "Frozen independent postrun audit has not passed")
    require(all(sha((root / p).read_bytes()) == digest for p, digest in run_hashes.items()),
            "Cumulative run differs from immutable after snapshot")
    gate = read(root, OUT / "root-git-readback-gate.json")
    intent = read(root, INTENT)
    require(intent["batch_id"] == BATCH and intent["expected_completed_before"] == BEFORE
            and intent["target_completed_after"] == AFTER
            and gate["batch_id"] == BATCH
            and re.fullmatch(r"[0-9a-f]{40}", gate["intent_and_preflight_commit"]),
            "Intent or launch readback gate identity differs")
    require([row["record_id"] for row in audit["new_completed_rows"]]
            == intent["planned_next_record_ids"], "Completed pair differs from fixed intent")
    pointer_raw = {p: (root / p).read_bytes() for p in [BASE_STATUS, NATIVE_STATUS, ACTIVE]}
    baseline, native, active = [json.loads(pointer_raw[p]) for p in [BASE_STATUS, NATIVE_STATUS, ACTIVE]]
    require(baseline["completed"] == BEFORE and native["baseline_completed"] == BEFORE
            and active["batch_id"] == BATCH,
            "Pointers changed beyond the expected predecessor; do not overwrite")
    require(baseline["public_transport_latest_completed_covers"] == 80
            and native["public_transport_latest_completed_covers"] == 80
            and baseline["first80_transport_execution"] == native["first80_transport_execution"]
            and native["first80_transport_execution"]["completed_covers"] == 80
            and native["first80_transport_execution"]["status"] == "completed_saved_verified"
            and native["first80_transport_execution"]["active"] is False
            and native["first60_transport_execution"]["completed_covers"] == 60
            and native["candidate_native_v2_execution"]["independently_verified_format_accepted_chunks"] == 0,
            "Existing first80/first60/candidate history is inconsistent")
    require(baseline["candidate_native_v3_execution"] == native["candidate_native_v3_execution"]
            and native["candidate_native_v3_execution"]["active"] is False
            and native["candidate_native_v3_execution"]["outcomes_known"] is True
            and native.get("active_model_process") is None
            and native["candidate_native_v3_execution"]["primary_commit"]
                == "4f116ed040ad9811af05401b60f5a7317dadde9f"
            and native["candidate_native_v3_execution"]["format_accepted_chunks"] == 1
            and native["candidate_native_v3_execution"]["empty_support_chunks"] == 1,
            "Preserve the already closed native-v3 experiment; do not overwrite concurrent native work")
    # The new recovery's directory limitation is retained independently of
    # named-file hashes and successful generation completion.
    limitation = {"strict_active_cache_inventory": False, "extra_temporary_file_cause": "unknown",
        "hermetic_execution_claimed": False,
        "named_resource_gate": "results/baseline60-cloud-recovery-20261026-v1/root-named-resource-gate.json",
        "scope": "Seven named frozen model/tokenizer resources and exact runtime versions verified; unexplained temporary resource prefixes retained separately."}
    receipt = {"schema": "large-campaign-verified-git-checkpoint-v3", "repository": REPO,
        "commit": primary_commit, "readback_verified": True, "expected_completed": AFTER,
        "run_files_sha256": run_hashes,
        "verification_basis": f"Root supplied exact actual Git readback of all save-plan paths and{RUN_COUNT} current run files, and fresh restoration of{PRIMARY_COUNT} primary members from actual fetched parts. This local receipt records coordinator verification, not an independent remote proof.",
        "actual_git_readback_receipt": str(proof_path), "actual_git_readback_receipt_sha256": sha(proof_raw),
        "primary_archive": archive["archive"],
        "primary_archive_storage": {"mode": "ordered_hash_bound_git_parts",
            "manifest": str(OUT / "artifacts/transport-manifest.json"),
            "chunk_count": transport["chunk_count"], "logical_zip_is_not_direct_git_blob": True},
        "resource_inventory_limitation": limitation}
    closure = {**intent, "phase": "completed_and_remote_verified", "active": False,
        "generation_outcome_known": True, "intent_git_commit": gate["intent_and_preflight_commit"],
        "actual_git_readback_verified": True, "completed_after": AFTER,
        "remote_checkpoint": primary_commit, "retry_permitted": False,
        "next_launch_requires_new_durable_intent": True,
        "primary_archive_path": archive["archive"]["path"], "audit_path": str(AUDIT),
        "root_actual_git_readback_receipt": str(proof_path),
        "resource_inventory_limitation": limitation}
    now = time.time()
    next_batch,next_orders,next_expected,next_gate=next_stage()
    baseline.update(phase=f"{BATCH}_verified_saved", completed=AFTER,
        full_denominator=300, not_attempted=300-AFTER, tinyllama_completed=AFTER,
        tinyllama_not_attempted=100-AFTER, last_completed_batch=BATCH,
        active_batch=None, active_intent_path=None, active_generation=False,
        last_remote_verified_checkpoint=primary_commit, latest_verified_git_commit=primary_commit,
        latest_primary_git_commit=primary_commit, last_remote_verified_completed=AFTER,
        next_large_batch=next_batch, next_large_source_orders=next_orders, next_batch=next_batch,
        expected_completed_for_next_batch=next_expected, completed_generations=AFTER,
        latest_primary_archive=archive["archive"]["path"],
        latest_primary_manifest=str(OUT / "artifacts/manifest.json"),
        latest_primary_transport_manifest=str(OUT / "artifacts/transport-manifest.json"),
        latest_verified_receipt=str(RECEIPT), latest_audit=str(AUDIT),
        cumulative_generated_output_tokens=audit["cumulative_generated_output_tokens"],
        model_counts={"tinyllama11":AFTER,"qwen05":0,"falcon3_1b":0},
        latest_primary_files=PRIMARY_COUNT, latest_run_files_verified=RUN_COUNT,
        latest_run_files_locally_verified=RUN_COUNT, latest_primary_remote_git_verified=True,
        latest_primary_archive_sha256=archive["archive"]["sha256"],
        latest_primary_archive_is_logical_reassembled=True, prior_completed_prefix_preserved=BEFORE,
        pending_remote_checkpoint_batch=None, next_launch_ready=False,
        next_launch_gate=next_gate,
        new_baseline_source_orders_awaiting_public_transport=sorted(set(baseline.get("new_baseline_source_orders_awaiting_public_transport",[])) | {BEFORE+1,AFTER}),
        pending_text_review_source_orders=sorted(set(baseline.get("pending_text_review_source_orders",[])) | {BEFORE+1,AFTER}),
        latest_run_files_verified_scope=f"Root actual Git readback of all{RUN_COUNT} current run files and fresh reconstruction of{PRIMARY_COUNT} primary members.",
        latest_receipt_closure_review_paths=[str(RECEIPT),str(CLOSURE)],
        previous_receipt_closure_review_commit=baseline.get("latest_receipt_closure_review_commit"),
        latest_receipt_closure_review_commit=None,
        latest_receipt_closure_review_commit_scope="New receipt/closure are pending their own Git save; immutable primary commit is recorded separately.",
        latest_initial_audit_failure_retained=None, latest_audit_recovered_without_inference_retry=False,
        latest_storage_list_failure_retained=None, updated_at_unix=now,
        resource_inventory_limitation=limitation)
    native.update(phase=f"baseline{AFTER}_transport80_mask_order_saved_verified",
        baseline_completed=AFTER, baseline_denominator=300,
        planned_next_source_orders=next_orders, active_baseline_batch=None, active_model_process=None,
        baseline_calls_status=f"{AFTER}_saved_no_active_baseline_model",
        baseline_last_checkpoint_commit=primary_commit,
        baseline_output_tokens=audit["cumulative_generated_output_tokens"],
        updated_at_unix=now, resource_inventory_limitation=limitation,
        new_baseline_source_orders_awaiting_public_transport=baseline["new_baseline_source_orders_awaiting_public_transport"])
    # Do not manufacture a completed source-fidelity review for the new pair.
    # Existing first80/first60, mask-order and native-v1/v2/v3 nested objects remain JSON-equivalent.
    snapshot = root / OUT / "pre-closure-pointer-snapshot"
    require(not snapshot.exists(), "Pointer snapshot already exists; do not repeat")
    snapshot.mkdir(parents=True)
    for p, raw in pointer_raw.items():
        target = snapshot / p
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    write_new(root / RECEIPT, receipt)
    write_new(root / CLOSURE, closure)
    for p, value in [(BASE_STATUS,baseline),(NATIVE_STATUS,native),(ACTIVE,closure)]:
        replace_pointer(root,p,pointer_raw[p],value)
    result = {"schema":"baseline81-100-post-git-closure-process-v1","batch_id":BATCH,"status":"closed_locally_pending_closure_git_save",
        "primary_commit":primary_commit,"actual_git_readback_receipt":str(proof_path),
        "updated_paths":[str(RECEIPT),str(CLOSURE),str(BASE_STATUS),str(NATIVE_STATUS),str(ACTIVE)],
        "pointer_previous_sha256":{str(p):sha(raw) for p,raw in pointer_raw.items()},
        "pointer_after_sha256":{str(p):sha((root/p).read_bytes()) for p in pointer_raw},
        "public_transport_completed_covers_unchanged":80,"candidate_v2_outcome_unchanged":True,"candidate_v3_outcome_unchanged":True,
        "generation_calls":0,"tokenizer_calls":0,"scientific_audit_repeated":False,
        "completed":AFTER,"full_denominator":300,"next_batch":next_batch,
        "resource_inventory_limitation":limitation}
    write_new(root / OUT / "post-git-closure.json",result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-number",type=int,choices=range(41,51),required=True)
    parser.add_argument("--primary-commit",required=True)
    parser.add_argument("--git-readback-receipt",type=Path,required=True)
    args=parser.parse_args()
    configure(args.batch_number)
    print(json.dumps(close(args.root,args.primary_commit,args.git_readback_receipt),indent=2))
