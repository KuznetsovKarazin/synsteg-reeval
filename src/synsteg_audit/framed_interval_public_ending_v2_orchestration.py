"""Explicit one-attempt worker for ending-v2, no experimental receiver inputs.

The coordinator owns freezing/intent. This installed receiver consumes exactly
public profile/spec/context/envelope, wire and key; expected plaintext and
experimental fixture/provenance files are never opened here.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time

from .canonical_channel_v2 import ContractError
from .framed_interval_public_ending_v2 import (
    VERSION, canonical_json, public_digest, load_public_json, validate_public_spec,
    validate_public_context, validate_envelope, generate, receive,
)
from .framed_interval_public_ending_v2_native import PublicNativeService,validate_profile,resource_descriptor
from .framed_interval_public_ending_v2_journal import DurableEndpoint
from .framed_interval_llm_v1_journal import atomic_json,recover_journal,result_record,sha,fsync_directory


def mkdir_durable(path, *, exclusive=False):
    path = Path(path)
    if path.exists():
        if exclusive: raise FileExistsError(path)
        if not path.is_dir() or path.is_symlink(): raise ValueError("directory required")
        return
    mkdir_durable(path.parent)
    path.mkdir()
    fsync_directory(path.parent); fsync_directory(path)


def write_once(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data: raise ValueError("saved artifact differs; no overwrite")
        return
    with path.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(path.parent)


def reconcile(directory):
    """Recover committed evidence without retry, rewriting or deleting raw traces."""
    directory = Path(directory)
    journal = directory / "token-journal.jsonl"
    if not journal.exists():
        return {"wire_available": False, "known_committed_tokens": 0, "unknown_native_step": True,
                "reason": "no_committed_journal", "retry_permitted": False}
    recovered = recover_journal(journal)
    raw = recovered["committed_wire"]
    write_once(directory / "committed-wire.bin", raw)
    convenience = directory / "wire.raw"
    raw_equal = convenience.exists() and convenience.read_bytes() == raw
    result_path = directory / "worker-result.json"
    result = json.loads(result_path.read_bytes()) if result_path.exists() else None
    core = (result.get("core") or {}) if result else {}
    core_equal = None
    if "wire_sha256" in core:
        core_equal = core["wire_sha256"] == sha(raw) and core["wire_bytes"] == len(raw)
    report = {"wire_available": True, "known_committed_tokens": len(recovered["committed_token_ids"]),
        "wire_bytes": len(raw), "wire_sha256": sha(raw), "last_event_sha256": recovered["last_event_sha256"],
        "committed_trace_sha256": sha(canonical_json(recovered["committed_traces"])),
        "unknown_native_step": recovered["unknown_native_step"], "torn_last_record": recovered["torn_last_record"],
        "convenience_raw_matches": raw_equal, "core_returned_wire_matches": core_equal,
        "core_process_outcome_preserved": None if result is None else result.get("status"),
        "retry_permitted": False}
    path = directory / "reconciliation.json"
    if path.exists():
        if json.loads(path.read_bytes()) != report: raise ValueError("reconciliation changed; explicit audit required")
    else: atomic_json(path, report)
    return report


def validate_request(request,profile,spec,context,envelope):
    if type(request) is not dict:raise ContractError("strict endpoint request required")
    role=request.get("role")
    fields={"role","key_hex"}|({"payload_hex","nonce_ledger"} if role=="generate" else {"wire_hex"})
    if role not in {"generate","receive"} or set(request)!=fields:
        raise ContractError("strict role-dependent request fields required")
    key=request["key_hex"]
    if type(key) is not str or re.fullmatch("[a-f0-9]{64}",key) is None:
        raise ContractError("separately supplied32-byte key required")
    validate_profile(profile);validate_public_spec(spec);validate_public_context(context,spec);validate_envelope(envelope)
    if spec["resources"]!=resource_descriptor(profile):
        raise ContractError("public spec/profile mismatch")
    if envelope["spec_sha256"]!=public_digest(spec) or envelope["context_sha256"]!=public_digest(context):
        raise ContractError("public envelope binding differs")
    if any(t>=profile["ready"]["vocabulary_size"] for t in context["prompt_ids"]):
        raise ContractError("public context ID out of vocabulary")
    if role=="receive":
        if envelope["arm"]!="stego":raise ContractError("receiver accepts only stego arm")
        wire=request["wire_hex"]
        if type(wire) is not str or len(wire)>2*spec["limits"]["max_wire_bytes"] or re.fullmatch("(?:[a-f0-9]{2})*",wire) is None:
            raise ContractError("bounded exact received bytes required")
    else:
        ledger=request["nonce_ledger"]
        if type(ledger) is not str or not ledger:raise ContractError("persistent sender nonce ledger required")
        payload=request["payload_hex"]
        if envelope["arm"]=="ordinary":
            if payload is not None:raise ContractError("ordinary arm has no payload")
        elif type(payload) is not str or len(payload)>32 or re.fullmatch("(?:[a-f0-9]{2})*",payload) is None:
            raise ContractError("payload must contain0..16bytes")
    return bytes.fromhex(key)


def execute_endpoint(request,root,profile,spec,context,envelope,directory,*,timeout=300,service_factory=PublicNativeService):
    """Does not inspect any experiment fixture; directory creation is one-attempt."""
    key=validate_request(request,profile,spec,context,envelope)
    if type(timeout) is not int or not 1<=timeout<=300:raise ContractError("bounded timeout required")
    directory=Path(directory)
    mkdir_durable(directory,exclusive=True)
    started=time.monotonic()
    atomic_json(directory/"execution-started.json",{"schema":VERSION+"-worker-start",
        "role":request["role"],"public_spec_sha256":public_digest(spec),"public_context_sha256":public_digest(context),
        "public_profile_sha256":public_digest(profile),"attempt":1,"retry_permitted":False})
    for name,obj in (("public-spec.json",spec),("public-context.json",context),("public-profile.json",profile),("envelope.json",envelope)):
        atomic_json(directory/name,obj)
    service=service_factory(root,profile,spec,context,directory,time.monotonic()+timeout)
    durable=None
    output={"schema":VERSION+"-native-worker-result","role":request["role"],"execution_kind":service.kind,
        "status":"worker_exception","core":None,"retry_permitted":False}
    try:
        if request["role"]=="generate":
            # Imported only by sender branch, never required by a receiver install.
            from .framed_interval_public_v1_nonce import reserve_nonce
            reservation=reserve_nonce(request["nonce_ledger"],key,bytes.fromhex(envelope["nonce_hex"]))
            atomic_json(directory/"nonce-reservation.json",reservation)
        endpoint=service.prepare()
        durable=DurableEndpoint(endpoint,envelope,key,directory,public_spec=spec,public_context=context)
        if request["role"]=="generate":
            payload=None if request["payload_hex"] is None else bytes.fromhex(request["payload_hex"])
            result=generate(key,envelope,durable,payload=payload,public_spec=spec,public_context=context)
        else:
            result=receive(bytes.fromhex(request["wire_hex"]),key,envelope,durable,public_spec=spec,public_context=context)
        core=result_record(result);traces=core.pop("traces")
        saved=recover_journal(directory/"token-journal.jsonl")
        committed=saved["committed_traces"]
        core["trace_sha256"]=sha(canonical_json(traces))
        core["committed_trace_matches_returned_prefix"]=canonical_json(traces[:len(committed)])==canonical_json(committed)
        core["noncommitted_trace_suffix"]=traces[len(committed):]
        atomic_json(directory/"traces.json",traces)
        if core.get("payload_hex") is not None:
            write_once(directory/"received-payload.bin",bytes.fromhex(core["payload_hex"]))
        output.update(status="returned",core=core)
    except BaseException as exc:
        output.update(error_type=type(exc).__name__,error=str(exc)[:1024])
    finally:
        if durable is not None:
            try:durable.close()
            except Exception as exc:output["journal_close_error"]=type(exc).__name__+": "+str(exc)[:512]
        try:service.close()
        except Exception as exc:output["service_close_error"]=type(exc).__name__+": "+str(exc)[:512]
        if "journal_close_error" in output or "service_close_error" in output:
            output["status"]="worker_exception"
        output["model_process_attempted"]=service.process_attempted
        output["wall_seconds"]=time.monotonic()-started
        atomic_json(directory/"worker-result.json",output)
        atomic_json(directory/"recovered-status.json",reconcile(directory))
    return output


def endpoint_main(argv=None):
    p=argparse.ArgumentParser()
    for name in ("root","profile","spec","context","envelope","key-file","output"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--role",required=True,choices=["generate","receive"])
    p.add_argument("--payload-file",type=Path);p.add_argument("--wire-file",type=Path)
    p.add_argument("--nonce-ledger",type=Path);p.add_argument("--timeout",type=int,default=300)
    a=p.parse_args(argv)
    if (a.role=="receive" and (a.payload_file is not None or a.nonce_ledger is not None or a.wire_file is None)
        or a.role=="generate" and (a.wire_file is not None or a.nonce_ledger is None)):
        p.error("role-specific input allowlist differs")
    objects=[load_public_json(path.read_bytes()) for path in (a.profile,a.spec,a.context,a.envelope)]
    key=a.key_file.read_bytes()
    if len(key)!=32:p.error("key file must be32bytes")
    request={"role":a.role,"key_hex":key.hex()}
    if a.role=="receive":request["wire_hex"]=a.wire_file.read_bytes().hex()
    else:request.update(payload_hex=None if a.payload_file is None else a.payload_file.read_bytes().hex(),
                        nonce_ledger=str(a.nonce_ledger))
    result=execute_endpoint(request,a.root,*objects,a.output,timeout=a.timeout)
    print(json.dumps({"status":result["status"],"core_status":(result.get("core") or {}).get("status")}))
    return 0 if result["status"]=="returned" else 1
