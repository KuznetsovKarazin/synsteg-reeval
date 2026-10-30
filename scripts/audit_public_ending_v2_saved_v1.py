#!/usr/bin/env python3
"""Read-only ending-v2 audit of saved laws and bytes, never native decode.

CLI requires an actual-Git restoration receipt supplied by the operator. The
receipt's existence is not proof of network provenance; coverage and bytes are
checked here. No collector/reconcile is imported (reconcile can write files).
"""
import argparse
from dataclasses import asdict
import hashlib
import hmac
import importlib
import json
from pathlib import Path
import re
import sys

ID = "framed-interval-public-ending-v2"
EXP, OUT = "experiments/" + ID, "results/" + ID
SCRIPT = "scripts/audit_public_ending_v2_saved_v1.py"
CLOSURE_SHA = "66accd06bede65fe00d6da7784e0edd34da59a350dd5209b5fac969820a7d9e8"
ORDERS = [13, 16, 19, 22]
CASE_IDS = [f"o{i:03d}-{arm}" for i in ORDERS for arm in ("stego", "ordinary")]
RECEIVER_SOURCES = [EXP + "/run_native.py"] + ["src/synsteg_audit/" + name for name in (
    "__init__.py", "canonical_channel_v2.py", "framed_interval_llm_v1.py", "framed_interval_llm_v1_journal.py",
    "framed_interval_llm_v1_qwen.py", "framed_interval_public_ending_v2.py", "framed_interval_public_ending_v2_journal.py",
    "framed_interval_public_ending_v2_native.py", "framed_interval_public_ending_v2_orchestration.py",
    "framed_interval_public_v1.py", "framed_interval_public_v1_native.py", "framed_interval_public_v1_nonce.py",
    "integer_interval_channel_v1.py", "proc_namespace_identity_v1.py", "proc_namespace_identity_v2.py", "seeded_pair_llm.py")]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    def __init__(self, root, receipt=None):
        self.root = Path(root).resolve()
        self.receipt = receipt
        self.reads = {}
        if receipt is not None:
            require(receipt.get("schema") == "ending-v2-audit-git-readback-v1", "readback schema differs")
            require(re.fullmatch("[0-9a-f]{40}", receipt.get("commit", "")) is not None, "commit required")
            require(type(receipt.get("files_sha256")) is dict, "readback hash map required")

    def path(self, relative):
        p = Path(relative)
        require(not p.is_absolute() and ".." not in p.parts, "unsafe evidence path")
        out = self.root / p
        require(out.resolve().is_relative_to(self.root), "evidence escaped restoration root")
        return out

    def exists(self, relative):
        return self.path(relative).is_file()

    def raw(self, relative):
        raw = self.path(relative).read_bytes()
        sha = digest(raw)
        if self.receipt is not None:
            require(self.receipt["files_sha256"].get(str(relative)) == sha,
                    "missing/different actual-Git readback hash: " + str(relative))
        self.reads[str(relative)] = {"sha256": sha, "bytes": len(raw)}
        return raw

    def json(self, relative):
        return json.loads(self.raw(relative))


def recover(raw):
    """Independent journal parser; retain an incomplete final event as unknown."""
    ids, suffixes, traces, prefix, pending = [], [], [], b"", None
    previous, torn, unknown = "0" * 64, False, False
    for sequence, line in enumerate(raw.splitlines(keepends=True)):
        if not line.endswith(b"\n"):
            torn = True
            break
        row = json.loads(line)
        claimed = row.pop("event_sha256")
        require(row.get("sequence") == sequence and row.get("previous_sha256") == previous
                and digest(canonical(row)) == claimed, "journal chain differs")
        require(not unknown, "event after unknown native STEP")
        previous = claimed
        kind = row.get("kind")
        if kind == "prepared":
            token, suffix, trace = row["token_id"], bytes.fromhex(row["suffix_hex"]), row["trace"]
            require(pending is None and type(token) is int and token >= 0 and bool(suffix), "invalid prepared event")
            require(row["token_index"] == len(ids) and row["wire_prefix_sha256"] == digest(prefix)
                    and trace["step"] == len(ids) and trace["prefix_ids"] == ids
                    and trace["observed_token"] == token, "prepared prefix differs")
            pending = (claimed, token, suffix, trace)
        elif kind == "committed":
            require(pending is not None and row["prepared_sha256"] == pending[0]
                    and row["token_index"] == len(ids) and row["token_id"] == pending[1]
                    and row["wire_after_sha256"] == digest(prefix + pending[2]), "commit differs")
            ids.append(pending[1]); suffixes.append(pending[2]); traces.append(pending[3])
            prefix += pending[2]; pending = None
        elif kind == "native_step_error":
            require(pending is not None and row["prepared_sha256"] == pending[0]
                    and row["token_index"] == len(ids), "native error lacks pending STEP")
            unknown = True
        else:
            raise ValueError("unknown journal event kind")
    return {"ids": ids, "suffixes": suffixes, "traces": traces, "wire": prefix,
            "unknown": unknown or pending is not None or torn, "torn": torn,
            "last_event_sha256": previous, "pending_token": None if pending is None else pending[1]}


def replay_saved(saved, core, interval, spec, envelope, fixture):
    """Pure evaluator replay conditional on recorded eligibility/weights."""
    config = core.validate_public_spec(spec)
    key = bytes.fromhex(fixture["key_hex"])
    _, ordering_key, tail_key = core.derive_keys(key, envelope)
    expected_payload = None if fixture["payload_hex"] is None else bytes.fromhex(fixture["payload_hex"])
    frame = None if expected_payload is None else core.seal_frame(expected_payload, key, envelope)
    bits = core.ordinary_bits(envelope, config) if frame is None else core.input_bits(frame, tail_key, envelope, config)
    sender, receiver = interval.Sender(bits, config), interval.Receiver(config)
    prefix, prefix_ids, first, ending, threshold_bytes = b"", [], None, None, None
    stream_mismatch = None
    for i, (token, suffix, trace) in enumerate(zip(saved["ids"], saved["suffixes"], saved["traces"]), 1):
        require(ending is None, "committed STEP after first ending")
        require(first is None or i - 1 - first < 96, "committed STEP after tail cap")
        require(i <= 768, "committed STEP after total cap")
        require(trace["step"] == i-1 and trace["prefix_ids"] == prefix_ids
                and trace["prefix_sha256"] == digest(prefix), "trace prefix differs")
        admission = trace["admission"]
        require(admission["status"] == "ok" and admission["prefix_ids"] == prefix_ids
                and admission["prefix_bytes_sha256"] == digest(prefix)
                and admission["codec_digest"] == spec["resources"]["codec_sha256"], "recorded admission binding differs")
        weights = admission["conditioned_weights"]
        expected_weights = [[v["token_id"], v["weight"]] for v in admission["verdicts"] if v["reason"] == "accepted"]
        require(weights == expected_weights, "recorded conditioned weights differ")
        ids_raw, context = canonical(prefix_ids), core.ordering_context_digest(envelope)
        def score(row):
            token_raw = canonical(row[0])
            data = (core.DOMAIN + b"order\0" + context + len(ids_raw).to_bytes(4, "big") + ids_raw
                    + len(token_raw).to_bytes(4, "big") + token_raw)
            return hmac.digest(ordering_key, data, "sha256"), row[0]
        rows = tuple(tuple(row) for row in sorted(weights, key=score))
        require([list(row) for row in rows] == trace["ordered_integer_law"], "keyed law order differs")
        law = interval.Law(rows)
        step = receiver.take(law, token)
        require(asdict(step.before) == trace["interval_before"] and asdict(step.after) == trace["interval_after"]
                and list(step.emitted_bits) == trace["emitted_bits"], "saved interval transition differs")
        bins = [{"token_id": t, "low": low, "high": high, "width": high-low+1}
                for t, low, high in interval.integer_bins(step.before, law, config)]
        require(bins == trace["integer_bins"], "saved integer bins differ")
        if stream_mismatch is None:
            try:
                if sender.take(law).symbol != token:
                    stream_mismatch = i
            except interval.CapacityLimit:
                stream_mismatch = i
        prefix += suffix; prefix.decode("utf-8"); prefix_ids.append(token)
        require(len(prefix) <= spec["limits"]["max_wire_bytes"], "committed wire byte cap exceeded")
        if first is None and len(receiver.recovered_bits) >= 272:
            first, threshold_bytes = i, len(prefix)
        if first is not None and prefix[-1] in b".!?":
            ending = i
    require(prefix == saved["wire"], "reconstructed wire differs")
    if saved["pending_token"] is not None:
        require(ending is None and len(prefix_ids) < 768 and (first is None or len(prefix_ids)-first < 96),
                "prepared native STEP after public stop")
    frame_verified = None
    if frame is not None and first is not None:
        recovered_frame = interval.exact_bytes(receiver.recovered_bits[:272])
        try:
            frame_verified = recovered_frame == frame and core.open_frame(recovered_frame, key, envelope) == expected_payload
        except Exception:
            frame_verified = False
    return {"committed_tokens": len(prefix_ids), "resolved_bits": len(receiver.recovered_bits),
            "first_threshold_token": first, "ending_token": ending,
            "tail_tokens": None if first is None else len(prefix_ids)-first,
            "wire_bytes": len(prefix), "wire_sha256": digest(prefix), "threshold_wire_bytes": threshold_bytes,
            "tail_wire_bytes": None if threshold_bytes is None else len(prefix)-threshold_bytes,
            "has_terminal_punctuation_exact": bool(prefix) and prefix[-1] in b".!?",
            "fixture_frame_aead_verified": frame_verified, "saved_law_full_prefix_replay": stream_mismatch is None,
            "first_fixture_stream_mismatch_token":stream_mismatch,
            "native_eligibility_independently_recomputed": False,
            "token_to_bytes_mapping_independently_recomputed":False, "tail_seconds": None}


def command_count_scope(saved, returned_traces, code, role, preprompt_verified):
    """Source-derived command arithmetic; deliberately not native telemetry.

    Two CanonicalGate constructors execute validate(empty). Each committed
    transition executes admit.validate, take.validate and fresh selected inspect.
    Candidate invalid_utf8 fails before TOKENIZE; special/zero skip both calls.
    Unrecorded failed admission/codec work is never guessed into an exact total.
    """
    count = len(saved["ids"])
    tokenizes = texts = 2 if preprompt_verified else 0
    receiver_initial_known = role == "receive" and count > 0
    if receiver_initial_known:
        tokenizes += 1; texts += 1
    def candidate_counts(admission):
        verdicts = admission["verdicts"]
        return (sum(v["reason"] in {"accepted","retokenization_mismatch"} for v in verdicts),
                sum(v["reason"] not in {"zero_weight","special_token"} for v in verdicts))
    for trace in saved["traces"]:
        enc, render = candidate_counts(trace["admission"])
        tokenizes += 3 + enc; texts += 3 + render
    suffix = returned_traces[count:]
    recorded_refusal = len(suffix) == 1 and "admission" in suffix[0]
    if recorded_refusal:
        enc, render = candidate_counts(suffix[0]["admission"])
        tokenizes += 1 + enc; texts += 1 + render
    normal_stop = code.get("status") in {"ending_pending_receiver","ordinary_ending_generated",
        "ordinary_cap_generated","insufficient_capacity","authenticated_delivery","incomplete_wire",
        "authentication_failed","invalid_frame"}
    normal_stop |= code.get("status") == "ending_not_reached" and code.get("reason") in {
        "tail96_without_terminal_punctuation","cap768_after_threshold_without_terminal_punctuation",
        "public_cap_without_terminal_punctuation"}
    normal_stop |= code.get("status") == "invalid_wire" and code.get("reason") == "trailing_tokens_after_first_ending"
    normal_stop |= recorded_refusal and code.get("status") in {"support_refusal","invalid_wire"}
    exact = bool(preprompt_verified and not saved["unknown"] and normal_stop
                 and (role == "generate" or receiver_initial_known))
    prompt = 1 if count > 0 or preprompt_verified and normal_stop else None
    returned_count = code.get("emitted_tokens" if role == "generate" else "processed_tokens")
    if type(returned_count) is not int:
        returned_count = None
    return {"scope":"source-derived protocol calls conditional on frozen branches and saved verdicts; not native telemetry",
        "PROMPT_source_derived":prompt,"STEP_journal_committed":count,
        "STEP_prepared_uncommitted_unknown":int(saved["pending_token"] is not None),
        "STEP_returned_core_count":returned_count,
        "STEP_returned_minus_journal":None if returned_count is None else returned_count-count,
        "DIST_recorded_lower_bound":count+int(recorded_refusal),
        "TOKENIZE_source_derived_total":tokenizes if exact else None,
        "TEXT_source_derived_total":texts if exact else None,
        "TOKENIZE_source_derived_lower_bound":tokenizes,"TEXT_source_derived_lower_bound":texts,
        "exact_total_from_source_not_telemetry":exact,
        "unrecorded_failed_codec_or_admission_calls_not_estimated":not exact}


def endpoint(inputs, base, role, fixture, expected_context, spec, core, interval):
    result = {"case_id": fixture["case_id"], "role": role, "classification": "not_attempted",
              "authenticated_delivery_counted": False, "errors": []}
    slot = "sender" if role == "generate" else "receiver"
    directory, process_dir = base + "/" + slot, base + "/" + slot + "-process"
    reserved = inputs.exists(process_dir + "/reservation.json")
    if reserved:
        inputs.json(process_dir + "/reservation.json")
        result["classification"] = "unknown"
    if not inputs.exists(directory + "/worker-result.json"):
        result["reason"] = "reserved_without_terminal_worker" if reserved else "not_attempted"
        return result
    try:
        worker = inputs.json(directory + "/worker-result.json")
        result.update(worker_status=worker.get("status"), model_process_attempted=worker.get("model_process_attempted"),
                      endpoint_wall_seconds=worker.get("wall_seconds"))
        require(worker.get("schema") == ID + "-native-worker-result" and worker.get("role") == role,
                "worker role/schema differs")
        context = inputs.json(directory + "/public-context.json")
        require(context == expected_context and inputs.json(directory + "/public-spec.json") == spec,
                "worker public context/spec differs")
        envelope = inputs.json(directory + "/envelope.json")
        expected_env = core.make_envelope(spec, context, bytes.fromhex(fixture["nonce_hex"]), arm=fixture["arm"],
                                        ordinary_seed=bytes.fromhex(fixture["ordinary_seed_hex"]))
        require(envelope == expected_env, "worker envelope differs from frozen fixture")
        inputs_dir = base + "/" + slot + "-inputs"
        require(inputs.raw(inputs_dir + "/key.bin") == bytes.fromhex(fixture["key_hex"]), "input fixture key differs")
        if role == "generate" and worker.get("model_process_attempted") is True:
            require(inputs.exists(directory + "/nonce-reservation.json"), "attempted sender lacks nonce burn evidence")
        if role == "generate" and inputs.exists(directory + "/nonce-reservation.json"):
            reservation = inputs.json(directory + "/nonce-reservation.json")
            tag = hmac.digest(bytes.fromhex(fixture["key_hex"]), b"framed-interval-public-v1\0nonce-ledger-reservation\0"
                              + bytes.fromhex(fixture["nonce_hex"]), "sha256").hex()
            require(reservation.get("reservation_id") == tag and reservation.get("nonce_hex") == fixture["nonce_hex"]
                    and reservation.get("state") == "burned_before_generation", "nonce reservation differs")
            require(inputs.json("runtime/" + ID + "/sender-nonces/" + tag + ".json") == reservation,
                    "durable nonce ledger differs from endpoint receipt")
            result["nonce_burn_verified"] = True
        if inputs.exists(directory + "/token-journal.jsonl"):
            saved = recover(inputs.raw(directory + "/token-journal.jsonl"))
            evidence = replay_saved(saved, core, interval, spec, envelope, fixture)
            if role == "generate":
                require(evidence["saved_law_full_prefix_replay"], "sender committed prefix differs from frozen fixture stream")
            result.update(evidence=evidence, unknown_native_step=saved["unknown"], torn_last_record=saved["torn"])
            if role == "generate":
                envelope_bytes, context_bytes = len(core.canonical_json(envelope)), len(core.canonical_json(context))
                result["communication_bytes"] = {"wire":len(saved["wire"]), "envelope":envelope_bytes,
                    "unshared_context":context_bytes, "wire_plus_envelope_plus_unshared_context":len(saved["wire"])+envelope_bytes+context_bytes,
                    "shared_compact_spec_reported_separately":len(core.canonical_json(spec)),
                    "pre_shared_key_reported_separately":32, "nonce_already_in_envelope_not_added_twice":True}
            if inputs.exists(directory + "/committed-wire.bin"):
                require(inputs.raw(directory + "/committed-wire.bin") == saved["wire"], "committed wire artifact differs")
            if inputs.exists(directory + "/wire.raw"):
                result["convenience_wire_matches"] = inputs.raw(directory + "/wire.raw") == saved["wire"]
            code = worker.get("core") or {}
            returned_traces = []
            if code:
                returned_traces = inputs.json(directory + "/traces.json")
                terminal_or_unknown = worker.get("status") != "returned" or code.get("status") in {"backend_error","contract_error"}
                prefix_matches = returned_traces[:len(saved["traces"])] == saved["traces"]
                if not terminal_or_unknown:
                    require(prefix_matches, "returned committed prefix differs")
                elif not prefix_matches:
                    result["terminal_returned_traces_disagree_with_authoritative_journal"] = True
                require(digest(canonical(returned_traces)) == code.get("trace_sha256"), "returned trace hash differs")
                require(code.get("noncommitted_trace_suffix") == returned_traces[len(saved["traces"]):], "refused trace suffix differs")
                disagrees = [field for field in ("first_threshold_token", "ending_token", "tail_tokens", "resolved_bits")
                             if code.get(field) != evidence[field]]
                if not terminal_or_unknown:
                    require(not disagrees, "returned worker counters differ: " + str(disagrees))
                elif disagrees:
                    result["terminal_returned_prefix_disagrees_with_authoritative_journal"] = disagrees
            preprompt_verified = False
            if inputs.exists(directory + "/native-preprompt-verification.json"):
                preprompt = inputs.json(directory + "/native-preprompt-verification.json")
                require(preprompt.get("schema") == ID + "-native-preprompt" and preprompt.get("status") == "passed"
                        and preprompt.get("public_spec_sha256") == core.public_digest(spec)
                        and preprompt.get("public_context_sha256") == core.public_digest(context)
                        and preprompt.get("prompt_ids") == context["prompt_ids"]
                        and preprompt.get("prompt_calls_so_far") == preprompt.get("distribution_calls_so_far") == preprompt.get("step_calls_so_far") == 0,
                        "native preprompt declaration differs")
                preprompt_verified = True
            result["command_count_scope"] = command_count_scope(saved, returned_traces, code, role, preprompt_verified)
            process = inputs.json(process_dir + "/supervisor-result.json") if inputs.exists(process_dir + "/supervisor-result.json") else {}
            clean = (worker.get("status") == "returned" and worker.get("execution_kind") == "native"
                     and worker.get("model_process_attempted") is True and process.get("status") == "completed_child"
                     and process.get("child_exit_code") == 0 and not any("error" in key for key in worker)
                     and not saved["unknown"])
            success = code.get("status") in {"ending_pending_receiver", "ordinary_ending_generated", "authenticated_delivery"}
            if success and clean:
                require(evidence["ending_token"] == evidence["committed_tokens"] and evidence["saved_law_full_prefix_replay"],
                        "success lacks complete ending and fixture-stream evidence")
                require(code.get("committed_trace_matches_returned_prefix") is True
                        and code.get("noncommitted_trace_suffix") == [], "success has incomplete trace evidence")
                count_field = "emitted_tokens" if role == "generate" else "processed_tokens"
                require(code.get(count_field) == evidence["committed_tokens"], "worker token count differs")
                if role == "receive":
                    require(code.get("status") == "authenticated_delivery" and code.get("authenticated_delivery") is True,
                            "receiver delivery status differs")
                    require(evidence["fixture_frame_aead_verified"] is True, "receiver successful frame differs")
                    wire = inputs.raw(inputs_dir + "/wire.bin")
                    require(wire == saved["wire"] and code.get("wire_tokens") == evidence["committed_tokens"], "receiver full wire differs")
                    sender_directory = base + "/sender"
                    require(wire == inputs.raw(sender_directory + "/committed-wire.bin"), "receiver input differs from matching sender wire")
                    sender_saved = recover(inputs.raw(sender_directory + "/token-journal.jsonl"))
                    require(saved["traces"] == sender_saved["traces"] and not sender_saved["unknown"],
                            "successful receiver full trace differs from matching sender")
                    result["matching_sender_wire_and_full_trace_verified"] = True
                    expected_payload = bytes.fromhex(fixture["payload_hex"])
                    require(code.get("payload_hex") == fixture["payload_hex"]
                            and inputs.raw(directory + "/received-payload.bin") == expected_payload, "receiver payload differs")
                    result["authenticated_delivery_counted"] = True
                else:
                    expected_status = "ordinary_ending_generated" if fixture["arm"] == "ordinary" else "ending_pending_receiver"
                    require(code.get("arm") == fixture["arm"] and code.get("status") == expected_status,
                            "sender arm/status differs")
                    require(result.get("nonce_burn_verified") is True, "successful sender lacks nonce burn evidence")
                    require(code.get("wire_sha256") == evidence["wire_sha256"] and code.get("wire_bytes") == evidence["wire_bytes"],
                            "sender returned wire differs")
                result["classification"] = "verified_saved_evidence_success"
            elif success:
                result["classification"] = "unknown"
                result["reason"] = "core_success_not_counted_without_clean_terminal_process"
            else:
                require(code.get("authenticated_delivery") is not True and code.get("payload_hex") is None,
                        "refusal advertises payload")
                known = ({"support_refusal", "capacity_limit", "wire_limit_exceeded", "ending_not_reached",
                          "insufficient_capacity", "ordinary_cap_generated"} if role == "generate" else
                         {"ending_not_reached", "capacity_limit", "invalid_wire", "incomplete_wire",
                          "insufficient_capacity", "authentication_failed", "invalid_frame"})
                result["classification"] = "known_refusal" if clean and code.get("status") in known else "unknown"
            result["core_status"] = code.get("status")
        else:
            result["classification"] = "unknown"
            result["reason"] = "no_committed_journal"
    except Exception as exc:
        result["classification"] = "evidence_error"
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
        result["authenticated_delivery_counted"] = False
    return result


def run(inputs, core, interval):
    fixtures = inputs.json(EXP + "/fixtures.json")["cases"]
    require([x["case_id"] for x in fixtures] == CASE_IDS, "frozen eight-case denominator differs")
    require(len({x["nonce_hex"] for x in fixtures}) == 8, "frozen nonces repeat")
    spec = inputs.json(EXP + "/native-public-spec.json")
    contexts = inputs.json(OUT + "/tokenization/contexts.json")["contexts"]
    require(len(contexts) == 4, "four public contexts required")
    selected = inputs.json(EXP + "/selected-context-texts.json")["rows"]
    require(len(selected) == 4, "four frozen context source rows required")
    preparations = []
    for i, context in enumerate(contexts):
        require(context["context_id"] == selected[i]["context_id"], "fixed public context identity differs")
        receipt = inputs.json(OUT + f"/tokenization/c{i:02d}/tokenization.json")
        supervisor = inputs.json(OUT + f"/tokenization/c{i:02d}/supervisor-result.json")
        require(supervisor.get("status") == "completed_child" and supervisor.get("child_exit_code") == 0
                and supervisor.get("session_verified_before_worker") is True
                and supervisor.get("group_shutdown_before_reap") is True
                and supervisor.get("owned_group_present_after_cleanup") is False
                and supervisor.get("shutdown_error") is None, "tokenization supervisor/cleanup differs")
        require(receipt.get("status") == "completed" and receipt.get("tokenizer_calls") == 1
                and receipt.get("prompt_calls") == receipt.get("step_calls") == 0
                and receipt.get("prompt_ids") == context["prompt_ids"]
                and receipt.get("context_sha256") == core.public_digest(context), "preparation tokenization evidence differs")
        preparations.append({"context_order":i,"prompt_ids":len(context["prompt_ids"]),
                             "declared_TOKENIZE":1,"declared_PROMPT":0,"declared_STEP":0})
    terminal_campaign = inputs.exists(OUT + "/generation/result.json")
    campaign = inputs.json(OUT + "/generation/result.json") if terminal_campaign else {
        "cases":[{"case_id":name} for name in CASE_IDS], "status":"missing_terminal_campaign"}
    require([x["case_id"] for x in campaign["cases"]] == CASE_IDS, "campaign case denominator differs")
    if terminal_campaign:
        for name, expected in {"planned_endpoints":12,"planned_senders":8,"planned_fresh_receivers":4,
                               "quality_denominator":8,"payload_delivery_denominator":4}.items():
            require(campaign.get(name) == expected, "campaign " + name + " differs")
    inventory_verified = None
    if inputs.exists(OUT + "/generation/receiver-install-inventory.json"):
        inventory = inputs.json(OUT + "/generation/receiver-install-inventory.json")
        closure = inputs.json(EXP + "/source-closure.json")["files_sha256"]
        profile_path = "experiments/framed-interval-public-v1/native-public-profile.json"
        profile = inputs.json(profile_path)
        expected_inventory = {path:closure[path] for path in RECEIVER_SOURCES + [profile_path, EXP + "/native-public-spec.json"]}
        expected_inventory.update({f"public-contexts/c{i:02d}.json":digest(canonical(context)) for i,context in enumerate(contexts)})
        expected_inventory.update({"resources/seeded-pair-llm-v1/"+name:meta["sha256"] for name,meta in profile["resource_files"].items()})
        require(inventory.get("files_sha256") == expected_inventory and inventory.get("fixture_present") is False
                and inventory.get("expected_plaintext_present") is False and inventory.get("source_proof_present") is False,
                "public-only receiver inventory differs")
        inventory_verified = True
    rows = []
    for fixture in fixtures:
        base = OUT + "/generation/" + fixture["case_id"]
        context = contexts[fixture["context_order"]]
        rows.append(endpoint(inputs, base, "generate", fixture, context, spec, core, interval))
        if fixture["arm"] == "stego":
            row = endpoint(inputs, base, "receive", fixture, context, spec, core, interval)
            declared = next(x for x in campaign["cases"] if x["case_id"] == fixture["case_id"])
            if row["classification"] == "not_attempted" and declared.get("receiver") == "dependency_blocked":
                row["classification"] = "dependency_blocked"
            rows.append(row)
    count = sum(row["authenticated_delivery_counted"] for row in rows)
    for row in rows:
        if row["role"] == "generate" and "communication_bytes" in row:
            delivered = any(other["case_id"] == row["case_id"] and other["role"] == "receive"
                            and other["authenticated_delivery_counted"] for other in rows)
            useful = 128 if delivered else 0
            costs = row["communication_bytes"]
            costs["delivered_payload_bits"] = useful
            costs["delivered_bits_per_wire_byte"] = None if costs["wire"] == 0 else useful/costs["wire"]
            costs["delivered_bits_per_wire_envelope_unshared_context_byte"] = useful/costs["wire_plus_envelope_plus_unshared_context"]
    errors = [row["case_id"] + ":" + row["role"] for row in rows if row["errors"]]
    if any(row.get("model_process_attempted") is True for row in rows) and inventory_verified is not True:
        errors.append("attempted campaign lacks required public-only receiver installation inventory")
    if terminal_campaign and campaign.get("exact_deliveries") != count:
        errors.append("campaign exact_deliveries differs from independently counted saved evidence")
    incomplete = not terminal_campaign or any(row["classification"] in {"unknown","not_attempted","dependency_blocked"} for row in rows)
    count_rows = [row["command_count_scope"] for row in rows if "command_count_scope" in row]
    totals = {"scope":"THIS ending-v2 pilot only; previous claimed/recorded STEP gaps are neither changed nor reconciled",
              "endpoints_with_count_evidence":len(count_rows),"planned_endpoints":12}
    for name in ("PROMPT_source_derived","STEP_journal_committed","STEP_prepared_uncommitted_unknown",
                 "STEP_returned_core_count","DIST_recorded_lower_bound","TOKENIZE_source_derived_total",
                 "TEXT_source_derived_total","TOKENIZE_source_derived_lower_bound","TEXT_source_derived_lower_bound"):
        known = [row[name] for row in count_rows if row[name] is not None]
        totals[name] = {"known_subtotal":sum(known),"endpoints_with_known_value":len(known),
                        "complete_12_endpoint_total":sum(known) if len(known) == 12 else None}
    deltas = [row["STEP_returned_minus_journal"] for row in count_rows if row["STEP_returned_minus_journal"] is not None]
    totals["STEP_returned_minus_journal_matched_endpoint_subtotal"] = sum(deltas)
    totals["STEP_gap_endpoints_with_both_counts"] = len(deltas)
    return {"schema": ID + "-saved-evidence-audit-v1", "status": "inconsistency" if errors else ("partial_unknown" if incomplete else "pass_saved_evidence"),
            "planned_senders":8,"planned_receivers":4,"planned_endpoints":12,"quality_denominator":8,
            "payload_delivery_denominator":4,"exact_deliveries_from_saved_evidence":count,"endpoints":rows,
            "preparation_tokenizations":preparations,"receiver_inventory_verified_against_frozen_public_inputs":inventory_verified,
            "journal_derived_committed_STEP_observations":sum(row.get("evidence",{}).get("committed_tokens",0) for row in rows),
            "new_pilot_only_command_totals":totals,
            "worker_declared_model_process_attempts":sum(row.get("model_process_attempted") is True for row in rows),
            "errors":errors,"model_calls":0,"native_tokenizer_calls":0,"native_decode_calls":0,
            "source_fidelity_evaluated":False,"tail_seconds":None,
            "limitations":["Replay is conditional on recorded eligibility/weights, not an independent model/tokenizer law reconstruction.",
                "Canonical token-to-bytes mapping remains runtime evidence; no independent tokenizer mapping check is performed.",
                "AEAD/frame comparison is an evaluator check using the frozen public fixture; it is not a receiver input.",
                "Git provenance requires actual operator fetch; this script checks readback hash coverage and exact bytes.",
                "No grammatical, factual truth, cross-machine or steganographic-security claim.",
                "No native-command telemetry inferred from journal STEP counts; only saved committed observations are counted."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--git-readback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "new output path required; no overwrite")
    receipt_raw = args.git_readback.read_bytes()
    inputs = Inputs(args.root, json.loads(receipt_raw))
    require(not args.output.resolve().is_relative_to(inputs.root), "audit output must be outside read-only restoration root")
    require(Path(__file__).resolve() == inputs.path(SCRIPT).resolve(), "run the Git-restored audit source")
    inputs.raw(SCRIPT)
    closure_raw = inputs.raw(EXP + "/source-closure.json")
    require(digest(closure_raw) == CLOSURE_SHA, "reviewed source closure differs")
    for path, expected in json.loads(closure_raw)["files_sha256"].items():
        require(digest(inputs.raw(path)) == expected, "frozen source bytes differ: " + path)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(inputs.root / "src"))
    core = importlib.import_module("synsteg_audit.framed_interval_public_ending_v2")
    interval = importlib.import_module("synsteg_audit.integer_interval_channel_v1")
    for module in (core, interval):
        require(Path(module.__file__).resolve().is_relative_to(inputs.root / "src"), "import escaped restored source")
    result = run(inputs, core, interval)
    result.update(actual_git_restoration_commit_declared=inputs.receipt["commit"],
                  git_readback_receipt_sha256=digest(receipt_raw), audited_inputs=inputs.reads,
                  audit_source_sha256=inputs.reads[SCRIPT]["sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"status":result["status"],"endpoints":len(result["endpoints"]),
                      "exact_deliveries":result["exact_deliveries_from_saved_evidence"],"native_calls":0}))
    return 2 if result["status"] == "inconsistency" else 0


if __name__ == "__main__":
    sys.exit(main())
