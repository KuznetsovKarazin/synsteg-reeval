#!/usr/bin/env python3
"""Posthoc ALL4 capacity audit from immutable saved journals; no model imports."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

ID = "framed-interval-llm-v1-posthoc-capacity-audit-v1"
RUNS = ("station-stego", "station-ordinary", "library-stego", "library-ordinary")


def canonical(x):
    return json.dumps(x, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw): return hashlib.sha256(raw).hexdigest()


def stats(values):
    return None if not values else {"count": len(values), "min": min(values),
        "mean": math.fsum(values)/len(values), "median": statistics.median(values), "max": max(values)}


def entropy(probabilities): return -math.fsum(p*math.log2(p) for p in probabilities if p > 0)


def committed(path):
    """Independent streaming check of saved prepared/commit hash chains."""
    previous, index, pending, wire, ids = "0"*64, 0, None, b"", []
    with path.open("rb") as stream:
        for sequence, raw in enumerate(stream):
            assert raw.endswith(b"\n"), "incomplete final journal"
            row = json.loads(raw); claimed = row.pop("event_sha256")
            assert row["sequence"] == sequence and row["previous_sha256"] == previous
            assert digest(canonical(row)) == claimed
            previous = claimed
            if row["kind"] == "prepared":
                assert pending is None and row["token_index"] == index
                assert row["wire_prefix_sha256"] == digest(wire)
                trace = row["trace"]
                assert trace["step"] == index and trace["prefix_ids"] == ids
                assert trace["observed_token"] == row["token_id"]
                assert trace["prefix_sha256"] == digest(wire)
                pending = (claimed, row["token_id"], bytes.fromhex(row["suffix_hex"]), trace)
            elif row["kind"] == "committed":
                assert pending is not None and row["prepared_sha256"] == pending[0]
                assert row["token_index"] == index and row["token_id"] == pending[1]
                wire += pending[2]; ids.append(pending[1])
                assert row["wire_after_sha256"] == digest(wire)
                yield pending[3]
                pending = None; index += 1
            else: raise ValueError("terminal unknown/error journal needs separate failure report")
    assert pending is None


def independent_transition(before, law, selected):
    low, high, pending = before["low"], before["high"], before["pending"]
    R, total, cumulative = high-low+1, sum(w for _, w in law), 0
    bins = []
    for token, weight in law:
        left = low + R*cumulative//total; cumulative += weight
        right = low + R*cumulative//total-1
        bins.append({"token_id": token, "low": left, "high": right, "width": right-left+1})
    chosen = next(row for row in bins if row["token_id"] == selected)
    low, high = chosen["low"], chosen["high"]
    bits, shifts = [], 0
    while True:
        if high < 1 << 31:
            bits += [0]+[1]*pending; pending = 0
        elif low >= 1 << 31:
            bits += [1]+[0]*pending; pending = 0; low -= 1 << 31; high -= 1 << 31
        elif low >= 1 << 30 and high < 3*(1 << 30):
            pending += 1; low -= 1 << 30; high -= 1 << 30
        else: break
        low *= 2; high = 2*high+1; shifts += 1
    after = {"low": low, "high": high, "pending": pending,
             "symbols": before["symbols"]+1, "shift_count": before["shift_count"]+shifts}
    return bins, after, bits


def one_endpoint(directory, run_id, role, inputs):
    result = json.loads((directory/"worker-result.json").read_bytes())
    envelope = json.loads((directory/"envelope.json").read_bytes())
    core = result["core"]
    rows, trace_fingerprints, reasons, reason_weights = [], [], Counter(), Counter()
    previous = {"low": 0, "high": (1 << 32)-1, "pending": 0, "symbols": 0, "shift_count": 0}
    resolved, first = 0, None
    for t in committed(directory/"token-journal.jsonl"):
        assert t["interval_before"] == previous
        admission = t["admission"]; verdicts = admission["verdicts"]
        base = {row["token_id"]: row["weight"] for row in verdicts}
        raw = t["base_law_evidence"]["float32_logits_and_integer_weights"]
        assert base == {row["token_id"]: row["weight"] for row in raw}
        assert len(raw) == 50 and sum(base.values()) == 1 << 24
        accepted = {row["token_id"]: row["weight"] for row in verdicts if row["reason"] == "accepted"}
        W, S = sum(base.values()), sum(accepted.values())
        assert S > 0 and dict(t["ordered_integer_law"]) == accepted
        assert dict(admission["conditioned_weights"]) == accepted
        assert admission["input_weight_total"] == W and admission["retained_weight_total"] == S
        assert admission["removed_weight_total"] == W-S
        assert Fraction(**admission["removed_mass"]) == Fraction(W-S, W)
        bins, after, bits = independent_transition(previous, t["ordered_integer_law"], t["observed_token"])
        assert bins == t["integer_bins"] and after == t["interval_after"] and bits == t["emitted_bits"]
        R = previous["high"]-previous["low"]+1
        widths = {row["token_id"]: row["width"] for row in bins}
        tv = Fraction(sum(abs(widths[token]*S-weight*R) for token, weight in accepted.items()), 2*R*S)
        assert tv == Fraction(**t["bin_law_vs_weights_TV"])
        combined_tv = Fraction(sum(abs(widths.get(token, 0)*W-weight*R) for token, weight in base.items()), 2*R*W)
        logits = [struct.unpack(">f", bytes.fromhex(row["logit_hex"]))[0] for row in raw]
        m = max(logits); float_weights = [math.exp((x-m)/0.8) for x in logits]
        normalizer = math.fsum(float_weights)
        quantization_tv = math.fsum(abs(row["weight"]/W-p/normalizer) for row,p in zip(raw,float_weights))/2
        for row in verdicts:
            reasons[row["reason"]] += 1; reason_weights[row["reason"]] += row["weight"]
        resolved += len(bits)
        assert resolved+after["pending"] == after["shift_count"]
        if first is None and resolved >= 272: first = after["symbols"]
        selected = t["observed_token"]
        row = {"run_id": run_id, "role": role, "token_number": after["symbols"],
            "accepted_candidates": len(accepted), "positive_candidates": sum(w > 0 for w in base.values()),
            "retained_mass": S/W, "removed_mass": (W-S)/W,
            "base_entropy_bits": entropy(w/W for w in base.values()),
            "conditioned_entropy_bits": entropy(w/S for w in accepted.values()),
            "bin_entropy_bits": entropy(width/R for width in widths.values()),
            "float_to_integer_top50_TV": quantization_tv, "integer_bin_TV": float(tv),
            "base_to_final_bin_TV": float(combined_tv),
            "selected_base_surprise_bits": -math.log2(base[selected]/W),
            "selected_conditioned_surprise_bits": -math.log2(accepted[selected]/S),
            "selected_bin_surprise_bits": -math.log2(widths[selected]/R),
            "max_conditioned_probability": max(accepted.values())/S,
            "resolved_cumulative": resolved, "consumed_cumulative": 32+after["shift_count"],
            "pending_bits": after["pending"], "resolved_this_token": len(bits),
            "prefix_ids_sha256": digest(canonical(t["prefix_ids"])),
            "interval_before_sha256": digest(canonical(previous)),
            "base_law_sha256": digest(canonical(sorted(base.items()))),
            "ordered_law_sha256": digest(canonical(t["ordered_integer_law"])),
            "bins_sha256": digest(canonical(bins))}
        rows.append(row); trace_fingerprints.append(digest(canonical(t))); previous = after
    assert result["status"] == "returned" and result["run_id"] == run_id
    assert len(rows) == core.get("emitted_tokens", core.get("processed_tokens"))
    assert core["resolved_bits"] == resolved and core["pending_bits"] == previous["pending"]
    if role == "sender": assert core["consumed_bits"] == 32+previous["shift_count"]
    if envelope["arm"] == "stego": assert core["first_frame_resolved_token"] == first
    else: assert core["first_frame_resolved_token"] is None
    L, P, C = resolved, previous["pending"], 32+previous["shift_count"]
    delta = 32-math.log2(previous["high"]-previous["low"]+1)
    surprise = math.fsum(r["selected_bin_surprise_bits"] for r in rows)
    assert abs(surprise-(L+P+delta)) < 1e-8 and C == L+P+32
    for name in ("token-journal.jsonl", "worker-result.json", "envelope.json", "committed-wire.bin"):
        path = directory/name; inputs[str(path)] = {"bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}
    wire = (directory/"committed-wire.bin").read_bytes()
    if role == "sender":
        assert digest(wire) == core["wire_sha256"] and len(wire) == core["wire_bytes"]
    else:
        request_path = directory/"request.json"
        request = json.loads(request_path.read_bytes())
        assert digest(wire) == request["wire"]["sha256"]
        assert Path(request["wire"]["path"]).read_bytes() == wire
        inputs[str(request_path)] = {"bytes": request_path.stat().st_size, "sha256": digest(request_path.read_bytes())}
    fields = ("accepted_candidates", "positive_candidates", "retained_mass", "removed_mass", "base_entropy_bits",
        "conditioned_entropy_bits", "bin_entropy_bits", "integer_bin_TV", "base_to_final_bin_TV", "float_to_integer_top50_TV")
    summary = {"run_id": run_id, "role": role, "status": core["status"], "tokens": len(rows),
        "resolved_bits": L, "consumed_bits_including_lookahead": C, "pending_bits": P,
        "lookahead_bits": 32, "final_interval_fractional_uncertainty_bits": delta,
        "bin_path_surprise_bits": surprise, "first_272_bits_resolved_token_diagnostic": first,
        "frame_first_resolved_token_recorded": core["first_frame_resolved_token"],
        "frame_bits_required": 272 if envelope["arm"] == "stego" else None,
        "frame_bit_deficit": max(0,272-L) if envelope["arm"] == "stego" else None,
        "tail_bits_read": max(0,C-272) if envelope["arm"] == "stego" else None,
        "tail_bits_resolved": max(0,L-272) if envelope["arm"] == "stego" else None,
        "authenticated_delivery": core["authenticated_delivery"], "wire_bytes": len(wire),
        "resolved_bits_per_token": L/len(rows), "resolved_bits_per_wire_byte": L/len(wire),
        "envelope_canonical_bytes": len(canonical(envelope)), "envelope_stored_bytes": (directory/"envelope.json").stat().st_size,
        "candidate_verdict_counts": dict(reasons), "candidate_weight_sums": dict(reason_weights),
        "candidate_denominator": sum(reasons.values()), "probability_mass_denominator": len(rows)*(1<<24),
        "singleton_support_steps": sum(r["accepted_candidates"]==1 for r in rows),
        "probability_above_0_99_steps": sum(r["max_conditioned_probability"]>0.99 for r in rows),
        "zero_new_resolved_bits_steps": sum(r["resolved_this_token"]==0 for r in rows),
        "descriptive_statistics": {name: stats([r[name] for r in rows]) for name in fields},
        "sum_base_entropy_bits": math.fsum(r["base_entropy_bits"] for r in rows),
        "sum_conditioned_entropy_bits": math.fsum(r["conditioned_entropy_bits"] for r in rows),
        "sum_base_selected_surprise_bits": math.fsum(r["selected_base_surprise_bits"] for r in rows),
        "sum_conditioned_selected_surprise_bits": math.fsum(r["selected_conditioned_surprise_bits"] for r in rows),
        "all_committed_trace_sha256": digest(canonical(trace_fingerprints))}
    summary["blocks_128_tokens"] = [{"token_start": start+1, "token_end": min(start+128,len(rows)),
        "resolved_bits": sum(r["resolved_this_token"] for r in rows[start:start+128]),
        "mean_retained_mass": statistics.mean(r["retained_mass"] for r in rows[start:start+128]),
        "mean_conditioned_entropy": statistics.mean(r["conditioned_entropy_bits"] for r in rows[start:start+128])}
        for start in range(0,len(rows),128)]
    context = {k:v for k,v in envelope.items() if k not in {"arm", "nonce_hex", "ordinary_seed_hex"}}
    summary["ordering_context_sha256"] = digest(canonical(context))
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.campaign/"campaign-status.json").read_bytes())
    assert status["state"] == "finished_all_planned_outcomes", "wait for ALL4 final outcomes"
    assert tuple(case["run_id"] for case in status["cases"]) == RUNS
    assert not args.output.exists(), "analysis output must be new"
    inputs, senders, receivers, token_rows, matching = {}, [], [], [], []
    for run in RUNS:
        summary, rows = one_endpoint(args.campaign/run/"sender", run, "sender", inputs)
        senders.append(summary); token_rows += rows
        if run.endswith("-stego"):
            receiver, rr = one_endpoint(args.campaign/run/"receiver", run, "receiver", inputs)
            assert receiver["all_committed_trace_sha256"] == summary["all_committed_trace_sha256"]
            receivers.append(receiver)
    for prompt in ("station", "library"):
        a,b = [r for r in senders if r["run_id"].startswith(prompt)]
        assert a["ordering_context_sha256"] == b["ordering_context_sha256"]
        aa,bb = [[r for r in token_rows if r["run_id"]==run] for run in (a["run_id"],b["run_id"])]
        common = [(x,y) for x,y in zip(aa,bb) if x["prefix_ids_sha256"]==y["prefix_ids_sha256"]]
        same_state = [(x,y) for x,y in common if x["interval_before_sha256"]==y["interval_before_sha256"]]
        assert all(x["base_law_sha256"]==y["base_law_sha256"] and x["ordered_law_sha256"]==y["ordered_law_sha256"] for x,y in common)
        assert all(x["bins_sha256"]==y["bins_sha256"] for x,y in same_state)
        matching.append({"prompt":prompt,"same_ordering_context":True,"common_observed_prefixes":len(common),
            "same_interval_state_prefixes":len(same_state),"matching_bins_at_same_prefix_and_state":True,
            "scope":"observed common histories only; trajectories after divergence are not counterfactual controls"})
    for name in ("campaign-status.json", "campaign-manifest.json"):
        p=args.campaign/name;inputs[str(p)]={"bytes":p.stat().st_size,"sha256":digest(p.read_bytes())}
    summary={"analysis_id":ID,"analysis_type":"posthoc; all four reported outcomes were known before this analysis",
        "generation_denominator":4,"fresh_receiver_denominator":2,"authenticated_deliveries":status["authenticated_payload_deliveries"],
        "senders":senders,"fresh_receivers":receivers,"ordinary_matching":matching,
        "overhead":{"payload_bits":128,"encrypted_length_bits":16,"AEAD_tag_bits":128,"frame_bits":272,
            "external_nonce_bits_per_message":96,"shared_key_bits":256,"public_resources_reusable":True,
            "nonce_in_envelope_already_do_not_double_count":True,"lookahead_32_is_buffering_not_32_bits_lost_against_path_surprise":True},
        "scope":["No neural or tokenizer call; reported eligibility reasons are saved evidence, not recomputed tokenizer predicates.",
            "Mass/TV reference is masked top50 integer law; full-softmax mask/topk mass remains unknown.",
            "Entropy summaries are along observed prefixes, not counterfactual generated paths or delivered payload.",
            "Mean conditional TV is not whole-text distinguishability or security.",
            "Consumed bits include32 lookahead+pending. Only authenticated full-wire recovery counts as payload delivery."],
        "checks":{"all_journal_chains":True,"all_integer_bins_and_transitions":True,"all_mass_and_bin_TV":True,
            "all_bit_accounting":True,"sender_receiver_trace_equality":True,"observed_conditional_ordinary_matching":True}}
    args.output.mkdir(parents=True)
    for name,obj in (("summary.json",summary),("inputs.json",inputs)):
        (args.output/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
    with (args.output/"per-token.jsonl").open("x") as stream:
        for row in token_rows:stream.write(json.dumps(row,sort_keys=True,separators=(",", ":"))+"\n")
    print(json.dumps({"status":"passed","analysis_id":ID,"senders":len(senders),"receivers":len(receivers),"tokens":len(token_rows)}))


if __name__ == "__main__": main()
