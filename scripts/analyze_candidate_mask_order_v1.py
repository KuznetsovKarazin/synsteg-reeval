#!/usr/bin/env python3
"""Pure recorded-prefix counterfactual, never an alternative model trajectory.

No model/tokenizer/native adapter is imported. The frozen schema parser is
reused explicitly and hash checked; this is not an independent parser audit.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import struct
import sys
import types

ID = "author-candidate-mask-order-v1"
TOTAL = 1 << 24
TEMPERATURE = 0.7
SCHEMA_PATH = "src/synsteg_audit/author_candidate_schema_v3.py"
SCHEMA_SHA = "cb17dd1780f5ee737e6b53454c0c7a911eb7867a9c2d2779f82790201d0b695f"
RUN = "results/author-candidate-local-native-v3/native-diagnostic-v3"
REQUESTS = "experiments/author-candidate-format-v2/requests.json"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    def bad(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def regular_bytes(path):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode), "nonregular input: " + str(path))
    raw = path.read_bytes()
    after = path.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(identity(before) == identity(after) and len(raw) == after.st_size,
         "input changed during read")
    return raw


def decode_rows(rows):
    need(type(rows) is list and 1 <= len(rows) <= 50, "one to fifty rows required")
    decoded = []
    ids = set()
    for row in rows:
        need(type(row) is dict and set(row) == {"token_id", "logit_hex"}, "closed logit row")
        token, encoded = row["token_id"], row["logit_hex"]
        need(type(token) is int and 0 <= token < 1 << 31 and token not in ids,
             "strict unique nonnegative token IDs required")
        need(type(encoded) is str and re.fullmatch("[0-9a-f]{8}", encoded) is not None,
             "exact network-order float32 hex required")
        value, = struct.unpack("!f", bytes.fromhex(encoded))
        need(math.isfinite(value), "finite logits required")
        decoded.append((value, token, encoded))
        ids.add(token)
    return sorted(decoded, key=lambda x: (-x[0], x[1]))


def quantize(rows, *, total=TOTAL, temperature=TEMPERATURE):
    """Largest remainder, logits desc/ID asc; remainder ties by token ID.

    Binary64 exp/fsum operation order matches the frozen scalar quantizer.
    Center on the largest logit *in the supplied support*. Underflow of other
    terms is recorded; the maximum has exp(0)=1, so nonempty finite support
    always has a nonzero normalizer. No per-token mass floor is introduced.
    """
    need(type(total) is int and 1 <= total <= TOTAL, "positive integer total required")
    need(type(temperature) is float and math.isfinite(temperature) and temperature > 0,
         "positive finite float temperature required")
    ordered = decode_rows(rows)
    exponential = [math.exp((x[0] - ordered[0][0]) / temperature) for x in ordered]
    normalizer = math.fsum(exponential)
    need(math.isfinite(normalizer) and normalizer >= 1, "invalid normalizer")
    ideal = [x / normalizer * total for x in exponential]
    masses = [math.floor(x) for x in ideal]
    missing = total - sum(masses)
    need(0 <= missing <= len(masses), "largest remainder rounding outside contract")
    remainder_order = sorted(range(len(masses)),
                             key=lambda i: (-(ideal[i] - masses[i]), ordered[i][1]))
    for i in remainder_order[:missing]:
        masses[i] += 1
    need(sum(masses) == total and any(masses), "mass conservation failed")
    return {
        "rows": [{"token_id": row[1], "logit_hex": row[2], "mass": mass}
                 for row, mass in zip(ordered, masses)],
        "exp_underflow_ids": [row[1] for row, exp in zip(ordered, exponential) if exp == 0],
    }


def fraction_json(value):
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator,
            "decimal_approximation": float(value)}


def compare_probe(weights, pieces, raw, words, classify, *, max_bytes=262144):
    need(type(weights) is list and len(weights) == 50, "complete top50 weights required")
    need(type(pieces) is list and len(pieces) == 50, "complete top50 pieces required")
    logits = []
    for row in weights:
        need(type(row) is dict and set(row) == {"token_id", "logit_hex", "weight"},
             "closed weight row")
        need(type(row["weight"]) is int and row["weight"] >= 0, "strict nonnegative mass")
        logits.append({"token_id": row["token_id"], "logit_hex": row["logit_hex"]})
    base = quantize(logits)
    expected = [{"token_id": r["token_id"], "logit_hex": r["logit_hex"], "weight": r["mass"]}
                for r in base["rows"]]
    need(canonical(weights) == canonical(expected), "saved base quantization/order differs")
    need(type(raw) is bytes and type(max_bytes) is int and 1 <= max_bytes <= 262144,
         "invalid prefix bytes or parser budget")
    prefix_state = classify(raw, words, max_bytes=max_bytes)["state"]
    need(prefix_state == "incomplete", "counterfactual requires a factual incomplete prefix")
    all_rows = []
    for w, piece in zip(weights, pieces):
        need(type(piece) is dict and set(piece) == {"token_id", "piece_hex"}
             and type(piece["token_id"]) is int and piece["token_id"] == w["token_id"],
             "piece row identity/order differs")
        encoded = piece["piece_hex"]
        need(type(encoded) is str and re.fullmatch(r"(?:[0-9a-f]{2})+", encoded) is not None
             and len(encoded) <= 2 * (262144 + 65536), "invalid or oversized piece")
        value = bytes.fromhex(encoded)
        state = classify(raw + value, words, max_bytes=max_bytes)["state"]
        need(state in {"incomplete", "complete", "invalid_prefix", "resource_refusal"},
             "unrecognized parser result")
        all_rows.append({**w, **piece, "prefix_state": state})
    # A budget refusal is never silently treated as syntactic ineligibility.
    if any(r["prefix_state"] == "resource_refusal" for r in all_rows):
        return {"state": "resource_refusal", "reason": "candidate parser byte budget",
                "rows": all_rows, "total_variation": None}
    eligible = [r for r in all_rows if r["prefix_state"] in {"incomplete", "complete"}]
    old_sum = sum(r["weight"] for r in eligible)
    alternative = quantize([{k: r[k] for k in ("token_id", "logit_hex")} for r in eligible]) if eligible else None
    new_masses = {r["token_id"]: r["mass"] for r in alternative["rows"]} if alternative else {}
    for row in all_rows:
        row["alternative_mass"] = new_masses.get(row["token_id"], 0)
    variation = (sum((abs(Fraction(r["weight"], old_sum)
                          - Fraction(new_masses[r["token_id"]], TOTAL))
                      for r in eligible), Fraction()) / 2 if old_sum else None)
    return {"state": "ready" if eligible else "empty_schema_support",
            "old_state": "ready" if old_sum else "empty_support",
            "eligible_count": len(eligible), "old_retained_mass": old_sum,
            "old_positive_eligible_count": sum(r["weight"] > 0 for r in eligible),
            "new_positive_eligible_count": sum(x > 0 for x in new_masses.values()),
            "eligible_zero_old_mass_ids": [r["token_id"] for r in eligible if r["weight"] == 0],
            "recovered_positive_ids": [r["token_id"] for r in eligible if r["weight"] == 0 and new_masses[r["token_id"]] > 0],
            "lost_positive_ids": [r["token_id"] for r in eligible if r["weight"] > 0 and new_masses[r["token_id"]] == 0],
            "base_exp_underflow_ids": base["exp_underflow_ids"],
            "alternative_exp_underflow_ids": alternative["exp_underflow_ids"] if alternative else [],
            "alternative_integer_total": TOTAL if eligible else 0,
            "total_variation": fraction_json(variation), "rows": all_rows}


def load_inputs(root, manifest):
    need(manifest.get("experiment_id") == ID, "wrong experiment manifest")
    inputs = {}
    for entry in manifest["inputs"]:
        path = entry["path"]
        need(type(path) is str and not Path(path).is_absolute() and ".." not in Path(path).parts,
             "unsafe input path")
        need(path not in inputs, "duplicate input path")
        raw = regular_bytes(root / path)
        need(len(raw) == entry["bytes"] and digest(raw) == entry["sha256"], "pinned input mismatch: " + path)
        inputs[path] = raw
    need(digest(inputs[SCHEMA_PATH]) == SCHEMA_SHA, "frozen parser hash differs")
    return inputs


def load_parser(raw):
    need(digest(raw) == SCHEMA_SHA, "frozen parser bytes differ")
    module = types.ModuleType("mask_order_frozen_schema")
    # Execute exactly the already hash-verified bytes. No second filesystem read,
    # package __init__, model module, or tokenizer import occurs.
    exec(compile(raw, SCHEMA_PATH, "exec"), module.__dict__)
    return module.classify_prefix


def analyze(root, manifest, manifest_raw):
    inputs = load_inputs(root, manifest)
    classify = load_parser(inputs[SCHEMA_PATH])
    requests = strict_json(inputs[REQUESTS])["requests"]
    need(len(requests) == 2, "complete fixed two-request denominator")
    rows = []
    attempt_summaries = []
    for chunk in range(2):
        request = requests[chunk]
        need(request["chunk_order"] == chunk, "request order")
        name = f"chunk{chunk}-attempt0"
        prefix = f"{RUN}/{name}/sealed/"
        binding = strict_json(inputs[prefix + "binding.json"])
        need(binding["request_sha256"] == request["request_sha256"], "bound request differs")
        terminal = strict_json(inputs[prefix + "terminal.json"])
        journal = inputs[prefix + "journal.jsonl"]
        need(journal.endswith(b"\n"), "torn journal")
        need(terminal["files"]["journal.jsonl"]["sha256"] == digest(journal), "terminal journal hash")
        raw = b""; ids = []; pending = None; previous = "0" * 64; count = 0; terminal_event = None
        for sequence, line in enumerate(journal.splitlines()):
            row = strict_json(line)
            need(canonical(row) == line, "noncanonical journal")
            need(row["sequence"] == sequence and row["previous_sha256"] == previous, "journal order")
            body = {k: row[k] for k in ("sequence", "previous_sha256", "event")}
            need(digest(canonical(body)) == row["event_sha256"], "journal chain digest")
            event = row["event"]
            kind = event["kind"]
            need(terminal_event is None, "event after terminal")
            if kind == "distribution_committed":
                need(pending is None, "unconsumed probe")
                evidence = event["sample_evidence"]
                need(evidence["prefix_ids"] == ids and evidence["prefix_raw_sha256"] == digest(raw), "factual prefix binding")
                need(evidence["request_sha256"] == request["request_sha256"]
                     and evidence["integer_total"] == TOTAL and evidence["temperature"] == TEMPERATURE,
                     "request/quantizer contract differs")
                comparison = compare_probe(evidence["weights"], evidence["contextual_pieces"], raw,
                                           request["words"], classify)
                if comparison["state"] != "resource_refusal":
                    old = evidence["conditional_law"]
                    need(old["state"] == comparison["old_state"] and old["retained_mass"] == comparison["old_retained_mass"], "replayed old law differs")
                    accepted = {r["token_id"]: r["mass"] for r in old["accepted"]}
                    expected = {r["token_id"]: r["weight"] for r in comparison["rows"]
                                if r["prefix_state"] in {"complete", "incomplete"} and r["weight"] > 0}
                    need(accepted == expected, "replayed old support differs")
                rows.append({"schema": ID + "-probe", "chunk_order": chunk,
                             "probe_order": count, "factual_step_index": len(ids),
                             "factual_prefix_sha256": digest(raw), "factual_prefix_bytes": len(raw),
                             "request_sha256": request["request_sha256"],
                             "factual_old_state": evidence["conditional_law"]["state"],
                             "journal_sequence": sequence, "journal_event_sha256": row["event_sha256"],
                             **comparison})
                count += 1; pending = evidence
            elif kind == "step_committed":
                need(pending is not None and pending["selection"]["state"] == "selected", "STEP without factual selection")
                selection = pending["selection"]
                ids.append(selection["token_id"])
                raw += bytes.fromhex(selection["piece_hex"])
                pending = None
            elif kind == "trace_complete":
                need(event["output_ids"] == ids and event["response_sha256"] == digest(raw)
                     and event["response_bytes"] == len(raw), "factual terminal differs")
                terminal_event = event
            previous = row["event_sha256"]
        need(terminal_event is not None and terminal["journal_head_sha256"] == previous, "terminal journal head")
        need(raw == inputs[prefix + "response.raw"], "factual response mismatch")
        expected = manifest["expected_attempts"][chunk]
        need(count == expected["probes"] and len(ids) == expected["steps"], "attempt denominator mismatch")
        attempt_summaries.append({"chunk_order": chunk, "factual_probes": count, "factual_steps": len(ids),
                                  "factual_stop": terminal_event["stop_label"]})
    need(len(rows) == 361 and sum(len(r["rows"]) for r in rows) == 18050, "full probe denominator mismatch")
    comparable = [r for r in rows if r["total_variation"] is not None]
    tv = sorted(Fraction(r["total_variation"]["numerator"], r["total_variation"]["denominator"]) for r in comparable)
    ready = [r for r in rows if r["state"] == "ready"]
    summary = {"schema": ID + "-summary", "experiment_id": ID,
               "input_manifest_sha256": digest(manifest_raw),
               "inputs": manifest["inputs"], "attempts": attempt_summaries,
               "observed_attempts": 2, "full_cover_chunks": 109, "outside_diagnostic_chunks": 107,
               "factual_probes": len(rows), "recorded_rows": sum(len(r["rows"]) for r in rows),
               "alternative_ready_probes": len(ready),
               "alternative_empty_schema_support": sum(r["state"] == "empty_schema_support" for r in rows),
               "alternative_resource_refusals": sum(r["state"] == "resource_refusal" for r in rows),
               "old_empty_support_probes": sum(r["factual_old_state"] == "empty_support" for r in rows),
               "old_empty_new_ready_probes": sum(r["factual_old_state"] == "empty_support" and r["state"] == "ready" for r in rows),
               "probes_recovering_positive_ids": sum(bool(r.get("recovered_positive_ids")) for r in rows),
               "recovered_positive_id_occurrences": sum(len(r.get("recovered_positive_ids", [])) for r in rows),
               "lost_positive_id_occurrences": sum(len(r.get("lost_positive_ids", [])) for r in rows),
               "base_exp_underflow_occurrences": sum(len(r.get("base_exp_underflow_ids", [])) for r in rows),
               "alternative_exp_underflow_occurrences": sum(len(r.get("alternative_exp_underflow_ids", [])) for r in rows),
               "total_variation_comparable_probes": len(tv),
               "exactly_equal_conditional_laws": sum(t == 0 for t in tv),
               "total_variation_mean": fraction_json(sum(tv, Fraction()) / len(tv)) if tv else None,
               "total_variation_max": fraction_json(max(tv)) if tv else None,
               "old_empty_details": [{k: v for k, v in r.items() if k != "rows"} for r in rows if r["factual_old_state"] == "empty_support"],
               "scope": {"native_calls": 0, "tokenizer_calls": 0, "new_generation_attempts": 0,
                         "factual_prefixes_only": True, "counterfactual_rollout_performed": False,
                         "full_vocabulary_mask_evaluated": False, "independent_parser": False,
                         "new_format_success_rate": None, "usable_candidate_rows": None,
                         "capacity_bits": None, "detectability_measured": False,
                         "hardware_provenance_attested": False},
               "limitations": ["All 361 probes are dependent observations on the two recorded factual trajectories.",
                               "Alternative token choices would change later logits; no counterfactual rollout or success rate follows.",
                               "Only the saved top50 support is available. Full-vocabulary grammar masking and pre-top-k masking remain unmeasured.",
                               "The frozen parser is reused; finite binary64 quantization may still give some eligible tokens zero mass.",
                               "A key selects available mass. It does not certify synonyms or embed an independent payload."]}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve output first; a failed analysis leaves an explicit failure, never
    # silently overwrites a result or promotes partial rows to success.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        manifest_raw = regular_bytes(args.input_manifest)
        rows, summary = analyze(args.root.resolve(), strict_json(manifest_raw), manifest_raw)
        rows_raw = b"".join(canonical(row) + b"\n" for row in rows)
        summary["probe_rows_sha256"] = digest(rows_raw)
        (args.output / "probe-comparisons.jsonl").write_bytes(rows_raw)
        (args.output / "summary.json").write_bytes(canonical(summary) + b"\n")
        print(json.dumps({"status": "complete", "factual_probes": len(rows),
                          "summary_sha256": digest(canonical(summary) + b"\n")}))
    except BaseException as exc:
        (args.output / "failure.json").write_bytes(canonical({"schema": ID + "-failure",
            "exception": type(exc).__name__, "message": str(exc),
            "native_calls": 0, "scientific_result_complete": False}) + b"\n")
        raise


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
