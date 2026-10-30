#!/usr/bin/env python3
"""Independent saved-metrics arithmetic. No project/decoder/model imports."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/framed-interval-bounded-v1-native-metrics"
CAMPAIGN = "results/framed-interval-bounded-v1/native-pilot-v1"
checks = 0
failures = []
evidence = {}


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        failures.append(label)


def raw(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    data = path.read_bytes()
    evidence[str(path.relative_to(ROOT))] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return data


def load(path):
    return json.loads(raw(path))


def equivalent(actual, expected):
    if type(expected) in (int, float) and type(actual) in (int, float):
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def descriptive(values):
    data = list(values)
    return {"n": len(data), "sum": sum(data), "mean": math.fsum(data)/len(data),
            "median": statistics.median(data), "min": min(data), "max": max(data)}


def verify_ratio(obj, numerator, denominator, label):
    check(obj.get("numerator") == numerator, label + ":numerator")
    check(obj.get("denominator") == denominator, label + ":denominator")
    check(equivalent(obj.get("value"), numerator/denominator), label + ":ratio")


def main():
    summary, rows, pairs, inputs, status = [
        load(OUT/name) for name in ("summary.json", "rows.json", "paired-costs.json", "inputs.json", "status.json")]
    collector = "scripts/collect_framed_interval_bounded_v1_native_metrics.py"
    check(hashlib.sha256(raw(collector)).hexdigest() == status["script_sha256"], "collector source identity")
    for name, descriptor in status["outputs"].items():
        data = raw(OUT/name)
        check({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} == descriptor,
              "saved metric output identity:" + name)
    for path, descriptor in inputs["files"].items():
        data = raw(path)
        check({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} == descriptor,
              "saved metric input identity:" + path)
    audit = load("results/framed-interval-bounded-v1-native-audit/independent-audit.json")
    process_check = load("results/framed-interval-bounded-v1-native-audit/process-completion-check.json")
    check(audit["status"] == "passed" and audit["failures"] == [], "global audit acceptance gate")
    check(audit["passed_checks"] == 89009, "recorded independent audit check count")
    check(audit["usable_delivery_numerator"] == audit["authenticated_deliveries_independently_verified"] == 8,
          "main audit usable numerator")
    check(process_check["status"] == "passed" and process_check["all24_worker_exit_zero"] is True,
          "process completion acceptance gate")
    protocol = load("experiments/framed-interval-bounded-v1/public-protocol.json")
    fixtures = load("experiments/framed-interval-bounded-v1/sender-fixtures.json")
    expected = [f"{p}-c{k:02d}-{arm}" for p in ("station", "library")
                for k in range(4) for arm in ("stego", "ordinary")]
    check([r["run_id"] for r in rows] == expected, "all16 rows fixed order")
    check([r["run_id"] for r in protocol["runs"]] == expected, "public16 fixed order")
    check(status["sender_rows"] == 16 and status["paired_rows"] == 8
          and len(pairs) == 8 and status["input_files"] == len(inputs["files"]) == 106, "status denominators")
    ledger = load(CAMPAIGN + "/campaign-status.json")
    check([r["run_id"] for r in ledger["cases"]] == expected, "ledger16 fixed order")
    cases = {row["run_id"]: row for row in ledger["cases"]}
    fixture_map = {row["run_id"]: row for row in fixtures["runs"]}
    recomputed, receiver_tokens, exit_zero, key_set = [], 0, 0, set()
    journal_steps = 0
    measures = ("sender_tokens", "sender_whitespace_words", "wire_bytes",
        "envelope_saved_json_bytes", "envelope_canonical_json_bytes", "wire_plus_canonical_envelope_bytes",
        "resolved_bits", "pending_bits", "sender_wall_seconds")
    for row in rows:
        run_id = row["run_id"]
        sender_base = CAMPAIGN + "/" + run_id + "/sender/"
        wire = raw(sender_base + "committed-wire.bin")
        envelope_raw = raw(sender_base + "envelope.json")
        envelope = json.loads(envelope_raw)
        compact = json.dumps(envelope, sort_keys=True, ensure_ascii=True,
                             separators=(",", ":"), allow_nan=False).encode("ascii")
        worker = load(sender_base + "worker-result.json")
        core = worker["core"]
        journal = [json.loads(line) for line in raw(sender_base + "token-journal.jsonl").splitlines()]
        count = sum(item["kind"] == "committed" for item in journal)
        journal_steps += count
        check(count == core["emitted_tokens"], "own committed journal count:" + run_id)
        check(worker == cases[run_id]["sender"]["worker"], "sender terminal record:" + run_id)
        check(cases[run_id]["sender"]["process"]["status"] == "exited"
              and cases[run_id]["sender"]["process"]["exit_code"] == 0, "sender clean exit:" + run_id)
        exit_zero += int(cases[run_id]["sender"]["process"]["exit_code"] == 0)
        key_set.add(load(sender_base + "request.json")["key_hex"])
        calculated = dict(row)
        calculated.update(sender_tokens=count, sender_whitespace_words=len(wire.decode("utf-8").split()),
            wire_bytes=len(wire), wire_sha256=hashlib.sha256(wire).hexdigest(),
            envelope_saved_json_bytes=len(envelope_raw), envelope_canonical_json_bytes=len(compact),
            wire_plus_canonical_envelope_bytes=len(wire)+len(compact),
            resolved_bits=core["resolved_bits"], consumed_bits=core["consumed_bits"],
            pending_bits=core["pending_bits"], sender_wall_seconds=cases[run_id]["sender"]["process"]["wall_seconds"])
        for field in (*measures, "wire_sha256", "consumed_bits"):
            check(equivalent(row[field], calculated[field]), "row arithmetic:" + run_id + ":" + field)
        check(row["consumed_bits"] == row["resolved_bits"] + row["pending_bits"] + 32,
              "lookahead is buffer:" + run_id)
        check(row["first_threshold_token"] == count and row["resolved_bits"] >= 272
              and count <= 768, "declared threshold stop counter:" + run_id)
        if row["arm"] == "stego":
            receiver_base = CAMPAIGN + "/" + run_id + "/receiver/"
            receiver = load(receiver_base + "worker-result.json")
            receiver_journal = [json.loads(line) for line in raw(receiver_base + "token-journal.jsonl").splitlines()]
            own_count = sum(item["kind"] == "committed" for item in receiver_journal)
            receiver_tokens += own_count
            journal_steps += own_count
            check(own_count == row["receiver_token_steps"] == count
                  == receiver["core"]["processed_tokens"], "fresh receiver committed count:" + run_id)
            payload = bytes.fromhex(receiver["core"]["payload_hex"])
            check(payload.hex() == fixture_map[run_id]["payload_hex"] and len(payload) == 16
                  and row["delivered_payload_bytes"] == 16, "posthoc expected payload comparison:" + run_id)
            check(row["authenticated_exact_delivery"] is True
                  and cases[run_id]["receiver"]["authenticated_delivery"] is True
                  and cases[run_id]["receiver"]["exact_expected_payload"] is True
                  and receiver["core"]["authenticated_delivery"] is True,
                  "all saved successful delivery flags:" + run_id)
            check(cases[run_id]["receiver"]["process"]["status"] == "exited"
                  and cases[run_id]["receiver"]["process"]["exit_code"] == 0, "receiver clean exit:" + run_id)
            exit_zero += int(cases[run_id]["receiver"]["process"]["exit_code"] == 0)
            verify_ratio(row["useful_bits_per_sender_token"], 128, count, run_id + ":token")
            verify_ratio(row["useful_bits_per_wire_byte"], 128, len(wire), run_id + ":wire")
        else:
            check(row["authenticated_exact_delivery"] is None
                  and row["delivered_payload_bytes"] is None
                  and row["receiver_token_steps"] == 0, "ordinary is not payload delivery:" + run_id)
        recomputed.append(calculated)
    groups = {"all16": recomputed, "stego8": [r for r in recomputed if r["arm"] == "stego"],
              "ordinary8": [r for r in recomputed if r["arm"] == "ordinary"]}
    for name, data in groups.items():
        for measure in measures:
            expected_stats = descriptive(row[measure] for row in data)
            for stat, value in expected_stats.items():
                check(equivalent(summary["descriptive_statistics"][name][measure][stat], value),
                      "descriptive:" + name + ":" + measure + ":" + stat)
    row_map = {r["run_id"]: r for r in recomputed}
    paired_measures = measures[:6]
    pair_order = [(p, f"c{k:02d}") for p in ("station", "library") for k in range(4)]
    check([(p["prompt_id"], p["config_id"]) for p in pairs] == pair_order, "all8 pair order")
    independent_deltas = {name: [] for name in paired_measures}
    for pair in pairs:
        s = row_map[pair["prompt_id"] + "-" + pair["config_id"] + "-stego"]
        o = row_map[pair["prompt_id"] + "-" + pair["config_id"] + "-ordinary"]
        check(pair["stego_run"] == s["run_id"] and pair["ordinary_run"] == o["run_id"], "paired identities")
        for name in paired_measures:
            expected_pair = {"stego": s[name], "ordinary": o[name], "stego_minus_ordinary": s[name]-o[name]}
            check(pair[name] == expected_pair, "paired cost:" + s["run_id"] + ":" + name)
            independent_deltas[name].append(s[name]-o[name])
    for name, data in independent_deltas.items():
        for stat, value in descriptive(data).items():
            check(equivalent(summary["paired_stego_minus_ordinary"][name][stat], value),
                  "paired descriptive:" + name + ":" + stat)
    t = summary["totals"]
    stego = groups["stego8"]
    for field, value in {
        "delivered_payload_bytes": 128, "delivered_payload_bits": 1024,
        "stego_sender_tokens": sum(r["sender_tokens"] for r in stego),
        "all_sender_tokens": sum(r["sender_tokens"] for r in recomputed),
        "ordinary_sender_tokens": sum(r["sender_tokens"] for r in groups["ordinary8"]),
        "receiver_token_steps": receiver_tokens,
        "all_committed_token_STEP_requests": journal_steps,
        "stego_wire_bytes": sum(r["wire_bytes"] for r in stego),
        "stego_canonical_envelope_bytes": sum(r["envelope_canonical_json_bytes"] for r in stego),
        "stego_saved_envelope_bytes": sum(r["envelope_saved_json_bytes"] for r in stego),
        "stego_wire_plus_canonical_envelope_bytes": sum(r["wire_plus_canonical_envelope_bytes"] for r in stego),
        "distinct_fixture_keys": len(key_set), "distinct_fixture_key_bytes": len(key_set)*32,
        "worker_exit_zero_count": exit_zero, "PROMPT_requests": 24,
        "nonce_bytes_all16": 16*12,
    }.items():
        check(t[field] == value, "aggregate:" + field)
    check(t["stego_sender_tokens"] == 2388 and t["stego_wire_bytes"] == 9876
          and t["stego_canonical_envelope_bytes"] == 20252 and journal_steps == 7330, "main reported totals")
    verify_ratio(t["useful_bits_per_stego_sender_token"], 1024, 2388, "allstego useful bits/token")
    verify_ratio(t["useful_bits_per_stego_wire_byte"], 1024, 9876, "allstego useful bits/wirebyte")
    verify_ratio(t["useful_payload_byte_fraction_of_stego_wire"], 128, 9876, "payload/wire fraction")
    verify_ratio(t["useful_payload_byte_fraction_of_wire_plus_canonical_envelope"], 128, 30128, "payload/allwire fraction")
    check(t["lower_level_graph_forward_invocations"] is None
          and t["shared_runtime_and_other_resources_bytes"] is None, "unknown totals remain null")
    check(summary["causal_comparison_to_fixed512_pilot_permitted"] is False, "no fixed512 causal claim")
    check(summary["quality"]["human_gold"] is False and summary["quality"]["validated_error_rate"] is None,
          "quality remains descriptive")
    limit = summary["deployment_binding_limitation"]
    check(limit["message_independent_public_binding"] is False
          and limit["arbitrary_message_deployment_established"] is False,
          "message-dependent activation limitation")
    check(all(summary[name] == status[name] == 0 for name in ("model_calls", "tokenizer_calls", "decoder_calls")),
          "arithmetic invokes no NN/tokenizer/decoder")
    report = {"schema": "framed-interval-bounded-v1-native-metrics-independent-review-v1",
        "status": "passed" if not failures else "failed",
        "checks": checks, "failures": failures, "evidence": evidence,
        "reviewer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reviewer_imports_from_project": 0, "model_calls": 0, "tokenizer_calls": 0, "decoder_calls": 0,
        "verified_counts": {"sender_rows": 16, "paired_rows": 8, "endpoint_journals": 24,
            "committed_STEP_rows": journal_steps, "payload_bytes": 128,
            "stego_tokens": 2388, "stego_wire_bytes": 9876, "stego_canonical_envelope_bytes": 20252},
        "scope": "Independent byte/word/envelope sizes, journal counts, pairing, descriptive arithmetic and saved acceptance flags. No model, tokenizer, interval or AEAD replay.",
        "limits": ["Prior main audit89009 acceptance gate is required; this arithmetic checker does not replace it.",
            "Fixture plaintext is read only for this separate posthoc reviewer, not supplied to receiver.",
            "This checker does not validate agent semantic annotation accuracy or establish causal/security/practical-deployment claims.",
            "Report prose was not yet available at review time; scope claims verified in saved metrics." ]}
    output = OUT / "independent-review.json"
    with output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "checks", "failures", "verified_counts")}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
