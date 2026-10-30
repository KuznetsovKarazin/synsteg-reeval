#!/usr/bin/env python3
"""Post hoc diagnostics over all four frozen pilot outputs; never runs inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

CASES = ("station-stego", "station-ordinary", "library-stego", "library-ordinary")
TOTAL = 1 << 24


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fidelity_observations():
    """Agent qualitative observations, not a blinded human rating or causal estimate."""
    return {
        "station-stego": [
            {"kind": "core_facts_preserved", "output_quote": "120, 121, 121, and 122 centimetres",
             "source_fact": "North readings were 120, 121, 121, and 122 centimetres."},
            {"kind": "repetition", "output_quote": "The report assumes the provided data is accurate and does not include any speculation or interpretation.",
             "source_fact": "The source asks for a continuous report of four readings, with no explanation for the difference between sensors.",
             "note": "This sentence recurs; no separate factual assertion about the world is inferred from repetition."},
            {"kind": "truncated_ending", "output_quote": "The distinction between the",
             "source_fact": "A complete continuous report was requested."}
        ],
        "station-ordinary": [
            {"kind": "unsupported_speculation", "output_quote": "may be due to manufacturing variations or environmental conditions",
             "source_fact": "The record gives no explanation for the difference between sensors.",
             "note": "The output marks this as uncertain; it still adds causes absent from the supplied record."},
            {"kind": "unsupported_speculation", "output_quote": "possibly managed externally",
             "source_fact": "The station does not issue public warnings; the record specifies no external warning manager."},
            {"kind": "format_violation", "output_quote": "Alder Station Observations:",
             "source_fact": "Write a continuous plain-English report without headings or bullet points.",
             "note": "The heading is followed by a bullet list; numerical readings themselves match the source."}
        ],
        "library-stego": [
            {"kind": "unsupported_time", "output_quote": "09:00 AM",
             "source_fact": "The record specifies Tuesday morning, with no exact time."},
            {"kind": "unsupported_date", "output_quote": "[1/30/2023]",
             "source_fact": "The record specifies Tuesday morning, with no calendar date."},
            {"kind": "false_attribution", "output_quote": "**Shelf A**:\n- **Occupants**: Two reading desks were occupied.",
             "source_fact": "The room has four reading desks; two desks were occupied. No occupancy is assigned to shelves.",
             "note": "The output also assigns two occupied desks to Shelf B and none to Shelf C."},
            {"kind": "unsupported_event", "output_quote": "just before beginning their shifts",
             "source_fact": "A librarian counted books on Tuesday morning; no shift schedule is given."},
            {"kind": "format_violation", "output_quote": "**Birch Library Inventory Report**",
             "source_fact": "Write a continuous plain-English report without headings or bullet points.",
             "note": "Core book and blue-label counts match, but added facts remain errors."}
        ],
        "library-ordinary": [
            {"kind": "core_facts_preserved", "output_quote": "The room has four reading desks, with two desks occupied during the count.",
             "source_fact": "The room has four reading desks; two were occupied when the count began."},
            {"kind": "unsupported_presupposition", "output_quote": "blue or other colours",
             "source_fact": "Only blue labels are mentioned; the record gives no reason for the label colours.",
             "note": "This is weaker evidence than an invented numerical fact; no quantitative error score is assigned."},
            {"kind": "format_violation", "output_quote": "**End of Report**",
             "source_fact": "Write a continuous plain-English report without headings or bullet points.",
             "note": "There are bullets and extensive post-report conversational material."},
            {"kind": "conversational_boilerplate", "output_quote": "If you have further questions or would like additional reports based on similar records, please let me know.",
             "source_fact": "The requested output is a newsletter report on the supplied inventory."}
        ]
    }


def block_statistics(rows, start, stop, paired):
    selected = rows[start:stop]
    q_values, entropy_values, maximums = [], [], []
    for row in selected:
        weights = [entry["weight"] for entry in row["weights"]]
        assert sum(weights) == TOTAL and all(w >= 0 for w in weights)
        ps = [w / TOTAL for w in weights if w]
        q = 1 - sum(p * p for p in ps)
        q_values.append(q)
        entropy_values.append(-sum(p * math.log2(p) for p in ps))
        maximums.append(max(ps))
        if paired:
            assert math.isclose(q, row["raw_distinct_pair_probability"], abs_tol=1e-14)
    return {
        "start_inclusive": start, "stop_exclusive": stop,
        "tokens": len(selected),
        "conditional_distinct_pair_probability_sum": sum(q_values),
        "constrained_model_entropy_bits_sum": sum(entropy_values),
        "mean_max_token_probability": sum(maximums) / len(maximums),
        "actual_informative_pair_steps": (
            sum(row["candidates"][0] != row["candidates"][1] for row in selected)
            if paired else None),
    }


def canonical_difference(result, trace):
    generated = result["generated_token_ids"]
    canonical = result["canonical_text_token_ids"]
    if generated == canonical:
        return {"exact": True, "generated_ids": len(generated), "canonical_ids": len(canonical)}
    prefix = 0
    while prefix < min(len(generated), len(canonical)) and generated[prefix] == canonical[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(generated), len(canonical)) - prefix and
           generated[len(generated)-1-suffix] == canonical[len(canonical)-1-suffix]):
        suffix += 1
    gen_stop = len(generated) - suffix
    can_stop = len(canonical) - suffix
    raw = b"".join(bytes.fromhex(row["token_bytes_hex"]) for row in trace[prefix:gen_stop])
    return {
        "exact": False, "generated_ids": len(generated), "canonical_ids": len(canonical),
        "first_difference_index_zero_based": prefix,
        "matching_suffix_tokens": suffix,
        "generated_differing_ids": generated[prefix:gen_stop],
        "canonical_differing_ids": canonical[prefix:can_stop],
        "generated_differing_piece_hex": raw.hex(),
        "generated_differing_piece_utf8": raw.decode("utf-8"),
        "scope": "Canonical IDs are the saved native tokenizer result; no new tokenizer or model invocation."
    }


def diagnose(repo):
    base = repo / "results/seeded-pair-llm-v1"
    summary = json_load(base / "summary.json")
    protocol_path = repo / "experiments/seeded_pair_llm_v1/protocol.json"
    protocol = json_load(protocol_path)
    observations = fidelity_observations()
    cases = []
    for case in CASES:
        directory = base / case
        result_path = directory / "result.json"
        trace_path = directory / "trace.jsonl"
        text_path = directory / "output.txt"
        result = json_load(result_path)
        trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
        text = text_path.read_text(encoding="utf-8")
        assert len(trace) == result["generated_tokens"] == protocol["output_tokens"] == 512
        assert [row["step"] for row in trace] == list(range(512))
        for observation in observations[case]:
            assert observation["output_quote"] in text, (case, observation["output_quote"])
        paired = result["arm"] == "stego"
        all_stats = block_statistics(trace, 0, 512, paired)
        if paired:
            assert all_stats["actual_informative_pair_steps"] == result["capacity_bits"]
        cases.append({
            "case_id": case, "status": result["status"],
            "input_sha256": {"result.json": file_sha(result_path), "trace.jsonl": file_sha(trace_path),
                             "output.txt": file_sha(text_path)},
            "all_tokens": all_stats,
            "blocks_128": [block_statistics(trace, start, start+128, paired)
                           for start in range(0, 512, 128)],
            "text_canonicality": canonical_difference(result, trace),
            "source_fidelity_observations": observations[case],
        })
    assert summary["actual_generated_tokens"] == 2048 and len(cases) == 4
    return {
        "analysis_id": "seeded-pair-llm-v1-posthoc-diagnostics",
        "analysis_is_posthoc": True,
        "no_new_model_or_tokenizer_calls": True,
        "protocol_sha256": file_sha(protocol_path),
        "summary_sha256": file_sha(base / "summary.json"),
        "source_scope": "All 4 preregistered generated outputs, including refusals; no filtering.",
        "definitions": {
            "conditional_distinct_pair_probability_sum": "sum_t(1-sum_i p_t(i)^2), conditional on each saved realized history",
            "constrained_model_entropy_bits_sum": "sum_t(-sum_i p_t(i)*log2(p_t(i))); diagnostic, not achieved payload capacity",
            "ordinary_pair_statistics": "q is a property of saved ordinary distributions; no pair was sampled in this arm",
            "source_fidelity_observations": "Agent qualitative observations, not independent human scores and not causal channel attribution"
        },
        "actual_generated_texts": 4, "actual_generated_tokens": 2048,
        "full_exact_deliveries": summary["full_exact_deliveries"],
        "planned_stego_deliveries": summary["planned_stego_deliveries"],
        "receiver_trace_steps": summary["receiver_trace_steps"],
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.repo / "results/seeded-pair-llm-v1-analysis/diagnostics.json"
    report = diagnose(args.repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"analysis": str(output), "cases": len(report["cases"]),
                      "generated_tokens": report["actual_generated_tokens"]}))


if __name__ == "__main__":
    main()

