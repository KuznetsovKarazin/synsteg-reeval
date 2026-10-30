#!/usr/bin/env python3
"""Measure the sealed test only after its one-opening evaluation is complete.

No models are fitted, thresholds chosen, or lost experimental results recreated.
The summary is descriptive and consumes only retained, hash-verified records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ratio(n, d):
    return n / d if d else None


def distribution(values):
    values = list(values)
    return {"n": len(values), "sum": sum(values),
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None}


def quality_proxy_limitation(protocol):
    method = protocol.get("method_config") or {}
    corrected = (
        method.get("candidate_filter_id") == "wordnet-first-sense-base-pos-rendered-single-gpt2-token-v2"
        and method.get("scorer_id") == "gpt2-verified-cover-prefix-next-token-nll-v2"
    )
    if corrected:
        return ("The declared adapter verifies rendered candidate tokens and causal prefixes; "
                "local scores still use the original cover and omit interactions among substitutions.")
    return ("Legacy artificial-leading-space scoring can mismatch newline/quote contexts; "
            "the declared filter/scorer IDs do not identify the rendered-token correction. "
            "Local scores use the original cover and omit interactions among substitutions.")


def load_completed_test(run):
    """All completion gates precede the first access to records/test."""
    run = Path(run)
    required = ("evaluation.json", "test_opened.json", "test_generation.json",
                "attack_lock.json", "freeze.json", "protocol.json", "events.jsonl")
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise ValueError("Refusing test access: completed evaluation required; missing " + ", ".join(missing))
    evaluation = read_json(run / "evaluation.json")
    opening = read_json(run / "test_opened.json")
    generation = read_json(run / "test_generation.json")
    lock = read_json(run / "attack_lock.json")
    frozen = read_json(run / "freeze.json")
    protocol = read_json(run / "protocol.json")
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines() if x.strip()]
    opened = [x for x in events if x.get("event") == "test_opened_once"]
    completed = [x for x in events if x.get("event") == "test_evaluation_completed"]
    if len(opened) != 1 or len(completed) != 1:
        raise ValueError("Refusing test access: exactly one opening and completed evaluation event required")
    if completed[0].get("evaluation_sha256") != file_hash(run / "evaluation.json"):
        raise ValueError("Refusing test access: evaluation digest differs from completion event")
    if completed[0]["time_unix"] < opened[0]["time_unix"]:
        raise ValueError("Refusing test access: completion precedes opening")
    if digest({k: v for k, v in frozen.items() if k != "freeze_id"}) != frozen["freeze_id"]:
        raise ValueError("Freeze integrity failure")
    if digest({k: v for k, v in lock.items() if k != "lock_id"}) != lock["lock_id"]:
        raise ValueError("Attack-lock integrity failure")
    if file_hash(run / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Protocol digest differs from freeze")
    if any(x.get("freeze_id") != frozen["freeze_id"] for x in (evaluation, opening, lock)):
        raise ValueError("Evaluation/freeze identities differ")
    if any(x.get("lock_id") != lock["lock_id"] for x in (evaluation, opening, opened[0])):
        raise ValueError("Evaluation/attack-lock identities differ")
    if generation.get("complete") is not True:
        raise ValueError("Refusing test access: generation is incomplete")
    expected_hash = generation.get("records_sha256")
    if not expected_hash or any(x.get("test_records_sha256") != expected_hash for x in (evaluation, opening)):
        raise ValueError("Evaluation/generation record identities differ")
    if not evaluation.get("attacks") or any(x.get("status") not in {"complete", "no_common_success_test_pairs"}
                                            for x in evaluation["attacks"].values()):
        raise ValueError("Refusing test access: attack statuses are not final")
    if set(evaluation["attacks"]) != set(lock.get("attacks", {})):
        raise ValueError("Refusing test access: final attack set differs from locked attacks")
    manifest_path = run / "manifests/test.json"
    if file_hash(manifest_path) != frozen["manifest_sha256"]["test"]:
        raise ValueError("Test manifest differs from freeze")
    samples = read_json(manifest_path)
    # First access to the test-record directory occurs only after every gate.
    paths = {p.stem: p for p in (run / "records/test").glob("*.json")}
    if set(paths) != {s["id"] for s in samples}:
        raise ValueError("Test record set differs from completed design")
    records = [read_json(paths[s["id"]]) for s in samples]
    if any(r.get("sample") != s or r.get("freeze_id") != frozen["freeze_id"]
           for r, s in zip(records, samples)):
        raise ValueError("Test record identity differs")
    if digest(records) != expected_hash:
        raise ValueError("Test records differ from evaluated bytes")
    return protocol, evaluation, records


def used_choice_measurements(record, output):
    metadata = output["metadata"]
    if metadata.get("framing") != "external_length":
        raise ValueError("This summary requires the pilot external-length framing")
    payload = record["synthetic_payload"]
    if set(payload) - {"0", "1"} or len(payload) != record["sample"]["payload_bits"]:
        raise ValueError("Invalid retained payload")
    positions, tokens = output["key"]["positions"], output["stego_text"].split()
    if len(tokens) != len(positions):
        raise ValueError("Text/codebook position count differs")
    used, offset = [], 0
    for index, position in enumerate(positions):
        groups = position["groups"]
        if not groups or offset >= len(payload):
            continue
        width = int(math.log2(len(groups)))
        if width < 1 or 2 ** width != len(groups):
            raise ValueError("Non-power-of-two codebook groups")
        group_index = int(payload[offset:offset + width].ljust(width, "0"), 2)
        group, scores = groups[group_index], position["scores"][group_index]
        candidate_index = group.index(tokens[index])
        if len(scores) != len(group) or not all(math.isfinite(x) for x in scores):
            raise ValueError("Invalid retained candidate scores")
        used.append({"choices": len(group), "edited": tokens[index] != position["original"],
                     "selected": scores[candidate_index], "canonical": scores[0], "minimum": min(scores)})
        offset += width
    if offset < len(payload) or len(used) != metadata["used_positions"]:
        raise ValueError("Used-position metadata differs from payload/codebook")
    return {"used_positions": len(used), "used_positions_with_choice": sum(x["choices"] > 1 for x in used),
            "used_singleton_positions": sum(x["choices"] == 1 for x in used),
            "choices_in_used_groups": [x["choices"] for x in used],
            "edited_used_positions": sum(x["edited"] for x in used),
            "selected_local_proxy_nll_sum": sum(x["selected"] for x in used),
            "canonical_local_proxy_nll_sum": sum(x["canonical"] for x in used),
            "minimum_local_proxy_nll_sum": sum(x["minimum"] for x in used)}


def measure_record(record, variant):
    sample = record["sample"]
    output = record.get("variants", {}).get(variant, {})
    metrics = output.get("metrics") or {}
    success, cover = bool(metrics.get("full_message_success")), record.get("cover_text")
    generation = record.get("generation_metadata", {})
    row = {"sample_id": sample["id"], "topic_id": sample["topic_id"], "genre_id": sample["genre_id"],
           "payload_bits": sample["payload_bits"], "variant": variant, "full_message_success": success,
           "delivered_bits": sample["payload_bits"] if success else 0,
           "cover_words": len(cover.split()) if cover else None,
           "generated_tokens": generation.get("generated_tokens"), "hit_token_limit": generation.get("hit_token_limit"),
           "failure": record.get("failure") or output.get("error"),
           "stego_words": len(output["stego_text"].split()) if output.get("stego_text") else None}
    if success:
        if output.get("decoded_payload") != record.get("synthetic_payload"):
            raise ValueError("Successful record lacks an exact decoded payload")
        if metrics["requested_bits"] != sample["payload_bits"] or metrics["stego_words"] != row["stego_words"]:
            raise ValueError("Retained delivery metrics disagree with record")
        raw_key = json.dumps(output["key"], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        if len(raw_key) != metrics["key_json_bytes"]:
            raise ValueError("Retained JSON key size disagrees with key")
        external_bits = output["metadata"].get("external_metadata_bits")
        if type(external_bits) is not int or external_bits < 0:
            raise ValueError("Missing external-length communication cost")
        row.update(key_json_bytes=metrics["key_json_bytes"], key_zlib9_bytes=metrics["key_zlib9_bytes"],
                   external_metadata_bits=external_bits,
                   key_json_plus_external_bits=8 * metrics["key_json_bytes"] + external_bits,
                   key_zlib9_plus_external_bits=8 * metrics["key_zlib9_bytes"] + external_bits,
                   choices=used_choice_measurements(record, output))
    return row


def aggregate(rows):
    success = [r for r in rows if r["full_message_success"]]
    generated = [r for r in rows if r["cover_words"] is not None]
    delivered, requested = sum(r["delivered_bits"] for r in rows), sum(r["payload_bits"] for r in rows)
    cover_words, successful_words = sum(r["cover_words"] for r in generated), sum(r["stego_words"] for r in success)
    result = {"attempted_samples": len(rows), "full_success_samples": len(success), "failed_samples": len(rows) - len(success),
              "full_success_rate": ratio(len(success), len(rows)), "attempted_payload_bits": requested,
              "delivered_payload_bits": delivered, "delivered_fraction_of_attempted_bits": ratio(delivered, requested),
              "mean_delivered_bits_per_planned_attempt": ratio(delivered, len(rows)),
              "generated_covers_with_word_denominator": len(generated),
              "attempts_without_generated_cover_word_denominator": len(rows) - len(generated),
              "all_generated_cover_words": cover_words, "successful_stego_words": successful_words,
              "full_message_goodput_per_all_generated_cover_word": ratio(delivered, cover_words),
              "successful_text_payload_bpw_weighted": ratio(delivered, successful_words),
              "cover_words": distribution(r["cover_words"] for r in generated),
              "generated_tokens": distribution(r["generated_tokens"] for r in rows if r["generated_tokens"] is not None),
              "token_limit_known_samples": sum(r["hit_token_limit"] is not None for r in rows),
              "hit_token_limit_samples": sum(r["hit_token_limit"] is True for r in rows),
              "successful_transmission_costs": {name: distribution(r[name] for r in success) for name in
                  ("key_json_bytes", "key_zlib9_bytes", "external_metadata_bits", "key_json_plus_external_bits", "key_zlib9_plus_external_bits")}}
    choices = [r["choices"] for r in success]
    used, freedom = sum(r["used_positions"] for r in choices), sum(r["used_positions_with_choice"] for r in choices)
    result["choice_freedom_on_successful_used_positions"] = {
        "used_positions": used, "positions_with_choice": freedom, "singleton_positions": used - freedom,
        "fraction_with_choice": ratio(freedom, used), "edited_positions": sum(r["edited_used_positions"] for r in choices),
        "choices_per_used_group": distribution(n for r in choices for n in r["choices_in_used_groups"]),
        "selected_local_proxy_nll_sum": sum(r["selected_local_proxy_nll_sum"] for r in choices),
        "canonical_local_proxy_nll_sum": sum(r["canonical_local_proxy_nll_sum"] for r in choices),
        "selected_minus_canonical_local_proxy_nll_sum": sum(r["selected_local_proxy_nll_sum"] - r["canonical_local_proxy_nll_sum"] for r in choices)}
    return result


def selector_comparison(records, variants):
    common = [r for r in records if all((r.get("variants", {}).get(v, {}).get("metrics") or {}).get("full_message_success") for v in variants)]
    identical = sum(len({r["variants"][v]["stego_text"] for v in variants}) == 1 for r in common)
    return {"common_success_pairs": len(common),
            "same_codebook_pairs": sum(len({digest(r["variants"][v]["key"]) for v in variants}) == 1 for r in common),
            "identical_selector_output_pairs": identical, "identical_selector_output_fraction": ratio(identical, len(common))}


def token_alignment_summary(records, variants):
    """Count shared pre-payload candidate admission once per attempted record."""
    names = ("candidate_pairs_considered", "candidate_pairs_admitted", "candidate_pairs_rejected")
    rows = []
    for record in records:
        prepared = record.get("prepared_methods") or {}
        values = [(prepared.get(v) or {}).get("token_alignment") for v in variants]
        row = {"sample_id": record["sample"]["id"], "available": all(v is not None for v in values)}
        if any(v is not None for v in values):
            if not row["available"]:
                raise ValueError("Token-alignment availability differs across shared selectors")
            if any(not isinstance(v, dict) or any(type(v.get(n)) is not int or v[n] < 0 for n in names)
                   for v in values):
                raise ValueError("Invalid token-alignment candidate counts")
            if len({tuple(v[n] for n in names) for v in values}) != 1:
                raise ValueError("Token-alignment counts differ across shared selectors")
            counts = {name: values[0][name] for name in names}
            if counts[names[0]] != counts[names[1]] + counts[names[2]]:
                raise ValueError("Token-alignment counts do not partition considered candidates")
            row.update(counts)
        else:
            row.update({name: None for name in names})
        rows.append(row)
    available = [row for row in rows if row["available"]]
    totals = {name: sum(row[name] for row in available) if available else None for name in names}
    return {"records_total": len(rows), "records_with_counts": len(available),
            "records_without_counts": len(rows) - len(available), "totals_over_available_records": totals,
            "admitted_fraction_of_considered": ratio(totals[names[1]], totals[names[0]]) if available else None,
            "counting_unit": "Candidate-position pairs, counted once per shared prepared codebook; not doubled across selectors.",
            "missing_count_policy": "Unavailable, including legacy records without retained counts; never imputed as zero.",
            "records": rows}


def summarize(run):
    protocol, evaluation, records = load_completed_test(run)
    variants = protocol["variants"]
    rows = [measure_record(r, v) for r in records for v in variants]
    report = {"status": "supplementary_measurements_after_completed_one_opening_evaluation",
              "freeze_id": evaluation["freeze_id"], "lock_id": evaluation["lock_id"],
              "test_records_sha256": evaluation["test_records_sha256"],
              "evaluation_sha256": file_hash(Path(run) / "evaluation.json"), "summary_script_sha256": file_hash(__file__),
              "scope": "prospective_test_only_no_refit_or_threshold_selection",
              "test_topic_clusters": len({r["sample"]["topic_id"] for r in records}),
              "prompt_requested_word_counts": {split: {genre: [int(n) for n in re.findall(r"\b(\d+)(?: English)? words\b", p["template"])]
                  for genre, p in prompts.items()} for split, prompts in protocol.get("prompts", {}).items()},
              "generation_token_ceiling": protocol.get("generation", {}).get("max_new_tokens"),
              "definitions": {
                  "word_count": "Python whitespace split; delivered payload bits require full-message success.",
                  "all_attempt_goodput": "Delivered bits / words in every generated cover, including failed encodings. Attempts with no cover have no word denominator and are counted explicitly; delivered bits per planned attempt includes them.",
                  "successful_bpw": "Sum delivered bits / sum successful stego words; excludes failures and codebook/length metadata costs.",
                  "key_cost": "Encoder-time compact JSON and zlib level-9 sizes from successful records. Retained compressed sizes preserve original key order; reserializing sorted record keys would change compression.",
                  "side_information": "Codebook bytes plus external length bits, not overall text-plus-side-channel efficiency or an entropy bound.",
                  "quality_proxy": "Retained original-cover candidate NLL at used positions; not actual stegotext perplexity, semantic quality, or independent quality evaluation."},
              "selector_comparison": selector_comparison(records, variants),
              "token_alignment": token_alignment_summary(records, variants), "variants": {}, "records": rows,
              "limitations": [
                  "Low-payload pilot and few test topics; no matched original-SynSteg baseline or broad secrecy conclusion.",
                  "Report actual cover lengths/token-limit hits alongside requested prompt lengths; model compliance is not assumed.",
                  quality_proxy_limitation(protocol),
                  "External-length framing has no CRC; exact decoding assumes intact codebook, length metadata, and formatting.",
                  "This descriptive supplement adds no hypotheses, p-values, confidence intervals, or attack tuning."]}
    for variant in variants:
        selected = [r for r in rows if r["variant"] == variant]
        report["variants"][variant] = {"overall": aggregate(selected),
            "by_payload_bits": {str(bits): aggregate([r for r in selected if r["payload_bits"] == bits]) for bits in sorted({r["payload_bits"] for r in selected})},
            "by_genre": {genre: aggregate([r for r in selected if r["genre_id"] == genre]) for genre in sorted({r["genre_id"] for r in selected})}}
    report["selector_comparison_by_payload_bits"] = {str(bits): selector_comparison([r for r in records if r["sample"]["payload_bits"] == bits], variants) for bits in protocol["payload_bits"]}
    return report


def markdown(report):
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
    lines = ["# Prospective pilot supplementary measurements", "",
             "Computed after the completed, hash-verified one-opening evaluation. No models or thresholds were changed.", "",
             "| Variant / bits | Full deliveries | Delivered / attempted bits | Successful text BPW | Goodput / all generated cover words |",
             "|---|---:|---:|---:|---:|"]
    for variant, value in report["variants"].items():
        for label, row in [("all", value["overall"]), *value["by_payload_bits"].items()]:
            lines.append(f"| {variant} / {label} | {row['full_success_samples']}/{row['attempted_samples']} | {row['delivered_payload_bits']}/{row['attempted_payload_bits']} | {fmt(row['successful_text_payload_bpw_weighted'])} | {fmt(row['full_message_goodput_per_all_generated_cover_word'])} |")
    lines += ["", "Successful-text BPW excludes failures and side information. All-generated-cover goodput includes covers from failed encodings; attempts with no cover are separately counted in JSON.", "",
              "| Variant | Median JSON / zlib key bytes | External bits per success (range) | Used positions with choice | Edited used positions |", "|---|---:|---:|---:|---:|"]
    for variant, value in report["variants"].items():
        row = value["overall"]; costs = row["successful_transmission_costs"]; choice = row["choice_freedom_on_successful_used_positions"]
        lines.append(f"| {variant} | {fmt(costs['key_json_bytes']['median'])} / {fmt(costs['key_zlib9_bytes']['median'])} | {fmt(costs['external_metadata_bits']['min'])}–{fmt(costs['external_metadata_bits']['max'])} | {choice['positions_with_choice']}/{choice['used_positions']} | {choice['edited_positions']} |")
    paired = report["selector_comparison"]
    lines += ["", f"Identical selector outputs: {paired['identical_selector_output_pairs']}/{paired['common_success_pairs']} common successful pairs. Same codebook: {paired['same_codebook_pairs']}/{paired['common_success_pairs']} pairs.", "",
              "Codebook sizes retain encoder-time serialization. Codebook bytes and external length bits are communication resources outside payload BPW.", "",
              "| Genre | Covers with word counts | Median words (min–max) | Token-limit hits / known |", "|---|---:|---:|---:|"]
    first = next(iter(report["variants"].values()))
    for genre, row in first["by_genre"].items():
        words = row["cover_words"]
        lines.append(f"| {genre} | {words['n']} | {fmt(words['median'])} ({fmt(words['min'])}–{fmt(words['max'])}) | {row['hit_token_limit_samples']}/{row['token_limit_known_samples']} |")
    alignment = report["token_alignment"]; totals = alignment["totals_over_available_records"]
    lines += ["", "Cover counts are shared by variants and shown once. JSON retains per-payload/per-genre measurements and original-context score sums. These sums are a quality proxy, not actual text perplexity or semantic assessment.", "",
              f"Token-alignment candidate counts are available for {alignment['records_with_counts']}/{alignment['records_total']} attempted records: considered {fmt(totals['candidate_pairs_considered'])}, admitted {fmt(totals['candidate_pairs_admitted'])}, rejected {fmt(totals['candidate_pairs_rejected'])}. Each shared codebook is counted once across both selectors; missing counts are unavailable, not zero.", ""]
    lines += ["- " + value for value in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New directory; default RUN/supplementary")
    args = parser.parse_args()
    report = summarize(args.run)
    output = args.output or args.run / "supplementary"
    files = {"measurements.json": canonical(report) + b"\n", "measurements.md": markdown(report).encode()}
    if output.exists() and any(not (output / name).is_file() or (output / name).read_bytes() != value for name, value in files.items()):
        raise ValueError("Different supplement already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (output / name).write_bytes(value)
    print(json.dumps({"status": report["status"], "output": str(output), "selector_comparison": report["selector_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
