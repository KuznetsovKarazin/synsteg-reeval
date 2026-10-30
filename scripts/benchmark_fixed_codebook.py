#!/usr/bin/env python3
"""Development-only feasibility of fixed codebooks on archived candidates.

This does not create new cover text and does not certify that archived candidate
generation was independent of the original payload. No detector is tuned here.
No language model is loaded; quality ties merely prefer retaining source words.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.fixed_codebook import (CapacityError, candidates_from_legacy_key,
                                         decode_bits, encode_bits, freeze_codebook)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/source/synsteg_reproduction")
    parser.add_argument("--limit-per-series", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configurations = [
        {"id": "strict_b1_g2", "max_bits_per_position": 1, "min_group_size": 2},
        {"id": "strict_b2_g2", "max_bits_per_position": 2, "min_group_size": 2},
        {"id": "lowrate_b1_g1", "max_bits_per_position": 1, "min_group_size": 1},
    ]
    rows = []
    for series, folder in [("luna", "reports/gpt6_luna_missing"), ("qwen", "reports/qwen_missing_b3")]:
        files = sorted((args.source / folder).glob("*.json"))
        if args.limit_per_series:
            files = files[:args.limit_per_series]
        for path in files:
            record = json.loads(path.read_text())
            for config in configurations:
                row = {"series": series, "uuid": record["uuid"], "configuration": config["id"],
                       "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "error": None}
                try:
                    options = candidates_from_legacy_key(record["container"], record["secret_key"])
                    book = freeze_codebook(record["container"], options, seed="development-feasibility-v1",
                                           **{k: v for k, v in config.items() if k != "id"})
                    row.update(capacity_bits=book.capacity_bits, payload_capacity_bits=book.payload_capacity_bits,
                               requested_archived_bits=len(record["message"]), eligibility=book.eligibility,
                               codebook_sha256=book.sha256, codebook_json_bytes=len(book.to_json().encode()))
                    for mode in ["length_crc32", "external_length"]:
                        try:
                            encoded = encode_bits(book, record["message"], framing=mode)
                            decoded = decode_bits(book, encoded.text, framing=mode,
                                                  **({"expected_bits": len(record["message"])} if mode == "external_length" else {}))
                            row["archived_payload_" + mode] = {"full_exact": decoded == record["message"],
                                                              "capacity_shortfall": False}
                        except CapacityError:
                            row["archived_payload_" + mode] = {"full_exact": False, "capacity_shortfall": True}
                    if book.capacity_bits >= 16:
                        a = "0101101000111100"
                        b = "".join("1" if bit == "0" else "0" for bit in a)
                        saved = book.to_json()
                        low = []
                        for payload in (a, b):
                            for selection in ("canonical", "quality"):
                                encoded = encode_bits(book, payload, framing="external_length", selection=selection)
                                low.append(decode_bits(book, encoded.text, framing="external_length", expected_bits=16) == payload)
                        row["external_16_bit_control"] = {"all_four_exact": all(low),
                                                           "codebook_unchanged": saved == book.to_json(),
                                                           "length_sidechannel_bits": 32}
                except (ValueError, TypeError, KeyError) as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    summaries = {}
    for series in ("luna", "qwen"):
        for config in configurations:
            subset = [r for r in rows if r["series"] == series and r["configuration"] == config["id"]]
            good = [r for r in subset if r["error"] is None]
            summaries[series + ":" + config["id"]] = {
                "records": len(subset), "input_errors": len(subset) - len(good),
                "median_capacity_bits": statistics.median(r["capacity_bits"] for r in good) if good else None,
                "framed_original_payload_successes": sum(r["archived_payload_length_crc32"]["full_exact"] for r in good),
                "external_original_payload_successes": sum(r["archived_payload_external_length"]["full_exact"] for r in good),
                "external_16bit_eligible": sum("external_16_bit_control" in r for r in good),
                "external_16bit_four_exact_and_immutable": sum(r.get("external_16_bit_control", {}).get("all_four_exact", False)
                                                               and r["external_16_bit_control"]["codebook_unchanged"] for r in good),
            }
    result = {"status": "completed", "scope": "archived_development_feasibility_only", "summaries": summaries, "rows": rows,
              "limitations": ["No new covers, no unseen test, no semantic or neural quality validation.",
                              "Archived candidate generation cannot be certified payload-independent.",
                              "External mode requires a separate 32-bit length field and the document-specific codebook."]}
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(output)
        print(json.dumps(summaries, indent=2))
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
