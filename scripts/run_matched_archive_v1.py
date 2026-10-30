#!/usr/bin/env python3
"""Frozen, retrospective matched transport comparison; no model calls.

The archived tables are post-embedding artifacts, including a shortened tail
slot. Relabeling them cannot certify prospective payload-independent inputs.
The recipient subprocess receives no payload or separately supplied cover.
Its production decoders provide process isolation, not independent algorithms.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import zlib
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit import fixed_codebook as fixed
from synsteg_audit import shared_dictionary as shared

VERSION = "matched-archive-v1"
ARCHIVE_SHA256 = "793c419e19b09a038af439fceb0c8727f2d809e2e855444d66839fd8132ef790"
SEED = "matched-archive-v1"
# Public deterministic fixture material: not a secret and not a security test.
FIXTURE_KEY = bytes(range(32))
PROTOCOL_SHA256 = "e289d0128f6b7eedc1193e751e98de3e0a241b572e9bdb0b1d8385aab05c3005"
FIXED_CONFIGS = (
    ("strict_b1_g2", 1, 2), ("strict_b2_g2", 2, 2),
    ("lowrate_b1_g1", 1, 1), ("ceiling_b5_g1", 5, 1),
)
SERIES = (("luna", "gpt6_luna_missing", 300, 5), ("qwen", "qwen_missing_b3", 75, 3))
SOURCE_PREFIX = "synsteg_reproduction/"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_protocol(path):
    raw = path.read_bytes()
    if sha(raw) != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol file identity mismatch")
    protocol = json.loads(raw)
    expected_configs = [{"id": name, "max_bits_per_position": width, "min_group_size": size}
                        for name, width, size in FIXED_CONFIGS]
    checks = {
        "experiment_id": VERSION, "archive_sha256": ARCHIVE_SHA256, "fixed_seed": SEED,
        "fixed_configurations": expected_configs,
        "cohorts": {"reports/" + folder: n for _, folder, n, _ in SERIES},
        "fixed_common": {"protect_capitalized": True, "candidate_filter_id": "none", "scorer_id": "none",
                         "selection": "canonical", "framing": "external_length", "external_metadata_bits": 32},
        "public_dictionaries": ["data/shared_dictionary_recovery_v1.json", "data/shared_dictionary_recovery_v2.json"],
        "public_common": {"min_group_size": 1, "max_bits_per_slot": 2, "selector": "canonical",
                          "framings": ["plain_length16", "aes_siv"], "fixture_key_hex": FIXTURE_KEY.hex()},
        "denominators": {"independent_new_cover_generations": 0, "archived_records": 375,
                         "primary_rows": 3375, "secondary_rows": 3000, "total_rows": 6375},
        "primary_payload": "complete_original_record_message_128_256_512_bits",
        "secondary_payload": "first_16_bits_of_each_original_message_no_original_arm",
    }
    for key, expected in checks.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Runtime configuration differs from frozen protocol: {key}")
    return protocol


def validate_frozen_manifest(archive, entries, path):
    frozen = json.loads(path.read_text())["records"]
    wanted = {row["member"]: row for row in frozen}
    if len(wanted) != 375 or set(wanted) != {row["archive_path"] for row in entries}:
        raise ValueError("Frozen record manifest membership mismatch")
    for entry in entries:
        row = wanted[entry["archive_path"]]
        if (row["sha256"] != entry["source_sha256"] or row["uuid"] != entry["uuid"]
                or row["payload_bits"] != entry["payload_bits"]
                or row["bytes"] != archive.getinfo(entry["archive_path"]).file_size):
            raise ValueError("Frozen record manifest content mismatch")


def write_json(path, value):
    path.write_bytes(canonical(value) + b"\n")


def write_rows(path, rows):
    payload = b"".join(canonical(row) + b"\n" for row in rows)
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0) if path.suffix == ".gz" else payload)


def read_rows(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.splitlines() if line]


def original_sources(archive):
    """Copy code only, never original covers/payloads, into recipient resources."""
    modules = {name: archive.read(SOURCE_PREFIX + "steganography/" + name + ".py").decode("utf-8")
               for name in ("helper", "secret_key", "core")}
    constants = ast.parse(archive.read(SOURCE_PREFIX + "utils/constants.py").decode("utf-8"))
    wanted = {"PUNCTUATION", "BRACKETS", "SPECIAL_TOKENS", "N_ASCII_BITS"}
    values = {}
    for node in constants.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != wanted:
        raise ValueError("Missing pure helper constants")
    return {"modules": modules, "constants": values,
            "source_sha256": {k: sha(v.encode()) for k, v in modules.items()}}


def load_original(bundle):
    """Execute only named pure AST functions; no API/model imports execute."""
    namespace = dict(bundle["constants"], time=time, math=math, product=product, SYNONYM_MAP=list)
    helper_names = {"clean_container", "check_endswith_special", "has_duplicates", "remove_brackets",
                    "is_token_replacable", "check_capitalization", "check_is_end_of_word", "get_text_from_binary"}
    def execute(source, names, target):
        parsed = ast.parse(source)
        selected = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in selected} != names:
            raise ValueError("Missing original pure function")
        exec(compile(ast.Module(body=selected, type_ignores=[]), "<archived-pure-code>", "exec"), target)
    execute(bundle["modules"]["helper"], helper_names, namespace)
    namespace["helper"] = SimpleNamespace(**{name: namespace[name] for name in helper_names})
    execute(bundle["modules"]["secret_key"], {"fix_token_container_size"}, namespace)
    execute(bundle["modules"]["core"], {"embed", "decode_message"}, namespace)
    return SimpleNamespace(embed=namespace["embed"], decode=namespace["decode_message"], helper=namespace["helper"])


def inventory(archive):
    entries = []
    loaded = []
    for series, folder, expected_n, nominal_b in SERIES:
        names = sorted(n for n in archive.namelist()
                       if n.startswith(SOURCE_PREFIX + "reports/" + folder + "/") and n.endswith(".json"))
        if len(names) != expected_n:
            raise ValueError(f"{series}: expected {expected_n} records, found {len(names)}")
        counts = Counter()
        for name in names:
            raw = archive.read(name)
            record = json.loads(raw)
            payload = record["message"]
            if not payload or set(payload) - {"0", "1"} or len(payload) not in (128, 256, 512):
                raise ValueError("Unexpected archived payload")
            if len(record["secret_key"]) != len(record["container"].split()):
                raise ValueError("Archived key does not align with cover")
            counts[len(payload)] += 1
            shortened = []
            candidate_values = []
            for i, entry in enumerate(record["secret_key"]):
                if len(entry) != 1:
                    raise ValueError("Invalid saved key entry")
                base, mapping = next(iter(entry.items()))
                widths = {len(code) for code in mapping}
                if len(widths) != 1:
                    raise ValueError("Mixed code width")
                candidate_values.append({base: list(mapping.values())})
                if len(set(mapping.values())) > 1 and next(iter(widths)) < nominal_b:
                    shortened.append(i)
            entries.append({"series": series, "uuid": record["uuid"], "archive_path": name,
                            "source_sha256": sha(raw), "payload_bits": len(payload),
                            "cover_sha256": sha(record["container"].encode()),
                            "saved_key_sha256": sha(canonical(record["secret_key"])),
                            "saved_candidate_values_sha256": sha(canonical(candidate_values)),
                            "shortened_saved_slot_indices": shortened})
            loaded.append((series, record, sha(raw)))
        if counts != Counter({128: expected_n // 3, 256: expected_n // 3, 512: expected_n // 3}):
            raise ValueError("Payload strata do not match frozen cohort")
    if len({(r["series"], r["uuid"]) for r in entries}) != 375:
        raise ValueError("Duplicate archive record identity")
    return entries, loaded


def book_resource(books, book_id, kind, value):
    encoded = canonical(value)
    books.append({"book_id": book_id, "kind": kind, "value": value, "sha256": sha(encoded)})
    return {"book_id": book_id, "book_sha256": sha(encoded),
            "book_json_bytes": len(encoded), "book_zlib9_bytes": len(zlib.compress(encoded, 9))}


def base_row(series, record, source_sha, grid, arm, payload):
    return {"row_id": ":".join((series, record["uuid"], grid, arm)), "series": series,
            "uuid": record["uuid"], "source_sha256": source_sha, "grid": grid, "arm": arm,
            "archived_payload_bits": len(record["message"]), "requested_payload_bits": len(payload),
            "payload_sha256": sha(payload.encode()), "cover_words": len(record["container"].split()),
            "status": "pending", "error": None, "delivered_payload_bits": 0,
            "full_exact": False, "fresh_receiver_exact": False}


def encode_fixed(record, payload, config, book):
    result = fixed.encode_bits(book, payload, selection="canonical", framing="external_length")
    return result.text, {"frame_bits": result.frame_bits, "padding_bits": result.padding_bits,
                         "used_positions": result.used_positions,
                         "changed_positions": sum(a != b for a, b in zip(record["container"].split(), result.text.split()))}


def receiver(output):
    books = {row["book_id"]: row for row in read_rows(output / "books.jsonl.gz")}
    original = load_original(json.loads((output / "recipient_resources/original_functions.json").read_text()))
    dictionaries = {version: shared.PublicDictionary.from_dict(json.loads(
        (output / f"recipient_resources/dictionary_{version}.json").read_text())) for version in ("v1", "v2")}
    results = []
    for job in read_rows(output / "receiver_jobs.jsonl.gz"):
        result = {"row_id": job["row_id"], "status": "error", "decoded_bits": None, "error": None}
        try:
            if job["kind"] == "original":
                decoded, _ = original.decode(job["stego_text"], books[job["book_id"]]["value"], clean_output=False)
            elif job["kind"] == "fixed":
                book = fixed.FrozenCodebook.from_dict(books[job["book_id"]]["value"])
                decoded = fixed.decode_bits(book, job["stego_text"], framing="external_length", expected_bits=job["expected_bits"])
            elif job["kind"] == "public":
                protocol = shared.Protocol(**job["protocol"])
                raw = shared.decode(job["stego_text"], bytes.fromhex(job["fixture_key_hex"]),
                                    dictionaries[job["dictionary_version"]], protocol)
                decoded = "".join(f"{b:08b}" for b in raw)
            else:
                raise ValueError("Unknown recipient job")
            result.update(status="decoded", decoded_bits=decoded)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
    write_rows(output / "receiver_results.jsonl.gz", results)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["grid"], row["series"], row["arm"], row["archived_payload_bits"])].append(row)
        groups[(row["grid"], row["series"], row["arm"], "all")].append(row)
    result = []
    for (grid, series, arm, stratum), subset in sorted(groups.items(), key=lambda item: str(item[0])):
        caps = [r["capacity_bits"] for r in subset if r.get("capacity_bits") is not None]
        costs = [r["book_zlib9_bytes"] for r in subset if r.get("book_zlib9_bytes") is not None]
        result.append({"grid": grid, "series": series, "arm": arm, "archived_payload_bits": stratum,
                       "denominator": len(subset), "statuses": dict(Counter(r["status"] for r in subset)),
                       "full_exact": sum(r["full_exact"] for r in subset),
                       "requested_bits": sum(r["requested_payload_bits"] for r in subset),
                       "delivered_bits": sum(r["delivered_payload_bits"] for r in subset),
                       "zero_capacity": sum(r.get("capacity_bits") == 0 for r in subset),
                       "median_capacity_bits": statistics.median(caps) if caps else None,
                       "mean_capacity_bits": statistics.mean(caps) if caps else None,
                       "median_book_zlib9_bytes": statistics.median(costs) if costs else None,
                       "unconditional_delivered_bits_per_source_word":
                           sum(r["delivered_payload_bits"] for r in subset) / sum(r["cover_words"] for r in subset)})
    return {"version": VERSION, "records": len(rows), "groups": result,
            "limitations": [
                "Retrospective accepted archive cohort, not prospective generation success.",
                "Every saved table has a payload-tail truncation; raw generated candidates cannot be restored.",
                "Fixed arms freeze saved candidate values without reading new payload; historical resource independence is not established.",
                "Native framing differs: original archived table; fixed external 32-bit length; public in-text 16/144-bit overhead.",
                "Public fixture key is published, not a deployed secret. No security, semantic quality, PPL or detectability claim.",
                "Secondary16 uses first16 archived bits and has no original-SynSteg arm.",
                "Fresh subprocess uses production decoders; separate implementation audit is required."]}


def run(archive_path, output, freeze_commit, source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", freeze_commit):
        raise ValueError("A persisted 40-character protocol freeze commit is required")
    if output.exists():
        raise FileExistsError("Output already exists; never overwrite a completed or partial stage")
    protocol_path = ROOT / "experiments/matched_archive_v1/protocol.json"
    frozen_manifest_path = ROOT / "experiments/matched_archive_v1/input_manifest.json"
    frozen_protocol = validate_protocol(protocol_path)
    if source_commit is not None and source_commit != frozen_protocol["base_main"]:
        raise ValueError("Source commit differs from protocol base")
    archive_bytes = archive_path.read_bytes()
    if sha(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError("Original archive SHA256 mismatch")
    output.mkdir(parents=True)
    (output / "recipient_resources").mkdir()
    write_json(output / "status.json", {"version": VERSION, "status": "running", "freeze_commit": freeze_commit})
    with ZipFile(archive_path) as archive:
        manifest, inputs = inventory(archive)
        validate_frozen_manifest(archive, manifest, frozen_manifest_path)
        sources = original_sources(archive)
    write_json(output / "recipient_resources/original_functions.json", sources)
    original = load_original(sources)
    dictionaries = {}
    resource_rows = []
    for version in ("v1", "v2"):
        path = ROOT / f"data/shared_dictionary_recovery_{version}.json"
        raw = path.read_bytes()
        dictionary = shared.PublicDictionary.from_dict(json.loads(raw))
        dictionaries[version] = dictionary
        (output / f"recipient_resources/dictionary_{version}.json").write_bytes(raw)
        resource_rows.append({"id": version, "path": str(path.relative_to(ROOT)), "file_sha256": sha(raw),
                              "dictionary_sha256": dictionary.sha256, "one_time_file_bytes": len(raw),
                              "one_time_file_zlib9_bytes": len(zlib.compress(raw, 9))})
    write_json(output / "inputs_manifest.json", {"archive_sha256": ARCHIVE_SHA256, "entries": manifest})
    write_json(output / "resources.json", {"public_dictionaries": resource_rows, "fixture_key_hex": FIXTURE_KEY.hex(),
                                           "per_public_receiver_secret_key_bytes": 32,
                                           "fixture_key_is_public": True,
                                           "protocol_identifiers_excluded_from_byte_counts": True})
    write_json(output / "provenance.json", {"version": VERSION, "freeze_commit": freeze_commit,
                                            "source_commit": source_commit, "archive_sha256": ARCHIVE_SHA256,
                                            "protocol_sha256": sha(protocol_path.read_bytes()),
                                            "frozen_input_manifest_sha256": sha(frozen_manifest_path.read_bytes()),
                                            "runtime_protocol_conformance": True,
                                            "python": sys.version, "script_sha256": sha(Path(__file__).read_bytes()),
                                            "fixed_codebook_sha256": sha(Path(fixed.__file__).read_bytes()),
                                            "shared_dictionary_sha256": sha(Path(shared.__file__).read_bytes()),
                                            "fixed_configs": FIXED_CONFIGS, "fixed_seed": SEED,
                                            "generation_calls": 0, "paid_calls": 0})
    rows, books, jobs, expected = [], [], [], {}
    for series, record, source_sha in inputs:
        payload = record["message"]
        row = base_row(series, record, source_sha, "primary", "original_archived", payload)
        row.update(book_resource(books, f"{series}:{record['uuid']}:original", "original", record["secret_key"]))
        row.update(external_length_bits=0, frame_overhead_bits=0, capacity_bits=None,
                   resource_contract="stego+document_table; archived variable-width final slot",
                   nominal_saved_table_bits=sum(len(next(iter(next(iter(e.values()))))) for e in record["secret_key"]
                                                if not original.helper.has_duplicates(e)))
        try:
            replay_key = deepcopy(record["secret_key"])
            replay_stego, embedded = original.embed(record["container"], replay_key, payload)
            row.update(reencoded_exact=replay_stego == record["encoded_message"], replay_embedded_bits=embedded,
                       replay_key_unchanged=replay_key == record["secret_key"],
                       archived_decoded_exact=record["decoded_message"] == payload)
            if not row["reencoded_exact"] or embedded != len(payload):
                raise ValueError("Original re-encoding differs from archived full delivery")
            jobs.append({"row_id": row["row_id"], "kind": "original", "book_id": row["book_id"],
                         "stego_text": record["encoded_message"]})
            expected[row["row_id"]] = payload
            row.update(status="awaiting_receiver", stego_sha256=sha(record["encoded_message"].encode()),
                       stego_words=len(record["encoded_message"].split()), frame_bits=len(payload), padding_bits=None)
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)

        for config_id, max_bits, min_group in FIXED_CONFIGS:
            book, book_meta, build_error = None, {}, None
            try:
                options = fixed.candidates_from_legacy_key(record["container"], record["secret_key"])
                book = fixed.freeze_codebook(record["container"], options, seed=SEED,
                                             max_bits_per_position=max_bits, min_group_size=min_group,
                                             protect_capitalized=True)
                book_meta = book_resource(books, f"{series}:{record['uuid']}:{config_id}", "fixed", book.to_dict())
            except Exception as exc:
                build_error = f"{type(exc).__name__}: {exc}"
            for grid, bits in (("primary", payload), ("secondary16", payload[:16])):
                row = base_row(series, record, source_sha, grid, "fixed_" + config_id, bits)
                row.update(book_meta, external_length_bits=32, frame_overhead_bits=0,
                           resource_contract="stego+document_codebook+external32bit_length",
                           capacity_bits=book.capacity_bits if book else None,
                           raw_payload_fits=book.capacity_bits >= len(bits) if book else False)
                try:
                    if build_error:
                        raise ValueError(build_error)
                    saved = book.to_json()
                    stego, details = encode_fixed(record, bits, config_id, book)
                    if saved != book.to_json():
                        raise ValueError("Codebook mutated during encoding")
                    row.update(details, status="awaiting_receiver", stego_sha256=sha(stego.encode()),
                               stego_words=len(stego.split()), codebook_unchanged=True)
                    jobs.append({"row_id": row["row_id"], "kind": "fixed", "book_id": row["book_id"],
                                 "stego_text": stego, "expected_bits": len(bits)})
                    expected[row["row_id"]] = bits
                except fixed.CapacityError as exc:
                    row.update(status="insufficient_capacity", error=str(exc))
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                rows.append(row)

        for version, dictionary in dictionaries.items():
            for framing in ("plain_length16", "aes_siv"):
                protocol = shared.Protocol(min_group_size=1, max_bits_per_slot=2, framing=framing)
                for grid, bits in (("primary", payload), ("secondary16", payload[:16])):
                    row = base_row(series, record, source_sha, grid, f"public_{version}_{framing}", bits)
                    row.update(external_length_bits=0, frame_overhead_bits=protocol.frame_overhead_bits,
                               resource_contract="stego+32byte_key+public_dictionary_and_protocol",
                               per_receiver_key_bytes=32, dictionary_sha256=dictionary.sha256,
                               protocol_sha256=protocol.sha256, capacity_bits=None)
                    try:
                        analysis = shared.analyze_text(record["container"], dictionary, protocol)
                        row.update(capacity_bits=analysis.capacity_bits,
                                   raw_payload_fits=analysis.capacity_bits >= len(bits),
                                   normalization_sha256=analysis.normalization_sha256)
                        raw = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
                        result = shared.encode(record["container"], raw, FIXTURE_KEY, dictionary, protocol)
                        row.update(status="awaiting_receiver", frame_bits=result.frame_bits,
                                   padding_bits=result.padding_bits, used_positions=result.consumed_slots,
                                   changed_positions=result.changed_slots,
                                   stego_sha256=sha(result.stego_text.encode()), stego_words=len(result.stego_text.split()))
                        jobs.append({"row_id": row["row_id"], "kind": "public", "dictionary_version": version,
                                     "stego_text": result.stego_text, "fixture_key_hex": FIXTURE_KEY.hex(),
                                     "protocol": {"min_group_size": 1, "max_bits_per_slot": 2, "framing": framing}})
                        expected[row["row_id"]] = bits
                    except shared.CapacityError as exc:
                        row.update(status="insufficient_capacity", error=str(exc))
                    except Exception as exc:
                        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
    if len(rows) != 375 * 17:
        raise AssertionError("Full arm denominator mismatch")
    write_rows(output / "books.jsonl.gz", books)
    write_rows(output / "receiver_jobs.jsonl.gz", jobs)
    write_rows(output / "records_before_receiver.jsonl.gz", rows)
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--receiver", str(output.resolve())],
                               capture_output=True, text=True)
    write_json(output / "receiver_process.json", {"returncode": completed.returncode,
                                                  "stdout": completed.stdout, "stderr": completed.stderr,
                                                  "job_count": len(jobs), "payload_or_cover_argument": False})
    if completed.returncode:
        raise RuntimeError("Fresh recipient process failed; preserve incomplete artifacts")
    decoded = {item["row_id"]: item for item in read_rows(output / "receiver_results.jsonl.gz")}
    if set(decoded) != set(expected):
        raise AssertionError("Receiver returned wrong job identities")
    for row in rows:
        if row["row_id"] in expected:
            outcome = decoded[row["row_id"]]
            exact = outcome["status"] == "decoded" and outcome["decoded_bits"] == expected[row["row_id"]]
            row.update(status="full_exact" if exact else "decode_error", full_exact=exact,
                       fresh_receiver_exact=exact, delivered_payload_bits=len(expected[row["row_id"]]) if exact else 0,
                       decoded_bits=len(outcome["decoded_bits"]) if outcome["decoded_bits"] is not None else None,
                       decoded_sha256=sha(outcome["decoded_bits"].encode()) if outcome["decoded_bits"] is not None else None,
                       error=outcome["error"] if not exact else None)
        row["delivered_bits_per_source_word"] = row["delivered_payload_bits"] / row["cover_words"]
        row["delivered_bits_per_stego_word"] = (row["delivered_payload_bits"] / row["stego_words"]
                                                 if row.get("stego_words") else None)
    write_rows(output / "records.jsonl", rows)
    write_json(output / "summary.json", summarize(rows))
    write_json(output / "status.json", {"version": VERSION, "status": "completed", "freeze_commit": freeze_commit,
                                         "input_records": 375, "primary_outcomes": 3375, "secondary_outcomes": 3000,
                                         "total_outcomes": len(rows), "receiver_jobs": len(jobs),
                                         "unexpected_errors": sum(r["status"] in ("error", "decode_error") for r in rows),
                                         "generation_calls": 0, "paid_calls": 0})
    hashes = {str(path.relative_to(output)): sha(path.read_bytes()) for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "output_manifest.json", hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-commit")
    parser.add_argument("--source-commit")
    parser.add_argument("--receiver", type=Path)
    args = parser.parse_args()
    if args.receiver is not None:
        receiver(args.receiver)
    else:
        if not all((args.archive, args.output, args.freeze_commit)):
            parser.error("--archive, --output, --freeze-commit are required")
        run(args.archive, args.output, args.freeze_commit, args.source_commit)


if __name__ == "__main__":
    main()
