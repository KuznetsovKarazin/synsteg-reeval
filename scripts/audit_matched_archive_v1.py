#!/usr/bin/env python3
"""Independent decoding/accounting audit for the frozen matched archive study.

No production codec functions are imported. Passing this checker validates
mechanical records, not meaning preservation or steganographic security.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import hmac
import json
from pathlib import Path
import re
import statistics
import unicodedata
import zipfile
import zlib

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESSIV
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def canonical(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")).encode("ascii")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixed_oracle(book: dict, text: str, length: int) -> str:
    """Invert saved groups; independently check layout, padding, and literals."""
    matches = list(re.finditer(r"\S+", text))
    separators, start = [], 0
    for match in matches:
        separators.append(text[start:match.start()])
        start = match.end()
    separators.append(text[start:])
    assert separators == list(book["separators"]), "fixed separators"
    assert len(matches) == len(book["positions"]), "fixed token count"
    assert type(length) is int and 0 <= length < 2**32
    bits = []
    bit_count = 0
    for match, position in zip(matches, book["positions"]):
        groups = position["groups"]
        token = match.group()
        if not groups or bit_count >= length:
            assert token == position["original"], "fixed literal/tail modification"
            continue
        assert len(groups) >= 2 and len(groups) & (len(groups) - 1) == 0
        width = len(groups).bit_length() - 1
        labels = [index for index, group in enumerate(groups) if token in group]
        assert len(labels) == 1, "fixed ambiguous or unknown token"
        bits.append(format(labels[0], f"0{width}b"))
        bit_count += width
    stream = "".join(bits)
    assert len(stream) >= length, "fixed truncated payload"
    assert not set(stream[length:]) - {"0"}, "fixed nonzero padding"
    return stream[:length]


def public_dictionary_canonical(raw: dict) -> dict:
    return {"schema": raw["schema"], "version": raw["version"],
            "classes": [{"id": entry["id"], "words": sorted(entry["words"])}
                        for entry in sorted(raw["classes"], key=lambda entry: entry["id"])]}


def public_analysis(text: str, dictionary: dict, protocol: dict) -> tuple[list[dict], str, int]:
    """Independent lexer and invariant representation for whole-word classes."""
    by_word = {word: entry for entry in dictionary["classes"] for word in entry["words"]}
    slots, normalized = [], []
    cursor = 0
    for token_index, match in enumerate(re.finditer(r"\S+", text)):
        lexical = re.fullmatch(r"([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)", match.group())
        if lexical is None:
            continue
        prefix, word, suffix = lexical.groups()
        if any(ch == "_" or not unicodedata.category(ch).startswith("P") for ch in prefix + suffix):
            continue
        if word.lower() not in by_word:
            continue
        style = ("lower" if word.islower() else "upper" if word.isupper()
                 else "title" if word.istitle() else None)
        if style is None:
            continue
        entry = by_word[word.lower()]
        width = min(protocol["max_bits_per_slot"],
                    max(0, (len(entry["words"]) // protocol["min_group_size"]).bit_length() - 1))
        if not width:
            continue
        begin, end = match.start() + len(prefix), match.end() - len(suffix)
        normalized.extend([["literal", text[cursor:begin]], ["class", entry["id"], style]])
        slots.append({"index": len(slots), "token_index": token_index,
                      "class_id": entry["id"], "width": width, "surface": word})
        cursor = end
    normalized.append(["literal", text[cursor:]])
    return slots, digest(canonical(normalized)), sum(s["width"] for s in slots)


def public_oracle(text: str, key: bytes, dictionary: dict, protocol: dict) -> bytes:
    """Independently derive labels, parse length and authenticate AES-SIV."""
    dictionary = public_dictionary_canonical(dictionary)
    version = "synsteg-shared-dictionary-aes-siv-v1"
    protocol = {"version": version, **protocol}
    dictionary_sha = digest(canonical(dictionary))
    protocol_sha = digest(canonical(protocol))
    slots, invariant_sha, capacity = public_analysis(text, dictionary, protocol)
    salt = hashlib.sha256(b"synsteg/shared-dictionary/hkdf-salt/v1\x00" +
                          bytes.fromhex(dictionary_sha)).digest()

    def derive(label: str, size: int) -> bytes:
        return HKDF(algorithm=hashes.SHA256(), length=size, salt=salt,
                    info=canonical([version, protocol_sha, label])).derive(key)

    mapping_key = derive("mapping-hmac-sha256", 32)
    classes = {entry["id"]: entry["words"] for entry in dictionary["classes"]}
    overhead = 144 if protocol["framing"] == "aes_siv" else 16
    assert capacity >= overhead, "public no complete frame capacity"
    stream = ""
    frame_size = None
    for slot in slots:
        context = [version, "word-rank", dictionary_sha, protocol_sha, invariant_sha,
                   slot["index"], slot["token_index"], slot["class_id"], slot["width"]]
        ordered = sorted(classes[slot["class_id"]], key=lambda word: (
            hmac.new(mapping_key, canonical(context + [word]), hashlib.sha256).digest(), word))
        label = ordered.index(slot["surface"].lower()) % (1 << slot["width"])
        stream += format(label, f"0{slot['width']}b")
        if frame_size is None and len(stream) >= 16:
            nbytes = int(stream[:16], 2)
            frame_size = overhead + 8 * nbytes
            assert frame_size <= capacity, "public length exceeds capacity"
        if frame_size is not None and len(stream) >= frame_size:
            break
    assert frame_size is not None and len(stream) >= frame_size, "public incomplete frame"
    assert not set(stream[frame_size:]) - {"0"}, "public nonzero padding"
    frame = bytes(int(stream[i:i + 8], 2) for i in range(0, frame_size, 8))
    if protocol["framing"] == "plain_length16":
        result = frame[2:]
    else:
        aad = [b"synsteg/shared-dictionary/authenticated-frame/v1", canonical({
            "version": version, "dictionary_sha256": dictionary_sha, "protocol": protocol,
            "normalization_sha256": invariant_sha, "payload_bytes": nbytes})]
        result = AESSIV(derive("aead-aes-siv-512", 64)).decrypt(frame[2:], aad)
    assert len(result) == nbytes, "public recovered length"
    return result


def fixed_book_accounting(book: dict) -> dict:
    serialized = json.dumps(book, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False).encode("utf-8")
    cover = "".join(s + p["original"] for s, p in zip(book["separators"], book["positions"]))
    cover += book["separators"][-1]
    assert digest(cover.encode()) == book["cover_sha256"]
    capacity = 0
    for position in book["positions"]:
        groups = position["groups"]
        if not groups:
            continue
        assert len(groups) & (len(groups) - 1) == 0
        width = len(groups).bit_length() - 1
        assert width <= book["max_bits_per_position"]
        forms = [form for group in groups for form in group]
        assert len(forms) == len(set(forms)), "duplicate fixed forms"
        assert all(len(group) >= book["min_group_size"] for group in groups)
        capacity += width
    return {"sha256": digest(serialized), "json_bytes": len(serialized),
            "zlib9_bytes": len(zlib.compress(serialized, 9)), "capacity_bits": capacity,
            "cover": cover}


def archived_oracle(text: str, key: list[dict]) -> str:
    """Independent reconstruction of the archive's longest-first parser.

    This deliberately preserves historical punctuation/duplicate semantics; it
    does not assert that arbitrary unseen keys are unambiguous or reliable.
    """
    normalized = text.translate(str.maketrans("", "", ",.?!"))
    offset = 0
    decoded = []
    for entry in key:
        assert len(entry) == 1
        original, labels = next(iter(entry.items()))
        if len(labels) != len(set(labels.values())):
            offset += len(original) + 1
            continue
        candidates = sorted(labels.items(), key=lambda item: len(item[1]), reverse=True)
        found = False
        for label, form in candidates:
            end = offset + len(form)
            if (normalized[offset:end].lower() == form.lower() and
                    (end >= len(normalized) or normalized[end] == " ")):
                decoded.append(label)
                offset = end + 1
                found = True
                break
        if not found:
            end = offset + len(original)
            if (normalized[offset:end].lower() == original.lower() and
                    (end >= len(normalized) or normalized[end] == " ")):
                offset = end + 1
            else:
                break
    return "".join(decoded)


def verify_fixed_mapping(book: dict, source: dict, width_limit: int, group_minimum: int) -> None:
    """Re-derive every accepted form/group from archived candidate *values*."""
    assert book["seed"] == "matched-archive-v1"
    assert book["max_bits_per_position"] == width_limit
    assert book["min_group_size"] == group_minimum
    assert book["protect_capitalized"] is True
    assert book["candidate_filter_id"] == book["scorer_id"] == "none"
    assert len(book["positions"]) == len(source["secret_key"])
    for index, (position, old_entry) in enumerate(zip(book["positions"], source["secret_key"])):
        token = position["original"]
        lexical = re.fullmatch(r"([^A-Za-z0-9_]*)([A-Za-z]+)([^A-Za-z0-9_]*)", token)
        expected_groups = []
        if lexical is not None:
            prefix, core, suffix = lexical.groups()
            if not any(c.isalpha() for c in prefix + suffix) and core.islower():
                old_mapping = next(iter(old_entry.values()))
                words = sorted({word.lower() for word in [core, *old_mapping.values()]
                                if re.fullmatch(r"[A-Za-z]+", word)})
                forms = [prefix + word + suffix for word in words]
                width = min(width_limit, max(0, (len(forms) // group_minimum).bit_length() - 1))
                if width:
                    def rank(form):
                        seed_input = json.dumps(["fixed-codebook-v1", "matched-archive-v1",
                                                 book["cover_sha256"], index, form], separators=(",", ":"))
                        return digest(seed_input.encode()), form
                    ordered = sorted(forms, key=rank)
                    expected_groups = [ordered[label::2**width] for label in range(2**width)]
        assert position["groups"] == expected_groups, f"fixed groups at slot {index}"
        assert position["scores"] == [[0.0] * len(group) for group in expected_groups]


def legacy_alignment_failure(source: dict) -> str | None:
    """Check the frozen adapter's failure condition without importing it."""
    tokens = source["container"].split()
    key = source["secret_key"]
    if len(tokens) != len(key):
        return "Legacy key does not align with cover token count"
    for token, entry in zip(tokens, key):
        if not isinstance(entry, dict) or len(entry) != 1:
            return "Legacy positions require one original-word mapping"
        original, mapping = next(iter(entry.items()))
        lexical = re.fullmatch(r"([^A-Za-z0-9_]*)([A-Za-z]+)([^A-Za-z0-9_]*)", token)
        core = lexical.group(2) if lexical else token
        edge_stripped = re.sub(r"^\W+|\W+$", "", token)
        if original.casefold() not in (core.casefold(), token.casefold(), edge_stripped.casefold()):
            return "Legacy original word does not match the aligned cover"
        if not isinstance(mapping, dict) or any(not isinstance(value, str) for value in mapping.values()):
            return "Invalid legacy candidate mapping"
    return None


def read_rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return [json.loads(line) for line in raw.splitlines() if line]


def audit(run_path: Path, archive_path: Path) -> dict:
    archive_sha = "793c419e19b09a038af439fceb0c8727f2d809e2e855444d66839fd8132ef790"
    assert digest(archive_path.read_bytes()) == archive_sha, "source archive identity"
    manifest = json.loads((run_path / "inputs_manifest.json").read_text())
    assert manifest["archive_sha256"] == archive_sha
    assert len(manifest["entries"]) == 375
    inputs = {}
    with zipfile.ZipFile(archive_path) as archive:
        for entry in manifest["entries"]:
            raw = archive.read(entry["archive_path"])
            assert digest(raw) == entry["source_sha256"]
            source = json.loads(raw)
            key = (entry["series"], source["uuid"])
            assert key not in inputs
            assert source["uuid"] == entry["uuid"]
            assert digest(source["container"].encode()) == entry["cover_sha256"]
            assert len(source["message"]) == entry["payload_bits"]
            inputs[key] = (source, entry)
    strata = Counter((series, len(value[0]["message"])) for (series, _), value in inputs.items())
    assert strata == Counter({("luna", 128): 100, ("luna", 256): 100, ("luna", 512): 100,
                              ("qwen", 128): 25, ("qwen", 256): 25, ("qwen", 512): 25})
    expected_names = {"synsteg_reproduction/reports/" + folder + "/" for folder in
                      ("gpt6_luna_missing", "qwen_missing_b3")}
    with zipfile.ZipFile(archive_path) as archive:
        all_names = {name for name in archive.namelist() if name.endswith(".json") and
                     any(name.startswith(prefix) for prefix in expected_names)}
    assert all_names == {entry["archive_path"] for entry in manifest["entries"]}

    rows = read_rows(run_path / "records.jsonl")
    books_list = read_rows(run_path / "books.jsonl.gz")
    books = {book["book_id"]: book for book in books_list}
    alignment_failures = {key: legacy_alignment_failure(value[0]) for key, value in inputs.items()}
    expected_book_count = 375 + 4 * sum(reason is None for reason in alignment_failures.values())
    assert len(books) == len(books_list) == expected_book_count
    jobs_list = read_rows(run_path / "receiver_jobs.jsonl.gz")
    jobs = {job["row_id"]: job for job in jobs_list}
    assert len(jobs) == len(jobs_list)
    outcomes_list = read_rows(run_path / "receiver_results.jsonl.gz")
    outcomes = {row["row_id"]: row for row in outcomes_list}
    assert len(outcomes) == len(outcomes_list) == len(jobs)
    assert set(outcomes) == set(jobs)
    assert len(rows) == len({row["row_id"] for row in rows}) == 6375
    configs = {"fixed_strict_b1_g2": (1, 2), "fixed_strict_b2_g2": (2, 2),
               "fixed_lowrate_b1_g1": (1, 1), "fixed_ceiling_b5_g1": (5, 1)}
    public_arms = {f"public_{v}_{f}" for v in ("v1", "v2") for f in ("plain_length16", "aes_siv")}
    variants = set(configs) | public_arms
    expected_row_ids = {f"{series}:{uuid}:{grid}:{arm}" for series, uuid in inputs
                        for grid, arms in (("primary", variants | {"original_archived"}),
                                           ("secondary16", variants)) for arm in arms}
    assert {row["row_id"] for row in rows} == expected_row_ids
    assert Counter(row["grid"] for row in rows) == Counter(primary=3375, secondary16=3000)
    dictionaries = {v: json.loads((run_path / f"recipient_resources/dictionary_{v}.json").read_text())
                    for v in ("v1", "v2")}
    dictionary_resources = json.loads((run_path / "resources.json").read_text())
    for resource in dictionary_resources["public_dictionaries"]:
        raw = (run_path / f"recipient_resources/dictionary_{resource['id']}.json").read_bytes()
        assert digest(raw) == resource["file_sha256"]
        assert len(raw) == resource["one_time_file_bytes"]
        assert len(zlib.compress(raw, 9)) == resource["one_time_file_zlib9_bytes"]
        assert digest(canonical(public_dictionary_canonical(json.loads(raw)))) == resource["dictionary_sha256"]
    assert dictionary_resources["per_public_receiver_secret_key_bytes"] == 32
    assert dictionary_resources["fixture_key_is_public"] is True
    assert dictionary_resources["fixture_key_hex"] == bytes(range(32)).hex(), "frozen fixture key"

    checked_books = set()
    decoded_counts = Counter()
    capacity_checks = Counter()
    row_statuses = Counter()
    verified_alignment_error_rows = 0
    for row in rows:
        assert row["row_id"] == ":".join((row["series"], row["uuid"], row["grid"], row["arm"]))
        source, source_entry = inputs[row["series"], row["uuid"]]
        expected = source["message"] if row["grid"] == "primary" else source["message"][:16]
        assert row["source_sha256"] == source_entry["source_sha256"]
        assert row["requested_payload_bits"] == len(expected)
        assert row["payload_sha256"] == digest(expected.encode())
        assert row["archived_payload_bits"] == len(source["message"])
        assert row["cover_words"] == len(source["container"].split())
        arm = row["arm"]
        alignment_failure = alignment_failures[row["series"], row["uuid"]]
        if arm in configs and alignment_failure:
            assert row["status"] == "error" and alignment_failure in row["error"]
            assert row["capacity_bits"] is None and "book_id" not in row
            assert row["row_id"] not in jobs
            assert row["raw_payload_fits"] is False
            assert row["external_length_bits"] == 32 and row["frame_overhead_bits"] == 0
            assert row["delivered_payload_bits"] == 0 and row["full_exact"] is False
            assert row["fresh_receiver_exact"] is False
            assert row["delivered_bits_per_source_word"] == 0
            assert row["delivered_bits_per_stego_word"] is None
            verified_alignment_error_rows += 1
            row_statuses["error"] += 1
            continue
        if arm in configs:
            book_entry = books[row["book_id"]]
            book = book_entry["value"]
            stats = fixed_book_accounting(book)
            assert stats["cover"] == source["container"]
            assert stats["sha256"] == row["book_sha256"] == book_entry["sha256"]
            assert stats["json_bytes"] == row["book_json_bytes"]
            assert stats["zlib9_bytes"] == row["book_zlib9_bytes"]
            assert stats["capacity_bits"] == row["capacity_bits"]
            if row["book_id"] not in checked_books:
                verify_fixed_mapping(book, source, *configs[arm])
                checked_books.add(row["book_id"])
            assert row["external_length_bits"] == 32 and row["frame_overhead_bits"] == 0
            assert row["raw_payload_fits"] == (stats["capacity_bits"] >= len(expected))
            capacity_checks["fixed"] += 1
        elif arm in public_arms:
            _, version, framing = arm.split("_", 2)
            protocol = {"version": "synsteg-shared-dictionary-aes-siv-v1", "min_group_size": 1,
                        "max_bits_per_slot": 2, "framing": framing}
            _, normalization, capacity = public_analysis(source["container"], dictionaries[version], protocol)
            assert capacity == row["capacity_bits"]
            assert normalization == row["normalization_sha256"]
            assert row["protocol_sha256"] == digest(canonical(protocol))
            assert row["dictionary_sha256"] == digest(canonical(public_dictionary_canonical(dictionaries[version])))
            assert row["external_length_bits"] == 0
            assert row["frame_overhead_bits"] == (144 if framing == "aes_siv" else 16)
            assert row["per_receiver_key_bytes"] == 32
            assert row["raw_payload_fits"] == (capacity >= len(expected))
            capacity_checks["public"] += 1
        else:
            assert arm == "original_archived" and row["grid"] == "primary"
            book_entry = books[row["book_id"]]
            assert book_entry["value"] == source["secret_key"]
            raw_key = json.dumps(source["secret_key"], ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode()
            assert digest(raw_key) == book_entry["sha256"] == row["book_sha256"]
            assert len(raw_key) == row["book_json_bytes"]
            assert len(zlib.compress(raw_key, 9)) == row["book_zlib9_bytes"]
            assert row["external_length_bits"] == row["frame_overhead_bits"] == 0
            assert row["reencoded_exact"] is True and row["replay_embedded_bits"] == len(expected)
        status = row["status"]
        row_statuses[status] += 1
        assert row["delivered_bits_per_source_word"] == row["delivered_payload_bits"] / row["cover_words"]
        assert row["delivered_bits_per_stego_word"] == (row["delivered_payload_bits"] / row["stego_words"]
                                                       if row.get("stego_words") else None)
        if status == "insufficient_capacity":
            assert row["capacity_bits"] < len(expected) + row["frame_overhead_bits"]
            assert row["row_id"] not in jobs
            assert row["delivered_payload_bits"] == 0 and row["full_exact"] is False
            continue
        assert status == "full_exact", f"unexpected recorded outcome {row['row_id']}: {status}"
        job = jobs[row["row_id"]]
        assert job["kind"] == ("fixed" if arm in configs else "public" if arm in public_arms else "original")
        if job["kind"] != "public":
            assert job["book_id"] == row["book_id"]
        assert digest(job["stego_text"].encode()) == row["stego_sha256"]
        assert len(job["stego_text"].split()) == row["stego_words"]
        if job["kind"] == "fixed":
            assert job["expected_bits"] == len(expected)
            recovered = fixed_oracle(books[job["book_id"]]["value"], job["stego_text"], job["expected_bits"])
        elif job["kind"] == "original":
            assert job["stego_text"] == source["encoded_message"]
            recovered = archived_oracle(job["stego_text"], books[job["book_id"]]["value"])
        else:
            assert job["kind"] == "public"
            assert job["fixture_key_hex"] == dictionary_resources["fixture_key_hex"]
            recovered_bytes = public_oracle(job["stego_text"], bytes.fromhex(job["fixture_key_hex"]),
                                            dictionaries[job["dictionary_version"]], job["protocol"])
            recovered = "".join(format(byte, "08b") for byte in recovered_bytes)
            _, after_normalization, after_capacity = public_analysis(
                job["stego_text"], dictionaries[job["dictionary_version"]], job["protocol"])
            assert after_normalization == row["normalization_sha256"] and after_capacity == row["capacity_bits"]
        assert recovered == expected == outcomes[row["row_id"]]["decoded_bits"]
        assert outcomes[row["row_id"]]["status"] == "decoded"
        assert row["decoded_bits"] == len(expected)
        assert row["decoded_sha256"] == digest(expected.encode())
        assert row["full_exact"] and row["fresh_receiver_exact"]
        assert row["delivered_payload_bits"] == len(expected)
        decoded_counts[job["kind"]] += 1

    summary = json.loads((run_path / "summary.json").read_text())
    assert summary["records"] == len(rows)
    expected_summary_groups = {(grid, series, arm, stratum) for grid, arms in (
        ("primary", variants | {"original_archived"}), ("secondary16", variants))
        for series in ("luna", "qwen") for arm in arms for stratum in (128, 256, 512, "all")}
    assert len(summary["groups"]) == len(expected_summary_groups) == 136
    assert {(g["grid"], g["series"], g["arm"], g["archived_payload_bits"])
            for g in summary["groups"]} == expected_summary_groups
    for group in summary["groups"]:
        subset = [r for r in rows if r["grid"] == group["grid"] and r["series"] == group["series"] and
                  r["arm"] == group["arm"] and (group["archived_payload_bits"] == "all" or
                                                 r["archived_payload_bits"] == group["archived_payload_bits"])]
        assert group["denominator"] == len(subset)
        assert group["statuses"] == dict(Counter(r["status"] for r in subset))
        assert group["full_exact"] == sum(r["full_exact"] for r in subset)
        assert group["requested_bits"] == sum(r["requested_payload_bits"] for r in subset)
        assert group["delivered_bits"] == sum(r["delivered_payload_bits"] for r in subset)
        assert group["zero_capacity"] == sum(r.get("capacity_bits") == 0 for r in subset)
        capacities = [r["capacity_bits"] for r in subset if r.get("capacity_bits") is not None]
        assert group["median_capacity_bits"] == (statistics.median(capacities) if capacities else None)
        assert group["mean_capacity_bits"] == (statistics.mean(capacities) if capacities else None)
        costs = [r["book_zlib9_bytes"] for r in subset if r.get("book_zlib9_bytes") is not None]
        assert group["median_book_zlib9_bytes"] == (statistics.median(costs) if costs else None)
        assert group["unconditional_delivered_bits_per_source_word"] == (
            sum(r["delivered_payload_bits"] for r in subset) / sum(r["cover_words"] for r in subset))
    output_hashes = json.loads((run_path / "output_manifest.json").read_text())
    for name, expected_hash in output_hashes.items():
        assert digest((run_path / name).read_bytes()) == expected_hash, f"output hash {name}"
    return {"schema": "matched-archive-independent-audit-v1", "status": "passed",
            "source_records": 375, "primary_outcomes": 3375, "secondary_outcomes": 3000,
            "outcome_statuses": dict(row_statuses), "independent_decodes": dict(decoded_counts),
            "independent_capacity_checks": dict(capacity_checks), "fixed_mappings_rederived": len(checked_books),
            "verified_native_adapter_rejected_sources": sum(value is not None for value in alignment_failures.values()),
            "verified_native_adapter_error_rows": verified_alignment_error_rows,
            "summary_groups_verified": len(summary["groups"]), "output_hashes_verified": len(output_hashes),
            "source_archive_sha256": archive_sha, "audit_script_sha256": digest(Path(__file__).read_bytes()),
            "limitations": ["All6375 outcomes reuse375 archived records, not6375 independent generated texts.",
                            "Same-payload comparisons retain different16/32/144-bit framing resource contracts.",
                            "No semantic, language-model-quality or steganalysis conclusion follows.",
                            "Original re-encode flag checked against runner; original decode independently reconstructed."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; preserve the previous audit")
    result = audit(args.run, args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
