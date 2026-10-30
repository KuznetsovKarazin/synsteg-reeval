#!/usr/bin/env python3
"""Independent mechanical audit of immutable prospective transport milestones.

No production encoder or decoder is imported. Public decoding re-derives the
lexer, class partition, HKDF/HMAC labels and frame with independently written
code. Passing this audit says nothing about meaning or steganographic security.
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
import zlib

from cryptography.hazmat.primitives.ciphers.aead import AESSIV


VERSION = "synsteg-shared-dictionary-aes-siv-v1"
BLOCKED = "BLOCKED_RESOURCE_OR_STAGE_NOT_RUN"
PUBLIC_ARMS = {f"public_{version}_{frame}" for version in ("v1", "v2")
               for frame in ("plain_length16", "aes_siv")}
FIXED_ARMS = {"strict_b1_g2", "strict_b2_g2", "lowrate_b1_g1", "ceiling_b5_g1"}
IMPLEMENTATION_FILES = {
    "scripts/run_prospective_development_transport_first60_v1.py",
    "scripts/audit_prospective_development_transport_first60_v1.py",
    "tests/test_prospective_transport_first60_v1.py",
    "docs/prospective_development_transport_first60_v1_ru.md",
}


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")).encode("ascii")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_bytes())


def read_rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return [json.loads(line) for line in raw.splitlines() if line]


def dictionary_canonical(raw: dict) -> dict:
    require(raw["schema"] == "synsteg-public-dictionary-v1", "dictionary schema")
    require(isinstance(raw["version"], str) and bool(raw["version"]), "dictionary version")
    seen_ids, seen_words, entries = set(), set(), []
    for entry in raw["classes"]:
        class_id, words = entry["id"], entry["words"]
        require(isinstance(class_id, str) and bool(class_id) and class_id not in seen_ids,
                "dictionary class identifier")
        require(isinstance(words, list) and len(words) >= 2, "dictionary class words")
        seen_ids.add(class_id)
        for word in words:
            require(isinstance(word, str) and re.fullmatch("[a-z]{2,}", word) is not None
                    and word not in seen_words, "dictionary word uniqueness/shape")
            seen_words.add(word)
        entries.append({"id": class_id, "words": sorted(words)})
    require(bool(entries), "empty dictionary")
    return {"schema": raw["schema"], "version": raw["version"],
            "classes": sorted(entries, key=lambda entry: entry["id"])}


def public_protocol(raw: dict) -> dict:
    allowed = {"version", "min_group_size", "max_bits_per_slot", "framing"}
    require(set(raw) <= allowed, "unknown protocol fields")
    require(raw.get("version", VERSION) == VERSION, "protocol version")
    require(type(raw["min_group_size"]) is int and raw["min_group_size"] >= 1,
            "protocol group size")
    require(type(raw["max_bits_per_slot"]) is int and raw["max_bits_per_slot"] in (1, 2),
            "protocol width")
    require(raw["framing"] in ("plain_length16", "aes_siv"), "protocol framing")
    return {"version": VERSION, "min_group_size": raw["min_group_size"],
            "max_bits_per_slot": raw["max_bits_per_slot"], "framing": raw["framing"]}


def public_analysis(text: str, dictionary: dict, protocol: dict) -> tuple[list[dict], str, int]:
    """Reconstruct whole-word eligibility and invariant literal/class sequence."""
    require(isinstance(text, str), "text type")
    lookup = {word: entry for entry in dictionary["classes"] for word in entry["words"]}
    slots, normalized, cursor = [], [], 0
    for token_index, match in enumerate(re.finditer(r"\S+", text)):
        lexical = re.fullmatch(r"([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)", match.group())
        if lexical is None:
            continue
        prefix, word, suffix = lexical.groups()
        if any(ch == "_" or not unicodedata.category(ch).startswith("P") for ch in prefix + suffix):
            continue
        entry = lookup.get(word.lower())
        if entry is None:
            continue
        style = "lower" if word.islower() else "upper" if word.isupper() else "title" if word.istitle() else None
        if style is None:
            continue
        groups = len(entry["words"]) // protocol["min_group_size"]
        width = min(protocol["max_bits_per_slot"], max(0, groups.bit_length() - 1))
        if not width:
            continue
        start, end = match.start() + len(prefix), match.end() - len(suffix)
        normalized.extend([["literal", text[cursor:start]], ["class", entry["id"], style]])
        slots.append({"index": len(slots), "token_index": token_index, "class_id": entry["id"],
                      "surface": word, "case_style": style, "width": width, "start": start, "end": end})
        cursor = end
    normalized.append(["literal", text[cursor:]])
    return slots, digest(canonical(normalized)), sum(slot["width"] for slot in slots)


def hkdf_sha256(key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """RFC5869 extract/expand using HMAC, without the codec's HKDF helper."""
    require(0 <= length <= 255 * 32, "HKDF length")
    prk = hmac.new(salt, key, hashlib.sha256).digest()
    block, output = b"", b""
    for counter in range(1, (length + 31) // 32 + 1):
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        output += block
    return output[:length]


def public_oracle(text: str, key: bytes, dictionary: dict, protocol: dict) -> bytes:
    """Decode from text+key+public resources only; never receives expected bits."""
    require(isinstance(key, bytes) and len(key) == 32, "fixture key length")
    dictionary, protocol = dictionary_canonical(dictionary), public_protocol(protocol)
    dsha, psha = digest(canonical(dictionary)), digest(canonical(protocol))
    slots, invariant, capacity = public_analysis(text, dictionary, protocol)
    overhead = 144 if protocol["framing"] == "aes_siv" else 16
    require(capacity >= overhead, "no complete frame capacity")
    salt = hashlib.sha256(b"synsteg/shared-dictionary/hkdf-salt/v1\x00" + bytes.fromhex(dsha)).digest()

    def derive(label, length):
        return hkdf_sha256(key, salt, canonical([VERSION, psha, label]), length)

    mapping_key = derive("mapping-hmac-sha256", 32)
    classes = {entry["id"]: entry["words"] for entry in dictionary["classes"]}
    stream, frame_size, payload_size = "", None, None
    for slot in slots:
        context = [VERSION, "word-rank", dsha, psha, invariant, slot["index"],
                   slot["token_index"], slot["class_id"], slot["width"]]
        ranked = sorted(classes[slot["class_id"]], key=lambda word: (
            hmac.new(mapping_key, canonical(context + [word]), hashlib.sha256).digest(), word))
        label = ranked.index(slot["surface"].lower()) % (2 ** slot["width"])
        stream += format(label, f"0{slot['width']}b")
        if frame_size is None and len(stream) >= 16:
            payload_size = int(stream[:16], 2)
            frame_size = overhead + 8 * payload_size
            require(frame_size <= capacity, "declared frame exceeds capacity")
        if frame_size is not None and len(stream) >= frame_size:
            break
    require(frame_size is not None and len(stream) >= frame_size, "incomplete frame")
    require(not set(stream[frame_size:]) - {"0"}, "nonzero final slot padding")
    frame = bytes(int(stream[i:i + 8], 2) for i in range(0, frame_size, 8))
    if protocol["framing"] == "plain_length16":
        result = frame[2:]
    else:
        aad = [b"synsteg/shared-dictionary/authenticated-frame/v1", canonical({
            "version": VERSION, "dictionary_sha256": dsha, "protocol": protocol,
            "normalization_sha256": invariant, "payload_bytes": payload_size})]
        result = AESSIV(derive("aead-aes-siv-512", 64)).decrypt(frame[2:], aad)
    require(len(result) == payload_size, "recovered payload length")
    return result


def verify_canonical_members(text: str, key: bytes, dictionary: dict, protocol: dict, used_slots: int) -> None:
    """Check sender's canonical selector separately from permissive decoding."""
    dictionary, protocol = dictionary_canonical(dictionary), public_protocol(protocol)
    dsha, psha = digest(canonical(dictionary)), digest(canonical(protocol))
    slots, invariant, _ = public_analysis(text, dictionary, protocol)
    require(type(used_slots) is int and 0 <= used_slots <= len(slots), "canonical used slot count")
    salt = hashlib.sha256(b"synsteg/shared-dictionary/hkdf-salt/v1\x00" + bytes.fromhex(dsha)).digest()
    mapping_key = hkdf_sha256(key, salt, canonical([VERSION, psha, "mapping-hmac-sha256"]), 32)
    classes = {entry["id"]: entry["words"] for entry in dictionary["classes"]}
    for slot in slots[:used_slots]:
        context = [VERSION, "word-rank", dsha, psha, invariant, slot["index"],
                   slot["token_index"], slot["class_id"], slot["width"]]
        ranked = sorted(classes[slot["class_id"]], key=lambda word: (
            hmac.new(mapping_key, canonical(context + [word]), hashlib.sha256).digest(), word))
        label = ranked.index(slot["surface"].lower()) % (2 ** slot["width"])
        group = ranked[label::2 ** slot["width"]]
        def render(word):
            return word.upper() if slot["case_style"] == "upper" else word.capitalize() if slot["case_style"] == "title" else word
        require(slot["surface"] == min(map(render, group)), "noncanonical encoder member")


def cell_identity(model_alias: str, record_id: str, parent: int, grid: str, arm: str) -> str:
    return digest(json.dumps([model_alias, record_id, parent, grid, arm], ensure_ascii=False,
                            sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def validate_cell_grid(cells: list[dict], records: list[dict], payload_map: dict) -> dict:
    """Validate all15300 identities and independent cover/resource/status axes.

    ``records`` are normalized input records with ``cover_status`` and are
    separately bound to immutable input checkpoints by ``audit``.
    """
    require(len(records) == 300, "input record denominator")
    lookup = {(row["model_alias"], row["record_id"]): row for row in records}
    require(len(lookup) == 300, "duplicate input model-record identity")
    require(Counter(row["model_alias"] for row in records) == Counter({
        "qwen05": 100, "tinyllama11": 100, "falcon3_1b": 100}), "model denominator")
    source_ids = {row["record_id"] for row in records}
    require(len(source_ids) == 100, "source denominator")
    for model in ("qwen05", "tinyllama11", "falcon3_1b"):
        require({row["record_id"] for row in records if row["model_alias"] == model} == source_ids,
                "models must share same100 sources")
    payload_rows = payload_map["rows"]
    payloads = {(row["record_id"], row["primary_payload_length"]): row for row in payload_rows}
    require(len(payload_rows) == len(payloads) == 300, "payload map denominator/duplicates")
    require(set(payloads) == {(record_id, length) for record_id in source_ids for length in (128, 256, 512)},
            "payload map identity set")
    for (record_id, parent), row in payloads.items():
        bits = row["payload_bits"]
        require(isinstance(bits, str) and len(bits) == parent and set(bits) <= {"0", "1"},
                "primary payload shape")
        require(row["secondary16_bits"] == bits[:16], "secondary payload prefix")
        require(row["payload_sha256"] == digest(bits.encode()), "mapped payload hash")
        require(row["secondary16_sha256"] == digest(bits[:16].encode()), "mapped secondary hash")
    variants = PUBLIC_ARMS | FIXED_ARMS
    expected = {cell_identity(model, record_id, parent, grid, arm)
                for model, record_id in lookup for parent in (128, 256, 512)
                for grid, arms in (("primary", variants | {"original_native"}), ("secondary16", variants))
                for arm in arms}
    require(len(expected) == len(cells) == 15300, "cell denominator")
    require({row["row_id"] for row in cells} == expected, "cell identity set/duplicates")
    resource_counts, execution_counts, cover_counts = Counter(), Counter(), Counter()
    for row in cells:
        model, record_id, parent, grid, arm = (row[k] for k in (
            "model_alias", "record_id", "parent_payload_length", "grid", "arm"))
        require(row["row_id"] == cell_identity(model, record_id, parent, grid, arm), "cell identity binding")
        record = lookup[(model, record_id)]
        for field in ("identity", "source_order"):
            require(row[field] == record[field], "cell input " + field)
        source_payload = payloads[(record_id, parent)]
        require(source_payload["source_order"] == record["source_order"], "payload source order")
        bits = source_payload["payload_bits"] if grid == "primary" else source_payload["secondary16_bits"]
        require(row["requested_payload_bits"] == len(bits) and row["payload_sha256"] == digest(bits.encode()),
                "cell requested payload")
        require(row["cover_status"] == record["cover_status"] and row["cover_status"] in
                ("completed", "not_attempted", "failed", "unknown"), "cover status axis")
        resource = "ready" if arm in PUBLIC_ARMS else BLOCKED
        require(row["resource_status"] == resource, "resource status axis")
        if row["cover_status"] != "completed":
            require(row["execution_status"] == "not_run_cover_unavailable", "cover status precedence")
        elif resource == BLOCKED:
            require(row["execution_status"] == "not_run_resource_blocked", "resource status precedence")
        else:
            require(row["execution_status"] in ("full_exact", "insufficient_capacity", "decoder_mismatch",
                    "implementation_error"), "executed public status")
        if row["execution_status"].startswith("not_run_"):
            for field in ("capacity_bits", "delivered_payload_bits"):
                require(row[field] is None, "unexecuted numeric outcome must be null: " + field)
        elif row["execution_status"] != "full_exact":
            require(row["delivered_payload_bits"] == 0, "unsuccessful execution delivers zero")
        else:
            require(row["delivered_payload_bits"] == len(bits), "exact delivered bits")
        resource_counts[resource] += 1
        execution_counts[row["execution_status"]] += 1
        cover_counts[row["cover_status"]] += 1
    require(resource_counts == Counter({"ready": 7200, BLOCKED: 8100}), "resource denominator")
    return {"cells": 15300, "input_records": 300, "source_groups": 100,
            "primary_cells": sum(row["grid"] == "primary" for row in cells),
            "secondary16_cells": sum(row["grid"] == "secondary16" for row in cells),
            "resource_statuses": dict(resource_counts), "execution_statuses": dict(execution_counts),
            "cover_statuses_per_cell": dict(cover_counts)}


def safe_file(directory: Path, relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe manifest path")
    value = directory / path
    require(value.is_file() and not value.is_symlink(), "missing/symlink manifest file: " + relative)
    require(directory.resolve() in value.resolve().parents, "manifest path escapes directory")
    return value


def check_summary(summary: dict, cells: list[dict], records: list[dict]) -> None:
    require(summary["planned_model_record_pairs"] == 300 and summary["source_groups"] == 100,
            "summary input denominators")
    require(summary["completed_covers"] == sum(r["status"] == "completed" for r in records), "summary completed covers")
    require(summary["cover_status_counts"] == dict(Counter(r["status"] for r in records)), "summary cover statuses")
    require(summary["total_cells"] == 15300 and summary["primary_cells"] == 8100 and
            summary["secondary16_cells"] == 7200, "summary cell denominators")
    for field in ("resource_status", "execution_status"):
        require(summary[field + "_counts"] == dict(Counter(r[field] for r in cells)), "summary " + field)
    groups = {}
    for row in cells:
        for model in ("all", row["model_alias"]):
            groups.setdefault((row["grid"], row["arm"], row["parent_payload_length"], model), []).append(row)
    require(len(summary["groups"]) == len(groups) == 204, "summary group denominator")
    seen = set()
    for group in summary["groups"]:
        identity = tuple(group[k] for k in ("grid", "arm", "parent_payload_length", "model_alias"))
        require(identity in groups and identity not in seen, "summary group identity")
        seen.add(identity)
        rows = groups[identity]
        measured = [r for r in rows if r["delivered_payload_bits"] is not None]
        capacities = [r["capacity_bits"] for r in measured if r["capacity_bits"] is not None]
        expected = {
            "planned_cells": len(rows), "executed_cells": len(measured),
            "cover_status_counts": dict(Counter(r["cover_status"] for r in rows)),
            "resource_status_counts": dict(Counter(r["resource_status"] for r in rows)),
            "execution_status_counts": dict(Counter(r["execution_status"] for r in rows)),
            "full_exact": sum(r["full_exact"] is True for r in rows),
            "requested_bits_all_planned": sum(r["requested_payload_bits"] for r in rows),
            "requested_bits_executed": sum(r["requested_payload_bits"] for r in measured),
            "confirmed_delivered_bits": sum(r["delivered_payload_bits"] for r in measured),
            "confirmed_useful_decoded_bytes": sum(r["useful_decoded_payload_bytes"] for r in measured),
            "unmeasured_cells": len(rows) - len(measured),
            "zero_capacity_executed": sum(r["capacity_bits"] == 0 for r in measured),
            "mean_capacity_bits_executed": statistics.mean(capacities) if capacities else None,
            "median_capacity_bits_executed": statistics.median(capacities) if capacities else None,
        }
        require(all(group[k] == value for k, value in expected.items()), "summary numeric/status group mismatch")


def validate_input_bound_rows(bound_rows, records, cells, dictionaries):
    """Check pre-encode metadata only; no cover analysis or payload decoder."""
    names = ('row_id', 'model_alias', 'record_id', 'source_group_id', 'source_order', 'identity',
             'attempt_sha256', 'cover_sha256', 'cover_status', 'resource_status',
             'parent_payload_length', 'grid', 'arm', 'requested_payload_bits', 'payload_sha256')
    require(len(bound_rows) == len(cells) == 15300, 'input binding full grid denominator')
    require(len({r['row_id'] for r in bound_rows}) == 15300, 'input binding duplicate identity')
    lookup = {(r['model_alias'], r['record_id']): r for r in records}
    require(len(lookup) == len(records) == 300, 'input binding records denominator')
    actions = Counter()
    for bound, cell in zip(bound_rows, cells):
        require(set(bound) == set(names) | {'source_record_sha256', 'planned_action', 'dictionary_sha256', 'protocol_sha256'},
                'input binding closed row schema')
        require({k: bound[k] for k in names} == {k: cell[k] for k in names}, 'input binding cell metadata')
        record = lookup[(cell['model_alias'], cell['record_id'])]
        raw = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        require(bound['source_record_sha256'] == digest(raw), 'input binding exact full record hash')
        if record['status'] != 'completed':
            action = 'not_run_cover_unavailable'
        elif cell['arm'] not in PUBLIC_ARMS:
            action = 'not_run_resource_blocked'
        else:
            require(record['model_alias'] == 'tinyllama11' and 1 <= record['source_order'] <= 60,
                    'input binding completed public cohort')
            action = 'reuse_exact_prior_outcome' if record['source_order'] <= 50 else 'new_encode_once'
        require(bound['planned_action'] == action, 'input binding planned action')
        actions[action] += 1
        if cell['arm'] in PUBLIC_ARMS:
            _, version, framing = cell['arm'].split('_', 2)
            require(bound['dictionary_sha256'] == digest(canonical(dictionaries[version])) and
                    bound['protocol_sha256'] == digest(canonical(public_protocol({'min_group_size': 1, 'max_bits_per_slot': 2, 'framing': framing}))),
                    'input binding public resource identity')
        else:
            require(bound['dictionary_sha256'] is None and bound['protocol_sha256'] is None,
                    'input binding blocked resource identity null')
    require(actions == Counter({'reuse_exact_prior_outcome': 1200, 'new_encode_once': 240,
                                'not_run_resource_blocked': 1620, 'not_run_cover_unavailable': 12240}),
            'input binding exact planned action denominators')
    return dict(actions)


def validate_input_binding(run, records, cells, plan, dictionaries, receipt):
    directory = run / 'input_binding'
    require({p.name for p in directory.iterdir()} == {'records.jsonl.gz', 'input_bound_grid.jsonl.gz', 'manifest.json'},
            'input binding exact file membership')
    manifest = read_json(directory / 'manifest.json')
    require(manifest['schema'] == 'prospective-first60-input-binding-v1' and
            manifest['experiment_id'] == 'prospective-development-transport-first60-v1' and
            manifest['checkpoint'] == 'results/large-campaign-development-v2/batches/batch30/after-checkpoint.json' and
            manifest['checkpoint_sha256'] == digest((run / 'input_checkpoint.json').read_bytes()) and
            manifest['prior_manifest_sha256'] == '5380b25e21c6f44e71bacaf9fef96b77d33ce43b35579f270b0d74345712cd85' and
            manifest['prior_audit_sha256'] == '9e0ed1cff0e540e4c6fdb73ebe35105b4dbd87969d96acea62df91ffb8422541' and
            manifest['feasibility_protocol_sha256'] == digest((run / 'feasibility-protocol.json').read_bytes()) and
            manifest['payload_map_sha256'] == plan['payloads']['map_sha256'], 'input binding source/plan commitment')
    require(manifest['model_record_pairs'] == 300 and manifest['cells'] == 15300 and manifest['completed_covers'] == 60 and
            manifest['reused_public_conditions'] == 1200 and manifest['new_public_conditions'] == 240 and
            manifest['encode_calls'] == manifest['decode_calls'] == manifest['model_calls'] == 0 and
            manifest['new_outcomes_known'] is False, 'input binding preparation scope')
    expected_files = {name: digest((directory / name).read_bytes()) for name in ('records.jsonl.gz', 'input_bound_grid.jsonl.gz')}
    require(manifest['files_sha256'] == expected_files, 'input binding manifest hashes')
    all_files = {**expected_files, 'manifest.json': digest((directory / 'manifest.json').read_bytes())}
    require(all(receipt['verified_files_sha256'].get('results/prospective-development-transport-first60-v1/input-binding/' + name) == sha
                for name, sha in all_files.items()), 'input binding actual Git receipt coverage')
    require(read_rows(directory / 'records.jsonl.gz') == records, 'input binding full records exact')
    return validate_input_bound_rows(read_rows(directory / 'input_bound_grid.jsonl.gz'), records, cells, dictionaries)


def audit_core(run: Path, frozen_root: Path, protocol_path: Path | None = None) -> dict:
    """Read immutable milestone only, checking every source and outcome cell."""
    run, frozen_root = Path(run).resolve(), Path(frozen_root).resolve()
    manifest = read_json(run / "output_manifest.json")
    require(isinstance(manifest, dict) and "output_manifest.json" not in manifest, "output manifest format")
    actual = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()}
    require(actual == set(manifest) | {"output_manifest.json"}, "output manifest exhaustive membership")
    for relative, expected in manifest.items():
        require(digest(safe_file(run, relative).read_bytes()) == expected, "output file hash: " + relative)
    plan_raw = (run / "feasibility-protocol.json").read_bytes()
    if protocol_path is not None:
        require(plan_raw == Path(protocol_path).read_bytes(), "supplied protocol differs")
    plan, activation, execution, receipt = (read_json(run / name) for name in
        ("feasibility-protocol.json", "activation.json", "execution.json", "git-readback-receipt.json"))
    require(plan["experiment_id"] == "prospective-development-transport-v1", "experiment identity")
    require(execution["feasibility_protocol_sha256"] == digest(plan_raw) == activation["feasibility_protocol_sha256"],
            "protocol hash binding")
    require(execution["activation_sha256"] == digest((run / "activation.json").read_bytes()), "activation hash")
    require(set(activation["implementation_files_sha256"]) == IMPLEMENTATION_FILES, "implementation freeze membership")
    require(activation["implementation_files_sha256"]["scripts/audit_prospective_development_transport_first60_v1.py"] ==
            digest(Path(__file__).read_bytes()), "independent auditor freeze identity")
    require(receipt["remote_readback_verified"] is True, "external Git readback flag")
    for name in ("implementation_freeze_commit", "checkpoint_commit"):
        require(execution[name] == receipt[name] and re.fullmatch("[0-9a-f]{40}", receipt[name]) is not None,
                "Git commit binding")
    require(execution["observed_completed_covers_at_implementation_freeze"] ==
            receipt["observed_completed_covers_at_implementation_freeze"], "prior data disclosure")
    require(activation["observed_completed_covers_at_implementation_freeze"] ==
            execution["observed_completed_covers_at_implementation_freeze"], "activation prior visibility")
    require(execution["runtime"] == activation["runtime"], "runtime identity")
    require(execution["fresh_receiver_has_cover_or_payload"] is False and
            execution["semantics_or_security_inferred"] is False, "scope assertions")
    require(execution["wall_seconds"] >= 0, "negative runtime")
    expected_pins = {pin["path"]: pin["sha256"] for pin in plan["resource_pins"]}
    require(execution["resource_files_sha256"] == expected_pins, "resource pin list")
    for path, expected in expected_pins.items():
        require(digest(safe_file(frozen_root, path).read_bytes()) == expected, "frozen resource SHA: " + path)
    for name, expected in (("input_config.json", plan["cover_cohort"]["plan_sha256"]),
                           ("payload-map.json", plan["payloads"]["map_sha256"])):
        require(digest((run / name).read_bytes()) == expected, "input artifact hash: " + name)
    require(execution["config_sha256"] == plan["cover_cohort"]["plan_sha256"] and
            execution["payload_map_sha256"] == plan["payloads"]["map_sha256"], "execution input hashes")
    cp_raw = (run / "input_checkpoint.json").read_bytes()
    cp, config, payload_map = json.loads(cp_raw), read_json(run / "input_config.json")["development_candidate"], read_json(run / "payload-map.json")
    require(execution["checkpoint_sha256"] == digest(cp_raw), "checkpoint hash")
    require(cp["status"] == "verified" and not cp["consistency_issues"] and cp["planned_model_record_pairs"] == 300,
            "checkpoint status")
    require(execution["source_checkpoint"] == activation["target"]["checkpoint"], "checkpoint source identity")
    target = activation["target"]
    require(cp["counts"] == target["cover_counts"] and cp["model_counts"] == target["model_counts"] and
            cp["completed_generations"] == target["completed_covers"], "activation target counts")
    expected_receipt = {**activation["implementation_files_sha256"],
        "experiments/prospective-development-transport-first60-v1/activation.json": execution["activation_sha256"],
        "experiments/prospective-development-transport-v1/protocol.json": digest(plan_raw),
        plan["payloads"]["map"]: plan["payloads"]["map_sha256"], target["checkpoint"]: digest(cp_raw)}
    require(all(receipt["verified_files_sha256"].get(k) == v for k, v in expected_receipt.items()), "Git receipt file coverage")
    actual_snapshot = {str(p.relative_to(run / "input_snapshot")) for p in (run / "input_snapshot").rglob("*") if p.is_file()}
    require(actual_snapshot == set(cp["run_files_sha256"]), "input snapshot exact membership")
    for path, expected in cp["run_files_sha256"].items():
        require(digest(safe_file(run / "input_snapshot", path).read_bytes()) == expected, "snapshot file hash")
    records = read_rows(run / "records.jsonl.gz")
    metadata = {(m["alias"], r["record_id"]): (m, r, index) for m in config["local_models"]
                for index, r in enumerate(config["selected_record_metadata"], 1)}
    require(len(records) == len(cp["records"]) == len(metadata) == 300, "checkpoint denominator")
    require(len({(r["model_alias"], r["record_id"]) for r in cp["records"]}) == 300, "checkpoint duplicate input")
    for row, source in zip(records, cp["records"]):
        model, meta, order = metadata[(source["model_alias"], source["record_id"])]
        require(all(source[k] == v for k, v in meta.items()) and source["model"] == model["model_id"] and
                source["revision"] == model["revision"], "input source metadata")
        expected = dict(source, source_order=order, cover_text=None, attempt_sha256=None)
        if source["status"] == "completed":
            relative = f"runs/large-campaign-development-v2/models/{source['model_alias']}/attempts/{source['identity']}.json"
            attempt_raw = safe_file(run / "input_snapshot", relative).read_bytes()
            attempt = json.loads(attempt_raw)
            text = attempt["result"]["cover_text"]
            require(attempt["status"] == "completed" and attempt["identity"] == source["identity"] and
                    attempt["record"] == meta, "attempt identity")
            require(digest(text.encode()) == source["cover_sha256"] and len(text.split()) == source["whitespace_words"],
                    "attempt cover content")
            expected.update(cover_text=text, attempt_sha256=digest(attempt_raw))
        require(row == expected, "full input record differs from checkpoint/attempt")
    require(Counter(r["status"] for r in records) == Counter(cp["counts"]), "checkpoint status totals")
    model_completed = {model: sum(row["model_alias"] == model and row["status"] == "completed" for row in records)
                       for model in ("qwen05", "tinyllama11", "falcon3_1b")}
    require(cp["model_counts"] == model_completed and cp["completed_generations"] == sum(model_completed.values()),
            "checkpoint per-model completion totals")
    cells = read_rows(run / "cells.jsonl.gz")
    grid = validate_cell_grid(cells, [dict(r, cover_status=r["status"]) for r in records], payload_map)
    before_rows = read_rows(run / "cells_before_receiver.jsonl.gz")
    before = {r["row_id"]: r for r in before_rows}
    require(len(before_rows) == len(before) == len(cells) and set(before) == {r["row_id"] for r in cells}, "pre-receiver cell membership")
    recipient = run / "recipient"
    require({p.name for p in recipient.iterdir()} == {"shared_dictionary.py", "dictionary_v1.json", "dictionary_v2.json",
        "receiver_jobs.jsonl.gz", "receiver_results.jsonl.gz"}, "recipient exact resources")
    require(digest((recipient / "shared_dictionary.py").read_bytes()) == expected_pins["src/synsteg_audit/shared_dictionary.py"],
            "recipient implementation identity")
    resources = read_json(run / "resources.json")
    resource_entries = {r["id"]: r for r in resources["public_dictionaries"]}
    require(len(resource_entries) == len(resources["public_dictionaries"]) == 2, "dictionary resource denominator")
    dictionaries, completed = {}, sum(r["status"] == "completed" for r in records)
    for resource in plan["resource_accounting"]["public_installation"]:
        version = resource["id"]
        raw = (recipient / f"dictionary_{version}.json").read_bytes()
        dictionary = dictionary_canonical(json.loads(raw))
        require(digest(raw) == resource["file_sha256"] and len(raw) == resource["one_time_file_bytes"] and
                len(zlib.compress(raw, 9)) == resource["one_time_file_zlib9_bytes"] and
                digest(canonical(dictionary)) == resource["dictionary_sha256"], "dictionary byte/semantic accounting")
        entry = resource_entries[version]
        expected = dict(resource,
            installation_bytes_per_completed_cover=len(raw) / completed if completed else None,
            installation_zlib9_bytes_per_completed_cover=len(zlib.compress(raw, 9)) / completed if completed else None,
            installation_bytes_per_planned_cover=len(raw) / 300,
            amortization_completed_cover_denominator=completed, amortization_planned_cover_denominator=300)
        require(entry == expected, "dictionary amortization denominator")
        dictionaries[version] = dictionary
    input_actions = validate_input_binding(run, records, cells, plan, dictionaries, receipt)
    for field, value in {"per_public_receiver_secret_key_bytes": 32, "fixture_key_hex": bytes(range(32)).hex(),
        "fixture_key_is_public": True, "public_per_text_map_bytes": 0, "blocked_original_fixed_map_bytes": None,
        "dictionary_installation_shared_across_framings_payloads_and_cells": True,
        "key_reused_across_fixture_cells_not_new_key_per_cell": True,
        "protocol_identifiers_and_transport_packaging_excluded_from_byte_counts": True}.items():
        require(resources[field] == value, "resource contract: " + field)
    cumulative = run / "cumulative_recipient"
    require({p.name for p in cumulative.iterdir()} == {"receiver_jobs.jsonl.gz", "receiver_results.jsonl.gz"},
            "cumulative recipient contains audit records only")
    jobs_list, results_list = (read_rows(cumulative / name) for name in ("receiver_jobs.jsonl.gz", "receiver_results.jsonl.gz"))
    new_jobs_list, new_results_list = (read_rows(recipient / name) for name in ("receiver_jobs.jsonl.gz", "receiver_results.jsonl.gz"))
    jobs, outcomes = {r["row_id"]: r for r in jobs_list}, {r["row_id"]: r for r in results_list}
    require(len(jobs_list) == len(jobs) == len(results_list) == len(outcomes) and set(jobs) == set(outcomes), "receiver membership")
    process = read_json(run / "receiver_process.json")
    require(process["returncode"] == 0 and process["job_count"] == len(new_jobs_list) and process["payload_or_cover_argument"] is False and
            process["receiver_cwd"] == "recipient" and set(process["receiver_resource_files"]) == {p.name for p in recipient.iterdir()},
            "fresh receiver process receipt")
    lookup = {(r["model_alias"], r["record_id"]): r for r in records}
    payloads = {(r["record_id"], r["primary_payload_length"]): r for r in payload_map["rows"]}
    after_fields = {"execution_status", "full_exact", "fresh_receiver_exact", "delivered_payload_bits", "useful_decoded_payload_bytes",
                    "decoded_bits", "decoded_sha256", "error", "delivered_bits_per_source_word", "delivered_bits_per_stego_word"}
    null_fields = ("capacity_bits", "delivered_payload_bits", "useful_decoded_payload_bytes", "book_json_bytes", "book_zlib9_bytes",
        "per_text_map_bytes", "max_payload_bytes", "can_frame_empty", "can_send_one_byte", "raw_payload_fits", "frame_bits",
        "padding_bits", "used_positions", "changed_positions", "normalization_sha256", "stego_sha256", "stego_words", "decoded_bits",
        "decoded_sha256", "full_exact", "fresh_receiver_exact", "error", "delivered_bits_per_source_word", "delivered_bits_per_stego_word", "encoder_called")
    independently_decoded, capacity_checks = 0, 0
    for row in cells:
        prior = before[row["row_id"]]
        require({k: v for k, v in row.items() if k not in after_fields} ==
                {k: v for k, v in prior.items() if k not in after_fields}, "receiver mutated sender/input metrics")
        record = lookup[(row["model_alias"], row["record_id"])]
        payload = payloads[(row["record_id"], row["parent_payload_length"])]
        expected_bits = payload["payload_bits"] if row["grid"] == "primary" else payload["secondary16_bits"]
        require(row["requested_payload_bytes"] == len(expected_bits) // 8 and
                row["attempt_sha256"] == record["attempt_sha256"] and row["source_group_id"] == record["source_group_id"], "cell source binding")
        text = record["cover_text"]
        require(row["cover_words"] == (len(text.split()) if text is not None else None) and
                row["cover_sha256"] == (digest(text.encode()) if text is not None else None), "cell cover accounting")
        public = row["arm"] in PUBLIC_ARMS
        if not public:
            require(row["resource_contract"] == "missing_author_generated_per_cover_candidates", "blocked resource contract")
            for field in ("frame_overhead_bits", "external_length_bits", "minimum_nonempty_frame_bits", "per_receiver_key_bytes", "dictionary_sha256", "protocol_sha256"):
                require(row[field] is None, "blocked resource metric must be null")
        else:
            _, version, framing = row["arm"].split("_", 2)
            require(row["resource_contract"] == "stego+32byte_key+public_dictionary_and_protocol", "public receiver resource contract")
            protocol = public_protocol({"min_group_size": 1, "max_bits_per_slot": 2, "framing": framing})
            overhead = 144 if framing == "aes_siv" else 16
            require(row["frame_overhead_bits"] == overhead and row["external_length_bits"] == 0 and
                    row["minimum_nonempty_frame_bits"] == overhead + 8 and row["per_receiver_key_bytes"] == 32,
                    "public framing accounting")
            require(row["dictionary_sha256"] == digest(canonical(dictionaries[version])) and
                    row["protocol_sha256"] == digest(canonical(protocol)), "public identities")
        if row["execution_status"].startswith("not_run_"):
            require(all(row[field] is None for field in null_fields), "nonexecution must keep all outcomes null")
            require(row["row_id"] not in jobs and prior == row, "unexecuted receiver leakage/change")
            continue
        require(public and text is not None, "only ready public cover can execute")
        require(type(row["encoder_called"]) is bool, "executed encoder call flag")
        if row['model_alias'] == 'tinyllama11' and 1 <= row['source_order'] <= 50:
            # Exact prior rows and pinned independent audit are checked below.
            # No historical encoder, receiver, or independent decoder replay.
            continue
        require(row['model_alias'] == 'tinyllama11' and 51 <= row['source_order'] <= 60,
                'independent new assessment outside fixed extension')
        slots, invariant, capacity = public_analysis(text, dictionaries[version], protocol)
        capacity_checks += 1
        if row["execution_status"] == "implementation_error" and row["encoder_called"] is False:
            require(row["row_id"] not in jobs and bool(row["error"]) and row["full_exact"] is False and
                    row["fresh_receiver_exact"] is False and row["useful_decoded_payload_bytes"] == 0, "pre-encoder analysis error")
            require(all(row[field] is None for field in ("capacity_bits", "normalization_sha256", "max_payload_bytes",
                    "can_frame_empty", "can_send_one_byte", "raw_payload_fits")), "analysis error must not invent metrics")
            continue
        require(row["encoder_called"] is True, "encoded public call flag")
        require(row["capacity_bits"] == capacity and row["normalization_sha256"] == invariant, "independent capacity/invariant")
        maximum = min(65535, (capacity - overhead) // 8) if capacity >= overhead else None
        require(row["max_payload_bytes"] == maximum and row["can_frame_empty"] == (capacity >= overhead) and
                row["can_send_one_byte"] == (capacity >= overhead + 8) and row["raw_payload_fits"] == (capacity >= len(expected_bits)),
                "minimum useful-byte and raw-fit accounting")
        require(row["book_json_bytes"] == row["book_zlib9_bytes"] == row["per_text_map_bytes"] == 0, "public per-text resource cost")
        require(row["delivered_bits_per_source_word"] == (row["delivered_payload_bits"] / len(text.split()) if text.split() else None),
                "source rate")
        require(row["delivered_bits_per_stego_word"] == (row["delivered_payload_bits"] / row["stego_words"] if row["stego_words"] else None),
                "stego rate")
        if row["execution_status"] == "insufficient_capacity":
            require(capacity < overhead + len(expected_bits) and row["row_id"] not in jobs, "capacity refusal correctness")
            require(row["full_exact"] is False and row["fresh_receiver_exact"] is False and row["useful_decoded_payload_bytes"] == 0,
                    "capacity refusal not success")
            require(all(row[field] is None for field in ("frame_bits", "padding_bits", "used_positions", "changed_positions",
                "stego_sha256", "stego_words", "decoded_bits", "decoded_sha256")), "capacity refusal has no output")
            continue
        if row["execution_status"] == "implementation_error":
            require(row["row_id"] not in jobs and bool(row["error"]) and row["full_exact"] is False and
                    row["fresh_receiver_exact"] is False and row["useful_decoded_payload_bytes"] == 0, "recorded implementation error")
            continue
        require(row["row_id"] in jobs and prior["execution_status"] == "awaiting_receiver", "completed receiver source")
        job, outcome = jobs[row["row_id"]], outcomes[row["row_id"]]
        require(set(job) == {"row_id", "kind", "dictionary_version", "stego_text", "fixture_key_hex", "protocol"}, "receiver field whitelist")
        require(job["kind"] == "public" and job["dictionary_version"] == version and job["fixture_key_hex"] == bytes(range(32)).hex() and
                job["protocol"] == {k: v for k, v in protocol.items() if k != "version"}, "receiver public contract")
        stego = job["stego_text"]
        require(row["stego_sha256"] == digest(stego.encode()) and row["stego_words"] == len(stego.split()), "stego identity/length")
        after_slots, after_invariant, after_capacity = public_analysis(stego, dictionaries[version], protocol)
        require((after_invariant, after_capacity) == (invariant, capacity), "eligibility invariant after replacement")
        frame_bits, used, observed = overhead + len(expected_bits), 0, 0
        while observed < frame_bits and used < len(slots):
            observed += slots[used]["width"]
            used += 1
        require(observed >= frame_bits and row["frame_bits"] == frame_bits and row["used_positions"] == used and
                row["padding_bits"] == observed - frame_bits, "frame/slot accounting")
        require(row["changed_positions"] == sum(a["surface"] != b["surface"] for a, b in zip(slots[:used], after_slots[:used])) and
                all(a["surface"] == b["surface"] for a, b in zip(slots[used:], after_slots[used:])), "replacement/tail accounting")
        verify_canonical_members(stego, bytes(range(32)), dictionaries[version], protocol, used)
        require(set(outcome) == {"row_id", "status", "decoded_bits", "error"}, "receiver outcome fields")
        recovered, oracle_error = None, None
        try:
            recovered = "".join(f"{value:08b}" for value in public_oracle(stego, bytes(range(32)), dictionaries[version], protocol))
        except Exception as exc:
            oracle_error = type(exc).__name__
        if oracle_error is None:
            require(outcome["status"] == "decoded" and outcome["decoded_bits"] == recovered and outcome["error"] is None,
                    "independent decoded result")
            independently_decoded += 1
        else:
            require(outcome["status"] == "error" and outcome["decoded_bits"] is None and bool(outcome["error"]), "independent decode rejection")
        exact = recovered == expected_bits
        require(row["execution_status"] == ("full_exact" if exact else "decoder_mismatch") and row["full_exact"] is exact and
                row["fresh_receiver_exact"] is exact, "exact delivery classification")
        require(row["decoded_bits"] == (len(recovered) if recovered is not None else None) and
                row["decoded_sha256"] == (digest(recovered.encode()) if recovered is not None else None), "decoded payload accounting")
        require(row["useful_decoded_payload_bytes"] == (len(expected_bits) // 8 if exact else 0), "useful decoded bytes")
        require(row["error"] is None if exact else bool(row["error"]), "receiver error classification")
    require(set(jobs) == {r["row_id"] for r in cells if r["execution_status"] in ("full_exact", "decoder_mismatch")}, "no unbound receiver jobs")
    summary, status = read_json(run / "summary.json"), read_json(run / "status.json")
    check_summary(summary, cells, records)
    for item in (execution, summary, status, cp):
        require(item["paid_calls"] == 0 and item["closed_test_opened"] is False, "paid/test boundary")
    for item in (execution, summary, status):
        require(item["model_calls"] == 0, "unexpected model call")
    actual_encoder_calls = sum(row["encoder_called"] is True for row in cells)
    require(execution["encoder_calls"] == actual_encoder_calls and capacity_checks == 240 and
            execution["receiver_jobs"] == len(jobs), "execution call accounting")
    require(status["status"] == "completed_pending_independent_audit" and status["completed_covers"] == completed and
            status["full_denominator"] == 300 and status["total_cells"] == 15300 and
            status["encoder_calls"] == actual_encoder_calls and status["receiver_jobs"] == len(jobs), "completion status")
    errors = sum(r["execution_status"] in ("decoder_mismatch", "implementation_error") for r in cells)
    require(status["unexpected_errors"] == errors, "recorded error denominator")
    return {"schema": "prospective-development-transport-independent-audit-v1", "status": "passed", **grid,
            "completed_covers": completed, "output_hashes_verified": len(manifest), "snapshot_hashes_verified": len(actual_snapshot),
            "independent_capacity_checks": capacity_checks, "encoder_calls_recorded": actual_encoder_calls,
            "independent_decodes": independently_decoded,
            "input_binding_cells_verified": 15300, "input_binding_actions_verified": input_actions,
            "independent_assessment_scope": "new orders51-60 only; exact prior1200 rows inherit pinned first50 audit",
            "prior_independent_decodes_replayed": 0,
            "receiver_jobs": len(jobs), "unexpected_errors_preserved": errors, "summary_groups_verified": 204,
            "output_manifest_sha256": digest((run / "output_manifest.json").read_bytes()),
            "audit_script_sha256": digest(Path(__file__).read_bytes()), "model_calls": 0, "paid_calls": 0,
            "limitations": ["Recorded subprocess receipts and hashes corroborate separation; no operating-system sandbox claim.",
                "Roundtrip is mechanical delivery, not meaning preservation, factuality or steganographic security.",
                "15300 cells reuse100 source groups and300 planned model-record pairs; missing/blocked outcomes remain unmeasured."]}


def validate_reuse(prior_records, records, prior_cells, cells, prior_before, before):
    """Independent row equality and full-grid accounting; no encoder imports."""
    key = lambda r: (r['model_alias'], r['record_id'])
    old_records = {key(r): r for r in prior_records}
    require(len(old_records) == len(records) == 300 and set(old_records) == {key(r) for r in records}, 'reuse record membership')
    new_keys = set()
    for row in records:
        old = old_records[key(row)]
        if row['model_alias'] == 'tinyllama11' and 51 <= row['source_order'] <= 60:
            require(old['status'] == 'not_attempted' and row['status'] == 'completed', 'new cover status transition')
            for field in ('model_alias', 'record_id', 'source_group_id', 'source_order', 'model', 'revision', 'target_words', 'genre', 'topic', 'split', 'identity', 'prompt_sha256'):
                require(old[field] == row[field], 'new cover metadata drift')
            new_keys.add(key(row))
        else:
            require(old == row, 'unchanged prior record differs')
    require(len(new_keys) == 10, 'new cover denominator')
    require([(r['model_alias'], r['source_order']) for r in records if r['status'] == 'completed'] == [('tinyllama11', n) for n in range(1, 61)], 'exact first60 cohort')
    require(len(prior_cells) == len(cells) == len(prior_before) == len(before) == 15300, 'reuse full grid denominator')
    require([r['row_id'] for r in prior_cells] == [r['row_id'] for r in cells] == [r['row_id'] for r in prior_before] == [r['row_id'] for r in before], 'reuse ordering/identity')
    reused_public, new_public, reused_all = [], [], 0
    for old, current, old_pre, current_pre in zip(prior_cells, cells, prior_before, before):
        if key(current) in new_keys:
            require(old['execution_status'] == 'not_run_cover_unavailable' and old['encoder_called'] is None, 'attempted prior row was replaced')
            if current['arm'] in PUBLIC_ARMS:
                new_public.append(current)
        else:
            require(old == current and old_pre == current_pre, 'old row or pre-receiver row changed')
            reused_all += 1
            if old['encoder_called'] is True:
                reused_public.append(old)
    require(reused_all == 14790 and len(reused_public) == 1200 and len(new_public) == 240, 'reuse/new cell denominators')
    # Use the frozen producer serialization, independently expressed here, to
    # bind exact UTF-8 rows (the oracle's canonical ASCII differs for Unicode).
    row_bytes = b''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n' for r in reused_public)
    return {'prior_public_rows_sha256': digest(row_bytes), 'new_encoder_calls': sum(r['encoder_called'] is True for r in new_public), 'new_keys': new_keys}


def validate_receiver_reuse(prior_jobs, prior_outcomes, new_jobs, new_outcomes, jobs, outcomes, new_keys, cells):
    """Pure concatenation/membership audit; never invokes a receiver or oracle."""
    require(len(prior_jobs) == len(prior_outcomes) == 126, 'prior receiver denominator')
    require(len(new_jobs) == len(new_outcomes), 'new receiver denominator')
    for name, rows in [('prior jobs', prior_jobs), ('prior outcomes', prior_outcomes),
                       ('new jobs', new_jobs), ('new outcomes', new_outcomes),
                       ('cumulative jobs', jobs), ('cumulative outcomes', outcomes)]:
        require(len({r['row_id'] for r in rows}) == len(rows), 'duplicate ' + name)
    require([r['row_id'] for r in prior_jobs] == [r['row_id'] for r in prior_outcomes], 'prior receiver ordering')
    require([r['row_id'] for r in new_jobs] == [r['row_id'] for r in new_outcomes], 'new receiver ordering')
    require(jobs == prior_jobs + new_jobs and outcomes == prior_outcomes + new_outcomes,
            'cumulative receiver records differ from exact prior plus new')
    require(not ({r['row_id'] for r in prior_jobs} & {r['row_id'] for r in new_jobs}), 'old receiver job replayed')
    by_id = {r['row_id']: r for r in cells}
    require(len(by_id) == len(cells), 'duplicate cell identity')
    require(all(j['row_id'] in by_id and
                (by_id[j['row_id']]['model_alias'], by_id[j['row_id']]['record_id']) in new_keys
                for j in new_jobs), 'receiver new job outside delta')
    return len(new_jobs)


def audit(run: Path, frozen_root: Path, protocol_path: Path | None = None) -> dict:
    """Independent new-only oracle plus exact inherited first50 evidence."""
    result = audit_core(run, frozen_root, protocol_path)
    run = Path(run).resolve()
    prior = run / 'prior_milestone'
    prior_manifest_sha = '5380b25e21c6f44e71bacaf9fef96b77d33ce43b35579f270b0d74345712cd85'
    prior_archive_sha = '7d780686c8cc43ecd5651488bdabe9e821737b29fed5e46255b05bb10a625814'
    prior_audit_sha = '9e0ed1cff0e540e4c6fdb73ebe35105b4dbd87969d96acea62df91ffb8422541'
    require(digest((prior / 'output_manifest.json').read_bytes()) == prior_manifest_sha, 'exact first50 output manifest')
    manifest = read_json(prior / 'output_manifest.json')
    require({str(p.relative_to(prior)) for p in prior.rglob('*') if p.is_file()} == set(manifest) | {'output_manifest.json'}, 'exact prior files')
    for relative, expected in manifest.items():
        require(digest(safe_file(prior, relative).read_bytes()) == expected, 'exact prior file bytes')
    require(digest((run / 'prior-independent-audit.json').read_bytes()) == prior_audit_sha, 'exact prior independent audit')
    prior_audit = read_json(run / 'prior-independent-audit.json')
    require(prior_audit['status'] == 'passed' and prior_audit['completed_covers'] == 50 and
            prior_audit['output_manifest_sha256'] == prior_manifest_sha and
            prior_audit['independent_capacity_checks'] == 240 and prior_audit['cumulative_public_conditions'] == 1200 and prior_audit['new_encoder_calls'] == 240 and prior_audit['reused_encoder_calls'] == 960 and prior_audit['prior_receiver_jobs_replayed'] == 0 and prior_audit['receiver_jobs'] == 126 and
            prior_audit['audit_script_sha256'] == '1f8410ea6a0b01c2dd0913eeccd26962d4b5b11db26927d94e9659a83e57f8c4',
            'inherited audit scope/identity')
    activation, execution, receipt, status, reuse, summary = (read_json(run / name) for name in ('activation.json', 'execution.json', 'git-readback-receipt.json', 'status.json', 'reuse.json', 'summary.json'))
    experiment = 'prospective-development-transport-first60-v1'
    for value in (activation, execution, status, summary):
        require(value['experiment_id'] == experiment, 'new experiment identity')
    require(activation['execution_enabled'] is True and activation['observed_completed_covers_at_implementation_freeze'] == 60, 'new activation and prior-output disclosure')
    require(activation['prior']['output_manifest_sha256'] == prior_manifest_sha and activation['prior']['archive_sha256'] == prior_archive_sha and activation['prior']['independent_audit_sha256'] == prior_audit_sha, 'activation prior binding')
    target = activation['target']
    require(target['batch'] == 'batch30' and target['checkpoint'] == 'results/large-campaign-development-v2/batches/batch30/after-checkpoint.json' and target['output'] == 'results/' + experiment + '/milestones/tinyllama-first60', 'fixed first60 target')
    prior_records, records = (read_rows(p / 'records.jsonl.gz') for p in (prior, run))
    prior_cells, cells = (read_rows(p / 'cells.jsonl.gz') for p in (prior, run))
    prior_before, before = (read_rows(p / 'cells_before_receiver.jsonl.gz') for p in (prior, run))
    checked = validate_reuse(prior_records, records, prior_cells, cells, prior_before, before)
    expected_reuse = {'schema': 'prospective-first60-reuse-v1', 'prior_output_manifest_sha256': prior_manifest_sha, 'prior_archive_sha256': prior_archive_sha, 'prior_public_rows_sha256': checked['prior_public_rows_sha256'], 'current_reused_public_rows_sha256': checked['prior_public_rows_sha256'], 'reused_encoder_calls': 1200, 'new_encoder_calls': checked['new_encoder_calls'], 'cumulative_encoder_calls': 1200 + checked['new_encoder_calls'], 'cumulative_public_conditions': 1440, 'new_public_conditions': 240, 'unchanged_record_rows': 290, 'reused_cell_rows': 14790, 'updated_cell_rows': 510, 'new_source_orders': list(range(51, 61))}
    require(all(reuse.get(k) == v for k, v in expected_reuse.items()), 'reuse report accounting')
    prior_jobs, prior_outcomes = (read_rows(prior / 'cumulative_recipient' / name) for name in ('receiver_jobs.jsonl.gz', 'receiver_results.jsonl.gz'))
    new_jobs, new_outcomes = (read_rows(run / 'recipient' / name) for name in ('receiver_jobs.jsonl.gz', 'receiver_results.jsonl.gz'))
    jobs, outcomes = (read_rows(run / 'cumulative_recipient' / name) for name in ('receiver_jobs.jsonl.gz', 'receiver_results.jsonl.gz'))
    new_job_count = validate_receiver_reuse(prior_jobs, prior_outcomes, new_jobs, new_outcomes, jobs, outcomes, checked['new_keys'], cells)
    for value in (reuse, execution):
        require(value['prior_receiver_jobs_replayed'] == 0 and value['prior_receiver_jobs_reused'] == 126 and value['new_receiver_jobs'] == new_job_count, 'receiver no-replay accounting')
    process = read_json(run / 'receiver_process.json')
    require(process['replayed_prior_jobs'] == 0 and process['reused_prior_jobs'] == 126 and process['new_jobs'] == new_job_count and process['job_count'] == new_job_count, 'receiver process new-only accounting')
    for value in (execution, status, summary):
        require(value['new_encoder_calls'] == checked['new_encoder_calls'] and value['reused_encoder_calls'] == 1200, 'new versus reused attempts')
    require(execution['encoder_calls'] == status['encoder_calls'] == 1200 + checked['new_encoder_calls'] and execution['encoder_calls_scope'] == 'cumulative_including_exact_prior1200', 'cumulative attempts label')
    require(execution['public_conditions'] == status['public_conditions'] == summary['cumulative_public_conditions'] == 1440, 'cumulative conditions')
    require(summary['cumulative_encoder_calls'] == 1200 + checked['new_encoder_calls'], 'summary attempts')
    intent_raw = (run / 'execution-intent.json').read_bytes()
    intent = json.loads(intent_raw)
    require(intent['experiment_id'] == experiment and intent['phase'] == 'prepared_before_encode' and intent['new_encoder_attempts'] == 240 and intent['reused_encoder_attempts'] == 1200 and intent['outcomes_known'] is False, 'new intent scope')
    require(execution['intent_sha256'] == digest(intent_raw) and execution['intent_commit'] == receipt['intent_commit'] and re.fullmatch('[0-9a-f]{40}', receipt['intent_commit']) is not None, 'intent Git identity')
    inherited = {'scripts/run_prospective_development_transport_v1.py': '2a927b07f0a375a77725b7b87a1a05fef7c87485b710e0bc3ffc44bf2b599f03', 'scripts/audit_prospective_development_transport_v1.py': 'f3919bfb86ff8063c722506c21e5fcc952a2c720c36fe1ee804f14e3b9020fab'}
    required_receipts = {**inherited, 'results/' + experiment + '/execution-intent.json': digest(intent_raw), 'results/prospective-development-transport-first50-v1/milestones/tinyllama-first50/output_manifest.json': prior_manifest_sha, 'results/prospective-development-transport-first50-v1/audits/tinyllama-first50.json': prior_audit_sha}
    require(all(receipt['verified_files_sha256'].get(k) == v for k, v in required_receipts.items()), 'additive provenance receipt')
    result.update(schema='prospective-development-transport-first60-independent-audit-v1', new_encoder_calls=checked['new_encoder_calls'], reused_encoder_calls=1200, cumulative_public_conditions=1440, prior1200_rows_exact=True, prior_receiver_jobs_replayed=0, prior_receiver_jobs_reused=126, new_receiver_jobs=new_job_count, old_encode_calls_in_extension=0, inherited_independent_audit_sha256=prior_audit_sha)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "audit output already exists")
    require(args.milestone.resolve() not in args.output.resolve().parents, "audit output must be outside immutable milestone")
    result = audit(args.milestone, args.frozen_root, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
