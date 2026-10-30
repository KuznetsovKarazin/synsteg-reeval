#!/usr/bin/env python3
"""Aggregate existing descriptive annotations; do not make new judgments.

The caller supplies verified immutable attempts, never live generation files.
Every declared quote/span is checked before labels are counted. Counts describe
saved annotations, not exhaustive claims, human ratings or factuality rates.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


SOURCE_PATH = "results/recovery-development-v1/linux_snapshot/development-prompts.jsonl"
SOURCE_SHA = "3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1"
BASELINE_PATH = "results/large-campaign-milestone-review-v1/inputs/batch01-text-observations-c24.json"
BASELINE_SHA = "e5f36fd7faab7efa9f2469aa2978b7fa451ce752c4c48764bbad285bdbfe2316"
REVIEW_DIRECTORY = "results/large-campaign-continuation-v3-text-review"
REQUIRED_ADDENDA = {
    "batch10-r020-addendum01.json": "63196d9be2228b5e51d116c0df575854598aaca2f35380063186b6fc2628ba31",
}
V1 = "synsteg-large-campaign-batch-text-observations-v1"
V3 = "synsteg-development-descriptive-text-review-v3"
ADDENDUM = "synsteg-development-descriptive-text-review-addendum-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(root, relative, evidence, expected_sha=None):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe evidence path")
    target = root / path
    require(target.is_file() and not target.is_symlink(), "missing or symlink evidence: " + relative)
    raw = target.read_bytes()
    digest = sha(raw)
    if expected_sha is not None:
        require(digest == expected_sha, "pinned evidence SHA mismatch: " + relative)
    require(relative not in evidence or evidence[relative] == digest, "evidence hash changed: " + relative)
    evidence[relative] = digest
    return raw, digest


def _span(text, quote, start, end, kind, counts):
    require(isinstance(quote, str) and bool(quote), kind + " empty quote")
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(text), kind + " invalid offsets")
    require(text[start:end] == quote, kind + " quote/offset mismatch")
    counts[kind] += 1
    return {"quote": quote, "start": start, "end": end}


def _modern_span(text, span, kind, counts):
    require(isinstance(span, dict), kind + " span object")
    normalized = _span(text, span["quote"], span["start"], span["end"], kind, counts)
    if "occurrences" in span:
        require(type(span["occurrences"]) is int and span["occurrences"] == text.count(span["quote"]),
                kind + " occurrence count mismatch")
        normalized["occurrences"] = span["occurrences"]
    return normalized


def _annotation(raw, category, index, order, legacy, cover, source, prompt, counts, descriptions):
    label_key = "classification" if legacy and category == "observations" else "label" if category == "observations" else "kind"
    label = raw[label_key]
    require(isinstance(label, str) and bool(label), "empty annotation label")
    identifier = (f"legacy-r{order:03d}-{'o' if category == 'observations' else 'f'}{index:02d}"
                  if legacy else raw["id"])
    require(isinstance(identifier, str) and bool(identifier), "annotation identifier")
    item = {"observation_id": identifier, "category": category, "raw_label": label,
            "reason_ru": raw.get("reason_ru"), "cover_spans": [], "source_spans": [], "prompt_spans": [],
            "qualified_by_addendum": False, "addenda": []}
    if legacy:
        cover_ranges = raw["cover_spans_chars_zero_based_half_open"]
        require(isinstance(cover_ranges, list) and bool(cover_ranges), "legacy cover spans absent")
        for start, end in cover_ranges:
            item["cover_spans"].append(_span(cover, raw["cover_quote"], start, end, "cover", counts))
        source_quote, source_range = raw.get("source_quote"), raw.get("source_span_chars_zero_based_half_open")
        require((source_quote is None) == (source_range is None), "legacy source quote/range pairing")
        if source_quote is not None:
            item["source_spans"].append(_span(source, source_quote, *source_range, "source", counts))
        if "prompt_constraint" in raw:
            description = raw["prompt_constraint"]
            require(isinstance(description, str) and bool(description), "legacy prompt description")
            item["legacy_prompt_constraint"] = description
            item["legacy_prompt_constraint_is_declared_quote"] = False
            item["legacy_prompt_constraint_exact_substring_found"] = description in prompt
            descriptions["legacy_prompt_constraints_without_offsets"] += 1
            descriptions["legacy_prompt_constraints_exact_substring_found"] += int(description in prompt)
    else:
        item["cover_spans"].append(_modern_span(cover, raw["cover_span"], "cover", counts))
        for key, text, kind in (("source_span", source, "source"), ("prompt_span", prompt, "prompt")):
            if key in raw:
                item[kind + "_spans"].append(_modern_span(text, raw[key], kind, counts))
    return item


def _stop_evidence(declared, attempt):
    result = attempt["result"]
    expected = {"hit_output_limit": result["hit_output_limit"], "stop_reason": result["stop_reason"],
                "output_tokens": result["usage"]["output_tokens"]}
    require(declared == expected, "generation stop evidence mismatch")


def collect_reviews(root, through_batch, attempts_by_record_id, sources_by_record_id, evidence):
    """Return ``(by_record, summary)`` and append relative-path SHA to evidence.

    ``attempts_by_record_id[rid]`` must contain the immutable parsed ``attempt``
    and independently verified raw ``attempt_sha256``. Missing reviews become
    unreviewed; corrupt or contradictory bindings fail the entire aggregation.
    Only completed attempts belong to this review denominator. The caller keeps
    ungenerated/failed/unknown cohort entries in its separate full300 table.
    """
    root = Path(root).resolve()
    if isinstance(through_batch, str):
        match = re.fullmatch(r"batch(\d+)", through_batch)
        require(match is not None, "through_batch must be integer or batchNN")
        through_batch = int(match.group(1))
    require(type(through_batch) is int and 1 <= through_batch <= 50, "through_batch range")
    require(isinstance(evidence, dict), "evidence must map relative paths to SHA256")
    source_raw, source_sha = _read(root, SOURCE_PATH, evidence, SOURCE_SHA)
    source_rows = [json.loads(line) for line in source_raw.splitlines() if line]
    source_ids = [row["record_id"] for row in source_rows]
    require(len(source_rows) == len(set(source_ids)) == 100, "fixed source denominator")
    require(set(sources_by_record_id) == set(source_ids), "provided source identity set")
    source_order = {rid: index for index, rid in enumerate(source_ids, 1)}
    for source in source_rows:
        rid = source["record_id"]
        require(sources_by_record_id[rid] == source, "provided source differs from frozen row")
        require(source["split"] == "development", "closed test source forbidden")
        require(sha(source["excerpt"].encode()) == source["excerpt_sha256"] and
                sha(source["prompt"].encode()) == source["prompt_sha256"], "source excerpt/prompt hashes")
    by_record = {}
    for rid, supplied in attempts_by_record_id.items():
        require(rid in sources_by_record_id and source_order[rid] <= through_batch * 2, "attempt outside selected milestone")
        attempt = supplied["attempt"]
        require(attempt["status"] == "completed" and isinstance(attempt["result"]["cover_text"], str), "only completed covers in review denominator")
        require(re.fullmatch("[0-9a-f]{64}", supplied["attempt_sha256"]) is not None, "attempt raw SHA shape")
        require(attempt["record"]["record_id"] == rid and attempt["record"]["prompt_sha256"] == sources_by_record_id[rid]["prompt_sha256"],
                "attempt record/prompt binding")
        for field in ("source_group_id", "split", "target_words"):
            require(attempt["record"][field] == sources_by_record_id[rid][field], "attempt source metadata")
        by_record[rid] = {"record_id": rid, "source_order": source_order[rid], "review_status": "unreviewed",
            "review_file": None, "review_sha256": None, "review_schema": None,
            "cover_sha256": sha(attempt["result"]["cover_text"].encode()), "attempt_sha256": supplied["attempt_sha256"],
            "observations": [], "format_observations": [], "addenda": []}
    counts, descriptions = Counter(), Counter()
    missing_files, loaded_reviews = [], {}
    for number in range(1, through_batch + 1):
        relative = BASELINE_PATH if number == 1 else f"{REVIEW_DIRECTORY}/batch{number:02d}.json"
        if not (root / relative).is_file():
            missing_files.append(relative)
            continue
        raw, digest = _read(root, relative, evidence, BASELINE_SHA if number == 1 else None)
        review = json.loads(raw)
        legacy = review["schema"] == V1
        require(review["schema"] == (V1 if number == 1 else V3), "review schema/batch mismatch")
        require(review["campaign_id"] == "large-campaign-development-v2" and review["model_id"] == "tinyllama11" and
                review["batch_id"] == f"batch{number:02d}", "review campaign/model/batch")
        require(review["source_file"] == SOURCE_PATH and review["source_file_sha256"] == source_sha, "review source file binding")
        require(review["records_included"] == len(review["records"]) == 2 and not review["excluded_records"], "review batch denominator")
        planned_key = "planned_batch_records" if legacy else "planned_batch_rows"
        require(review[planned_key] == 2, "planned review denominator")
        if legacy:
            require(review["quantitative_factuality_metric"] is False and review["new_generation_calls"] == 0, "legacy scope")
        else:
            require(review["fixed_development_source_count"] == 100 and review["batch_source_orders"] == [number * 2 - 1, number * 2], "review source orders")
            require(review["method"]["quantitative_factuality_metric"] is False and review["method"]["new_generation_calls"] == 0, "review scope")
        loaded_reviews[relative] = (digest, review)
        for position, record in enumerate(review["records"], 1):
            rid = record["record_id"]
            require(rid in by_record, "review names unavailable cover")
            entry, source, supplied = by_record[rid], sources_by_record_id[rid], attempts_by_record_id[rid]
            attempt, order = supplied["attempt"], source_order[rid]
            cover, excerpt, prompt = attempt["result"]["cover_text"], source["excerpt"], source["prompt"]
            require(order == number * 2 - 2 + position and entry["review_status"] == "unreviewed", "review duplicate/order mismatch")
            require(record["title"] == source["title"] and record["target_words"] == source["target_words"], "review source metadata")
            require(record["source_excerpt"] == excerpt and record["source_excerpt_sha256"] == source["excerpt_sha256"], "review excerpt identity")
            require(record["cover_sha256"] == sha(cover.encode()) and record["attempt_file_sha256"] == supplied["attempt_sha256"], "review cover/attempt SHA")
            path_key = "attempt_file" if legacy else "attempt_path"
            require(record[path_key] == f"runs/large-campaign-development-v2/models/tinyllama11/attempts/{attempt['identity']}.json", "review attempt path")
            words_key = "cover_words_whitespace" if legacy else "cover_whitespace_words"
            require(record[words_key] == len(cover.split()), "review cover word count")
            if legacy:
                require(record["batch_position_one_based"] == position and record["full_cover_read_by_agent"] is True and
                        record["attempt_status"] == "completed", "legacy review completeness")
                require(record["model"] == attempt["model"] and record["revision"] == attempt["result"]["revision"], "legacy model identity")
                require(record["split"] == "development" and record["source_words_recorded"] == source["excerpt_words"], "legacy source fields")
                require(record["generated_tokens"] == attempt["result"]["usage"]["output_tokens"] and
                        record["stop_reason"] == attempt["result"]["stop_reason"] and
                        record["hit_output_limit"] == attempt["result"]["hit_output_limit"], "legacy stop fields")
            else:
                require(record["source_order"] == order and record["source_group_id"] == source["source_group_id"] and
                        record["prompt_sha256"] == source["prompt_sha256"], "review prompt/order binding")
                require(record["generation_status"] == "completed" and record["full_cover_and_excerpt_read"] is True and
                        record["review_status"] == "reviewed_complete_text", "review completeness")
            if "generation_stop_evidence" in record:
                _stop_evidence(record["generation_stop_evidence"], attempt)
            entry.update(review_status="reviewed_complete_text", review_file=relative, review_sha256=digest, review_schema=review["schema"])
            identifiers = set()
            for category in ("observations", "format_observations"):
                for index, annotation in enumerate(record[category], 1):
                    normalized = _annotation(annotation, category, index, order, legacy, cover, excerpt, prompt, counts, descriptions)
                    require(normalized["observation_id"] not in identifiers, "duplicate observation identifier")
                    identifiers.add(normalized["observation_id"])
                    entry[category].append(normalized)
    review_dir = root / REVIEW_DIRECTORY
    for name, digest in REQUIRED_ADDENDA.items():
        number = int(re.match(r"batch(\d+)", name).group(1))
        if number <= through_batch:
            _read(root, f"{REVIEW_DIRECTORY}/{name}", evidence, digest)
    addendum_paths = sorted(review_dir.glob("batch*-addendum*.json")) if review_dir.exists() else []
    addenda_count, qualified_ids = 0, set()
    addendum_spans = Counter()
    for path in addendum_paths:
        match = re.fullmatch(r"batch(\d+)-.+-addendum\d+\.json", path.name)
        require(match is not None, "unknown addendum filename")
        number = int(match.group(1))
        if number > through_batch:
            continue
        relative = str(path.relative_to(root))
        raw, digest = _read(root, relative, evidence, REQUIRED_ADDENDA.get(path.name))
        addendum = json.loads(raw)
        require(addendum["schema"] == ADDENDUM, "addendum schema")
        review_path = addendum["review_path"]
        require(review_path == f"{REVIEW_DIRECTORY}/batch{number:02d}.json" and review_path in loaded_reviews,
                "addendum target review unavailable")
        require(addendum["review_sha256"] == loaded_reviews[review_path][0], "addendum review hash binding")
        rid = addendum["record_id"]
        require(rid in by_record and by_record[rid]["review_file"] == review_path, "addendum record binding")
        entry, supplied = by_record[rid], attempts_by_record_id[rid]
        attempt, source = supplied["attempt"], sources_by_record_id[rid]
        require(addendum["source_order"] == source_order[rid] and addendum["cover_sha256"] == entry["cover_sha256"] and
                addendum["attempt_file_sha256"] == supplied["attempt_sha256"], "addendum source/cover/attempt binding")
        require(addendum["attempt_path"] == f"runs/large-campaign-development-v2/models/tinyllama11/attempts/{attempt['identity']}.json", "addendum attempt path")
        matches = [item for category in ("observations", "format_observations") for item in entry[category]
                   if item["observation_id"] == addendum["observation_id"]]
        require(len(matches) == 1 and matches[0]["raw_label"] == addendum["original_label"], "addendum observation/label binding")
        annotation = matches[0]
        _modern_span(attempt["result"]["cover_text"], addendum["cover_span"], "cover", addendum_spans)
        require(any(all(addendum["cover_span"][k] == span[k] for k in ("quote", "start", "end"))
                    for span in annotation["cover_spans"]), "addendum quote differs from target observation")
        for key, text, kind in (("source_span", source["excerpt"], "source"), ("prompt_span", source["prompt"], "prompt")):
            if key in addendum:
                _modern_span(text, addendum[key], kind, addendum_spans)
        if "verified_stop_evidence" in addendum:
            _stop_evidence(addendum["verified_stop_evidence"], attempt)
        detail = {"path": relative, "sha256": digest, "observation_id": addendum["observation_id"],
                  "clarification_ru": addendum["clarification_ru"], "aggregation_note_ru": addendum["aggregation_note_ru"]}
        annotation["qualified_by_addendum"] = True
        annotation["addenda"].append(detail)
        entry["addenda"].append(detail)
        qualified_ids.add((rid, annotation["observation_id"]))
        addenda_count += 1
    observations, labels_by_cover, formats, formats_by_cover, qualified_formats = Counter(), Counter(), Counter(), Counter(), Counter()
    reviewed = [entry for entry in by_record.values() if entry["review_status"] == "reviewed_complete_text"]
    for entry in reviewed:
        observations.update(item["raw_label"] for item in entry["observations"])
        labels_by_cover.update({item["raw_label"] for item in entry["observations"]})
        formats.update(item["raw_label"] for item in entry["format_observations"])
        formats_by_cover.update({item["raw_label"] for item in entry["format_observations"]})
        qualified_formats.update(item["raw_label"] for item in entry["format_observations"] if item["qualified_by_addendum"])
    summary = {"schema": "saved-development-text-review-summary-v1", "through_batch": f"batch{through_batch:02d}",
        "source_groups_available": 100, "completed_covers_supplied": len(by_record), "reviewed_covers": len(reviewed),
        "unreviewed_covers": len(by_record) - len(reviewed), "missing_review_files": missing_files,
        "covers_with_label": dict(sorted(labels_by_cover.items())), "observation_counts_by_label": dict(sorted(observations.items())),
        "total_observations": sum(observations.values()), "covers_with_raw_format_kind": dict(sorted(formats_by_cover.items())),
        "raw_format_observation_counts": dict(sorted(formats.items())), "total_raw_format_observations": sum(formats.values()),
        "qualified_raw_format_observation_counts": dict(sorted(qualified_formats.items())),
        "addendum_files": addenda_count, "qualified_existing_observations": len(qualified_ids), "new_observations_from_addenda": 0,
        "validated_annotation_spans": {kind: counts[kind] for kind in ("cover", "source", "prompt")},
        "validated_addendum_spans": {kind: addendum_spans[kind] for kind in ("cover", "source", "prompt")},
        "legacy_prompt_constraints_without_offsets": descriptions["legacy_prompt_constraints_without_offsets"],
        "legacy_prompt_constraints_exact_substring_found": descriptions["legacy_prompt_constraints_exact_substring_found"],
        "review_schema_cover_counts": dict(Counter(entry["review_schema"] for entry in reviewed)),
        "factuality_rate_computed": False, "new_semantic_judgments": 0, "new_generation_calls": 0,
        "limitations": ["Counts summarize saved descriptive agent annotations, not exhaustive atomic claims or human ratings.",
            "No factuality success/error rate or world-truth conclusion is inferred; unsupported_by_excerpt means only absent support in that excerpt.",
            "Raw format kinds are preserved across schemas; annotation grouping differs and counts are not normalized quality scores.",
            "Unreviewed does not mean no observations or no problems. The caller separately retains all300 generation-status rows.",
            "Addenda qualify existing observations without increasing the cover or observation denominator.",
            "The qualified incomplete_final_sentence label must not be counted as a confirmed grammatical error.",
            "Legacy prompt_constraint is an unpositioned descriptive field, not a verified quoted prompt span."]}
    return by_record, summary
