"""Synthetic-only integrity tests for aggregation of saved annotations."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("saved_review_summary_test", ROOT / "scripts/summarize_development_text_reviews_v1.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def span(text, quote):
    start = text.index(quote)
    return {"quote": quote, "start": start, "end": start + len(quote), "occurrences": text.count(quote)}


@pytest.fixture
def sample(tmp_path, monkeypatch):
    sources, attempts = {}, {}
    for order in range(1, 101):
        rid = f"synthetic-source-{order:03d}"
        excerpt = "Item exists. It is cited."
        prompt = "No headings. Paraphrase naturally and use the source."
        source = {"record_id": rid, "source_group_id": f"group-{order}", "title": f"Title {order}",
            "split": "development", "target_words": 250, "excerpt": excerpt, "excerpt_words": len(excerpt.split()),
            "excerpt_sha256": module.sha(excerpt.encode()), "prompt": prompt, "prompt_sha256": module.sha(prompt.encode())}
        sources[rid] = source
        if order <= 4:
            cover = "Heading Item states facts. Heading"
            attempt = {"status": "completed", "identity": module.sha(f"identity-{order}".encode()),
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "record": {k: source[k] for k in ("record_id", "source_group_id", "split", "target_words", "prompt_sha256")},
                "result": {"cover_text": cover, "revision": "synthetic-revision", "hit_output_limit": order == 4,
                    "stop_reason": "max_tokens" if order == 4 else "end_turn", "usage": {"output_tokens": 64}}}
            raw_sha = module.sha(json.dumps(attempt, sort_keys=True).encode())
            attempts[rid] = {"attempt": attempt, "attempt_sha256": raw_sha}
    source_path = tmp_path / module.SOURCE_PATH
    source_path.parent.mkdir(parents=True)
    source_path.write_text("".join(json.dumps(row) + "\n" for row in sources.values()))
    source_sha = module.sha(source_path.read_bytes())
    monkeypatch.setattr(module, "SOURCE_SHA", source_sha)
    monkeypatch.setattr(module, "REQUIRED_ADDENDA", {})
    reviews = []
    for number in (1, 2):
        legacy = number == 1
        review = {"schema": module.V1 if legacy else module.V3, "campaign_id": "large-campaign-development-v2",
            "model_id": "tinyllama11", "batch_id": f"batch{number:02d}", "source_file": module.SOURCE_PATH,
            "source_file_sha256": source_sha, "records_included": 2, "excluded_records": [], "records": []}
        if legacy:
            review.update(planned_batch_records=2, quantitative_factuality_metric=False, new_generation_calls=0)
        else:
            review.update(planned_batch_rows=2, fixed_development_source_count=100, batch_source_orders=[3, 4],
                          method={"quantitative_factuality_metric": False, "new_generation_calls": 0})
        for position, order in enumerate(range(number * 2 - 1, number * 2 + 1), 1):
            rid = f"synthetic-source-{order:03d}"
            source, supplied = sources[rid], attempts[rid]
            attempt = supplied["attempt"]
            cover = attempt["result"]["cover_text"]
            path = f"runs/large-campaign-development-v2/models/tinyllama11/attempts/{attempt['identity']}.json"
            record = {"record_id": rid, "title": source["title"], "target_words": 250,
                "source_excerpt": source["excerpt"], "source_excerpt_sha256": source["excerpt_sha256"],
                "cover_sha256": module.sha(cover.encode()), "attempt_file_sha256": supplied["attempt_sha256"]}
            if legacy:
                quote = "Item states facts."
                start = cover.index(quote)
                record.update(batch_position_one_based=position, attempt_file=path, cover_words_whitespace=len(cover.split()),
                    full_cover_read_by_agent=True, attempt_status="completed", model=attempt["model"], revision="synthetic-revision",
                    split="development", source_words_recorded=source["excerpt_words"], generated_tokens=64,
                    stop_reason="end_turn", hit_output_limit=False,
                    observations=[{"classification": "unsupported_by_excerpt", "cover_quote": quote,
                        "cover_spans_chars_zero_based_half_open": [[start, start + len(quote)]],
                        "source_quote": "Item exists.", "source_span_chars_zero_based_half_open": [0, 12]}],
                    format_observations=[{"kind": "heading", "cover_quote": "Heading",
                        "cover_spans_chars_zero_based_half_open": [[0, 7], [cover.rindex("Heading"), len(cover)]],
                        "prompt_constraint": "Paraphrase naturally."}])
            else:
                record.update(source_order=order, source_group_id=source["source_group_id"], prompt_sha256=source["prompt_sha256"],
                    attempt_path=path, cover_whitespace_words=len(cover.split()), generation_status="completed",
                    full_cover_and_excerpt_read=True, review_status="reviewed_complete_text",
                    observations=[{"id": f"r{order:03d}-o01", "label": "unsupported_by_excerpt",
                        "cover_span": span(cover, "Item states facts."), "source_span": span(source["excerpt"], "Item exists.")},
                        {"id": f"r{order:03d}-o02", "label": "unsupported_by_excerpt", "cover_span": span(cover, "Item states facts.")}],
                    format_observations=[{"id": f"r{order:03d}-f01", "kind": "incomplete_final_sentence" if order == 4 else "heading",
                        "cover_span": span(cover, "Heading"), "prompt_span": span(source["prompt"], "No headings")}])
            review["records"].append(record)
        relative = module.BASELINE_PATH if legacy else module.REVIEW_DIRECTORY + "/batch02.json"
        write(tmp_path / relative, review)
        reviews.append(review)
    monkeypatch.setattr(module, "BASELINE_SHA", module.sha((tmp_path / module.BASELINE_PATH).read_bytes()))
    return tmp_path, attempts, sources, reviews


def make_addendum(sample):
    root, attempts, _, reviews = sample
    rid = "synthetic-source-004"
    record = reviews[1]["records"][1]
    observation = record["format_observations"][0]
    value = {"schema": module.ADDENDUM, "review_path": module.REVIEW_DIRECTORY + "/batch02.json",
        "review_sha256": module.sha((root / (module.REVIEW_DIRECTORY + "/batch02.json")).read_bytes()),
        "record_id": rid, "source_order": 4, "observation_id": observation["id"], "original_label": observation["kind"],
        "cover_sha256": record["cover_sha256"], "attempt_path": record["attempt_path"],
        "attempt_file_sha256": attempts[rid]["attempt_sha256"], "cover_span": observation["cover_span"],
        "clarification_ru": "This saved label does not establish grammatical incompleteness.",
        "aggregation_note_ru": "A clarification, no additional observation or cover.",
        "verified_stop_evidence": {"hit_output_limit": True, "stop_reason": "max_tokens", "output_tokens": 64}}
    path = root / module.REVIEW_DIRECTORY / "batch02-r004-addendum01.json"
    write(path, value)
    return path, value


def test_both_review_schemas_preserve_raw_counts(sample):
    root, attempts, sources, _ = sample
    evidence = {}
    rows, summary = module.collect_reviews(root, 2, attempts, sources, evidence)
    assert summary["reviewed_covers"] == 4 and summary["unreviewed_covers"] == 0
    assert summary["covers_with_label"] == {"unsupported_by_excerpt": 4}
    assert summary["observation_counts_by_label"] == {"unsupported_by_excerpt": 6}
    assert summary["raw_format_observation_counts"] == {"heading": 3, "incomplete_final_sentence": 1}
    assert summary["validated_annotation_spans"] == {"cover": 12, "source": 4, "prompt": 2}
    assert summary["legacy_prompt_constraints_without_offsets"] == 2
    assert summary["legacy_prompt_constraints_exact_substring_found"] == 0
    assert summary["factuality_rate_computed"] is False
    assert len(rows) == 4 and len(evidence) == 3


@pytest.mark.parametrize("mutation", ["cover_offset", "source_offset", "prompt_offset", "cover_hash", "attempt_hash", "source_hash", "prompt_hash", "occurrences", "duplicate_observation"])
def test_corrupt_annotation_fails_entire_aggregation(sample, mutation):
    root, attempts, sources, reviews = sample
    review = reviews[1]
    record = review["records"][0]
    if mutation == "cover_offset": record["observations"][0]["cover_span"]["start"] += 1
    elif mutation == "source_offset": record["observations"][0]["source_span"]["end"] -= 1
    elif mutation == "prompt_offset": record["format_observations"][0]["prompt_span"]["start"] += 1
    elif mutation == "cover_hash": record["cover_sha256"] = "0" * 64
    elif mutation == "attempt_hash": record["attempt_file_sha256"] = "0" * 64
    elif mutation == "source_hash": record["source_excerpt_sha256"] = "0" * 64
    elif mutation == "prompt_hash": record["prompt_sha256"] = "0" * 64
    elif mutation == "occurrences": record["format_observations"][0]["cover_span"]["occurrences"] = 1
    else: record["observations"][1]["id"] = record["observations"][0]["id"]
    write(root / module.REVIEW_DIRECTORY / "batch02.json", review)
    expected = {"cover_offset": "cover quote/offset", "source_offset": "source quote/offset", "prompt_offset": "prompt quote/offset",
                "cover_hash": "review cover/attempt", "attempt_hash": "review cover/attempt", "source_hash": "review excerpt",
                "prompt_hash": "review prompt", "occurrences": "occurrence count", "duplicate_observation": "duplicate observation"}
    with pytest.raises(ValueError, match=expected[mutation]):
        module.collect_reviews(root, "batch02", attempts, sources, {})


def test_missing_review_means_unreviewed_not_error_free(sample):
    root, attempts, sources, _ = sample
    (root / module.REVIEW_DIRECTORY / "batch02.json").unlink()
    records, summary = module.collect_reviews(root, 2, attempts, sources, {})
    assert summary["reviewed_covers"] == summary["unreviewed_covers"] == 2
    assert records["synthetic-source-003"]["review_status"] == "unreviewed"
    assert summary["covers_with_label"] == {"unsupported_by_excerpt": 2}
    assert summary["missing_review_files"] == [module.REVIEW_DIRECTORY + "/batch02.json"]


def test_addendum_qualifies_without_double_count(sample):
    root, attempts, sources, _ = sample
    before = module.collect_reviews(root, 2, attempts, sources, {})[1]
    path, _ = make_addendum(sample)
    evidence = {}
    records, after = module.collect_reviews(root, 2, attempts, sources, evidence)
    for field in ("reviewed_covers", "unreviewed_covers", "covers_with_label", "observation_counts_by_label",
                  "total_observations", "raw_format_observation_counts", "total_raw_format_observations"):
        assert after[field] == before[field]
    assert after["addendum_files"] == after["qualified_existing_observations"] == 1
    assert after["new_observations_from_addenda"] == 0
    assert after["qualified_raw_format_observation_counts"] == {"incomplete_final_sentence": 1}
    assert records["synthetic-source-004"]["format_observations"][0]["qualified_by_addendum"] is True
    assert str(path.relative_to(root)) in evidence


@pytest.mark.parametrize("mutation", ["review_sha", "observation_id", "cover_sha", "attempt_sha", "span", "stop"])
def test_invalid_addendum_fails(sample, mutation):
    root, attempts, sources, _ = sample
    path, value = make_addendum(sample)
    if mutation == "review_sha": value["review_sha256"] = "0" * 64
    elif mutation == "observation_id": value["observation_id"] = "unknown"
    elif mutation == "cover_sha": value["cover_sha256"] = "0" * 64
    elif mutation == "attempt_sha": value["attempt_file_sha256"] = "0" * 64
    elif mutation == "span": value["cover_span"]["end"] -= 1
    else: value["verified_stop_evidence"]["output_tokens"] += 1
    write(path, value)
    with pytest.raises(ValueError):
        module.collect_reviews(root, 2, attempts, sources, {})


def test_known_mandatory_addendum_cannot_disappear(sample, monkeypatch):
    root, attempts, sources, _ = sample
    path, _ = make_addendum(sample)
    monkeypatch.setattr(module, "REQUIRED_ADDENDA", {path.name: module.sha(path.read_bytes())})
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        module.collect_reviews(root, 2, attempts, sources, {})


def test_pinned_baseline_and_source_are_verified(sample):
    root, attempts, sources, _ = sample
    (root / module.BASELINE_PATH).write_text("{}")
    with pytest.raises(ValueError, match="pinned evidence SHA"):
        module.collect_reviews(root, 2, attempts, sources, {})


def test_wrong_frozen_source_rows_are_rejected(sample):
    root, attempts, sources, _ = sample
    sources["synthetic-source-001"]["title"] = "Changed"
    with pytest.raises(ValueError, match="provided source"):
        module.collect_reviews(root, 2, attempts, sources, {})
