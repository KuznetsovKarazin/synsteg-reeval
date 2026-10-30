"""Synthetic summary-integrity checks; no neural or prospective data are read."""
import importlib.util
import json
from pathlib import Path
import random

import pytest
from sklearn.metrics import roc_auc_score

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_neural_results.py"
SPEC = importlib.util.spec_from_file_location("neural_summary", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fixture(tmp_path):
    """Two fake scores, only for testing integrity and pairing rules."""
    identity = {"job": {"kind": "D1"}, "runtime": {}, "device": "cpu", "script_sha256": "test"}
    fingerprint = summary.stable_digest(identity)
    source, scores = {}, []
    for arm, value in (("cover", .2), ("stego", .8)):
        text = f"synthetic {arm} test fixture"
        digest = summary.sha(text.encode())
        row = {"id": arm, "text": text, "text_sha256": digest, "dataset": "synthetic",
               "arm": arm, "uuid": "pair", "evaluation_role": "archived_development_only"}
        source[arm] = row
        scores.append({"id": arm, "text_sha256": digest,
                       "cache_key": summary.stable_digest({"job_fingerprint": fingerprint, "text_sha256": digest}),
                       "score": value, "details": {"input_tokens": 5, "scored_input_tokens": 5,
                                                   "truncated": False, "orientation": "higher_is_more_AI"}})
    data = "".join(json.dumps(r)+"\n" for r in scores).encode()
    manifest = {"status": "complete", "identity": identity, "job_fingerprint": fingerprint,
                "corpus_sha256": "corpus", "scores_sha256": summary.sha(data), "n_records": 2,
                "n_unique_texts": 2, "n_complete_unique_texts": 2}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    (tmp_path/"scores.jsonl").write_bytes(data)
    corpora = {"corpus": {"scope": "primary", "rows": source}}
    return manifest, scores, corpora


def write_scores(tmp_path, manifest, scores):
    data = "".join(json.dumps(r)+"\n" for r in scores).encode()
    manifest["scores_sha256"] = summary.sha(data)
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    (tmp_path/"scores.jsonl").write_bytes(data)


def test_auc_direction_ties_and_sklearn_parity():
    assert summary.auc([0, 1], [2, 3]) == 1
    assert summary.auc([2, 3], [0, 1]) == 0
    assert summary.auc([1, 1], [1, 1]) == .5
    rng = random.Random(702)
    for _ in range(50):
        human = [rng.randrange(7) for _ in range(23)]
        machine = [rng.randrange(7) for _ in range(17)]
        assert summary.auc(human, machine) == pytest.approx(roc_auc_score([0]*23+[1]*17, human+machine), abs=1e-14)


def test_incomplete_run_never_reads_scores(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps({"status": "incomplete", "n_complete_unique_texts": 2}))
    (tmp_path/"scores.jsonl").write_text("INVALID JSON; MUST NOT BE READ")
    progress, run, hashes = summary.load_complete_run(tmp_path, {})
    assert progress["status"] == "incomplete" and run is None and hashes == {}


def test_complete_scores_and_dataset_pairing(tmp_path):
    _, _, corpora = fixture(tmp_path)
    _, run, hashes = summary.load_complete_run(tmp_path, corpora)
    assert len(hashes) == 2
    metrics = summary.metric_summary(run["rows"], "score", "ai")
    delta = metrics["paired_deltas"][0]
    assert delta["n_pairs"] == 1
    assert delta["target_minus_reference"]["mean"] == pytest.approx(.6)
    other = dict(run["rows"][1], dataset="different")
    separate = summary.metric_summary([run["rows"][0], other], "score", "ai")
    assert separate["paired_deltas"] == []


def test_corrupt_complete_scores_rejected(tmp_path):
    _, _, corpora = fixture(tmp_path)
    (tmp_path/"scores.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Score-file SHA"):
        summary.load_complete_run(tmp_path, corpora)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "text", "cache", "orientation", "truncation"])
def test_complete_manifest_cannot_hide_score_identity_errors(tmp_path, mutation):
    manifest, scores, corpora = fixture(tmp_path)
    if mutation == "missing":
        scores.pop()
    elif mutation == "duplicate":
        scores.append(scores[0])
    elif mutation == "text":
        scores[0]["text_sha256"] = "wrong"
    elif mutation == "cache":
        scores[0]["cache_key"] = "wrong"
    elif mutation == "orientation":
        scores[0]["details"]["orientation"] = "chosen_after_test"
    elif mutation == "truncation":
        scores[0]["details"]["truncated"] = True
    write_scores(tmp_path, manifest, scores)
    with pytest.raises(ValueError):
        summary.load_complete_run(tmp_path, corpora)


def test_calibration_is_explicitly_in_sample_and_direction_is_fixed():
    result = summary.threshold_metrics([0, 0, 1, 2], [0, 3], 1)
    assert result["rule"] == "score > threshold"
    assert result["human_FPR"] == .25 and result["machine_TPR"] == .5
    assert summary.quantile([0, 10], .95) == 9.5
