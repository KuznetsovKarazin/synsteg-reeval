"""Completed-cache integrity and no-inference suite planning."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("suite", ROOT / "scripts/run_neural_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


@pytest.mark.parametrize("profile,d4,dtype", [("cpu", "D4-bf16", "bfloat16"), ("gpu-fp32", "D4-fp32", "float32")])
def test_plan_does_not_import_torch_or_write_outputs(tmp_path, profile, d4, dtype):
    # An import blocker makes this more specific than checking printed output.
    code = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('plan imported torch')
sys.meta_path.insert(0, BlockTorch())
import runpy
runpy.run_path(sys.argv.pop(1), run_name='__main__')
"""
    output = tmp_path / "absent"
    result = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts/run_neural_suite.py"),
                             "--profile", profile, "--output", str(output), "--plan-only"],
                            capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert not output.exists()
    assert plan["scope"]["primary"] == {"records": 1949, "unique_texts": 1949}
    assert plan["scope"]["quality"] == {"records": 3449, "unique_texts": 3443}
    assert plan["jobs"][-1]["name"] == d4
    assert plan["jobs"][-1]["job"]["dtype"] == dtype


def fixture_cache(folder):
    runner = suite.runner_module()
    records = [{"id": "a", "text_sha256": "a" * 64}, {"id": "b", "text_sha256": "b" * 64}]
    identity = {"job": {"kind": "D1"}, "runtime": {"mock": True}, "device": "cpu", "script_sha256": "script"}
    fingerprint = runner.digest(identity)
    scores = [{"id": row["id"], "text_sha256": row["text_sha256"],
               "cache_key": runner.cache_key(fingerprint, row["text_sha256"]), "score": 0.4, "details": {}}
              for row in records]
    (folder / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scores))
    manifest = {"status": "complete", "identity": identity, "job_fingerprint": fingerprint,
                "corpus_sha256": "corpus", "n_records": 2, "n_unique_texts": 2, "n_complete_unique_texts": 2,
                "scores_sha256": suite.sha(folder / "scores.jsonl")}
    runner.atomic_json(folder / "manifest.json", manifest)
    return runner, records, identity, scores, manifest


def test_complete_cache_reusable_and_partial_scores_never_read(tmp_path):
    runner, records, identity, scores, manifest = fixture_cache(tmp_path)
    assert suite.validate_completed_job(tmp_path, records, identity, "corpus", runner)
    manifest["status"] = "incomplete"
    runner.atomic_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "scores.jsonl").write_text("not JSON and intentionally unreadable as scores")
    assert suite.validate_completed_job(tmp_path, records, identity, "corpus", runner) is False


@pytest.mark.parametrize("damage", ["order", "nonfinite", "identity", "hash", "count"])
def test_completed_cache_fails_closed(tmp_path, damage):
    runner, records, identity, scores, manifest = fixture_cache(tmp_path)
    if damage == "order":
        scores.reverse()
    elif damage == "nonfinite":
        scores[0]["score"] = float("nan")
    elif damage == "identity":
        manifest["identity"]["device"] = "cuda"
        identity = dict(identity, device="cpu")
    elif damage == "count":
        manifest["n_complete_unique_texts"] = 1
    (tmp_path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in scores))
    manifest["scores_sha256"] = "wrong" if damage == "hash" else suite.sha(tmp_path / "scores.jsonl")
    runner.atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        suite.validate_completed_job(tmp_path, records, identity, "corpus", runner)
