"""Runner policy and checkpoint tests; these do not validate model inference."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rescore_models.py"
SPEC = importlib.util.spec_from_file_location("rescore_models", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def job():
    return {"kind": "ppl", "model_id": "openai-community/gpt2", "model_revision": "a" * 40,
            "tokenizer_id": "openai-community/gpt2", "tokenizer_revision": "a" * 40}


@pytest.mark.parametrize("length", [2, 3, 7, 8, 9, 14, 19, 1025])
@pytest.mark.parametrize("stride", [1, 3, 7])
def test_sliding_windows_count_every_target_once(length, stride):
    covered = []
    for begin, end, first in runner.windows(length, 8, stride):
        assert begin < first < end <= length
        assert end - begin <= 8
        covered.extend(range(first, end))
    assert covered == list(range(1, length))


def test_checkpoint_identity_includes_text_revision_and_protocol():
    base = runner.validate_job(job())
    fingerprint = runner.digest(base)
    sha = runner.text_hash("test document")
    first = runner.cache_key(fingerprint, sha)
    other = dict(base, model_revision="b" * 40)
    assert first != runner.cache_key(runner.digest(other), sha)
    assert first != runner.cache_key(fingerprint, runner.text_hash("test document "))
    assert first != runner.cache_key(runner.digest(dict(base, ppl_policy="prefix")), sha)


def test_no_branch_pins_or_wrong_detector_backbones():
    with pytest.raises(ValueError, match="40-character"):
        runner.validate_job(dict(job(), model_revision="main"))
    with pytest.raises(ValueError, match="requires Qwen"):
        runner.validate_job(dict(job(), kind="D5"))


def test_partial_and_stale_checkpoints_cannot_be_finalized(tmp_path):
    sha = runner.text_hash("abc")
    records = [{"id": "a", "text_sha256": sha}]
    fingerprint = "job"
    key = runner.cache_key(fingerprint, sha)
    with pytest.raises(ValueError, match="incomplete"):
        runner.complete_rows(records, {}, fingerprint)
    checkpoint = {"status": "complete", "score": 2., "job_fingerprint": "other", "text_sha256": sha}
    path = tmp_path / "checkpoint.json"
    runner.atomic_json(path, checkpoint)
    with pytest.raises(ValueError, match="mismatch"):
        runner.read_checkpoint(path, fingerprint, sha)
    checkpoint["job_fingerprint"] = fingerprint
    runner.atomic_json(path, checkpoint)
    result = runner.read_checkpoint(path, fingerprint, sha)
    assert runner.complete_rows(records, {key: result}, fingerprint)[0]["score"] == 2.


def test_empty_or_duplicate_corpus_is_rejected(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id":"same","text":"one"}\n{"id":"same","text":"two"}\n')
    with pytest.raises(ValueError, match="unique"):
        runner.read_corpus(path)
    path.write_text('{"id":"one","text":"   "}\n')
    with pytest.raises(ValueError, match="nonempty"):
        runner.read_corpus(path)


def test_resume_and_completeness_without_neural_dependencies(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(''.join(json.dumps({"id": f"id{i}", "text": text}) + "\n" for i, text in enumerate(["first", "second", "first"])))
    settings = tmp_path / "job.json"
    settings.write_text(json.dumps(job()))
    out = tmp_path / "run"
    calls = []

    class FakeScorer:
        def __init__(self, job, device, allow_download):
            assert allow_download is False

        def score(self, text):
            calls.append(text)
            if len(calls) == 2:
                raise RuntimeError("simulated interrupted inference")
            return 3.0, {"mock": True}

    monkeypatch.setattr(runner, "Scorer", FakeScorer)
    args = ["--corpus", str(corpus), "--job", str(settings), "--output", str(out)]
    with pytest.raises(RuntimeError):
        runner.main(args)
    assert not (out / "scores.jsonl").exists()
    assert json.loads((out / "manifest.json").read_text())["status"] == "incomplete"
    assert len(list((out / "checkpoints").glob("*.json"))) == 1
    runner.main(args)
    assert calls == ["first", "second", "second"]
    assert len((out / "scores.jsonl").read_text().splitlines()) == 3
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["n_records"] == 3 and manifest["n_complete_unique_texts"] == 2
    runner.main(args)
    assert len(calls) == 3
    settings.write_text(json.dumps(dict(job(), model_revision="b" * 40)))
    with pytest.raises(ValueError, match="another job"):
        runner.main(args)
