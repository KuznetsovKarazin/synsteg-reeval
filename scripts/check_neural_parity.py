#!/usr/bin/env python3
"""Checkpoint-level smoke parity on archived development text, never held-out data.

Uses cached immutable checkpoints and independent reference call paths. Loads one
model at a time. This is a small integration check, not corpus-wide validation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parity_runner", ROOT / "scripts/rescore_models.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def check(kind, samples):
    import torch
    from transformers import pipeline
    settings = ROOT / "configs/neural" / ("gpt2-prefix.json" if kind == "GPT2" else kind + ".json")
    raw = json.loads(settings.read_text())
    raw.update(cpu_threads=1)
    job = r.validate_job(raw)
    scorer = r.Scorer(job, "cpu", False)
    checks = []
    with torch.inference_mode():
        if kind in {"D1", "D2"}:
            reference = pipeline("text-classification", model=scorer.model, tokenizer=scorer.tokenizer,
                                 device=-1, top_k=None, truncation=True, max_length=512)
        for sample in samples:
            text = sample["text"]
            actual, details = scorer.score(text)
            if kind == "GPT2":
                ids = scorer.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :512]
                loss = scorer.model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids, use_cache=False).loss
                expected = math.exp(float(loss.item()))
                method = "HuggingFace causal-LM built-in labels loss.exp (all short-text targets)"
                assert len(ids[0]) <= 512
            elif kind in {"D1", "D2"}:
                scores = reference(text)
                if isinstance(scores[0], list):
                    scores = scores[0]
                by_label = {x["label"]: x["score"] for x in scores}
                ai_label = job["expected_id2label"][str(job["ai_label_index"])]
                expected = float(by_label[ai_label])
                method = f"HuggingFace text-classification pipeline label {ai_label}"
                assert sum(by_label.values()) == pytest_approx_one(sum(by_label.values()))
            else:
                encoded = scorer.tokenizer(text, truncation=True, max_length=768, return_tensors="pt")
                hidden = scorer.model.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])[0]
                # Desklib's model-card reference expands the mask over hidden dims.
                expanded = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = torch.sum(hidden * expanded, dim=1) / torch.clamp(expanded.sum(dim=1), min=1e-9)
                expected = float(torch.sigmoid(scorer.model.classifier(pooled)).item())
                method = "Desklib model-card expanded attention-mask mean pooling + linear head + sigmoid"
            absolute = abs(actual - expected)
            relative = absolute / max(abs(expected), 1e-30)
            tolerance = 1e-5 if kind == "GPT2" else 1e-6
            passed = relative <= tolerance if kind == "GPT2" else absolute <= tolerance
            checks.append({"source_id": sample["source_id"], "text_sha256": r.text_hash(text),
                           "reference": method, "runner_score": actual, "reference_score": expected,
                           "absolute_error": absolute, "relative_error": relative,
                           "tolerance": tolerance, "tolerance_kind": "relative" if kind == "GPT2" else "absolute",
                           "passed": passed, "runner_details": details})
            if not passed:
                raise AssertionError(f"{kind} parity failed: {checks[-1]}")
    out = {"kind": kind, "job": job, "checks": checks, "passed": all(x["passed"] for x in checks),
           "scope": "two short archived development samples; no heldout test access"}
    del scorer
    gc.collect()
    return out


def pytest_approx_one(value):
    # Explicit invariant without requiring pytest in inference environment.
    if abs(value - 1.0) > 2e-6:
        raise AssertionError("classification probabilities do not sum to one")
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=ROOT / "results/neural-suite-2026-10-22/corpora/primary/corpus.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=["GPT2", "D1", "D2", "D3"], action="append")
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = [json.loads(x) for x in a.corpus.read_text().splitlines() if x.strip()]
    samples = []
    for dataset in ("gpt4o", "luna"):
        row = next(x for x in rows if x["dataset"] == dataset and x["arm"] == "cover")
        # Short-text integration parity only; keep crop policy and source ID.
        samples.append({"source_id": row["id"], "source_text_sha256": row["text_sha256"],
                        "crop_policy": "first 40 whitespace-delimited words", "text": " ".join(row["text"].split()[:40])})
    a.output.mkdir(parents=True)
    manifest = {"status": "incomplete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": r.versions(), "runner_sha256": hashlib.sha256((ROOT / "scripts/rescore_models.py").read_bytes()).hexdigest(),
                "parity_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "corpus_sha256": hashlib.sha256(a.corpus.read_bytes()).hexdigest(), "cpu_threads": 1,
                "samples": samples, "kinds": a.kind or ["GPT2", "D1", "D2", "D3"], "checks": {}}
    r.atomic_json(a.output / "manifest.json", manifest)
    for kind in manifest["kinds"]:
        print("Checking", kind, flush=True)
        result = check(kind, samples)
        r.atomic_json(a.output / (kind + ".json"), result)
        manifest["checks"][kind] = {"passed": result["passed"], "result_sha256": hashlib.sha256((a.output / (kind + ".json")).read_bytes()).hexdigest()}
        r.atomic_json(a.output / "manifest.json", manifest)
        print("Passed", kind, flush=True)
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    r.atomic_json(a.output / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "checks": manifest["checks"]}, indent=2))


if __name__ == "__main__":
    main()
