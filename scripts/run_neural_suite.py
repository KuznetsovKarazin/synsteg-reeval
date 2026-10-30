#!/usr/bin/env python3
"""Sequential, resumable declared neural scope: primary1949 and quality3449.

Runs one model job at a time. D4 also emits same-forward Qwen PPL and corrected
D5; BF16 is a separate CPU protocol, never an FP32 replication claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"primary": (1949, 1949), "quality": (3449, 3443)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def runner_module():
    spec = importlib.util.spec_from_file_location("suite_runner", ROOT / "scripts/rescore_models.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def profile_jobs(profile):
    if profile not in {"cpu", "gpu-fp32"}:
        raise ValueError("unknown profile")
    d4 = "D4-bf16" if profile == "cpu" else "D4-fp32"
    return [("gpt2-full", "quality"), ("D1", "primary"), ("D2", "primary"),
            ("D3", "primary"), (d4, "primary")]


def validate_corpus(folder, scope, runner):
    manifest = json.loads((folder / "manifest.json").read_text())
    records = runner.read_corpus(folder / "corpus.jsonl")
    expected_records, expected_unique = EXPECTED[scope]
    if manifest.get("scope") != scope or (len(records), len({r["text_sha256"] for r in records})) != EXPECTED[scope]:
        raise ValueError(f"{scope} corpus scope/count mismatch")
    if (manifest.get("n_records"), manifest.get("n_unique_texts")) != EXPECTED[scope]:
        raise ValueError(f"{scope} manifest count mismatch")
    if sha(folder / "corpus.jsonl") != manifest.get("corpus_sha256"):
        raise ValueError(f"{scope} corpus hash mismatch")
    if sha(folder / "texts.jsonl") != manifest.get("unique_texts_sha256"):
        raise ValueError(f"{scope} unique-text table hash mismatch")
    texts = [json.loads(line) for line in (folder / "texts.jsonl").read_text().splitlines() if line.strip()]
    if len(texts) != expected_unique or {x["text_sha256"] for x in texts} != {x["text_sha256"] for x in records}:
        raise ValueError(f"{scope} unique-text inventory mismatch")
    if any(runner.text_hash(x["text"]) != x["text_sha256"] for x in texts):
        raise ValueError(f"{scope} unique text content mismatch")
    return manifest, records


def validate_completed_job(folder, records, expected_identity, corpus_sha, runner):
    """Return False for missing/incomplete output; reject contradictory identity.

    A completed cache is reusable only after exact scope, hashes and ordered
    record identity checks. Partial score tables are never read as results.
    """
    path = folder / "manifest.json"
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    fingerprint = runner.digest(expected_identity)
    if (manifest.get("job_fingerprint") != fingerprint
            or manifest.get("identity") != expected_identity
            or manifest.get("corpus_sha256") != corpus_sha):
        raise ValueError(f"incompatible job/runtime/corpus: {folder.name}")
    if manifest.get("status") != "complete":
        return False
    unique_count = len({row["text_sha256"] for row in records})
    if (manifest.get("n_records"), manifest.get("n_unique_texts"), manifest.get("n_complete_unique_texts")) != (len(records), unique_count, unique_count):
        raise ValueError(f"completed count mismatch: {folder.name}")
    scores_path = folder / "scores.jsonl"
    if not scores_path.is_file() or sha(scores_path) != manifest.get("scores_sha256"):
        raise ValueError(f"completed score hash mismatch: {folder.name}")
    scores = [json.loads(line) for line in scores_path.read_text().splitlines() if line.strip()]
    if len(scores) != len(records):
        raise ValueError(f"completed row count mismatch: {folder.name}")
    for source, score in zip(records, scores):
        expected_key = runner.cache_key(fingerprint, source["text_sha256"])
        if (score.get("id"), score.get("text_sha256"), score.get("cache_key")) != (source["id"], source["text_sha256"], expected_key):
            raise ValueError(f"completed row identity mismatch: {folder.name}")
        value = score.get("score")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"nonfinite score: {folder.name}")
        if expected_identity["job"]["kind"] == "D4":
            details = score.get("details", {})
            for key in ("observer_prefix_ppl", "performer_prefix_ppl", "observer_fast_detectgpt", "binoculars_ratio"):
                value = details.get(key)
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"missing/nonfinite D4 auxiliary {key}")
            if not math.isclose(score["score"], -details["binoculars_ratio"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("D4 score orientation mismatch")
    return True


def hardware(device):
    info = {"machine": platform.machine(), "processor": platform.processor()}
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("gpu-fp32 requires CUDA; no silent CPU/BF16 fallback")
        info.update(cuda_runtime=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0),
                    gpu_capability=list(torch.cuda.get_device_capability(0)),
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    float32_matmul_precision=torch.get_float32_matmul_precision())
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["cpu", "gpu-fp32"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    runner = runner_module()
    device = "cpu" if args.profile == "cpu" else "cuda"
    specs = profile_jobs(args.profile)
    configs = {name: runner.validate_job(json.loads((ROOT / "configs/neural" / f"{name}.json").read_text()))
               for name, scope in specs}
    output = args.output.resolve()
    plan = {"profile": args.profile, "device": device, "sequential_jobs": True,
            "scope": {s: {"records": n, "unique_texts": u} for s, (n, u) in EXPECTED.items()},
            "jobs": [{"name": name, "corpus_scope": scope, "job": configs[name]} for name, scope in specs],
            "scope_limitation": "Declared primary1949/quality3449 only; not all13460 archived search candidates; no prospective data."}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    # Separate processes are used for model jobs, so each model is unloaded
    # before the next starts. Direct rescore jobs must not run in these folders.
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".suite.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another suite owns this output directory") from error
        execution_hardware = hardware(device)
        subprocess.run([sys.executable, str(ROOT / "scripts/import_source.py")], cwd=ROOT, check=True)
        corpora = {}
        for scope in EXPECTED:
            folder = output / "corpora" / scope
            if not folder.exists():
                subprocess.run([sys.executable, str(ROOT / "scripts/build_neural_corpus.py"),
                                "--scope", scope, "--output", str(folder)], cwd=ROOT, check=True)
            corpora[scope] = validate_corpus(folder, scope, runner)
        runtime = runner.versions()
        identity = {"schema_version": 2, "profile": args.profile, "device": device,
                    "jobs": configs, "runtime": runtime, "hardware": execution_hardware,
                    "runner_sha256": sha(ROOT / "scripts/rescore_models.py"),
                    "suite_sha256": sha(__file__), "corpus_builder_sha256": sha(ROOT / "scripts/build_neural_corpus.py"),
                    "corpora": {scope: manifest["corpus_sha256"] for scope, (manifest, records) in corpora.items()}}
        path = output / "suite_manifest.json"
        manifest = {"status": "incomplete", "created_utc": utc(), "identity": identity,
                    "fingerprint": runner.digest(identity), "plan": plan, "jobs": {}}
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("identity") != identity or prior.get("fingerprint") != runner.digest(identity):
                raise ValueError("suite identity/environment changed; choose a new output directory")
            manifest["created_utc"] = prior["created_utc"]
        runner.atomic_json(path, manifest)
        try:
            for name, scope in specs:
                corpus_manifest, records = corpora[scope]
                folder = output / name
                expected = {"job": configs[name], "runtime": runtime, "device": device,
                            "script_sha256": identity["runner_sha256"]}
                reused = validate_completed_job(folder, records, expected, corpus_manifest["corpus_sha256"], runner)
                manifest["jobs"][name] = {"status": "complete" if reused else "running", "reused_completed_cache": reused}
                runner.atomic_json(path, manifest)
                if not reused:
                    command = [sys.executable, str(ROOT / "scripts/rescore_models.py"), "--corpus",
                               str(output / "corpora" / scope / "corpus.jsonl"), "--job",
                               str(ROOT / "configs/neural" / f"{name}.json"), "--output", str(folder), "--device", device]
                    if args.allow_download:
                        command.append("--allow-download")
                    print(f"Running {name}: {len(records)} {scope} records", flush=True)
                    subprocess.run(command, cwd=ROOT, check=True)
                if not validate_completed_job(folder, records, expected, corpus_manifest["corpus_sha256"], runner):
                    raise ValueError(f"job returned without complete validated output: {name}")
                manifest["jobs"][name].update(status="complete", manifest_sha256=sha(folder / "manifest.json"))
                runner.atomic_json(path, manifest)
            manifest.update(status="complete", finished_utc=utc())
            runner.atomic_json(path, manifest)
        except BaseException as error:
            manifest.update(status="incomplete", failure_type=type(error).__name__, failure_message=str(error))
            runner.atomic_json(path, manifest)
            raise
    print(json.dumps({"status": "complete", "profile": args.profile, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
