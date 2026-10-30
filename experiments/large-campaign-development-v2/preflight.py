#!/usr/bin/env python3
"""Public resource preparation and inference-free preflight for one new campaign.

No model forward/generate or ledger reservation is implemented here. Local resource
paths are a reconstructible cache; manifests and receipts are the durable evidence.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "large-campaign-development-v2"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
PROMPTS = ROOT / "results/recovery-development-v1/linux_snapshot/development-prompts.jsonl"
PROMPTS_SHA = "3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1"
RUNNER_SHA = "4d5ab74c78c58214c1b5e11a346b4339eb0c6d08e713e8909d0d5fd9947c7e02"
OUT = ROOT / "results" / CAMPAIGN / "preflight"
RUNTIME = ROOT / "runtime" / CAMPAIGN
HF_HOME = RUNTIME / "hf"
CACHE = HF_HOME / "hub"
SNAPSHOT = CACHE / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
FILES = ("config.json", "generation_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "model.safetensors")
SYSTEM = "Write clear English prose. Return only the requested text, without a heading or commentary."

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def download():
    info_url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(info_url, timeout=60) as response:
        info = json.load(response)
    if info.get("sha") != REVISION:
        raise RuntimeError("Upstream revision mismatch")
    write("upstream-model-info.json", info)
    records = {row["rfilename"]: row for row in info["siblings"]}
    receipt = {"schema": "synsteg-public-model-download-v2", "campaign_id": CAMPAIGN,
               "started_utc": now(), "model": MODEL, "revision": REVISION,
               "metadata_url": info_url, "authentication_used": False,
               "automatic_retries": 0, "files": [], "status": "in_progress"}
    write("download-receipt.json", receipt)
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        meta = records[name]
        expected_size = meta["size"]
        expected_sha = meta.get("lfs", {}).get("sha256")
        path = SNAPSHOT / name
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        row = {"name": name, "source_url": url, "expected_size": expected_size,
               "expected_lfs_sha256": expected_sha, "upstream_git_blob_id": meta.get("blobId")}
        start = time.monotonic()
        try:
            if not path.is_file():
                partial = path.with_suffix(path.suffix + ".partial")
                with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as target:
                    while chunk := response.read(8 * 1024 * 1024):
                        target.write(chunk)
                partial.replace(path)
                row["action"] = "downloaded_public_file"
            else:
                row["action"] = "verified_existing_file"
            row.update(size=path.stat().st_size, sha256=sha(path))
            if row["size"] != expected_size or (expected_sha and row["sha256"] != expected_sha):
                raise RuntimeError("Downloaded asset size/hash mismatch")
            if not expected_sha and meta.get("blobId"):
                raw = path.read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                if blob != meta["blobId"]:
                    raise RuntimeError("Downloaded Git blob hash mismatch")
                row["verified_git_blob_sha1"] = blob
            row["status"] = "verified"
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            receipt["status"] = "blocked"
            receipt["files"].append(row)
            receipt["finished_utc"] = now()
            write("download-receipt.json", receipt)
            raise
        row["elapsed_seconds"] = round(time.monotonic() - start, 6)
        receipt["files"].append(row)
        write("download-receipt.json", receipt)
        print(json.dumps({k: row[k] for k in ("name", "size", "sha256", "status")}), flush=True)
    receipt.update(status="all_public_assets_verified", finished_utc=now())
    write("download-receipt.json", receipt)

def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    receipt = {"schema": "synsteg-large-development-preflight-v2", "campaign_id": CAMPAIGN,
               "started_utc": now(), "model": MODEL, "revision": REVISION,
               "status": "in_progress", "generation_calls": 0, "forward_calls": 0,
               "attempt_reservations": 0, "paid_calls": 0, "closed_test_accessed": False,
               "dtype": "float32", "cpu_threads": 6, "max_output_tokens": 1024,
               "hf_home_relative": str(HF_HOME.relative_to(ROOT)),
               "hf_cache_relative": str(CACHE.relative_to(ROOT)),
               "python": sys.version, "platform": platform.platform(),
               "source_script_sha256": sha(__file__)}
    write("preflight-receipt.json", receipt)
    try:
        if sha(PROMPTS) != PROMPTS_SHA or sha(ROOT / "scripts/run_budget_generation.py") != RUNNER_SHA:
            raise RuntimeError("Frozen prompt or runner hash mismatch")
        downloads = json.loads((OUT / "download-receipt.json").read_text())
        if downloads["status"] != "all_public_assets_verified":
            raise RuntimeError("Incomplete model download")
        for item in downloads["files"]:
            if sha(SNAPSHOT / item["name"]) != item["sha256"]:
                raise RuntimeError("Cached model resource drift")
        receipt["model_resource_manifest_sha256"] = sha(OUT / "download-receipt.json")
        receipt["packages"] = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(6)
        torch.use_deterministic_algorithms(True)
        options = dict(revision=REVISION, local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, **options)
        config = AutoConfig.from_pretrained(MODEL, **options)
        rows = [json.loads(line) for line in PROMPTS.read_text().splitlines() if line.strip()]
        if len(rows) != 100 or any(r["split"] != "development" for r in rows):
            raise RuntimeError("Require exactly all 100 development prompts")
        if len({r["record_id"] for r in rows}) != 100 or len({r["source_group_id"] for r in rows}) != 100:
            raise RuntimeError("Duplicate record/source_group ID")
        lengths = []
        for index, row in enumerate(rows):
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            count = inputs["input_ids"].shape[1]
            context = getattr(config, "max_position_embeddings", None)
            lengths.append({"source_order": index + 1, "record_id": row["record_id"],
                "source_group_id": row["source_group_id"], "target_words": row["target_words"],
                "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                "chat_template_render_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_tokens": count, "max_output_tokens": 1024, "model_context_tokens": context,
                "remaining_after_requested_output": context - count - 1024,
                "context_fits": count + 1024 <= context})
        write("all-100-context-checks.json", {"schema": "synsteg-development-context-checks-v2",
             "model": MODEL, "revision": REVISION, "prompts_sha256": PROMPTS_SHA,
             "denominator": 100, "rows": lengths})
        receipt["all_100_context_checks_sha256"] = sha(OUT / "all-100-context-checks.json")
        receipt["context_summary"] = {
            "denominator": len(lengths), "context_fits": sum(r["context_fits"] for r in lengths),
            "input_tokens_min": min(r["input_tokens"] for r in lengths),
            "input_tokens_max": max(r["input_tokens"] for r in lengths),
            "model_context_tokens": config.max_position_embeddings,
            "target_words_counts": dict(collections.Counter(str(r["target_words"]) for r in rows))}
        write("preflight-receipt.json", receipt)
        started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, **options).eval().cpu()
        receipt["model_preload"] = {"status": "loaded_without_forward_or_generate",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "devices": sorted({str(p.device) for p in model.parameters()}),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        del model
        fits = receipt["context_summary"]["context_fits"] == 100
        receipt.update(status="preflight_passed_awaiting_git_freeze" if fits else "blocked_context_overflow",
                       ready_for_attempt_reservation=False,
                       readiness_note="Requires independent audit and Git-frozen launch protocol before any generation")
    except Exception as exc:
        receipt.update(status="blocked", error_type=type(exc).__name__, error=str(exc),
                       ready_for_attempt_reservation=False)
        raise
    finally:
        receipt["finished_utc"] = now()
        write("preflight-receipt.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "preflight"))
    args = parser.parse_args()
    download() if args.action == "download" else preflight()

