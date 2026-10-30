#!/usr/bin/env python3
"""Exclusive D4 BF16 checkpoint parity and CPU benchmark on archived dev only.

NumPy checks the exact logits captured from runner forwards. This tests BF16
integration, not equivalence to FP32 model inference. No frozen runner edits.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import statistics
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("d4_checkpoint_runner", ROOT / "scripts/rescore_models.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def independent_statistics(observer, performer, ids, mask, pad_id=None):
    """Independent NumPy float64 row reductions using E[L²]−E[L]² variance."""
    import numpy as np
    ids, mask = np.asarray(ids)[0], np.asarray(mask, dtype=bool)[0]
    if observer.shape != performer.shape or observer.shape[:2] != (1, len(ids)):
        raise ValueError("unaligned logits and tokens")
    targets = set(np.flatnonzero(mask[:-1] & mask[1:]).tolist())
    cross_positions = set(np.flatnonzero(mask & (ids != pad_id if pad_id is not None else True)).tolist())
    observed_o, observed_p, means, variances, crosses = [], [], [], [], []
    for t in range(len(ids)):
        if t not in targets and t not in cross_positions:
            continue
        logs = []
        for tensor in (observer, performer):
            raw = tensor[0, t]
            if hasattr(raw, "detach"):
                # BF16 -> float32 preserves every value before NumPy float64.
                raw = raw.detach().float().cpu().numpy()
            x = np.asarray(raw, dtype=np.float64)
            shifted = x - x.max()
            logs.append(shifted - np.log(np.exp(shifted).sum()))
        lo, lp = logs
        probabilities = np.exp(lo)
        if t in cross_positions:
            crosses.append(float(-np.dot(probabilities, lp)))
        if t in targets:
            observed_o.append(float(lo[ids[t + 1]]))
            observed_p.append(float(lp[ids[t + 1]]))
            mean = float(np.dot(probabilities, lo))
            means.append(mean)
            variances.append(0.0 if np.max(lo) == np.min(lo)
                             else float(np.dot(probabilities, lo * lo) - mean * mean))
    if not observed_o or not crosses or math.fsum(variances) <= 0:
        raise ValueError("observed targets, cross positions and positive variance required")
    nll_o, nll_p = -math.fsum(observed_o) / len(observed_o), -math.fsum(observed_p) / len(observed_p)
    cross = math.fsum(crosses) / len(crosses)
    numerator, variance = math.fsum(observed_o) - math.fsum(means), math.fsum(variances)
    return {"observer_mean_nll": nll_o, "performer_mean_nll": nll_p,
            "observer_prefix_ppl": math.exp(nll_o), "performer_prefix_ppl": math.exp(nll_p),
            "observer_fast_detectgpt": numerator / math.sqrt(variance),
            "observer_performer_cross_entropy": cross, "binoculars_ratio": nll_p / cross,
            "d5_numerator": numerator, "d5_variance_sum": variance,
            "predicted_tokens": len(observed_o), "cross_entropy_positions": len(crosses)}


class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peak_rss_bytes = self.peak_cgroup_bytes = self.samples = 0
        self.initial_events = self.events()

    @staticmethod
    def events():
        try:
            return {line.split()[0]: int(line.split()[1]) for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()}
        except OSError:
            return {}

    def sample(self):
        try:
            lines = Path("/proc/self/status").read_text().splitlines()
            rss = next(int(x.split()[1]) * 1024 for x in lines if x.startswith("VmRSS:"))
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.peak_cgroup_bytes = max(self.peak_cgroup_bytes, int(Path("/sys/fs/cgroup/memory.current").read_text()))
            self.samples += 1
        except (OSError, StopIteration):
            pass

    def run(self):
        while not self.stop.wait(0.2):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.sample()
        self.stop.set()
        self.thread.join()

    def result(self):
        current = self.events()
        return {"sampled_peak_rss_bytes": self.peak_rss_bytes, "sampled_peak_cgroup_bytes": self.peak_cgroup_bytes,
                "process_lifetime_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "memory_samples": self.samples, "sample_interval_seconds": 0.2,
                "cgroup_memory_events": current,
                "cgroup_memory_event_deltas": {key: value - self.initial_events.get(key, value) for key, value in current.items()}}


def text_prefix(tokenizer, text, maximum):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(encoded["input_ids"]) <= maximum:
        return text
    cropped = text[:encoded["offset_mapping"][maximum - 1][1]]
    while len(tokenizer(cropped, add_special_tokens=False)["input_ids"]) > maximum:
        cropped = cropped[:-1]
    return cropped


@contextmanager
def capture_forwards(scorer):
    captured = {}
    def hook(role):
        def save(module, args, kwargs, output):
            captured[role] = {"logits": output.logits.detach(), "ids": kwargs["input_ids"].detach(),
                              "mask": kwargs["attention_mask"].detach()}
        return save
    handles = [scorer.model.register_forward_hook(hook("observer"), with_kwargs=True),
               scorer.performer.register_forward_hook(hook("performer"), with_kwargs=True)]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def parity(scorer, row):
    import torch
    text = text_prefix(scorer.tokenizer, row["text"], 40)
    with capture_forwards(scorer) as tensors:
        value, details = scorer.score(text)
    observer, performer = tensors["observer"], tensors["performer"]
    if not torch.equal(observer["ids"], performer["ids"]) or not torch.equal(observer["mask"], performer["mask"]):
        raise AssertionError("different model token IDs or masks")
    pad = scorer.tokenizer.pad_token_id if scorer.job["denominator_mask"] == "token_id" else None
    expected = independent_statistics(observer["logits"], performer["logits"],
                                      observer["ids"].cpu().numpy(), observer["mask"].cpu().numpy(), pad)
    checks = {key: {"runner": details[key], "reference": reference, "absolute_error": abs(details[key] - reference),
                    "passed": math.isclose(details[key], reference, rel_tol=1e-9, abs_tol=1e-10)}
              for key, reference in expected.items()}
    checks["negative_D4_orientation"] = {"runner": value, "reference": -expected["binoculars_ratio"],
                                          "passed": math.isclose(value, -expected["binoculars_ratio"], rel_tol=1e-9, abs_tol=1e-10)}
    result = {"source_id": row["id"], "source_text_sha256": row["text_sha256"],
              "crop_policy": "verbatim source prefix ending at or before first40 tokens",
              "sample_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "token_ids_sha256": hashlib.sha256(observer["ids"].cpu().numpy().tobytes()).hexdigest(),
              "input_tokens": details["scored_input_tokens"], "observer_dtype": str(observer["logits"].dtype),
              "performer_dtype": str(performer["logits"].dtype), "vocabulary_size": observer["logits"].shape[-1],
              "model_token_ids_and_masks_equal": True, "checks": checks,
              "passed": all(x["passed"] for x in checks.values())}
    if not result["passed"]:
        raise AssertionError(json.dumps(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "results/neural-suite-2026-10-22/corpora/primary/corpus.jsonl")
    parser.add_argument("--job", type=Path, default=ROOT / "configs/neural/D4-bf16.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[3, 8])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--exclusive-confirmed", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if any(t < 1 for t in args.threads) or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    runner = load_runner()
    job = runner.validate_job(json.loads(args.job.read_text()))
    if job["kind"] != "D4" or job["dtype"] != "bfloat16":
        parser.error("this8GiB CPU check requires explicitly declared D4 BF16")
    if args.plan_only:
        print(json.dumps({"job": job, "threads": args.threads, "repeats": args.repeats,
                          "parity_samples": "two archived cover prefixes, <=40tokens",
                          "benchmark": "same long archived cover at40/512tokens",
                          "requires_exclusive_model_workload": True}, indent=2))
        return 0
    if not args.exclusive_confirmed:
        parser.error("coordinate with all model job owners, then pass --exclusive-confirmed")
    if args.output.exists():
        raise FileExistsError(args.output)
    corpus_manifest = json.loads(args.corpus.with_name("manifest.json").read_text())
    roles = {"archived_development_only", "legacy_human_calibration_reference_not_holdout"}
    declared = set(corpus_manifest.get("counts_by_evaluation_role", {}))
    if not declared or not declared <= roles:
        raise ValueError("only frozen archived-development corpus is permitted")
    if digest_file(args.corpus) != corpus_manifest.get("corpus_sha256"):
        raise ValueError("archived corpus hash mismatch")
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    if any(row.get("evaluation_role") not in roles for row in rows):
        raise ValueError("prospective or unclassified records excluded")
    samples = [next(row for row in rows if row["dataset"] == dataset and row["arm"] == "cover")
               for dataset in ("gpt4o", "luna")]
    long_row = max((row for row in rows if row["dataset"] in {"gpt4o", "luna"} and row["arm"] == "cover"), key=lambda row: len(row["text"]))
    args.output.mkdir(parents=True)
    manifest = {"status": "incomplete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "job": job, "runtime": runner.versions(), "runner_sha256": digest_file(ROOT / "scripts/rescore_models.py"),
                "check_script_sha256": digest_file(__file__), "corpus_sha256": digest_file(args.corpus),
                "corpus_manifest_sha256": digest_file(args.corpus.with_name("manifest.json")),
                "scope": "archived development only; BF16 integration parity, not FP32 equivalence",
                "precision_limitation": "BF16 forwards; NumPy and runner reductions float64. No FP32 model comparison.",
                "parity": [], "benchmarks": []}
    runner.atomic_json(args.output / "manifest.json", manifest)
    try:
        with MemoryMonitor() as memory:
            start = time.perf_counter()
            scorer = runner.Scorer(job, "cpu", False)
            manifest["load_seconds"] = time.perf_counter() - start
            print(json.dumps({"stage": "loaded", "seconds": manifest["load_seconds"], **memory.result()}), flush=True)
            for row in samples:
                result = parity(scorer, row)
                manifest["parity"].append(result)
                gc.collect()
                manifest["memory"] = memory.result()
                runner.atomic_json(args.output / "manifest.json", manifest)
                print(json.dumps({"stage": "parity", "source_id": row["id"], "passed": result["passed"]}), flush=True)
            lengths = [len(scorer.tokenizer(row["text"], add_special_tokens=False)["input_ids"]) for row in rows]
            manifest["corpus_token_counts"] = {"records": len(rows), "input_tokens": sum(lengths),
                                                "scored_tokens": sum(min(n, 512) for n in lengths),
                                                "truncated_records": sum(n > 512 for n in lengths),
                                                "mean_scored_tokens": statistics.mean(min(n, 512) for n in lengths)}
            import torch
            for threads in args.threads:
                torch.set_num_threads(threads)
                for limit in (40, 512):
                    text = text_prefix(scorer.tokenizer, long_row["text"], limit)
                    observations = []
                    for repeat in range(args.repeats + 1):
                        start = time.perf_counter()
                        score, details = scorer.score(text)
                        elapsed = time.perf_counter() - start
                        if details["scored_input_tokens"] != limit:
                            raise AssertionError(f"benchmark requires exactly{limit} input tokens")
                        item = {"seconds": elapsed, "score": score, "scored_input_tokens": limit, "warmup": repeat == 0}
                        observations.append(item)
                        print(json.dumps({"stage": "benchmark", "threads": threads, "token_limit": limit, **item}), flush=True)
                    manifest["benchmarks"].append({"cpu_threads": threads, "token_limit": limit,
                        "source_id": long_row["id"], "source_text_sha256": long_row["text_sha256"],
                        "sample_text_sha256": runner.text_hash(text), "observations": observations,
                        "median_seconds_excluding_warmup": statistics.median(x["seconds"] for x in observations[1:])})
                    manifest["memory"] = memory.result()
                    runner.atomic_json(args.output / "manifest.json", manifest)
            best = min((x for x in manifest["benchmarks"] if x["token_limit"] == 512), key=lambda x: x["median_seconds_excluding_warmup"])
            short = next(x for x in manifest["benchmarks"] if x["cpu_threads"] == best["cpu_threads"] and x["token_limit"] == 40)
            t_short, t_long = short["median_seconds_excluding_warmup"], best["median_seconds_excluding_warmup"]
            slope = max(0.0, (t_long - t_short) / (512 - 40))
            manifest["recommended_cpu_threads"] = best["cpu_threads"]
            manifest["estimated_1949_full_prefix_seconds"] = t_long * 1949
            manifest["estimated_actual_corpus_seconds_linear_interpolation"] = sum(t_short + max(0, min(n, 512) - 40) * slope for n in lengths)
            manifest["estimate_caveat"] = "One archived source at40/512tokens, linear interpolation over actual corpus lengths; nonlinear attention and thermal variation affect runtime."
            manifest["memory"] = memory.result()
            del scorer
            gc.collect()
        manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        runner.atomic_json(args.output / "manifest.json", manifest)
        print(json.dumps({key: manifest[key] for key in ("status", "recommended_cpu_threads", "estimated_actual_corpus_seconds_linear_interpolation", "memory")}), flush=True)
    except BaseException as error:
        manifest.update(status="incomplete", error_type=type(error).__name__, error=str(error))
        runner.atomic_json(args.output / "manifest.json", manifest)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
