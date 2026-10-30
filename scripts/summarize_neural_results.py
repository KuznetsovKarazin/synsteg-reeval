#!/usr/bin/env python3
"""Summarize complete archived-corpus neural runs; never read prospective tests.

Only status=complete runs with valid manifests, score hashes, corpus identities,
and exact record coverage receive numeric summaries. No training, tuning, sign
selection, or new neural inference is performed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ("D1", "D2", "D3", "gpt2-full", "D4-bf16")
LIMITATIONS = [
    "These are archived development data, not a new closed test.",
    "Legacy human references are historical calibration data and are not genre-matched cover negatives.",
    "AI-versus-human AUC measures AI detection, not hidden-payload detection or steganographic security.",
    "The human 95th-percentile threshold is descriptive historical calibration; its human FPR is in-sample, not independent validation.",
    "Cover, base stego, and optimized variants share documents. Reused human references and duplicate texts are not independent replications.",
    "Paired deltas match dataset and UUID; optimized texts are archived outputs, not newly optimized texts. Payload recovery must be evaluated separately.",
    "No confidence intervals or p-values are inferred from these dependent historical records.",
    "Perplexity is model-dependent and does not establish semantic fidelity, grammar, or undetectability.",
    "bfloat16 is a separately named precision protocol, not float32 reference reproduction.",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable_digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode())


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected finite numeric value")
    return float(value)


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered or not 0 <= probability <= 1:
        raise ValueError("Invalid quantile sample/probability")
    position = (len(ordered)-1)*probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)


def describe(values):
    values = [finite(v) for v in values]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd_population": statistics.pstdev(values), "median": statistics.median(values),
            "q05": quantile(values, .05), "q95": quantile(values, .95),
            "min": min(values), "max": max(values)}


def auc(human, machine):
    """Mann-Whitney AUC; machine positive, fixed higher-is-more-AI orientation."""
    if not human or not machine:
        return None
    ordered = sorted([(finite(v), 0) for v in human] + [(finite(v), 1) for v in machine])
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i+1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += sum(label for _, label in ordered[i:j])*(i+1+j)/2
        i = j
    n = len(machine)
    return (rank_sum-n*(n+1)/2)/(n*len(human))


def read_json(path):
    data = path.read_bytes()
    return json.loads(data), sha(data)


def load_corpora(suite):
    corpora, hashes = {}, {}
    # Explicit allowlist; no recursive discovery or reading of pilot/test data.
    for scope in ("primary", "quality"):
        directory = suite / "corpora" / scope
        manifest, manifest_sha = read_json(directory / "manifest.json")
        data = (directory / "corpus.jsonl").read_bytes()
        digest = sha(data)
        if manifest.get("scope") != scope or manifest.get("corpus_sha256") != digest:
            raise ValueError(f"Corpus integrity mismatch: {scope}")
        rows = {}
        for line in data.decode().splitlines():
            row = json.loads(line)
            if row["id"] in rows or sha(row["text"].encode()) != row["text_sha256"]:
                raise ValueError(f"Duplicate ID or corrupt text: {scope}")
            if row.get("evaluation_role") not in {"archived_development_only", "legacy_human_calibration_reference_not_holdout"}:
                raise ValueError("Only archived development/calibration roles are allowed")
            rows[row["id"]] = row
        unique = len({r["text_sha256"] for r in rows.values()})
        if len(rows) != manifest["n_records"] or unique != manifest["n_unique_texts"]:
            raise ValueError(f"Corpus count mismatch: {scope}")
        corpora[digest] = {"scope": scope, "manifest": manifest, "rows": rows}
        hashes[str(directory / "manifest.json")] = manifest_sha
        hashes[str(directory / "corpus.jsonl")] = digest
    return corpora, hashes


def load_complete_run(directory, corpora):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "not_started", "n_complete_unique_texts": 0}, None, {}
    manifest, manifest_sha = read_json(manifest_path)
    progress = {key: manifest.get(key) for key in ("status", "n_records", "n_unique_texts", "n_complete_unique_texts")}
    progress["observed_manifest_sha256"] = manifest_sha
    if manifest.get("status") != "complete":
        progress["note"] = "Volatile progress snapshot only; no incomplete scores were read."
        return progress, None, {}
    corpus = corpora.get(manifest["corpus_sha256"])
    if corpus is None:
        raise ValueError(f"No verified primary/quality corpus: {directory.name}")
    if stable_digest(manifest["identity"]) != manifest["job_fingerprint"]:
        raise ValueError(f"Job fingerprint mismatch: {directory.name}")
    data = (directory / "scores.jsonl").read_bytes()
    if sha(data) != manifest["scores_sha256"]:
        raise ValueError(f"Score-file SHA mismatch: {directory.name}")
    scores, cache_values = {}, {}
    kind = manifest["identity"]["job"]["kind"]
    for line in data.decode().splitlines():
        row = json.loads(line)
        identity = row["id"]
        if identity in scores or identity not in corpus["rows"]:
            raise ValueError(f"Unexpected or duplicate score ID: {directory.name}")
        source = corpus["rows"][identity]
        if row["text_sha256"] != source["text_sha256"]:
            raise ValueError(f"Text/score mismatch: {directory.name}")
        expected = stable_digest({"job_fingerprint": manifest["job_fingerprint"], "text_sha256": row["text_sha256"]})
        if row.get("cache_key") != expected:
            raise ValueError(f"Cache identity mismatch: {directory.name}")
        value = finite(row["score"])
        details = row["details"]
        n, scored = details["input_tokens"], details["scored_input_tokens"]
        if isinstance(n, bool) or isinstance(scored, bool) or not isinstance(n, int) or not isinstance(scored, int) or not 0 < scored <= n:
            raise ValueError(f"Invalid token coverage: {directory.name}")
        if details["truncated"] != (scored < n):
            raise ValueError(f"Inconsistent truncation flag: {directory.name}")
        if kind == "ppl":
            if value <= 0 or details["predicted_tokens"] != scored-1:
                raise ValueError(f"Invalid causal PPL coverage: {directory.name}")
            if not math.isclose(math.log(value), finite(details["mean_nll"]), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"PPL/NLL mismatch: {directory.name}")
        elif details.get("orientation") != "higher_is_more_AI":
            raise ValueError(f"Unexpected AI-score orientation: {directory.name}")
        if kind == "D4" and not math.isclose(value, -finite(details["binoculars_ratio"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"D4 score/ratio mismatch: {directory.name}")
        signature = stable_digest({"score": row["score"], "details": details})
        if expected in cache_values and cache_values[expected] != signature:
            raise ValueError(f"Identical cached text has conflicting scores: {directory.name}")
        cache_values[expected] = signature
        scores[identity] = {**source, **row}
    if set(scores) != set(corpus["rows"]) or len(scores) != manifest["n_records"]:
        raise ValueError(f"Incomplete record coverage: {directory.name}")
    unique = len({r["text_sha256"] for r in scores.values()})
    if unique != manifest["n_unique_texts"] or unique != manifest["n_complete_unique_texts"]:
        raise ValueError(f"Incomplete unique-text coverage: {directory.name}")
    return progress, {"manifest": manifest, "corpus": corpus, "rows": list(scores.values())}, {
        str(manifest_path): manifest_sha, str(directory / "scores.jsonl"): sha(data)}


def coverage(rows):
    input_tokens = sum(r["details"]["input_tokens"] for r in rows)
    scored_tokens = sum(r["details"]["scored_input_tokens"] for r in rows)
    return {"n_records": len(rows), "n_unique_texts": len({r["text_sha256"] for r in rows}),
            "truncated_records": sum(r["details"]["truncated"] for r in rows),
            "input_tokens_sum_over_records": input_tokens,
            "scored_input_tokens_sum_over_records": scored_tokens,
            "scored_input_token_fraction": scored_tokens/input_tokens if input_tokens else None,
            "input_tokens": describe([r["details"]["input_tokens"] for r in rows])}


def threshold_metrics(human, machine, threshold):
    fpr = sum(v > threshold for v in human)/len(human)
    tpr = sum(v > threshold for v in machine)/len(machine)
    return {"threshold": threshold, "rule": "score > threshold", "human_FPR": fpr,
            "machine_TPR": tpr, "balanced_accuracy": (1-fpr+tpr)/2,
            "accuracy": (sum(v <= threshold for v in human)+sum(v > threshold for v in machine))/(len(human)+len(machine))}


def metric_summary(rows, field, kind, probability=False):
    def value(row):
        return finite(row["score"] if field == "score" else row["details"][field])
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["arm"])].append(row)
    summaries = {}
    for (dataset, arm), group in sorted(groups.items()):
        values = [value(r) for r in group]
        result = {"dataset": dataset, "arm": arm, "scores": describe(values), **coverage(group)}
        if kind == "ppl":
            if min(values) <= 0:
                raise ValueError("Nonpositive perplexity")
            result["geometric_mean_ppl_over_records"] = math.exp(statistics.fmean(math.log(v) for v in values))
            if all("predicted_tokens" in r["details"] for r in group):
                count = sum(r["details"]["predicted_tokens"] for r in group)
                if count <= 0:
                    raise ValueError("No PPL target tokens")
                result["predicted_tokens_sum_over_records"] = count
                result["token_weighted_ppl_over_records"] = math.exp(sum(math.log(value(r))*r["details"]["predicted_tokens"] for r in group)/count)
        summaries[dataset+"/"+arm] = result
    paired = []
    for dataset in sorted({dataset for dataset, _ in groups}):
        arms = sorted(arm for ds, arm in groups if ds == dataset)
        if "cover" not in arms:
            continue
        comparisons = [("cover", a) for a in arms if a != "cover"]
        comparisons += [("stego", a) for a in arms if a.startswith("stego_opt")]
        for reference, target in comparisons:
            if reference not in arms:
                continue
            left, right = {}, {}
            for arm, destination in ((reference, left), (target, right)):
                for row in groups[(dataset, arm)]:
                    if row["uuid"] in destination:
                        raise ValueError(f"Ambiguous paired UUID: {dataset}/{arm}")
                    destination[row["uuid"]] = row
            common = sorted(left.keys() & right.keys())
            delta = [value(right[u])-value(left[u]) for u in common]
            item = {"dataset": dataset, "reference_arm": reference, "target_arm": target,
                    "n_pairs": len(common), "reference_only": len(left.keys()-right.keys()),
                    "target_only": len(right.keys()-left.keys()), "target_minus_reference": describe(delta),
                    "target_lower": sum(d < 0 for d in delta), "equal": sum(d == 0 for d in delta),
                    "target_higher": sum(d > 0 for d in delta),
                    "pairs_with_either_arm_truncated": sum(left[u]["details"]["truncated"] or right[u]["details"]["truncated"] for u in common)}
            if kind == "ppl" and common:
                ratios = [value(right[u])/value(left[u]) for u in common]
                item["target_over_reference_ppl_ratio"] = describe(ratios)
                item["geometric_mean_ppl_ratio"] = math.exp(statistics.fmean(math.log(r) for r in ratios))
            paired.append(item)
    comparisons = []
    human_rows = [r for r in rows if r["arm"] == "human"]
    if kind == "ai" and human_rows:
        human = [value(r) for r in human_rows]
        threshold = quantile(human, .95)
        for (dataset, arm), group in sorted(groups.items()):
            if arm == "human":
                continue
            machine = [value(r) for r in group]
            item = {"machine_dataset": dataset, "machine_arm": arm, "n_human": len(human),
                    "n_machine": len(machine), "positive_class": "machine", "orientation": "higher_is_more_AI",
                    "AUC": auc(human, machine), "historical_human_q95_calibration": threshold_metrics(human, machine, threshold),
                    "evaluation_role": "archived_development_with_legacy_human_calibration_not_holdout"}
            if probability:
                item["fixed_probability_0_5"] = threshold_metrics(human, machine, .5)
            comparisons.append(item)
    return {"kind": kind, "source_field": field, "groups": summaries,
            "paired_deltas": paired, "AI_vs_historical_human": comparisons}


def summarize_run(run):
    manifest, rows = run["manifest"], run["rows"]
    job = manifest["identity"]["job"]
    kind = job["kind"]
    name = "perplexity" if kind == "ppl" else "negative_binoculars_ratio" if kind == "D4" else "AI_score"
    metrics = {name: metric_summary(rows, "score", "ppl" if kind == "ppl" else "ai", kind in {"D1", "D2", "D3"})}
    if kind == "D4":
        for field in ("observer_prefix_ppl", "performer_prefix_ppl", "observer_fast_detectgpt"):
            metrics[field] = metric_summary(rows, field, "ppl" if field.endswith("ppl") else "ai")
    return {"status": "complete_verified", "corpus_scope": run["corpus"]["scope"],
            "job_fingerprint": manifest["job_fingerprint"], "job": job,
            "runtime": manifest["identity"]["runtime"], "device": manifest["identity"]["device"],
            "scoring_script_sha256": manifest["identity"]["script_sha256"],
            "created_utc": manifest["created_utc"], "finished_utc": manifest["finished_utc"],
            "scores_sha256": manifest["scores_sha256"], "corpus_sha256": manifest["corpus_sha256"],
            "coverage": coverage(rows), "metrics": metrics,
            "notes": (["D4 is the negative Binoculars ratio; sign is fixed by the runner, never chosen from results.",
                       "Auxiliary Qwen PPL and D5 reuse the same forwards; they are not extra independent inference runs.",
                       "Auxiliary D5 uses the observer as both reference and scoring model: a self-reference configuration."] if kind == "D4" else [])}


def number(value):
    return "—" if value is None else f"{value:.6g}"


def markdown(summary):
    lines = ["# Neural re-evaluation of archived SynSteg data", "", f"Generated: {summary['created_utc']}", "",
             f"Suite status: **{summary['suite_status']}**. Numeric tables contain only complete, integrity-checked runs.", "",
             "Historical development/calibration analysis only. AI-versus-human detection is not a steganalysis benchmark or closed test.", "",
             "| Run | Status | Records | Unique complete / expected |", "|---|---|---:|---:|"]
    for name, p in summary["run_progress"].items():
        lines.append(f"| {name} | {p['status']} | {p.get('n_records', '—')} | {p.get('n_complete_unique_texts', 0)} / {p.get('n_unique_texts', '—')} |")
    for name, run in summary["runs"].items():
        job, cov = run["job"], run["coverage"]
        lines += ["", f"## {name}", "", f"Model: `{job['model_id']}`; revision `{job['model_revision']}`. "
                  f"Precision: **{job['dtype']}**, `{job['precision_protocol']}`. Corpus: `{run['corpus_scope']}`.", "",
                  f"Coverage: {cov['n_records']} records / {cov['n_unique_texts']} unique texts; "
                  f"{cov['truncated_records']} records truncated. Input-token coverage: {number(cov['scored_input_token_fraction'])}.", "",
                  (f"PPL policy: `{job.get('ppl_policy')}`; context {job.get('context_length')}; stride {job.get('stride')}." if job['kind'] == 'ppl'
                   else f"Maximum input length: {job.get('max_length')} model tokens; longer texts are evaluated on a prefix."), ""]
        for metric_name, metric in run["metrics"].items():
            lines += [f"### {metric_name}", "", "| Dataset / arm | N | Mean | Median | Truncated |", "|---|---:|---:|---:|---:|"]
            for label, group in metric["groups"].items():
                s = group["scores"]
                lines.append(f"| {label} | {s['n']} | {number(s.get('mean'))} | {number(s.get('median'))} | {group['truncated_records']} |")
            lines += ["", "Paired differences are target minus reference. A lower AI score does not establish hidden-payload secrecy.", "",
                      "| Dataset | Reference → target | Pairs | Mean difference | Target lower |", "|---|---|---:|---:|---:|"]
            for p in metric["paired_deltas"]:
                lines.append(f"| {p['dataset']} | {p['reference_arm']} → {p['target_arm']} | {p['n_pairs']} | {number(p['target_minus_reference'].get('mean'))} | {p['target_lower']} |")
            if metric["AI_vs_historical_human"]:
                lines += ["", "Historical AI-versus-human comparisons. Human FPR uses the same records used to set the q95 threshold; it is not independently validated.", "",
                          "| Machine group | AI-vs-human AUC | TPR at human q95 | In-sample human FPR |", "|---|---:|---:|---:|"]
                for c in metric["AI_vs_historical_human"]:
                    t = c["historical_human_q95_calibration"]
                    lines.append(f"| {c['machine_dataset']}/{c['machine_arm']} | {number(c['AUC'])} | {number(t['machine_TPR'])} | {number(t['human_FPR'])} |")
            lines.append("")
        lines += run["notes"]
    lines += ["", "## Limits", ""] + ["- "+note for note in summary["limitations"]]
    lines += ["", "JSON retains precision settings, revisions, token coverage, paired missingness, distributions and input hashes. This script tuned no parameters.", ""]
    return "\n".join(lines)


def write_atomic(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "results/neural-suite-2026-10-22")
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--require-complete", action="store_true", help="Fail before writing if any requested run is incomplete")
    args = parser.parse_args(argv)
    suite = args.suite.resolve()
    if len(set(args.runs)) != len(args.runs) or any(Path(n).name != n for n in args.runs):
        raise ValueError("Run names must be unique directory basenames")
    corpora, inputs = load_corpora(suite)
    progress, runs = {}, {}
    for name in args.runs:
        p, run, hashes = load_complete_run(suite/name, corpora)
        progress[name] = p
        if run is not None:
            runs[name] = summarize_run(run)
            inputs.update(hashes)
    if args.require_complete and len(runs) != len(args.runs):
        raise ValueError("Not all requested runs are complete; no output written")
    summary = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "suite_status": "complete" if len(runs) == len(args.runs) else "partial_suite_complete_runs_only",
               "evaluation_role": "historical_development_and_calibration_not_closed_test",
               "run_progress": progress, "runs": runs, "input_sha256": inputs,
               "summary_script_sha256": sha(Path(__file__).read_bytes()), "limitations": LIMITATIONS}
    outputs = {"summary.json": (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode(),
               "summary.md": markdown(summary).encode()}
    for name, data in outputs.items():
        write_atomic(suite/name, data)
    manifest = {"created_utc": summary["created_utc"], "suite_status": summary["suite_status"],
                "summary_script_sha256": summary["summary_script_sha256"], "input_sha256": inputs,
                "outputs_sha256": {name: sha(data) for name, data in outputs.items()},
                "completed_runs": list(runs), "requested_runs": args.runs,
                "note": "Incomplete progress is a volatile snapshot; incomplete score files were not read."}
    write_atomic(suite/"summary_manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2)+"\n").encode())
    print(json.dumps({"suite_status": summary["suite_status"], "completed_runs": list(runs),
                      "outputs": [str(suite/n) for n in (*outputs, "summary_manifest.json")]}, indent=2))


if __name__ == "__main__":
    main()
