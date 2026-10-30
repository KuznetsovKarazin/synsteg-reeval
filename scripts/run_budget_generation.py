#!/usr/bin/env python3
"""Resumable development generation with a shared <= USD 20 reservation ledger.

Preview is the default. Paid attempts are serialized across all runs sharing
the campaign ledger. Failures and interrupted requests are never auto-retried.
No model, prompt or cover is selected by capacity or generation success.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.budget_ledger import (BudgetError, CampaignLedger, UNIT,
    atomic_json, canonical_hash, money, token_cost)
import run_claude_smoke as smoke

SYSTEM = "Write clear English prose. Return only the requested text, without a heading or commentary."
SPLITS = ("development", "train", "validation", "test")
PINNED_LOCAL = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "tiiuae/Falcon3-1B-Instruct": "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf",
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_runtime():
    """Read installed versions without importing/loading any neural runtime."""
    packages = ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")
    try:
        versions = {name: importlib.metadata.version(name) for name in packages}
    except importlib.metadata.PackageNotFoundError:
        raise BudgetError("Local runtime dependencies are missing; install the registered requirements before execution") from None
    return {"schema": "synsteg-local-runtime-recovery-v1", "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "packages": versions}


def read_prompts(path, split, max_records):
    if split not in SPLITS or type(max_records) is not int or not 1 <= max_records <= 1600:
        raise BudgetError("Invalid split or max-records (1..1600)")
    rows, ids, groups = [], set(), set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        required = ("record_id", "source_group_id", "split", "topic", "genre", "prompt")
        if not isinstance(obj, dict) or any(not isinstance(obj.get(k), str) or not obj[k].strip() for k in required):
            raise BudgetError("Malformed prompt record")
        if (obj["split"] not in SPLITS or type(obj.get("target_words")) is not int
                or not 1 <= obj["target_words"] <= 2000 or len(obj["prompt"]) > 20000):
            raise BudgetError("Invalid prompt split, length or target words")
        if obj["record_id"] in ids or obj["source_group_id"] in groups:
            raise BudgetError("Duplicate record or source group in prompts")
        ids.add(obj["record_id"])
        groups.add(obj["source_group_id"])
        if obj["split"] == split:
            rows.append(obj)
    if not rows:
        raise BudgetError("No records in requested split")
    return rows[:max_records]


def read_pricing(path, model, max_output_tokens):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("schema") != "synsteg-generation-pricing-v1":
        raise BudgetError("Invalid pricing schema")
    source = obj.get("source_url", "")
    if not isinstance(source, str) or not source.startswith(("https://www.anthropic.com/", "https://platform.claude.com/")):
        raise BudgetError("Pricing requires an official source URL")
    try:
        date.fromisoformat(obj["checked_at"])
    except (KeyError, TypeError, ValueError):
        raise BudgetError("Pricing requires checked_at in YYYY-MM-DD form") from None
    entry = obj.get("models", {}).get(model)
    if not isinstance(entry, dict) or not re.fullmatch(r"claude-[a-z0-9.-]+", model):
        raise BudgetError("Exact Claude model is absent from supplied pricing")
    for field in ("input_usd_per_million", "output_usd_per_million"):
        if money(entry.get(field)) <= 0:
            raise BudgetError("Pricing rates must be positive, covering all applicable tiers")
    for field in ("input_context_tokens", "max_output_tokens"):
        if type(entry.get(field)) is not int or not 1 <= entry[field] <= 2_000_000:
            raise BudgetError("Pricing requires bounded model context/output limits")
    if max_output_tokens > entry["max_output_tokens"]:
        raise BudgetError("Output limit exceeds supplied model limit")
    options = entry.get("request_options", {})
    if not isinstance(options, dict) or set(options) - {"thinking", "output_config"}:
        raise BudgetError("Only explicit thinking/output_config request options are allowed")
    if "thinking" in options and options["thinking"] not in ({"type": "disabled"}, {"type": "between_tools"}):
        raise BudgetError("Unbounded or unknown thinking configuration refused")
    if "output_config" in options:
        setting = options["output_config"]
        if not isinstance(setting, dict) or set(setting) != {"effort"} or setting["effort"] not in {"low", "medium", "high"}:
            raise BudgetError("Unsupported output configuration")
    return obj, entry


def record_metadata(row):
    return {**{k: row[k] for k in ("record_id", "source_group_id", "split", "topic", "genre", "target_words")},
            "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest()}


def check_model_access(model, key, opener=None):
    """One no-generation GET; retain only an exact ID/status, never raw errors."""
    request = urllib.request.Request(smoke.API_ROOT + "/v1/models/" + model,
        method="GET", headers={"x-api-key": key, "anthropic-version": smoke.API_VERSION,
                               "accept": "application/json"})
    send = urllib.request.build_opener(smoke.NoRedirect()).open if opener is None else opener
    try:
        with send(request, timeout=30) as response:
            status = response.status
            raw = response.read(smoke.MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        return {"status": "http_error", "http_status": status, "generation_requests_sent": 0}
    except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException):
        return {"status": "transport_error", "generation_requests_sent": 0}
    try:
        obj = json.loads(raw) if len(raw) <= smoke.MAX_RESPONSE_BYTES else None
    except (ValueError, RecursionError):
        obj = None
    okay = (type(status) is int and 200 <= status < 300 and isinstance(obj, dict)
            and obj.get("id") == model and key not in model)
    return {"status": "model_access_verified_no_generation" if okay else "invalid_model_response",
            "http_status": status if type(status) is int else None,
            "model": model, "generation_requests_sent": 0, "inference_verified": False}


def _zero_usage(value):
    if value is None or type(value) is int and value == 0:
        return True
    return isinstance(value, dict) and all(_zero_usage(v) for v in value.values())


def validate_claude(reply, model, rate, max_output_tokens):
    if reply.get("status") != "http_success" or reply.get("credential_redacted"):
        raise BudgetError("HTTP failure, unknown outcome or credential echo; retained without retry")
    raw = reply.get("retained_response")
    if (not isinstance(raw, dict) or raw.get("type") != "message" or raw.get("role") != "assistant"
            or raw.get("model") != model or not isinstance(raw.get("id"), str) or not raw["id"]):
        raise BudgetError("Invalid response identity")
    usage = raw.get("usage")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
        raise BudgetError("Missing token usage")
    # No caching, tools, priority/fast mode or other billed features requested.
    # Unexpected usage fields that indicate such features invalidate settlement.
    for key, value in usage.items():
        if key == "service_tier" and value == "standard":
            continue
        if key not in ("input_tokens", "output_tokens") and not _zero_usage(value):
            raise BudgetError("Unrecognized nonzero usage category; conservative reservation retained")
    if usage["input_tokens"] > rate["input_context_tokens"] or usage["output_tokens"] > max_output_tokens:
        raise BudgetError("Token usage exceeds configured model/request bound")
    blocks = raw.get("content")
    if not isinstance(blocks, list) or any(not isinstance(b, dict) for b in blocks):
        raise BudgetError("Invalid content blocks")
    text = "".join(b["text"] for b in blocks if b.get("type") == "text" and isinstance(b.get("text"), str)).strip()
    if not text or raw.get("stop_reason") not in ("end_turn", "max_tokens"):
        raise BudgetError("Empty text, refusal or unexpected stop reason")
    cost = token_cost(usage["input_tokens"], usage["output_tokens"], rate["input_usd_per_million"], rate["output_usd_per_million"])
    return {"cover_text": text, "stop_reason": raw["stop_reason"], "usage": usage,
            "hit_output_limit": raw["stop_reason"] == "max_tokens", "response_id": raw["id"],
            "request_id": reply.get("request_id"), "returned_model": raw["model"],
            "accounted_standard_token_cost_usd": cost / UNIT,
            "retained_response_sha256": reply.get("retained_response_sha256")}, cost


class LocalGenerator:
    def __init__(self, model, revision, dtype, cpu_threads, seed):
        self.model_id, self.revision, self.dtype = model, revision, dtype
        self.cpu_threads, self.seed = cpu_threads, seed
        self.tokenizer = self.model = None

    def __call__(self, row, max_output_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(self.cpu_threads)
        torch.use_deterministic_algorithms(True)
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision,
                local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, revision=self.revision,
                local_files_only=True, trust_remote_code=False,
                torch_dtype=getattr(torch, self.dtype)).eval().cpu()
        seed = int(canonical_hash([self.seed, self.model_id, row["record_id"]])[:8], 16)
        torch.manual_seed(seed)
        prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_count = inputs["input_ids"].shape[1]
        context = getattr(self.model.config, "max_position_embeddings", None)
        if not isinstance(context, int) or input_count + max_output_tokens > context:
            raise BudgetError("Local context is insufficient; no truncation or regeneration")
        started = time.monotonic()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_output_tokens,
                do_sample=True, temperature=0.8, top_p=0.95, top_k=50,
                pad_token_id=self.tokenizer.eos_token_id)
        token_ids = outputs[0, input_count:]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if not text:
            raise BudgetError("Local generation returned no text; no retry")
        limited = len(token_ids) == max_output_tokens
        return {"cover_text": text, "stop_reason": "max_tokens" if limited else "end_turn",
                "usage": {"input_tokens": input_count, "output_tokens": len(token_ids)},
                "hit_output_limit": limited, "seed": seed,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "returned_model": self.model_id, "revision": self.revision, "dtype": self.dtype}


def run(*, prompts, output, provider, model, max_records=30, max_output_tokens=2048,
        split="development", pricing=None, campaign_ledger=None, budget_usd="20",
        revision=None, dtype="float32", cpu_threads=6, seed=20261024,
        frozen_manifest=None, execute=False, environ=None, sender=None, local_generator=None,
        preflight_sender=None, runtime_reader=None, max_new_records=None):
    if provider not in ("claude", "local") or type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 8192:
        raise BudgetError("Invalid provider or output limit (1..8192)")
    if dtype not in ("float32", "bfloat16") or type(cpu_threads) is not int or not 1 <= cpu_threads <= 64:
        raise BudgetError("Invalid local dtype/thread configuration")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise BudgetError("Seed must be a 32-bit unsigned integer")
    if max_new_records is not None and (type(max_new_records) is not int or not 1 <= max_new_records <= 1600):
        raise BudgetError("max-new-records must be an integer in [1, 1600]")
    rows = read_prompts(prompts, split, max_records)
    pricing_obj = rate = None
    if provider == "claude":
        if not pricing or not campaign_ledger:
            raise BudgetError("Claude requires explicit pricing and a shared campaign ledger")
        pricing_obj, rate = read_pricing(pricing, model, max_output_tokens)
        ceiling = token_cost(rate["input_context_tokens"], max_output_tokens,
                             rate["input_usd_per_million"], rate["output_usd_per_million"])
    else:
        if PINNED_LOCAL.get(model) != revision:
            raise BudgetError("Use one of the three registered local models with its exact revision")
        ceiling = 0
    settings = {"max_output_tokens": max_output_tokens, "system": SYSTEM,
                "api_version": smoke.API_VERSION if provider == "claude" else None,
                "request_options": rate.get("request_options", {}) if rate else None,
                "seed": seed if provider == "local" else None,
                "cpu_threads": cpu_threads if provider == "local" else None,
                "temperature": 0.8 if provider == "local" else None,
                "top_p": 0.95 if provider == "local" else None,
                "top_k": 50 if provider == "local" else None}
    definition = {"schema": "synsteg-budget-generation-run-v1", "provider": provider, "model": model,
        "revision": revision, "dtype": dtype if provider == "local" else None,
        "settings": settings, "split": split, "prompts_sha256": file_hash(prompts),
        "pricing_sha256": file_hash(pricing) if pricing else None,
        "selected_records": [record_metadata(row) for row in rows], "selected_records_count": len(rows),
        "runner_sha256": file_hash(__file__), "ledger_code_sha256": file_hash(ROOT / "src/synsteg_audit/budget_ledger.py"),
        "transport_code_sha256": {name: file_hash(ROOT / "scripts" / name) for name in
                                  ("run_claude_smoke.py", "check_claude_access.py")} if provider == "claude" else None,
        "selection": "First N records in supplied frozen prompt order; no outcome/capacity selection",
        "automatic_retries": 0, "billing_balance_verified": False,
        "reservation_per_attempt_usd": ceiling / UNIT, "budget_usd": money(budget_usd) / UNIT}
    if split != "development":
        if frozen_manifest is None:
            raise BudgetError("Nondevelopment execution requires an explicit frozen manifest")
        frozen = json.loads(Path(frozen_manifest).read_text(encoding="utf-8"))
        expected = {"provider": provider, "model": model, "revision": revision,
                    "dtype": definition["dtype"], "settings": settings}
        if (frozen.get("schema") != "synsteg-generation-freeze-v1"
                or frozen.get("prompts_sha256") != definition["prompts_sha256"]
                or expected not in frozen.get("generators", [])
                or frozen.get("execution_enabled") is not True):
            raise BudgetError("Frozen manifest does not authorize these prompts and generator settings")
        definition["frozen_manifest_sha256"] = file_hash(frozen_manifest)
    ledger = CampaignLedger(campaign_ledger if provider == "claude" else Path(output) / "local_ledger", budget_usd)
    if not execute:
        return 0, {"status": "preview_only_no_requests", "plan": definition,
                   "provider_billing_limit": False}
    env = os.environ if environ is None else environ
    key = env.get("ANTHROPIC_API_KEY", "") if provider == "claude" else ""
    if provider == "claude":
        if (not isinstance(key, str) or not 20 <= len(key) <= 512
                or any(not 33 <= ord(char) <= 126 for char in key)):
            raise BudgetError("ANTHROPIC_API_KEY is missing or invalid; no requests sent")
        if key in json.dumps(rows) or key in str(Path(output).resolve()) or key in str(Path(campaign_ledger).resolve()):
            raise BudgetError("Credential present in prompt or output path; no files or requests created")
    output = Path(output)
    sender = smoke.send_once if sender is None else sender
    generator = local_generator or LocalGenerator(model, revision, dtype, cpu_threads, seed)
    preflight_sender = check_model_access if preflight_sender is None else preflight_sender
    with ledger.locked(), CampaignLedger(output / "run_guard").locked():
        output.mkdir(parents=True, exist_ok=True)
        runtime_sha256 = None
        if provider == "local":
            # Prepare/dry-run may precede installation of neural dependencies.
            # Freeze the actual environment only at first execution, separately
            # from run.json, and reject changes before reserving any attempt.
            runtime = (runtime_reader or local_runtime)()
            runtime_file = output / "runtime.json"
            if runtime_file.exists():
                if json.loads(runtime_file.read_text(encoding="utf-8")) != runtime:
                    raise BudgetError("Local runtime differs from first execution; refusing to resume")
            else:
                atomic_json(runtime_file, runtime)
            runtime_sha256 = canonical_hash(runtime)
        run_file = output / "run.json"
        if run_file.exists():
            if json.loads(run_file.read_text(encoding="utf-8")) != definition:
                raise BudgetError("Existing run differs; refusing to overwrite/resume with changed settings")
        else:
            atomic_json(run_file, definition)
        summary = {"schema": "synsteg-budget-generation-summary-v1", "status": "running",
                   "selected_records_count": len(rows), "completed": 0, "resumed_completed": 0,
                   "requests_attempted_this_invocation": 0, "automatic_retries": 0,
                   "attempts": [], "provider_billing_limit": False}
        if max_new_records is not None:
            summary["max_new_records_this_invocation"] = max_new_records
        code = 0
        access_checked = False
        for row in rows:
            identity = canonical_hash([provider, model, row["record_id"]])
            meta = record_metadata(row)
            fingerprint = canonical_hash({"record": meta, "provider": provider, "model": model,
                "revision": revision, "dtype": definition["dtype"], "settings": settings,
                "pricing_sha256": definition["pricing_sha256"], "prompts_sha256": definition["prompts_sha256"],
                "runtime_sha256": runtime_sha256,
                "implementation": {name: definition[name] for name in
                    ("runner_sha256", "ledger_code_sha256", "transport_code_sha256")}})
            existing = ledger.read()["attempts"].get(identity)
            # This is an operational checkpoint, not a change to the frozen
            # denominator/settings. Existing failed/unknown attempts still go
            # through the usual no-retry stop below, even after a full batch.
            if (existing is None and max_new_records is not None
                    and summary["requests_attempted_this_invocation"] >= max_new_records):
                summary["status"] = "batch_completed"
                break
            # Check access only if a new generation would be needed. Completed
            # resumptions do not contact the provider or depend on today's access.
            if provider == "claude" and not access_checked and existing is None:
                access = preflight_sender(model, key)
                atomic_json(output / "model_access.json", access)
                if access.get("status") != "model_access_verified_no_generation":
                    summary.update(status="model_access_unverified_no_generation")
                    code = 4
                    break
                access_checked = True
            try:
                entry, created = ledger.reserve(identity, fingerprint, ceiling,
                    {"record": meta, "provider": provider, "model": model})
            except BudgetError as exc:
                summary.update(status="stopped_before_request", detail=str(exc))
                code = 3
                break
            attempt_file = output / "attempts" / (identity + ".json")
            attempt = {"schema": "synsteg-budget-generation-attempt-v1", "identity": identity,
                       "record": meta, "provider": provider, "model": model,
                       "status": entry["status"], "request_fingerprint": fingerprint,
                       "result": entry.get("result"), "automatic_retries": 0}
            if attempt_file.exists():
                previous = json.loads(attempt_file.read_text(encoding="utf-8"))
                if any(previous.get(field) != attempt[field] for field in
                       ("schema", "identity", "record", "provider", "model", "request_fingerprint")):
                    raise BudgetError("Existing attempt metadata differs from its ledger identity")
                # Resume must not discard the retained provider response.
                attempt = {**previous, **attempt}
            atomic_json(attempt_file, attempt)
            if not created:
                summary["attempts"].append({"identity": identity, "record_id": row["record_id"], "status": entry["status"]})
                if entry["status"] == "completed":
                    summary["completed"] += 1
                    summary["resumed_completed"] += 1
                    continue
                summary.update(status="stopped_existing_unknown_or_failed_no_retry")
                code = 3
                break
            summary["requests_attempted_this_invocation"] += 1
            atomic_json(output / "summary.json", summary)
            result = None
            try:
                if provider == "claude":
                    body = {"model": model, "max_tokens": max_output_tokens, "system": SYSTEM,
                            "messages": [{"role": "user", "content": row["prompt"]}],
                            **rate.get("request_options", {})}
                    reply = sender(body, key)
                    # Persist wire evidence before settlement. The shared ledger
                    # holds a full reservation if persistence is interrupted.
                    attempt["response"] = reply
                    atomic_json(attempt_file, attempt)
                    result, cost = validate_claude(reply, model, rate, max_output_tokens)
                else:
                    result, cost = generator(row, max_output_tokens), 0
            except KeyboardInterrupt:
                summary.update(status="interrupted_unknown_no_retry")
                code = 130
                break
            except Exception as exc:
                # Do not persist arbitrary exception text; it can contain
                # request headers, credentials or user-specific cache paths.
                result = {"error_type": type(exc).__name__, "generation_outcome_known": False}
                if isinstance(exc, BudgetError):
                    result["error_code"] = str(exc)
                entry = ledger.finish(identity, result)
                attempt.update(status=entry["status"], result=result)
                atomic_json(attempt_file, attempt)
                summary["attempts"].append({"identity": identity, "record_id": row["record_id"], "status": entry["status"]})
                summary.update(status="stopped_on_failure_no_retry")
                code = 3
                break
            # Settlement is outside the transport catch: a ledger persistence
            # or bound error must not be followed by a second settlement.
            entry = ledger.finish(identity, result, cost)
            attempt.update(status=entry["status"], result=result)
            atomic_json(attempt_file, attempt)
            summary["completed"] += 1
            summary["attempts"].append({"identity": identity, "record_id": row["record_id"], "status": entry["status"]})
            atomic_json(output / "summary.json", summary)
        else:
            summary["status"] = "selected_records_completed"
        summary.update(finished_at_unix=time.time(), exit_code=code, ledger=ledger.summary())
        atomic_json(output / "summary.json", summary)
        return code, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", required=True, choices=("claude", "local"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-records", type=int, default=30)
    parser.add_argument("--max-new-records", type=int,
                        help="Checkpoint after at most N new attempts; preserves the full selected run plan")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--split", choices=SPLITS, default="development")
    parser.add_argument("--pricing", type=Path)
    parser.add_argument("--campaign-ledger", type=Path)
    parser.add_argument("--budget-usd", default="20")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        code, summary = run(**vars(args))
    except (BudgetError, OSError, ValueError):
        print("Not started or persistence failed; inspect configuration and retain existing ledger/attempts.", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
