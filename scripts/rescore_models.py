#!/usr/bin/env python3
"""Optional, resumable neural scoring. No neural dependencies on import.

This runner implements causal perplexity, supervised detectors D1–D3, corrected
Qwen-pair Binoculars (D4), and analytic Fast-DetectGPT (D5). Implemented inference
is not evidence of a completed corpus run. See docs/neural_runs.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

MODELS = {
    "D1": ("openai-community/roberta-base-openai-detector", {"0": "Fake", "1": "Real"}, 0, 512),
    "D2": ("Hello-SimpleAI/chatgpt-detector-roberta", {"0": "Human", "1": "ChatGPT"}, 1, 512),
    "D3": ("desklib/ai-text-detector-v1.01", None, None, 768),
}


def digest(value):
    """Stable JSON identity, with non-finite numbers forbidden."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_job(job):
    job = dict(job)
    kind = job.get("kind")
    if kind not in {"ppl", "D4", "D5", *MODELS}:
        raise ValueError("kind must be ppl or D1–D5")
    for key in ("model_id", "model_revision", "tokenizer_id", "tokenizer_revision"):
        if not isinstance(job.get(key), str) or not job[key]:
            raise ValueError(f"nonempty {key} is required")
    for key in ("model_revision", "tokenizer_revision"):
        if not re.fullmatch(r"[a-f0-9]{40}", job[key]):
            raise ValueError(f"{key} must be a full 40-character commit SHA, not a branch/tag")
    if kind in MODELS:
        name, labels, ai_index, maximum = MODELS[kind]
        if job["model_id"] != name or job["tokenizer_id"] != name:
            raise ValueError(f"{kind} requires model and tokenizer {name}")
        if job.get("max_length", maximum) != maximum:
            raise ValueError(f"{kind} max_length is fixed to {maximum}; use a separate named protocol for an ablation")
        job.update(max_length=maximum, ai_label_index=ai_index, expected_id2label=labels)
    elif kind in {"D4", "D5"}:
        if job["model_id"] != "Qwen/Qwen2.5-1.5B" or job["tokenizer_id"] != job["model_id"]:
            raise ValueError(f"{kind} requires Qwen/Qwen2.5-1.5B; another backbone is a separate detector")
        if job.get("max_length", 512) != 512:
            raise ValueError(f"{kind} archived comparison requires max_length=512")
        job.update(max_length=512, add_special_tokens=False)
        if kind == "D4":
            for key in ("performer_model_id", "performer_model_revision", "performer_tokenizer_id", "performer_tokenizer_revision"):
                if not isinstance(job.get(key), str) or not job[key]:
                    raise ValueError(f"nonempty {key} is required for D4")
            for key in ("performer_model_revision", "performer_tokenizer_revision"):
                if not re.fullmatch(r"[a-f0-9]{40}", job[key]):
                    raise ValueError(f"{key} must be a full 40-character commit SHA")
            if job["performer_model_id"] != "Qwen/Qwen2.5-1.5B-Instruct" or job["performer_tokenizer_id"] != job["performer_model_id"]:
                raise ValueError("D4 performer must be Qwen/Qwen2.5-1.5B-Instruct")
            if job.get("cross_entropy_positions", "all") != "all":
                raise ValueError("D4 reference protocol includes all input positions")
            job["cross_entropy_positions"] = "all"
            if job.setdefault("denominator_mask", "attention") not in {"attention", "token_id"}:
                raise ValueError("denominator_mask must be attention or token_id")
            job["auxiliary_outputs"] = ["observer_prefix_ppl", "performer_prefix_ppl", "observer_fast_detectgpt"]
        else:
            job["auxiliary_outputs"] = ["scoring_prefix_ppl"]
    else:
        policy = job.setdefault("ppl_policy", "sliding_window")
        if policy not in {"sliding_window", "prefix"}:
            raise ValueError("ppl_policy must be sliding_window or prefix")
        window = job.setdefault("context_length", 1024)
        stride = job.setdefault("stride", 512)
        if not isinstance(window, int) or window < 2 or not isinstance(stride, int) or not 1 <= stride < window:
            raise ValueError("require context_length >= 2 and 1 <= stride < context_length")
        if policy == "prefix":
            limit = job.setdefault("prefix_tokens", 512)
            if not isinstance(limit, int) or not 2 <= limit <= window:
                raise ValueError("prefix_tokens must be between 2 and context_length")
    dtype = job.setdefault("dtype", "float32")
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError("dtype must be float32 or explicitly declared bfloat16")
    if dtype == "bfloat16" and (not job.get("precision_protocol") or job["precision_protocol"] == "float32_reference"):
        raise ValueError("bfloat16 requires a separately named precision_protocol")
    job.setdefault("precision_protocol", "float32_reference")
    threads = job.setdefault("cpu_threads", 1)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    job["batch_size"] = 1
    job["trust_remote_code"] = False
    job["implementation_version"] = 2
    return job


def windows(n_tokens, context_length, stride):
    """Yield (begin, end, first_new_target), counting targets 1..N-1 once.

    All indices are global token positions; the input at position zero has no
    preceding context and is never counted as a predicted token.
    """
    if n_tokens < 2 or context_length < 2 or not 1 <= stride < context_length:
        raise ValueError("invalid token count or window/stride")
    previous_end = 1
    for begin in range(0, n_tokens, stride):
        end = min(begin + context_length, n_tokens)
        target_start = max(previous_end, begin + 1)
        if target_start < end:
            yield begin, end, target_start
        previous_end = end
        if end == n_tokens:
            break


def read_corpus(path):
    records = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"corpus line {number}: nonempty text required")
        identity = row.get("id")
        if identity is None and "uuid" in row and "arm" in row:
            # Tuple serialization avoids collisions from delimiters in source IDs.
            identity = json.dumps([row["uuid"], row["arm"], row.get("cand", -1)], separators=(",", ":"))
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"corpus line {number}: unique string id (or uuid/arm/cand) required")
        seen.add(identity)
        records.append({"id": identity, "text": text, "text_sha256": text_hash(text)})
    if not records:
        raise ValueError("empty corpus")
    return records


def cache_key(job_fingerprint, sha):
    return digest({"job_fingerprint": job_fingerprint, "text_sha256": sha})


def read_checkpoint(path, fingerprint, sha):
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("job_fingerprint") != fingerprint or row.get("text_sha256") != sha or row.get("status") != "complete":
        raise ValueError(f"checkpoint identity/status mismatch: {path.name}")
    score = row.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError(f"checkpoint lacks finite score: {path.name}")
    return row


def complete_rows(records, checkpoints, fingerprint):
    """Fail closed: partial inference cannot become a final score table."""
    output = []
    for row in records:
        key = cache_key(fingerprint, row["text_sha256"])
        score = checkpoints.get(key)
        if score is None or score.get("status") != "complete" or score.get("job_fingerprint") != fingerprint or score.get("text_sha256") != row["text_sha256"]:
            raise ValueError(f"incomplete or mismatched inference for {row['id']}")
        output.append({"id": row["id"], "text_sha256": row["text_sha256"], "cache_key": key,
                       "score": score["score"], "details": score.get("details", {})})
    return output


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "huggingface-hub", "safetensors", "sentencepiece", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def causal_detector_statistics(observer_logits, input_ids, attention_mask, performer_logits=None,
                               *, pad_token_id=None, chunk_positions=16):
    """Independent double-precision reductions over bounded vocabulary chunks.

    D5 uses observer as both reference and scoring model. D4 additionally returns
    performer NLL/H(observer, performer), including the last input position.
    This routine receives actual neural logits; it does not generate or infer
    missing values. Chunking changes only reduction grouping, not token scope.
    """
    import torch

    if observer_logits.ndim != 3 or observer_logits.shape[0] != 1:
        raise ValueError("single-example causal logits [1,T,V] required")
    if input_ids.shape != observer_logits.shape[:2] or attention_mask.shape != input_ids.shape:
        raise ValueError("token/logit shape mismatch")
    if performer_logits is not None and performer_logits.shape != observer_logits.shape:
        raise ValueError("aligned observer/performer logits required")
    if chunk_positions < 1 or observer_logits.shape[1] < 2:
        raise ValueError("at least two positions and a positive chunk size required")
    mask = attention_mask.bool()
    adjacent = mask[:, :-1] & mask[:, 1:]
    n_targets = int(adjacent.sum().item())
    if n_targets == 0:
        raise ValueError("no observed next-token transitions")
    observed_ids = input_ids[mask]
    if bool(((observed_ids < 0) | (observed_ids >= observer_logits.shape[-1])).any()):
        raise ValueError("observed token ID outside vocabulary")
    nll_o = nll_p = numerator = variance = cross_sum = 0.0
    cross_count = 0
    length = observer_logits.shape[1]
    for start in range(0, length, chunk_positions):
        end = min(length, start + chunk_positions)
        xo = observer_logits[:, start:end].double()
        if not bool(torch.isfinite(xo[:, mask[0, start:end]]).all()):
            raise ValueError("non-finite observed logits")
        xo = torch.where(mask[:, start:end, None], xo, 0.0)
        lo = torch.log_softmax(xo, dim=-1)
        po = lo.exp()
        if performer_logits is not None:
            xp = performer_logits[:, start:end].double()
            if not bool(torch.isfinite(xp[:, mask[0, start:end]]).all()):
                raise ValueError("non-finite performer logits")
            xp = torch.where(mask[:, start:end, None], xp, 0.0)
            lp = torch.log_softmax(xp, dim=-1)
            cross = -(po * lp).sum(-1)
            cmask = mask[:, start:end].clone()
            if pad_token_id is not None:
                cmask &= input_ids[:, start:end] != pad_token_id
            cross_sum += float(cross[cmask].sum().item())
            cross_count += int(cmask.sum().item())
        n = min(end, length - 1) - start
        if n <= 0:
            continue
        valid = adjacent[:, start:start+n]
        # Pad IDs may be invalid; they are excluded and replaced before gather.
        target = input_ids[:, start+1:start+1+n].clone()
        target[~valid] = 0
        obs = lo[:, :n].gather(-1, target[..., None])[..., 0]
        mu = (po[:, :n] * lo[:, :n]).sum(-1)
        var = (po[:, :n] * (lo[:, :n] - mu[..., None]).square()).sum(-1)
        # Uniform logits have exactly zero mathematical variance; roundoff in
        # summing 1/V weights must not turn that into a usable denominator.
        uniform = lo[:, :n].amax(-1) == lo[:, :n].amin(-1)
        var = torch.where(uniform, 0.0, var)
        nll_o += float((-obs[valid]).sum().item())
        numerator += float((obs - mu)[valid].sum().item())
        variance += float(var[valid].sum().item())
        if performer_logits is not None:
            observed_p = lp[:, :n].gather(-1, target[..., None])[..., 0]
            nll_p += float((-observed_p[valid]).sum().item())
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("sampling discrepancy requires positive finite variance")
    result = {"observer_mean_nll": nll_o / n_targets, "observer_prefix_ppl": math.exp(nll_o / n_targets),
              "observer_fast_detectgpt": numerator / math.sqrt(variance),
              "predicted_tokens": n_targets, "d5_numerator": numerator, "d5_variance_sum": variance}
    if performer_logits is not None:
        if cross_count == 0 or not math.isfinite(cross_sum) or cross_sum <= 0:
            raise ValueError("positive finite cross-entropy required")
        mean_p = nll_p / n_targets
        mean_cross = cross_sum / cross_count
        result.update(performer_mean_nll=mean_p, performer_prefix_ppl=math.exp(mean_p),
                      observer_performer_cross_entropy=mean_cross, cross_entropy_positions=cross_count,
                      binoculars_ratio=mean_p / mean_cross)
    return result


class Scorer:
    def __init__(self, job, device, allow_download):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig, AutoModel

        self.torch, self.job = torch, job
        self.device = torch.device(device)
        self.dtype = getattr(torch, job["dtype"])
        torch.set_num_threads(job["cpu_threads"])
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        kwargs = {"local_files_only": not allow_download, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(job["tokenizer_id"], revision=job["tokenizer_revision"], **kwargs)
        self.tokenizer.truncation_side = "right"
        self.tokenizer.padding_side = "right"
        if job["kind"] in {"ppl", "D4", "D5"}:
            self.model = AutoModelForCausalLM.from_pretrained(job["model_id"], revision=job["model_revision"], torch_dtype=self.dtype, **kwargs)
            limit = getattr(self.model.config, "max_position_embeddings", getattr(self.model.config, "n_positions", None))
            if isinstance(limit, int) and job.get("context_length", job.get("max_length")) > limit:
                raise ValueError(f"requested context exceeds model limit {limit}")
            if job["kind"] == "D4":
                self.performer_tokenizer = AutoTokenizer.from_pretrained(job["performer_tokenizer_id"], revision=job["performer_tokenizer_revision"], **kwargs)
                if self.performer_tokenizer.get_vocab() != self.tokenizer.get_vocab():
                    raise ValueError("observer/performer token-to-ID vocabularies differ")
                self.performer = AutoModelForCausalLM.from_pretrained(job["performer_model_id"], revision=job["performer_model_revision"], torch_dtype=self.dtype, **kwargs)
                if self.performer.config.vocab_size != self.model.config.vocab_size:
                    raise ValueError("observer/performer model vocabulary dimensions differ")
                self.performer = self.performer.to(device=self.device, dtype=self.dtype).eval()
        elif job["kind"] in {"D1", "D2"}:
            self.model = AutoModelForSequenceClassification.from_pretrained(job["model_id"], revision=job["model_revision"], torch_dtype=self.dtype, **kwargs)
            actual = {str(k): v for k, v in self.model.config.id2label.items()}
            if actual != job["expected_id2label"]:
                raise ValueError(f"label mapping differs from audited orientation: {actual}")
        else:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            config = AutoConfig.from_pretrained(job["model_id"], revision=job["model_revision"], **kwargs)

            class DesklibModel(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.model = AutoModel.from_config(config, trust_remote_code=False)
                    self.classifier = torch.nn.Linear(config.hidden_size, 1)

                def forward(self, input_ids, attention_mask):
                    hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
                    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    return self.classifier((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9))

            self.model = DesklibModel(config)
            state_path = hf_hub_download(job["model_id"], "model.safetensors", revision=job["model_revision"], local_files_only=not allow_download)
            missing, unexpected = self.model.load_state_dict(load_file(state_path), strict=False)
            if [k for k in missing if not k.endswith("position_ids")] or unexpected:
                raise ValueError(f"Desklib checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()

    def score(self, text):
        torch, job = self.torch, self.job
        with torch.inference_mode():
            if job["kind"] in {"D4", "D5"}:
                ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
                original_n = ids.shape[1]
                if job["kind"] == "D4":
                    ids_p = self.performer_tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
                    if not torch.equal(ids, ids_p):
                        raise ValueError("observer/performer text tokenization differs")
                ids = ids[:, :job["max_length"]].to(self.device)
                mask = torch.ones_like(ids)
                observer_logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                performer_logits = (self.performer(input_ids=ids, attention_mask=mask, use_cache=False).logits
                                    if job["kind"] == "D4" else None)
                pad_id = self.tokenizer.pad_token_id if job.get("denominator_mask") == "token_id" else None
                stats = causal_detector_statistics(observer_logits, ids, mask, performer_logits, pad_token_id=pad_id)
                if job["kind"] == "D4":
                    value = -stats["binoculars_ratio"]
                else:
                    value = stats["observer_fast_detectgpt"]
                return value, {**stats, "input_tokens": original_n, "scored_input_tokens": int(ids.shape[1]),
                               "truncated": original_n > job["max_length"], "orientation": "higher_is_more_AI",
                               "dtype": job["dtype"], "precision_protocol": job["precision_protocol"],
                               "denominator_mask": job.get("denominator_mask"),
                               "auxiliary_scores_from_same_forward": True}
            if job["kind"] != "ppl":
                original_n = len(self.tokenizer(text, add_special_tokens=True)["input_ids"])
                encoded = self.tokenizer(text, truncation=True, max_length=job["max_length"], return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}
                if job["kind"] == "D3":
                    value = torch.sigmoid(self.model(**encoded)).item()
                else:
                    value = self.model(**encoded).logits.float().softmax(-1)[0, job["ai_label_index"]].item()
                return value, {"input_tokens": original_n, "scored_input_tokens": int(encoded["input_ids"].shape[1]), "truncated": original_n > job["max_length"], "orientation": "higher_is_more_AI"}

            ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
            original_n = ids.shape[1]
            if job["ppl_policy"] == "prefix":
                ids = ids[:, :job["prefix_tokens"]]
            n = ids.shape[1]
            total_nll, counted = 0.0, 0
            for begin, end, first in windows(n, job["context_length"], job["stride"]):
                part = ids[:, begin:end].to(self.device)
                logits = self.model(input_ids=part, attention_mask=torch.ones_like(part), use_cache=False).logits[:, :-1].float()
                targets = part[:, 1:]
                # Global target `first` corresponds to shifted index first-begin-1.
                offset = first - begin - 1
                loss = torch.nn.functional.cross_entropy(logits[:, offset:].reshape(-1, logits.shape[-1]), targets[:, offset:].reshape(-1), reduction="sum")
                total_nll += float(loss.item())
                counted += end - first
            if counted != n - 1:
                raise AssertionError("causal target coverage mismatch")
            mean_nll = total_nll / counted
            value = math.exp(mean_nll)
            return value, {"input_tokens": original_n, "scored_input_tokens": n, "predicted_tokens": counted, "sum_nll": total_nll, "mean_nll": mean_nll, "truncated": n < original_n, "ppl_policy": job["ppl_policy"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="JSONL with id/text or uuid/arm/cand/text")
    parser.add_argument("--job", type=Path, required=True, help="JSON manifest with immutable model/tokenizer revisions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--allow-download", action="store_true", help="Explicitly allow model/tokenizer downloads; default is local cache only")
    parser.add_argument("--cpu-threads", type=int, help="Override job cpu_threads; included in immutable fingerprint")
    parser.add_argument("--plan-only", action="store_true", help="Validate inputs and print work inventory without importing torch or writing scores")
    args = parser.parse_args(argv)
    settings = json.loads(args.job.read_text(encoding="utf-8"))
    if args.cpu_threads is not None:
        settings["cpu_threads"] = args.cpu_threads
    job = validate_job(settings)
    records = read_corpus(args.corpus)
    source_sha = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    runtime = versions()
    identity = {"job": job, "runtime": runtime, "device": args.device,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = digest(identity)
    unique = {r["text_sha256"]: r["text"] for r in records}
    manifest = {"status": "planned", "job_fingerprint": fingerprint, "identity": identity,
                "corpus_sha256": source_sha, "n_records": len(records), "n_unique_texts": len(unique),
                "n_complete_unique_texts": 0, "allow_download": args.allow_download,
                "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv}
    if args.plan_only:
        print(json.dumps(manifest, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("job_fingerprint") != fingerprint or prior.get("corpus_sha256") != source_sha:
            raise ValueError("output belongs to another job/runtime/corpus; use a new output directory")
        manifest["created_utc"] = prior["created_utc"]
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoints = {}
    for sha in unique:
        key = cache_key(fingerprint, sha)
        stored = read_checkpoint(checkpoint_dir / f"{key}.json", fingerprint, sha)
        if stored is not None:
            checkpoints[key] = stored
    manifest.update(status="incomplete", n_complete_unique_texts=len(checkpoints))
    atomic_json(manifest_path, manifest)
    try:
        pending = [(sha, text) for sha, text in unique.items() if cache_key(fingerprint, sha) not in checkpoints]
        scorer = Scorer(job, args.device, args.allow_download) if pending else None
        for sha, text in pending:
            started = time.monotonic()
            value, details = scorer.score(text)
            if not math.isfinite(value):
                raise ValueError(f"non-finite score for text {sha}")
            key = cache_key(fingerprint, sha)
            row = {"status": "complete", "job_fingerprint": fingerprint, "text_sha256": sha,
                   "score": value, "details": details, "elapsed_seconds": time.monotonic() - started}
            atomic_json(checkpoint_dir / f"{key}.json", row)
            checkpoints[key] = row
            manifest["n_complete_unique_texts"] = len(checkpoints)
            atomic_json(manifest_path, manifest)
            print(f"{len(checkpoints)}/{len(unique)} unique texts", flush=True)
        final = complete_rows(records, checkpoints, fingerprint)
        temporary = args.output / "scores.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in final), encoding="utf-8")
        temporary.replace(args.output / "scores.jsonl")
        manifest.update(status="complete", n_complete_unique_texts=len(checkpoints), finished_utc=datetime.now(timezone.utc).isoformat(),
                        scores_sha256=hashlib.sha256((args.output / "scores.jsonl").read_bytes()).hexdigest())
        atomic_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="incomplete", failure_type=type(error).__name__, failure_message=str(error))
        atomic_json(manifest_path, manifest)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
