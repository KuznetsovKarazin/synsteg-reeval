#!/usr/bin/env python3
"""Prospective, single-opening SynSteg experiment with a frozen local adapter.

This enforces a workflow, not OS-level secrecy from the experiment operator.
The shipped protocol is a template until model and adapter identities are set.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.metrics import delivery_metrics

SPLITS = ("train", "development", "test")
SCHEMA = "synsteg-prospective-v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def event(run: Path, name: str, **details: Any) -> None:
    # One writer per run, guarded by the CLI. Each attempted sample is also saved
    # in a separate atomic record; the JSONL ledger is never the sole evidence.
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(canonical({"event": name, "time_unix": time.time(), **details}).decode() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def writer_lock(run: Path):
    lock = run / ".writer.lock"
    run.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("Another writer or a stale .writer.lock exists; inspect it before resuming") from error
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def environment() -> dict:
    versions = {}
    for package in ("numpy", "scikit-learn", "scipy", "joblib", "torch", "transformers",
                    "nltk", "tokenizers", "huggingface-hub", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "packages": versions}


def safe_repo_file(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"A frozen code file must exist inside the repository: {relative}")
    return path


def validate_protocol(protocol: dict, *, executable: bool = False) -> None:
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Unsupported protocol schema")
    if protocol.get("paid_api_allowed") is not False:
        raise ValueError("This runner permits only local execution, with no paid APIs")
    variants = protocol.get("variants", [])
    if len(variants) < 2 or len(set(variants)) != len(variants):
        raise ValueError("At least two distinct, preregistered variants are required")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", variant) for variant in variants):
        raise ValueError("Variant IDs must be safe lowercase identifiers")
    payloads = protocol.get("payload_bits", [])
    if not payloads or any(type(n) is not int or n <= 0 for n in payloads) or len(set(payloads)) != len(payloads):
        raise ValueError("payload_bits must contain distinct positive integers")
    if type(protocol.get("repetitions")) is not int or protocol["repetitions"] < 1:
        raise ValueError("repetitions must be a positive integer")
    if protocol.get("maximum_attempts") != 1:
        raise ValueError("v1 fixes one attempt; failures are counted, never silently replaced")
    genres = protocol.get("genres", [])
    if not genres or len({g["id"] for g in genres}) != len(genres):
        raise ValueError("Distinct genres are required")
    topic_ids, topic_texts, prompt_ids, prompt_texts = set(), set(), set(), set()
    for split in SPLITS:
        topics = protocol["topics"][split]
        if not topics:
            raise ValueError("Every split must have topics")
        for topic in topics:
            normalized = " ".join(topic["text"].casefold().split())
            if topic["id"] in topic_ids or normalized in topic_texts:
                raise ValueError("Topic families or topic text overlap across the design")
            topic_ids.add(topic["id"])
            topic_texts.add(normalized)
        for genre in genres:
            prompt = protocol["prompts"][split][genre["id"]]
            normalized = " ".join(prompt["template"].casefold().split())
            if prompt["family_id"] in prompt_ids or normalized in prompt_texts:
                raise ValueError("Prompt families overlap across the design")
            if "{topic}" not in prompt["template"]:
                raise ValueError("Every prompt requires the {topic} placeholder")
            prompt_ids.add(prompt["family_id"])
            prompt_texts.add(normalized)
    if not 0 < protocol["attack"]["target_fpr"] < 1:
        raise ValueError("target_fpr must lie between zero and one")
    if not 0 < protocol["duplicate_policy"]["bigram_jaccard"] <= 1:
        raise ValueError("Invalid near-duplicate threshold")
    if not protocol["attack"]["features"] or set(protocol["attack"]["features"]) - {"word", "char"}:
        raise ValueError("Only word and char attack features are implemented")
    if type(protocol["statistics"]["bootstrap_resamples"]) is not int or protocol["statistics"]["bootstrap_resamples"] < 1:
        raise ValueError("At least one bootstrap resample is required")
    if executable:
        if protocol.get("execution_enabled") is not True:
            raise ValueError("Protocol execution is disabled")
        if protocol.get("adapter_factory") is None:
            raise ValueError("Configure an adapter_factory")
        safe_repo_file(protocol["adapter_file"])
        model = protocol["model"]
        if model.get("backend") != "local_hf":
            if not (model.get("backend") == "fixture" and protocol.get("fixture_only") is True):
                raise ValueError("Only local_hf or explicitly labeled test fixtures are supported")
        else:
            for name in ("model_revision", "tokenizer_revision"):
                if not re.fullmatch(r"[0-9a-f]{40}", model.get(name, "")):
                    raise ValueError(f"An immutable 40-character {name} is required")
            if not model.get("model_id") or not model.get("tokenizer_id"):
                raise ValueError("Model and tokenizer IDs are required")


def make_manifest(protocol: dict) -> dict[str, list[dict]]:
    validate_protocol(protocol)
    result = {split: [] for split in SPLITS}
    for split in SPLITS:
        for topic in protocol["topics"][split]:
            for genre in protocol["genres"]:
                template = protocol["prompts"][split][genre["id"]]
                for bits in protocol["payload_bits"]:
                    for repetition in range(protocol["repetitions"]):
                        identifier = f"{split}-{topic['id']}-{genre['id']}-{bits}-{repetition}"
                        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
                            raise ValueError("Unsafe sample ID")
                        prompt = template["template"].format(topic=topic["text"])
                        seed = int(digest([protocol["generation_seed"], identifier])[:8], 16)
                        result[split].append({"id": identifier, "document_family_id": identifier,
                            "split": split, "topic_id": topic["id"], "topic": topic["text"],
                            "genre_id": genre["id"], "prompt_family_id": template["family_id"],
                            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "payload_bits": bits, "repetition": repetition, "generation_seed": seed})
    return result


def prepare(protocol_path: Path, run: Path) -> dict:
    if (run / "protocol.json").exists():
        raise ValueError("Run already prepared; use a new run directory")
    protocol = read_json(protocol_path)
    manifest = make_manifest(protocol)
    atomic_json(run / "protocol.json", protocol)
    for split, samples in manifest.items():
        atomic_json(run / "manifests" / f"{split}.json", samples)
    event(run, "prepared", counts={s: len(x) for s, x in manifest.items()})
    return {"status": "prepared_not_frozen", "counts": {s: len(x) for s, x in manifest.items()}}


def code_snapshot(protocol: dict) -> dict[str, str]:
    paths = {Path(__file__).resolve(), safe_repo_file(protocol["adapter_file"])}
    paths.update((ROOT / "src").rglob("*.py"))
    paths.update(safe_repo_file(p) for p in protocol.get("additional_code_files", []))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def resource_snapshot(protocol: dict) -> dict[str, str]:
    """Verify explicitly configured resources such as the local WordNet ZIP.

    Large HF weights are identified by immutable model revisions separately.
    The adapter must not introduce undeclared lexical resources or downloads.
    """
    result = {}
    for resource in protocol.get("external_resource_files", []):
        path = Path(resource["path"]).expanduser().resolve()
        expected = resource["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Every external resource requires a full SHA256")
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"External resource is missing or changed: {path}")
        if str(path) in result:
            raise ValueError("Duplicate external resource")
        result[str(path)] = expected
    return result


def freeze(run: Path) -> dict:
    if (run / "freeze.json").exists():
        raise ValueError("Run is already frozen")
    if (run / "records").exists():
        raise ValueError("Cannot freeze after generation has begun")
    protocol = read_json(run / "protocol.json")
    validate_protocol(protocol, executable=True)
    expected = make_manifest(protocol)
    for split in SPLITS:
        if read_json(run / "manifests" / f"{split}.json") != expected[split]:
            raise ValueError("Manifest does not match the prespecified design")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    result = {"schema": SCHEMA, "protocol_sha256": file_hash(run / "protocol.json"),
              "manifest_sha256": {s: file_hash(run / "manifests" / f"{s}.json") for s in SPLITS},
              "code_sha256": code_snapshot(protocol), "git_commit_context": commit,
              "external_resource_sha256": resource_snapshot(protocol),
              "environment": environment(), "frozen_at_unix": time.time(),
              "claim_scope": protocol["claim_scope"],
              "note": "File hashes are authoritative even when code was frozen before a Git commit."}
    result["freeze_id"] = digest(result)
    atomic_json(run / "freeze.json", result)
    event(run, "frozen", freeze_id=result["freeze_id"])
    return result


def verify_frozen(run: Path) -> tuple[dict, dict]:
    frozen = read_json(run / "freeze.json")
    signed = {k: v for k, v in frozen.items() if k != "freeze_id"}
    if digest(signed) != frozen["freeze_id"]:
        raise ValueError("Freeze record has changed")
    protocol = read_json(run / "protocol.json")
    if file_hash(run / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Frozen protocol has changed; start a new experiment")
    if code_snapshot(protocol) != frozen["code_sha256"]:
        raise ValueError("Frozen code has changed; start a new experiment")
    if resource_snapshot(protocol) != frozen["external_resource_sha256"]:
        raise ValueError("Frozen external resources have changed")
    if environment() != frozen["environment"]:
        raise ValueError("Frozen environment has changed")
    for split in SPLITS:
        if file_hash(run / "manifests" / f"{split}.json") != frozen["manifest_sha256"][split]:
            raise ValueError("Frozen sample manifest has changed")
    return protocol, frozen


def load_adapter(protocol: dict):
    path = safe_repo_file(protocol["adapter_file"])
    spec = importlib.util.spec_from_file_location("synsteg_prospective_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, protocol["adapter_factory"])()


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def near_duplicate(left: str, right: str, threshold: float) -> bool:
    a, b = normalized(left), normalized(right)
    if a == b:
        return True
    # At least eight lexical bigrams avoid declaring short incidental phrases
    # duplicates. Exact normalized equality is always rejected.
    def bigrams(text):
        words = text.split()
        return set(zip(words[:-1], words[1:]))
    x, y = bigrams(a), bigrams(b)
    return len(x) >= 8 and len(y) >= 8 and len(x & y) / len(x | y) >= threshold


def read_records(run: Path, split: str, *, require_complete: bool = True) -> list[dict]:
    samples = read_json(run / "manifests" / f"{split}.json")
    expected = {sample["id"] for sample in samples}
    directory = run / "records" / split
    paths = list(directory.glob("*.json")) if directory.exists() else []
    if {p.stem for p in paths} - expected:
        raise ValueError("Unexpected sample records")
    if require_complete and len(paths) != len(samples):
        raise ValueError(f"Split {split} is incomplete")
    values = {path.stem: read_json(path) for path in paths}
    for sample in samples:
        record = values.get(sample["id"])
        if record is not None and record.get("sample") != sample:
            raise ValueError("Sample record identity mismatch")
    return [values[s["id"]] for s in samples if s["id"] in values]


def payload_for(protocol: dict, sample: dict) -> str:
    rng = random.Random(digest([protocol["payload_seed"], sample["id"]]))
    return "".join(str(rng.randrange(2)) for _ in range(sample["payload_bits"]))


def generate(run: Path, split: str, *, adapter=None, limit: int | None = None) -> dict:
    protocol, frozen = verify_frozen(run)
    if split not in SPLITS:
        raise ValueError("Unknown split")
    if (run / "test_opened.json").exists() or (run / "evaluation.json").exists():
        raise ValueError("Opened test cannot be regenerated")
    if split == "test":
        verify_attack_lock(run, protocol, frozen)
    elif (run / "attack_lock.json").exists():
        raise ValueError("Train/development records cannot change after attack lock")
    adapter = adapter or load_adapter(protocol)
    samples = read_json(run / "manifests" / f"{split}.json")
    # These texts are used only by the fixed collision-quarantine rule, never
    # to tune generation settings or an attack on test data.
    previous = []
    for other in SPLITS:
        if other == "test" and split != "test":
            continue
        previous += read_records(run, other, require_complete=False)
    prior_covers = [(r["sample"]["id"], r["cover_text"]) for r in previous if r.get("cover_text")]
    generated = 0
    for sample in samples:
        path = run / "records" / split / f"{sample['id']}.json"
        if path.exists():
            continue
        if limit is not None and generated >= limit:
            break
        # Starting events survive a process failure; only a completed record is
        # counted as a scientific attempt. Interrupted samples are explicitly
        # recorded as interrupted on resume, never silently regenerated.
        started = run / "attempt_started" / split / f"{sample['id']}.json"
        if started.exists():
            record = {"sample": sample, "freeze_id": frozen["freeze_id"], "attempt": 1,
                      "status": "failed", "failure": "interrupted_attempt_not_retried",
                      "cover_text": None, "variants": {}, "started_at": read_json(started)["time_unix"],
                      "finished_at": time.time()}
            atomic_json(path, record)
            event(run, "attempt_interrupted", sample_id=sample["id"], split=split)
            generated += 1
            continue
        atomic_json(started, {"time_unix": time.time(), "freeze_id": frozen["freeze_id"]})
        event(run, "attempt_started", sample_id=sample["id"], split=split)
        record = {"sample": sample, "freeze_id": frozen["freeze_id"], "attempt": 1,
                  "started_at": time.time(), "status": "failed", "cover_text": None,
                  "variants": {}, "failure": None}
        try:
            settings = copy.deepcopy(protocol["generation"])
            settings["model"] = copy.deepcopy(protocol["model"])
            generation = adapter.generate_cover(copy.deepcopy(sample), settings)
            cover = generation["text"]
            if not isinstance(cover, str) or not cover.strip():
                raise ValueError("Generator returned an empty cover")
            record["cover_text"] = cover
            record["generation_metadata"] = generation.get("metadata", {})
            collision = next((sid for sid, old in prior_covers if near_duplicate(
                cover, old, protocol["duplicate_policy"]["bigram_jaccard"])), None)
            if collision is not None:
                raise ValueError(f"quarantined_duplicate_of:{collision}")
            prior_covers.append((sample["id"], cover))
            prepared = adapter.prepare_methods(cover, copy.deepcopy(sample),
                                               copy.deepcopy(protocol["method_config"]))
            if set(prepared) != set(protocol["variants"]):
                raise ValueError("Adapter returned unexpected/missing variants")
            before = {variant: digest(value) for variant, value in prepared.items()}
            record["prepared_method_sha256"] = before
            record["prepared_methods"] = copy.deepcopy(prepared)
            # Persist preparation event before even constructing synthetic bits.
            event(run, "methods_prepared_before_payload", sample_id=sample["id"], sha256=before)
            payload = payload_for(protocol, sample)
            record["payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
            record["synthetic_payload"] = payload
            for variant in protocol["variants"]:
                try:
                    state = copy.deepcopy(prepared[variant])
                    output = adapter.encode_decode(variant, state, payload, cover,
                        copy.deepcopy(sample), copy.deepcopy(protocol["method_config"]))
                    if digest(state) != before[variant]:
                        raise ValueError("Prepared codebook/state changed after payload")
                    stego = output["stego_text"]
                    decoded = output["decoded_payload"]
                    metrics = delivery_metrics(payload, decoded, used_bits=output["embedded_bits"],
                        cover=cover, stego=stego, key=output.get("key"))
                    record["variants"][variant] = {"stego_text": stego, "metrics": metrics,
                        "metadata": output.get("metadata", {}), "error": None,
                        "decoded_payload": decoded, "key": output.get("key"),
                        "prepared_state_sha256": before[variant]}
                except Exception as error:
                    record["variants"][variant] = {"stego_text": None, "metrics": None,
                        "error": f"{type(error).__name__}: {error}",
                        "prepared_state_sha256": before[variant]}
            record["status"] = "completed"
        except Exception as error:
            record["failure"] = f"{type(error).__name__}: {error}"
        record["finished_at"] = time.time()
        atomic_json(path, record)
        event(run, "attempt_completed", sample_id=sample["id"], split=split,
              status=record["status"], failure=record["failure"])
        generated += 1
    records = read_records(run, split, require_complete=False)
    report = {"split": split, "records": len(records), "expected": len(samples),
              "complete": len(records) == len(samples), "new_attempts": generated}
    if report["complete"]:
        atomic_json(run / f"{split}_generation.json", {**report, "records_sha256": digest(records)})
    return report


def common_success(records: list[dict], variants: list[str]) -> list[dict]:
    return [record for record in records if all(
        record.get("variants", {}).get(v, {}).get("metrics") is not None
        and record["variants"][v]["metrics"]["full_message_success"] for v in variants)]


def paired_rows(records: list[dict], variant: str) -> tuple[list[str], list[int], list[dict]]:
    texts, labels, identity = [], [], []
    for record in records:
        for arm, label, text in (("cover", 0, record["cover_text"]),
                                 ("stego", 1, record["variants"][variant]["stego_text"])):
            texts.append(text)
            labels.append(label)
            identity.append({"sample_id": record["sample"]["id"],
                             "topic_id": record["sample"]["topic_id"], "arm": arm})
    return texts, labels, identity


def conservative_fpr_threshold(negative_scores, target_fpr: float) -> float:
    import numpy as np
    values = np.sort(np.asarray(negative_scores, dtype=float))[::-1]
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Invalid development-negative scores")
    allowed = math.floor(target_fpr * len(values))
    return float(np.nextafter(values[allowed], math.inf))


def data_hashes(run: Path, splits: tuple[str, ...]) -> dict[str, str]:
    return {split: digest(read_records(run, split)) for split in splits}


def train_attacks(run: Path) -> dict:
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.pipeline import Pipeline
    protocol, frozen = verify_frozen(run)
    if (run / "attack_lock.json").exists() or (run / "records" / "test").exists():
        raise ValueError("Attacks already locked or test generation already exists")
    raw = {s: read_records(run, s) for s in ("train", "development")}
    cohort = {s: common_success(r, protocol["variants"]) for s, r in raw.items()}
    for split in cohort:
        if len(cohort[split]) < protocol["attack"]["minimum_common_success_pairs"]:
            raise ValueError(f"Insufficient common-success pairs in {split}; do not open test")
    attacks = {}
    for variant in protocol["variants"]:
        tx, ty, _ = paired_rows(cohort["train"], variant)
        dx, dy, _ = paired_rows(cohort["development"], variant)
        for feature in protocol["attack"]["features"]:
            if feature == "word":
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                            lowercase=True, sublinear_tf=True)
            elif feature == "char":
                vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                            min_df=1, lowercase=True, sublinear_tf=True)
            else:
                raise ValueError("Unregistered feature family")
            model = Pipeline([("tfidf", vectorizer), ("lr", LogisticRegression(
                C=protocol["attack"]["C"], solver="lbfgs", max_iter=2000, random_state=0))])
            model.fit(tx, ty)
            raw_scores = model.predict_proba(dx)[:, 1]
            raw_auc = roc_auc_score(dy, raw_scores)
            direction = 1 if raw_auc >= 0.5 else -1
            scores = raw_scores if direction == 1 else 1 - raw_scores
            threshold = conservative_fpr_threshold(scores[np.asarray(dy) == 0], protocol["attack"]["target_fpr"])
            identifier = f"{variant}__{feature}"
            path = run / "attacks" / f"{identifier}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
            attacks[identifier] = {"variant": variant, "features": feature, "score_direction": direction,
                "threshold": threshold, "decision_threshold": 0.5, "artifact": str(path.relative_to(run)),
                "artifact_sha256": file_hash(path), "development_accuracy": float(accuracy_score(dy, scores >= .5)),
                "development_auc": float(roc_auc_score(dy, scores)), "fit_pairs": len(cohort["train"]),
                "development_pairs": len(cohort["development"])}
    lock = {"freeze_id": frozen["freeze_id"], "train_development_sha256": data_hashes(run, ("train", "development")),
            "attacks": attacks, "cohort_rule": "common_full_success_across_all_prespecified_variants",
            "created_at_unix": time.time()}
    lock["lock_id"] = digest(lock)
    atomic_json(run / "attack_lock.json", lock)
    event(run, "attacks_locked_before_test_generation", lock_id=lock["lock_id"])
    return lock


def verify_attack_lock(run: Path, protocol: dict, frozen: dict) -> dict:
    if not (run / "attack_lock.json").is_file():
        raise ValueError("Attacks must be trained and locked before test generation or evaluation")
    lock = read_json(run / "attack_lock.json")
    if digest({k: v for k, v in lock.items() if k != "lock_id"}) != lock["lock_id"]:
        raise ValueError("Attack lock changed")
    if lock["freeze_id"] != frozen["freeze_id"]:
        raise ValueError("Attacks belong to another frozen experiment")
    if lock["train_development_sha256"] != data_hashes(run, ("train", "development")):
        raise ValueError("Training/development data changed after attack lock")
    for attack in lock["attacks"].values():
        if file_hash(run / attack["artifact"]) != attack["artifact_sha256"]:
            raise ValueError("Attack model artifact changed after lock")
    return lock


def metrics_from_scores(labels, scores, fpr_threshold: float) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(labels), np.asarray(scores)
    if not np.isfinite(p).all() or set(y) != {0, 1}:
        raise ValueError("Invalid paired scores")
    return {"accuracy": float(np.mean((p >= .5) == y)), "auc": float(roc_auc_score(y, p)),
            "fpr_at_development_threshold": float(np.mean(p[y == 0] >= fpr_threshold)),
            "tpr_at_development_threshold": float(np.mean(p[y == 1] >= fpr_threshold))}


def cluster_intervals(labels, scores, identities: list[dict], threshold: float, *, repetitions: int, seed: int) -> dict:
    import numpy as np
    topics = sorted({row["topic_id"] for row in identities})
    if len(topics) < 2:
        return {"status": "unavailable_less_than_two_topic_clusters", "clusters": len(topics)}
    blocks = {topic: [i for i, row in enumerate(identities) if row["topic_id"] == topic] for topic in topics}
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        indices = [i for topic in rng.choices(topics, k=len(topics)) for i in blocks[topic]]
        values.append(metrics_from_scores(np.asarray(labels)[indices], np.asarray(scores)[indices], threshold))
    return {"status": "computed", "method": "topic_cluster_percentile_bootstrap",
            "clusters": len(topics), "resamples": repetitions,
            "caution": "Very few topic clusters yield unstable intervals; not a precision guarantee.",
            "intervals_95": {name: [float(x) for x in np.quantile([v[name] for v in values], [.025, .975])]
                             for name in values[0]}}


def reliability(records: list[dict], variants: list[str]) -> dict:
    result = {}
    for variant in variants:
        output = [r.get("variants", {}).get(variant, {}) for r in records]
        success = [v for v in output if v.get("metrics") and v["metrics"]["full_message_success"]]
        result[variant] = {"attempted_samples": len(records), "full_success_samples": len(success),
            "full_success_rate": len(success) / len(records) if records else None,
            "failed_samples": len(records) - len(success),
            "generation_or_preparation_failures": sum(bool(r.get("failure")) for r in records),
            "capacity_or_encoding_or_decode_failures": sum(bool(v.get("error")) for v in output),
            "successful_payload_bits": sum(v["metrics"]["requested_bits"] for v in success),
            "attempted_payload_bits": sum(r["sample"]["payload_bits"] for r in records),
            "mean_successful_bpw_stego": (sum(v["metrics"]["used_bpw_stego"] for v in success) / len(success)
                                          if success else None)}
    return result


def evaluate(run: Path) -> dict:
    import joblib
    protocol, frozen = verify_frozen(run)
    lock = verify_attack_lock(run, protocol, frozen)
    if (run / "evaluation.json").exists():
        raise ValueError("Test already evaluated once; read evaluation.json, do not refit")
    generation_path = run / "test_generation.json"
    if not generation_path.is_file() or not read_json(generation_path).get("complete"):
        raise ValueError("Complete test generation is required before opening")
    generation = read_json(generation_path)
    # The opening marker precedes reading test text. Resume is possible with the
    # exact same frozen code, records and attack artifacts, and is logged.
    marker = run / "test_opened.json"
    if marker.exists():
        opening = read_json(marker)
        if opening["freeze_id"] != frozen["freeze_id"] or opening["lock_id"] != lock["lock_id"]:
            raise ValueError("Cannot resume evaluation with changed identity")
        if opening["test_records_sha256"] != generation["records_sha256"]:
            raise ValueError("Cannot resume with changed test generation")
        event(run, "test_evaluation_resumed", lock_id=lock["lock_id"])
    else:
        atomic_json(marker, {"freeze_id": frozen["freeze_id"], "lock_id": lock["lock_id"],
                             "test_records_sha256": generation["records_sha256"],
                             "opened_at_unix": time.time()})
        event(run, "test_opened_once", lock_id=lock["lock_id"])
    records = read_records(run, "test")
    if digest(records) != generation["records_sha256"]:
        raise ValueError("Test records changed after generation completion")
    cohort = common_success(records, protocol["variants"])
    report = {"freeze_id": frozen["freeze_id"], "lock_id": lock["lock_id"],
        "claim_scope": protocol["claim_scope"], "fixture_only": protocol.get("fixture_only", False),
        "test_records_sha256": digest(records), "reliability": reliability(records, protocol["variants"]),
        "reliability_by_payload_bits": {str(bits): reliability(
            [r for r in records if r["sample"]["payload_bits"] == bits], protocol["variants"])
            for bits in protocol["payload_bits"]},
        "attack_cohort": {"rule": "common_full_success_across_all_prespecified_variants",
                          "pairs": len(cohort), "attempted_pairs": len(records)},
        "attacks": {}, "limitations": [
            "Conditional steganalysis excludes failed transmissions; reliability uses all attempted samples.",
            "This workflow prevents accidental stage mixing but cannot hide files from the operator.",
            "The default pilot does not establish superiority or publication-ready generalization.",
            "Semantic separation of topic families requires prospective review; unique IDs alone do not prove it."]}
    for identifier, attack in lock["attacks"].items():
        if not cohort:
            report["attacks"][identifier] = {"status": "no_common_success_test_pairs"}
            continue
        texts, labels, identities = paired_rows(cohort, attack["variant"])
        model = joblib.load(run / attack["artifact"])
        scores = model.predict_proba(texts)[:, 1]
        if attack["score_direction"] == -1:
            scores = 1 - scores
        predictions = [{**identity, "label": int(y), "score": float(p),
                        "prediction": int(p >= .5), "prediction_at_dev_fpr_threshold": int(p >= attack["threshold"])}
                       for identity, y, p in zip(identities, labels, scores)]
        atomic_json(run / "test_predictions" / f"{identifier}.json", predictions)
        report["attacks"][identifier] = {"status": "complete", **metrics_from_scores(labels, scores, attack["threshold"]),
            "development_threshold": attack["threshold"], "score_direction_fixed_on_development": attack["score_direction"],
            "uncertainty": cluster_intervals(labels, scores, identities, attack["threshold"],
                repetitions=protocol["statistics"]["bootstrap_resamples"], seed=protocol["statistics"]["bootstrap_seed"])}
    atomic_json(run / "evaluation.json", report)
    event(run, "test_evaluation_completed", evaluation_sha256=file_hash(run / "evaluation.json"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze", "generate", "train", "evaluate", "verify"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--limit", type=int, help="Stop after N new attempts; does not change the frozen design")
    args = parser.parse_args()
    run = args.run.resolve()
    with writer_lock(run):
        if args.command == "prepare":
            if args.protocol is None:
                parser.error("prepare needs --protocol")
            result = prepare(args.protocol, run)
        elif args.command == "freeze":
            result = freeze(run)
        elif args.command == "generate":
            if args.split is None:
                parser.error("generate needs --split")
            if args.limit is not None and args.limit <= 0:
                parser.error("--limit must be positive")
            result = generate(run, args.split, limit=args.limit)
        elif args.command == "train":
            result = train_attacks(run)
        elif args.command == "evaluate":
            result = evaluate(run)
        else:
            protocol, frozen = verify_frozen(run)
            result = {"verified": True, "freeze_id": frozen["freeze_id"], "claim_scope": protocol["claim_scope"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
