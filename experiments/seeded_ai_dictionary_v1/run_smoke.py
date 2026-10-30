#!/usr/bin/env python3
"""Exactly two fresh-process, fixed-prompt, public-seed dictionary samples."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PUBLIC_DEMO_SEED = 20261023
MESSAGES = [{"role": "user", "content": (
    "Return only a JSON array containing exactly 4 arrays. Each inner array must "
    "contain exactly 4 distinct single lowercase English words that are synonyms "
    "of each other. No word may appear in more than one array. Use only letters "
    "a-z inside each word. Do not include explanations, Markdown, numbers, or "
    "additional text."
)}]
SETTINGS = {"max_new_tokens": 192, "do_sample": True, "temperature": 0.8,
            "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.0}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def validate(text):
    result = {"json_parse_success": False, "strict_structure_success": False,
              "semantic_synonymy_verified": False, "english_vocabulary_verified": False,
              "checks": {}, "parsed_json": None}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as error:
        result["parse_error"] = str(error)
        return result
    result["json_parse_success"] = True
    result["parsed_json"] = parsed
    checks = result["checks"]
    checks["exactly_four_groups"] = isinstance(parsed, list) and len(parsed) == 4
    checks["all_groups_arrays_of_four"] = (isinstance(parsed, list)
        and all(isinstance(group, list) and len(group) == 4 for group in parsed))
    flat = [word for group in parsed for word in group] if checks["all_groups_arrays_of_four"] else []
    checks["exactly_sixteen_single_lowercase_ascii_words"] = (len(flat) == 16
        and all(isinstance(word, str) and re.fullmatch(r"[a-z]+", word) is not None for word in flat))
    checks["all_sixteen_words_globally_distinct"] = (checks["exactly_sixteen_single_lowercase_ascii_words"]
        and len(set(flat)) == 16)
    result["strict_structure_success"] = all(checks.values())
    return result


def child(run_number):
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    destination = HERE / f"run_{run_number:02d}.json"
    if destination.exists():
        raise RuntimeError("Existing sample must never be overwritten or regenerated")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(PUBLIC_DEMO_SEED)
    np.random.seed(PUBLIC_DEMO_SEED)
    torch.manual_seed(PUBLIC_DEMO_SEED)
    started = time.time()
    load_start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
        local_files_only=True, token=False, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION,
        local_files_only=True, token=False, trust_remote_code=False,
        dtype=torch.bfloat16).eval()
    load_seconds = time.monotonic() - load_start
    prompt = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    torch.manual_seed(PUBLIC_DEMO_SEED)
    generation_start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, **SETTINGS, pad_token_id=tokenizer.eos_token_id)
    generation_seconds = time.monotonic() - generation_start
    input_ids = inputs["input_ids"][0].tolist()
    full_ids = output[0].tolist()
    generated_ids = full_ids[len(input_ids):]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    record = {"schema": "seeded-ai-dictionary-single-sample-v1", "run_number": run_number,
        "public_demo_seed": PUBLIC_DEMO_SEED, "seed_is_secret": False,
        "model_id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16", "device": "cpu",
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "local_files_only": True, "trust_remote_code": False, "authentication_used": False,
        "messages": MESSAGES, "rendered_prompt": prompt, "generation_settings": SETTINGS,
        "pad_token_id": tokenizer.eos_token_id, "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "model_generation_config": model.generation_config.to_dict(),
        "input_token_ids": input_ids, "generated_token_ids": generated_ids, "full_token_ids": full_ids,
        "generated_text": text,
        "generated_text_with_special_tokens": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "full_sequence_text_with_special_tokens": tokenizer.decode(full_ids, skip_special_tokens=False),
        "generated_tokens": len(generated_ids), "hit_token_limit": len(generated_ids) == SETTINGS["max_new_tokens"],
        "validation": validate(text), "process_pid": os.getpid(), "process_parent_pid": os.getppid(),
        "started_at_unix": started, "finished_at_unix": time.time(),
        "model_load_seconds": load_seconds, "generation_seconds": generation_seconds,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "runner_sha256": sha256(__file__), "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy")},
        "environment": {name: os.environ.get(name) for name in
            ("PYTHONHASHSEED", "HF_HOME", "HF_HUB_DISABLE_XET", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    write_new(destination, record)
    print(json.dumps({"run": run_number, "file": str(destination), "generation_seconds": generation_seconds,
        "tokens": len(generated_ids), "strict_structure_success": record["validation"]["strict_structure_success"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-child", type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.single_child:
        child(args.single_child)
        return
    if any((HERE / name).exists() for name in ("run_01.json", "run_02.json", "comparison.json", "launch.json")):
        raise RuntimeError("One-shot smoke already started; no retry, replacement, or overwrite is allowed")
    environment = {**os.environ, "PYTHONHASHSEED": str(PUBLIC_DEMO_SEED),
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    started = time.time()
    write_new(HERE / "launch.json", {"started_at_unix": started, "maximum_runs": 2,
        "runner_sha256": sha256(__file__), "public_demo_seed": PUBLIC_DEMO_SEED,
        "messages": MESSAGES, "generation_settings": SETTINGS,
        "model_id": MODEL_ID, "revision": REVISION,
        "design": "Two separate fresh subprocesses, sequential, fixed prompt/seed/settings; no retries or selection."})
    child_exits = []
    for run_number in (1, 2):
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                    "--single-child", str(run_number)], env=environment)
        code = process.wait()
        child_exits.append({"run_number": run_number, "owned_child_pid": process.pid, "exit_code": code})
        if code != 0:
            write_new(HERE / "execution_failure.json", {"child_exits": child_exits,
                "status": "stopped_without_retry", "failed_run": run_number})
            raise SystemExit(code)
    first, second = [json.loads((HERE / f"run_{i:02d}.json").read_text()) for i in (1, 2)]
    both_parsed = all(x["validation"]["json_parse_success"] for x in (first, second))
    comparison = {"schema": "seeded-ai-dictionary-comparison-v1", "status": "complete",
        "child_exits": child_exits, "fresh_processes_distinct": first["process_pid"] != second["process_pid"],
        "input_token_ids_equal": first["input_token_ids"] == second["input_token_ids"],
        "generated_token_ids_equal": first["generated_token_ids"] == second["generated_token_ids"],
        "generated_text_exactly_equal": first["generated_text"] == second["generated_text"],
        "both_json_parse_success": both_parsed,
        "parsed_json_exactly_equal": first["validation"]["parsed_json"] == second["validation"]["parsed_json"] if both_parsed else None,
        "both_strict_structure_success": all(x["validation"]["strict_structure_success"] for x in (first, second)),
        "run_sha256": {f"run_{i:02d}.json": sha256(HERE / f"run_{i:02d}.json") for i in (1, 2)},
        "runner_sha256": sha256(__file__), "elapsed_seconds": time.time() - started,
        "scope": "One public seed and one prompt, same software/hardware environment only.",
        "limitations": ["Two samples do not establish general or cross-platform determinism.",
            "Structural validation cannot establish English vocabulary or semantic synonymy.",
            "This is not an encryption key, security test, communication experiment, or deployment-ready codebook.",
            "No prompt tuning, retries, valid-output selection, payload embedding, or attack fitting was performed."]}
    write_new(HERE / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
