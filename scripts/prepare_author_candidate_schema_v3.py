"""Prepare a pure, disabled component from the immutable v2 request cohort."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/author-candidate-schema-v3"
V2_REQUESTS = "experiments/author-candidate-format-v2/requests.json"
V2_REQUESTS_SHA256 = "0cc5750f8251cb7b10bb29d00ec858d5b005d0df42fb7d2781152c807a4ae421"
V2_FORMAT_SHA256 = "20f70a75be1a7ab12825a00714b72e96e11c2fc8ae24a09d85af78508fc62273"
BASE_MAIN = "3495354ac9b6951cd259b3d325b007d649e51772"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as existing:
            if not stat.S_ISREG(os.fstat(existing.fileno()).st_mode):
                raise ValueError("prepared destination must be a regular file")
            if existing.read(len(raw) + 1) != raw:
                raise ValueError("refusing to overwrite different prepared file: " + str(path))
        return
    with os.fdopen(fd, "wb") as created:
        created.write(raw)
        created.flush()
        os.fsync(created.fileno())


def main():
    source = (ROOT / V2_REQUESTS).read_bytes()
    if sha(source) != V2_REQUESTS_SHA256:
        raise ValueError("frozen v2 request bytes differ")
    if sha((ROOT / "src/synsteg_audit/author_candidate_format_v2.py").read_bytes()) != V2_FORMAT_SHA256:
        raise ValueError("frozen validator bytes differ")
    old = json.loads(source)
    expected = [["Popular", "Knowledge", "Article:", "Understanding", "Chinese"],
                ["Astronomy", "For", "a", "general", "audience,"]]
    if ([r["words"] for r in old["requests"]] != expected or old["full_cover_chunks"] != 109
            or old["diagnostic_chunks"] != 2 or old["outside_diagnostic_chunks"] != 107):
        raise ValueError("cohort identity differs")
    for request in old["requests"]:
        if sha(canonical({k: v for k, v in request.items() if k != "request_sha256"})) != request["request_sha256"]:
            raise ValueError("v2 request digest differs")
    requests = {"schema": "author-candidate-schema-v3-cohort", "native_enabled": False,
                "source_path": V2_REQUESTS, "source_sha256": V2_REQUESTS_SHA256,
                "source_git_commit": BASE_MAIN, "cover_sha256": old["cover_sha256"],
                "full_cover_chunks": 109, "diagnostic_chunks": 2, "outside_diagnostic_chunks": 107,
                "selection": "same first two chunks; no new selection or response-dependent expansion",
                "frozen_v2_requests_unchanged": old["requests"]}
    save(BASE / "cohort.json", requests)
    protocol = {
        "schema": "author-candidate-schema-v3-protocol", "id": "author-candidate-schema-v3",
        "scope": "pure schema-prefix, token-support conditioning and deterministic draw component",
        "native_execution_enabled": False, "native_runner_implemented": False,
        "model_calls": 0, "tokenizer_calls": 0, "encoder_calls": 0, "decoder_calls": 0,
        "base_git_main": BASE_MAIN,
        "frozen_dependencies": {V2_REQUESTS: V2_REQUESTS_SHA256,
            "src/synsteg_audit/author_candidate_format_v2.py": V2_FORMAT_SHA256},
        "cohort": {"path": "experiments/author-candidate-schema-v3/cohort.json",
                   "sha256": sha((BASE / "cohort.json").read_bytes()), "chunks": 2,
                   "full_cover_chunks": 109, "not_attempted_chunks": 109,
                   "proposed_outside_diagnostic_chunks": 107, "new_selection": False},
        "schema_contract": {"root": "only words key",
            "rows": "one ordered singleton object per exact input token, including repeated tokens",
            "values": "[] or exactly eight strings; no silent filling, deletion or repair",
            "candidate_surface": "nonempty trimmed; distinct after whitespace collapse and casefold; exclude source surface",
            "forbidden_unicode_categories": ["Cc", "Cf", "Zl", "Zp"],
            "unicode": "strict UTF-8; JSON surrogate escapes only as valid pairs; partial escapes/UTF-8 require a legal schema continuation",
            "not_checked": ["meaning", "part of speech", "morphology", "reference", "punctuation preservation", "world factuality"]},
        "limits": {"raw_bytes": 262144, "source_tokens": 5, "support_rows": 65536,
                   "mass_per_row_max": 16777216, "HMAC_blocks": 128,
                   "draw_attempt_index": [0, 2], "draw_step_index": [0, 511]},
        "prefix_semantics": "abstract schema viability, not existence of a completion in model vocabulary/top50 or remaining512token/bytebudget",
        "support": {"input": "explicit list of unique nonnegative token IDs, nonempty exact bytepieces, positive integer masses",
            "EOS_and_special_tokens": "caller must exclude; component cannot identify them from bytes",
            "ordering": "ascending token ID, preserving distinct IDs even if bytepieces equal",
            "mask": "allow only whole token pieces extending to incomplete or complete required schema",
            "resource_limit": "refuse whole step if any supplied candidate exceeds bytebudget; never silently grammar-mask it",
            "empty_support": "known refusal; no top_k expansion, repair, fallback, seed search or automatic retry",
            "duplicate_piece_ids": "retain separate categorical events; do not merge their mass",
            "complete_prefix": "stop without generating EOS or trailing tokens"},
        "probability": {"base": "supplied discrete law q_i=m_i/M; no logits or full-vocabulary distribution observed",
            "conditional": "q_i_given_schema=m_i/S for eligible IDs, S=sum eligible masses; zero otherwise",
            "retained_fraction": "S/M; null when input support is empty (0/0 is undefined)",
            "rounding": "no re-quantization; exact integer numerator/denominator",
            "native_proposal_not_run": "condition the complete EOS/special-masked temperature0.7 top50 integer2^24 law; no full-vocabulary recovery or top50 widening",
            "top50_limitation": "empty support is possible even where a legal token exists outside top50",
            "distribution_shift": "schema-conditioning generally changes the base distribution; no stealth or indistinguishability claim"},
        "draw": {"key_bytes": 32, "domain": "author-candidate-schema-v3/draw-v1",
            "message": "UTF8 compact sorted-key JSON of domain,request_sha256,attempt,step,law_sha256,counter",
            "hash": "HMAC-SHA256; big-endian unsigned256 block",
            "rejection": "accept block<floor(2^256/S)*S then ticket=block%S; ascending-ID mass intervals",
            "counter": "0..127, then resource refusal; never substitute key or seed",
            "bindings": "law digest includes sourceword hash, rawprefix hash, base-law hash and complete conditional metadata",
            "integrity_limit": "hash is not authentication or proof of genuine model/tokenizer evidence; adapter/auditor must reconstruct law",
            "fixture_key": "bytes(range(32)) is a public synthetic-test fixture only; no native key or run is activated",
            "seed_search": False},
        "abstention": {"all_empty_rows": "format acceptance only", "usable_candidate_rows": None,
            "capacity_bits": None, "semantic_equivalence_verified": False, "full_cover_resource_complete": False},
        "native_followup_gates": ["separate new native ID and bounded protocol",
            "full vocabulary/tokenpiece and logits evidence binding with independent law reconstruction",
            "journal guard and unknown/no-retry policy", "independent mock/fault review",
            "Git-pinned source, disabled activation, separate enabled intent and actual readback",
            "single project NN lease; no old-v1/v2 rerun"],
        "nonclaims": ["not GPT-4o original-provider replication", "not prompt-only causal comparison",
            "not a complete109chunk dictionary", "not a useful stego codec or semantic certification",
            "synthetic fixture tests are not model attempts"],
        "paid_API_allowed": False, "closed_test_allowed": False}
    save(BASE / "protocol.json", protocol)
    save(BASE / "activation.disabled.json", {"schema": "author-candidate-schema-v3-activation",
        "enabled": False, "native_runner_implemented": False, "model_calls_authorized_by_this_file": False})
    print(json.dumps({"status": "prepared_disabled", "cohort_sha256": protocol["cohort"]["sha256"],
                      "model_calls": 0, "tokenizer_calls": 0}))


if __name__ == "__main__":
    main()
