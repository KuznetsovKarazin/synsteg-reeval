#!/usr/bin/env python3
"""Freeze archived text records for neural rescoring, without reusing old scores.

This is a development-corpus export, never a new closed test. Text is copied
verbatim; identical texts receive one SHA256 for inference-cache deduplication,
while their original dataset/UUID/arm identities remain separate records.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ("primary", "quality", "archived-ai", "baseline", "all")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


class Inventory:
    def __init__(self, source):
        self.source = Path(source).resolve()
        self.records = {}
        self.source_files = {}
        self.excluded_records = []

    def read(self, relative, jsonl=False):
        p = self.source / relative
        data = p.read_bytes()
        self.source_files[relative] = {"sha256": sha256_bytes(data), "bytes": len(data)}
        if jsonl:
            return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        return json.loads(data)

    def add(self, dataset, uuid, arm, text, source_file, *, candidate=-1, payload=None, role=None, metadata=None, record_index=None):
        if not isinstance(uuid, str) or not uuid:
            raise ValueError(f"Missing record identity in {source_file}")
        if not isinstance(text, str) or not text.strip():
            self.excluded_records.append({"dataset": dataset, "uuid": uuid, "arm": arm,
                                          "source_file": source_file, "source_record_index": record_index,
                                          "reason": "no_nonempty_stored_text"})
            return
        identity_parts = [dataset, uuid, arm, candidate]
        if record_index is not None:
            identity_parts.append(record_index)
        identity = json.dumps(identity_parts, ensure_ascii=False, separators=(",", ":"))
        row = {"id": identity, "dataset": dataset, "uuid": uuid, "arm": arm,
               "cand": candidate, "text": text, "text_sha256": sha256_bytes(text.encode("utf-8")),
               "source_file": source_file, "evaluation_role": role or "archived_development_only"}
        if payload is not None:
            row["requested_payload_bits"] = payload
        if record_index is not None:
            row["source_record_index"] = record_index
        if metadata:
            row.update(metadata)
        if identity in self.records and self.records[identity] != row:
            raise ValueError(f"Conflicting duplicate record ID: {identity}")
        self.records[identity] = row

    def reports(self, relative, dataset, arms=("cover", "stego"), *, variant=None):
        files = sorted((self.source / relative).glob("*.json"))
        if not files:
            raise ValueError(f"No source reports: {relative}")
        for path in files:
            rel = str(path.relative_to(self.source))
            report = self.read(rel)
            uuid = report.get("uuid")
            payload = len(report["message"]) if isinstance(report.get("message"), str) else None
            for arm in arms:
                field = "container" if arm == "cover" else "encoded_message"
                self.add(dataset, uuid, variant or arm, report.get(field), rel, payload=payload)

    def ai_records(self, complete=False):
        relative = "experiments/ai_detectors/corpus.jsonl"
        rows = self.read(relative, jsonl=True)
        index = {}
        for row in rows:
            key = (row["uuid"], row["arm"], row.get("cand", -1))
            if key in index:
                raise ValueError(f"Duplicate archived AI-corpus tuple: {key}")
            index[key] = row
            if complete or row["arm"] == "human":
                self.add("archived_ai", row["uuid"], row["arm"], row["text"], relative,
                         candidate=row.get("cand", -1),
                         role="legacy_human_calibration_reference_not_holdout" if row["arm"] == "human" else None)
        if complete:
            selection_file = "experiments/ai_detectors/results.json"
            selections = self.read(selection_file)["selected"]
            for uuid in sorted(selections):
                for arm, candidate in (("stego_rand", 0), ("stego_sel", selections[uuid])):
                    selected = index[(uuid, "cand_b5", candidate)]
                    self.add("archived_ai", uuid, arm, selected["text"], relative, candidate=candidate,
                             metadata={"selection_source": selection_file if arm == "stego_sel" else None,
                                       "selection_status": "historical_detector_guided_choice" if arm == "stego_sel" else "historical_first_candidate"})

    def primary(self):
        self.ai_records(complete=False)
        self.reports("reports/gpt_omni_reports", "gpt4o")
        self.reports("experiments/ppl_opt/series/opt_k32", "gpt4o", arms=("stego",), variant="stego_opt_k32")
        self.reports("reports/gpt6_luna_missing", "luna")
        self.reports("reports/qwen_missing_b3", "qwen_missing_b3")

    def quality(self):
        self.primary()
        for k in (4, 8):
            self.reports(f"experiments/ppl_opt/series/opt_k{k}", "gpt4o", arms=("stego",), variant=f"stego_opt_k{k}")
        self.reports("reports/json_reports", "deepstego2023")
        self.reports("experiments/ppl_opt/series/ds2023_opt", "deepstego2023", arms=("stego",), variant="stego_opt")

    def baseline(self):
        directory = self.source / "experiments/conversation_bench/results"
        files = sorted(directory.glob("*.jsonl"))
        if not files:
            raise ValueError("No stored conversation benchmark text files")
        for path in files:
            rel = str(path.relative_to(self.source))
            kind, config = path.stem.split("_", 1)
            if kind not in {"covers", "corpus", "capacity", "chain"}:
                continue
            for index, row in enumerate(self.read(rel, jsonl=True)):
                uuid = str(row.get("Key", row.get("i", "")))
                arm = "cover" if kind == "covers" else "stego"
                self.add(f"conversation_{config}_{kind}", uuid, arm,
                         row.get("text") if kind == "covers" else row.get("stegotext"), rel,
                         payload=row.get("L", row.get("bits")),
                         record_index=index,
                         metadata={"paired_with_cover": False,
                                   "pairing_note": "Archived streams are not asserted to be matched pairs."})


SCOPE_NOTES = {
    "primary": "299 legacy human reference texts; GPT-4o cover/base/K32; new luna and Qwen-missing cover/stego. All are archived development data.",
    "quality": "Primary plus GPT-4o K4/K8 and DeepStego2023 cover/base/optimized; full stored documents, not sentence fragments.",
    "archived-ai": "Every stored AI corpus record, including all 16 detector-search candidates, plus explicit random/selected aliases. Selection is historical and not a held-out test.",
    "baseline": "All stored Conversation-Steganography text streams including capacity and chain attempts; missing texts retained in exclusion ledger, no assumed pair matching.",
    "all": "Union of quality, complete archived AI corpus, exploratory Qwen series, and all stored Conversation-Steganography text streams. Never pooled into one scientific endpoint.",
}


def build(source, output, scope="primary"):
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope: {scope}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen corpus: {output}")
    inventory = Inventory(source)
    if scope == "primary":
        inventory.primary()
    elif scope == "quality":
        inventory.quality()
    elif scope == "archived-ai":
        inventory.ai_records(complete=True)
    elif scope == "baseline":
        inventory.baseline()
    else:
        inventory.quality()
        inventory.ai_records(complete=True)
        inventory.reports("reports/qwen_series", "qwen_exploratory")
        inventory.baseline()
    rows = [inventory.records[k] for k in sorted(inventory.records)]
    if not rows:
        raise ValueError("No corpus records")
    unique = {}
    for row in rows:
        sha = row["text_sha256"]
        if sha in unique and unique[sha] != row["text"]:
            raise ValueError("Text digest collision")
        unique[sha] = row["text"]
    corpus = b"".join(json_bytes(row) for row in rows)
    texts = b"".join(json_bytes({"id": sha, "text_sha256": sha, "text": unique[sha]}) for sha in sorted(unique))
    manifest = {
        "schema_version": 1, "status": "frozen_archived_corpus", "scope": scope,
        "scope_note": SCOPE_NOTES[scope], "n_records": len(rows), "n_unique_texts": len(unique),
        "counts_by_dataset_arm": dict(sorted(Counter(row["dataset"] + "/" + row["arm"] for row in rows).items())),
        "counts_by_evaluation_role": dict(sorted(Counter(row["evaluation_role"] for row in rows).items())),
        "corpus_sha256": sha256_bytes(corpus), "unique_texts_sha256": sha256_bytes(texts),
        "source_files": dict(sorted(inventory.source_files.items())),
        "excluded_records": inventory.excluded_records,
        "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "exclusions": [
            "Stored detector/PPL scores are not copied or reused.",
            "Sentence fragments are omitted because corresponding full documents are included in relevant scopes.",
            "qwen_probe prompts and simulated/probe outcomes are not part of the evaluated report series.",
            "Raw IMDB CSV is not resampled: the exact archived 299 human reference texts are retained.",
            "Scope-specific omissions are explicit in scope_note; primary/quality do not cover every detector-search candidate or exploratory baseline stream.",
        ],
        "limitations": [
            "All exported records are archived development data; no fresh test-set claim is implied.",
            "Legacy human references are calibration data, not independent matched-cover negatives.",
            "Identical text is cached once, but duplicate records must not be treated as independent observations.",
            "Counts include named historical selections; they do not count new optimization or generation runs.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".corpus-") as temp:
        stage = Path(temp) / "frozen"
        stage.mkdir()
        (stage / "corpus.jsonl").write_bytes(corpus)
        (stage / "texts.jsonl").write_bytes(texts)
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/source/synsteg_reproduction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, default="primary")
    args = parser.parse_args(argv)
    manifest = build(args.source, args.output, args.scope)
    print(json.dumps({k: manifest[k] for k in ("status", "scope", "n_records", "n_unique_texts", "corpus_sha256", "counts_by_dataset_arm")}, indent=2))


if __name__ == "__main__":
    main()
