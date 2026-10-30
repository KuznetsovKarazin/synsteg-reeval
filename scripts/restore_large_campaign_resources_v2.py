#!/usr/bin/env python3
"""Verify or restore only the seven frozen public TinyLlama cache resources.

Default is read-only verification. --download-missing permits network retrieval
only for absent resources. Existing mismatches and symlinks are refused. Frozen
receipts, logs, configs and result files are never written. No model is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "large-campaign-development-v2"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
RECEIPT = ROOT / "results" / CAMPAIGN / "preflight/download-receipt.json"
RECEIPT_SHA256 = "8d72f0b2280d68c9cd1c41f7482ebda6b5a8ea48a221c43ede51448ce83f5477"
SNAPSHOT = (ROOT / "runtime" / CAMPAIGN / "hf/hub" /
            ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION)
FILES = {"config.json", "generation_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "model.safetensors"}
CHUNK = 1024 * 1024


class RestoreError(ValueError):
    pass


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            value.update(chunk)
    return value.hexdigest()


def read_frozen_receipt(path=RECEIPT):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECEIPT_SHA256:
        raise RestoreError("Frozen download receipt hash mismatch")
    obj = json.loads(raw)
    if (obj.get("schema") != "synsteg-public-model-download-v2" or
            obj.get("campaign_id") != CAMPAIGN or obj.get("model") != MODEL or
            obj.get("revision") != REVISION or obj.get("status") != "all_public_assets_verified"):
        raise RestoreError("Frozen receipt identity/status mismatch")
    rows = obj.get("files")
    if not isinstance(rows, list) or len(rows) != len(FILES):
        raise RestoreError("Require exactly seven frozen resources")
    if {row.get("name") for row in rows} != FILES:
        raise RestoreError("Resource names differ from the frozen seven")
    for row in rows:
        expected_url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{row['name']}"
        if (row.get("source_url") != expected_url or row.get("status") != "verified" or
                type(row.get("size")) is not int or row["size"] <= 0 or
                row.get("expected_size") != row["size"] or
                not isinstance(row.get("sha256"), str) or
                re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise RestoreError("Invalid frozen resource metadata")
    return rows


def reject_symlink_parents(path):
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise RestoreError("Symlink cache path refused")


def check_cached(path, row):
    reject_symlink_parents(path)
    if not path.exists():
        return {"name": row["name"], "status": "missing"}
    if not path.is_file():
        return {"name": row["name"], "status": "mismatch", "reason": "not_regular_file"}
    size = path.stat().st_size
    if size != row["size"]:
        return {"name": row["name"], "status": "mismatch", "reason": "size",
                "actual_size": size, "expected_size": row["size"]}
    actual = digest(path)
    if actual != row["sha256"]:
        return {"name": row["name"], "status": "mismatch", "reason": "sha256",
                "actual_sha256": actual, "expected_sha256": row["sha256"]}
    return {"name": row["name"], "status": "verified", "size": size, "sha256": actual}


def validate_public_url(url):
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    allowed = (host == "huggingface.co" or host.endswith(".huggingface.co") or
               host == "hf.co" or host.endswith(".hf.co"))
    if (parts.scheme != "https" or not allowed or parts.username is not None or
            parts.password is not None or parts.port not in (None, 443)):
        raise RestoreError("Only HTTPS public Hugging Face resource redirects are allowed")


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_missing(path, row, opener=None):
    """Publish one verified new file atomically without replacing any path."""
    reject_symlink_parents(path)
    if path.exists():
        raise RestoreError("Resource appeared before download; refusing replacement")
    validate_public_url(row["source_url"])
    path.parent.mkdir(parents=True, exist_ok=True)
    sender = opener or urllib.request.build_opener(PublicRedirect()).open
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + path.name + ".", suffix=".part",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            request = urllib.request.Request(row["source_url"], headers={"Accept": "application/octet-stream"})
            value = hashlib.sha256()
            size = 0
            with sender(request, timeout=120) as response:
                while chunk := response.read(CHUNK):
                    size += len(chunk)
                    if size > row["size"]:
                        raise RestoreError("Public download exceeds frozen byte count")
                    value.update(chunk)
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if size != row["size"] or value.hexdigest() != row["sha256"]:
            raise RestoreError("Public download bytes/hash differ from frozen receipt")
        # Hard-link creation is atomic and refuses a concurrently created destination.
        # If unsupported, fail closed; never fall back to replacing an existing path.
        os.link(temporary, path)
        return {"name": row["name"], "status": "restored_verified",
                "size": size, "sha256": value.hexdigest()}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore(rows, snapshot=SNAPSHOT, allow_download=False, opener=None):
    report = {"schema": "synsteg-large-resource-restore-v2", "campaign_id": CAMPAIGN,
              "model": MODEL, "revision": REVISION, "frozen_receipt_sha256": RECEIPT_SHA256,
              "mode": "download_missing" if allow_download else "verify_only",
              "generation_calls": 0, "model_loads": 0, "paid_calls": 0,
              "frozen_artifacts_written": 0, "public_download_attempts": 0,
              "automatic_retries": 0, "files": [], "status": "checking"}
    try:
        # Check the whole cache before any network or directory creation.
        report["files"] = [check_cached(snapshot / row["name"], row) for row in rows]
        mismatches = [row for row in report["files"] if row["status"] == "mismatch"]
        missing = [row for row in report["files"] if row["status"] == "missing"]
        if mismatches:
            report["status"] = "blocked_existing_mismatch"
            return report
        if missing and not allow_download:
            report["status"] = "blocked_missing_resources"
            return report
        for index, row in enumerate(rows):
            if report["files"][index]["status"] == "missing":
                report["public_download_attempts"] += 1
                report["files"][index] = download_missing(snapshot / row["name"], row, opener)
        report["status"] = "all_seven_resources_verified"
    except (OSError, ValueError, http.client.HTTPException) as exc:
        report.update(status="blocked", error_type=type(exc).__name__, error=str(exc))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify-only", action="store_true", help="Default: never access network or write cache")
    group.add_argument("--download-missing", action="store_true", help="Restore absent public files only")
    args = parser.parse_args(argv)
    try:
        report = restore(read_frozen_receipt(), allow_download=args.download_missing)
    except (OSError, ValueError) as exc:
        report = {"schema": "synsteg-large-resource-restore-v2", "status": "blocked",
                  "error_type": type(exc).__name__, "error": str(exc),
                  "generation_calls": 0, "model_loads": 0, "public_download_attempts": 0,
                  "frozen_artifacts_written": 0}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "all_seven_resources_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())

