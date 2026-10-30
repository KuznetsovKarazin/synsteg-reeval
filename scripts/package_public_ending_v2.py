#!/usr/bin/env python3
"""Preserve terminated public-ending-v2 bytes using unchanged generic storage.

Two explicit source trees retain results and burned nonces. Resource bytes are
not archived. Packaging is not scientific acceptance and performs no inference.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

ID = "framed-interval-public-ending-v2"
STORAGE_SHA = "7a791b2685220f58c6b637b3642bf6e46c5f5d08269592e31739914d1193ffa6"
CANONICAL_LOCK = "/workspace/scratch/af7adce5db85/recovery60/restored/results/project-nn.lock"
BASES = ["results/" + ID, "runtime/" + ID + "/sender-nonces"]


def storage(path):
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != STORAGE_SHA:
        raise ValueError("frozen storage bytes differ")
    spec = importlib.util.spec_from_file_location("_public_ending_v2_storage", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.SCHEMA = ID + "-raw-artifacts-v1"
    return module


def package(root, helper, terminal_receipt):
    root = Path(root).resolve(strict=True); s = storage(helper)
    terminal_raw = s.regular_bytes(terminal_receipt)
    terminal = s.read_json(terminal_receipt)
    if type(terminal.get("exit_code")) is not int:
        raise ValueError("actual outer process exit must be known before packaging")
    if not (root / BASES[0]).is_dir():
        raise ValueError("actual results tree required")
    local_lock = root / "results/project-nn.lock"
    if local_lock.exists() and os.path.samefile(CANONICAL_LOCK, local_lock):
        raise ValueError("canonical and checkout storage locks must differ")
    s.no_symlink_ancestors(CANONICAL_LOCK)
    fd = os.open(CANONICAL_LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("canonical lock must be regular")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = root / "artifacts" / ID / "raw-v1"
        out.mkdir(parents=True, exist_ok=False)
        s.put(out / "outer-process-source.json", terminal_raw)
        records = []
        for label, base in zip(("results", "sender-nonces"), BASES):
            primary = root / base
            if not primary.exists():
                records.append({"source_base": base, "status": "absent_at_snapshot",
                    "absence_does_not_authorize_retry": True}); continue
            s.no_symlink_ancestors(primary)
            if not primary.is_dir(): raise ValueError("source base must be a real directory")
            if not any(primary.iterdir()):
                records.append({"source_base": base, "status": "empty_directory_at_snapshot",
                    "directory_mode": stat.S_IMODE(primary.stat().st_mode),
                    "absence_does_not_authorize_retry": True}); continue
            # Neither root of these trees has result.json. All nested outcomes,
            # unknowns and empty directories remain bytes, not inferred success.
            records.append(s.package(root, out / label, base=base, allow_incomplete=True))
        receipt = {"schema": ID + "-packaging-v1", "source_bases": BASES,
            "storage_helper_sha256": STORAGE_SHA, "outer_process": terminal,
            "preserved": records, "resource_bytes_archived": False,
            "generation_initial_denominator_present": (root / BASES[0] / "generation/initial-denominator.json").is_file(),
            "receiver_install_inventory_present": (root / BASES[0] / "generation/receiver-install-inventory.json").is_file(),
            "receiver_install_preserved_as": "source closure + public profile/spec/contexts + complete file inventory; resource identities restored separately",
            "canonical_cooperative_lease_held": True,
            "quiescence_scope": "actual outer process exit and cooperative lease, not independent OS isolation",
            "actual_Git_readback_completed": False,
            "scientific_acceptance_inferred": False, "model_calls": 0, "tokenizer_calls": 0}
        s.put(out / "packaging-receipt.json", s.json_bytes(receipt))
        return receipt
    finally:
        os.close(fd)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--storage-helper", type=Path, required=True)
    p.add_argument("--terminal-receipt", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(package(a.root, a.storage_helper, a.terminal_receipt), sort_keys=True))
