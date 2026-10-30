#!/usr/bin/env python3
"""Restore archived synthetic pytest receipts without overwriting existing bytes."""
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/framed-interval-llm-v1-launcher-preflight"


def main():
    manifest = json.loads((BASE / "synthetic-checks-archive.json").read_bytes())
    raw = (BASE / "synthetic-checks.zip").read_bytes()
    sha = lambda data: hashlib.sha256(data).hexdigest()
    if len(raw) != manifest["archive_bytes"] or sha(raw) != manifest["archive_sha256"]:
        raise ValueError("synthetic check archive identity differs")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(manifest["members"]):
            raise ValueError("synthetic check archive members differ")
        for name, entry in manifest["members"].items():
            if (Path(name).name != name or name in {".", ".."} or entry["bytes"] > 2_000_000):
                raise ValueError("unsafe archived receipt")
            data = archive.read(name)
            if len(data) != entry["bytes"] or sha(data) != entry["sha256"]:
                raise ValueError("archived receipt identity differs")
            target = BASE / "final-check" / name
            if target.is_symlink() or target.parent.is_symlink(): raise ValueError("symlinked restore target")
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                if target.read_bytes() != data: raise ValueError("existing receipt differs; refusing overwrite")
                continue
            temporary = target.with_name(target.name+".restore-writing")
            with temporary.open("xb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, target); temporary.unlink()
    print(json.dumps({"status": "verified_or_restored", "members": len(manifest["members"]),
                      "model_calls": 0, "network_calls": 0}, sort_keys=True))


if __name__ == "__main__": main()
