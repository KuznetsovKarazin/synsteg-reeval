"""Fresh-process synthetic receiver: stdin is only wire + key + public envelope."""
from __future__ import annotations

import sys

DENIED = []
FORBIDDEN = {"torch", "transformers", "tokenizers", "huggingface_hub", "requests", "httpx"}


def guard(event, args):
    if event.startswith("socket.") or event in {"subprocess.Popen", "os.system", "os.fork",
                                                "os.forkpty", "os.exec", "os.posix_spawn"}:
        DENIED.append(event)
        raise RuntimeError("receiver network/process action forbidden")
    if event == "import" and str(args[0]).split(".")[0] in FORBIDDEN:
        DENIED.append("import:" + str(args[0]))
        raise RuntimeError("neural/network module forbidden")


sys.addaudithook(guard)

from dataclasses import asdict
import importlib.metadata
import json
import os

from synsteg_audit.framed_interval_packet_v1 import decode
from synsteg_audit.framed_interval_packet_v1_fixtures import public_bundle


def main():
    raw = sys.stdin.buffer.read(4194305)
    if len(raw) > 4194304:
        raise ValueError("receiver input bound exceeded")
    request = json.loads(raw)
    if type(request) is not dict or set(request) != {"wire_hex", "key_hex", "envelope"}:
        raise ValueError("receiver accepts only wire, key and public envelope")
    envelope = request["envelope"]
    result = decode(bytes.fromhex(request["wire_hex"]), bytes.fromhex(request["key_hex"]),
                    envelope, public_bundle(envelope["resource_id"]))
    out = asdict(result)
    out["payload_hex"] = None if result.payload is None else result.payload.hex()
    del out["payload"]
    print(json.dumps({"schema": "framed-interval-packet-v1-fresh-receiver",
                      "pid": os.getpid(), "input_fields": sorted(request),
                      "cryptography_version": importlib.metadata.version("cryptography"),
                      "denied_events": DENIED,
                      "forbidden_modules_loaded": sorted(set(sys.modules) & FORBIDDEN),
                      "result": out}, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
