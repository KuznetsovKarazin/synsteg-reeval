"""Synthetic subprocess fixture. Not reachable from the production launcher CLI."""
import json
import os
from pathlib import Path
import sys

from synsteg_audit.framed_interval_llm_v1_activation import Activation
from synsteg_audit.framed_interval_llm_v1_fixtures import fake_endpoint
from synsteg_audit.framed_interval_llm_v1_orchestration import execute_endpoint

root, campaign, endpoint, request_path = map(Path, sys.argv[1:5])
spec = json.loads((root / "fake-activation.json").read_bytes())
activation = Activation(root, spec["sha256"], {}, spec["public_protocol"], spec["preflight"], {})
request = json.loads(request_path.read_bytes())


class FakeService:
    kind = "synthetic"
    def __init__(self, act, directory, deadline): pass
    def prepare(self): return fake_endpoint(spec["profile"])
    def close(self): pass


if request["role"] == "receive":
    # Audit hook observes read access below Python's pathlib wrappers as well.
    # Receiver gets wire/envelope + public resources only; expected payload and
    # sender execution traces are deliberately inaccessible in this fake process.
    forbidden = set()
    forbidden.add(str(root / "experiments/framed-interval-llm-v1/proposed-protocol.json"))
    forbidden.update(str(endpoint.parent / "sender" / name) for name in
                     ("request.json", "worker-result.json", "token-journal.jsonl", "wire.raw"))
    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            if str(Path(os.fsdecode(args[0])).resolve()) in forbidden:
                raise AssertionError("fresh receiver attempted to read private sender artifact")
    sys.addaudithook(audit)
    assert "payload_hex" not in request

result = execute_endpoint(request, activation, campaign, endpoint, service_factory=FakeService)
raise SystemExit(0 if result["status"] == "returned" else 2)
