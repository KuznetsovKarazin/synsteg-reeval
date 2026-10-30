"""Fake-only four-case launcher checks; no native process/model/resource access."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from synsteg_audit.framed_interval_llm_v1_activation import Activation, INTENT, PROPOSAL_PATH, PUBLIC_PATH
from synsteg_audit.framed_interval_llm_v1_fixtures import fake_endpoint
from synsteg_audit.framed_interval_llm_v1_journal import atomic_json, sha
from synsteg_audit.framed_interval_llm_v1_orchestration import (
    _collect, execute_endpoint, mkdir_durable, reconcile, run_campaign, validate_request)

REPO = Path(__file__).resolve().parents[1]


def fixture(root, profile="uniform16"):
    for name in (PROPOSAL_PATH, PUBLIC_PATH):
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    protocol = json.loads((root / PUBLIC_PATH).read_bytes())
    spec = {"sha256": "a"*64, "public_protocol": protocol, "profile": profile,
            "preflight": {"prompt_ids": {"station": [10000, 10001], "library": [10000, 10001]}}}
    (root / "fake-activation.json").write_text(json.dumps(spec))
    act = Activation(root, spec["sha256"], {}, protocol, spec["preflight"], {})
    return act, lambda *args: act


class FakeService:
    kind = "synthetic"
    def __init__(self, act, directory, deadline): self.act = act
    def prepare(self): return fake_endpoint()
    def close(self): pass


def endpoint_request(root, act, *, role="generate"):
    campaign = root / "results/framed-interval-llm-v1/unit"
    mkdir_durable(campaign)
    manifest = {"activation_sha256": act.sha256, "public_runs": act.public_protocol["runs"],
                "planned_cases": 4, "planned_fresh_stego_receivers": 2}
    atomic_json(campaign / "campaign-manifest.json", manifest)
    request = {"schema": "framed-interval-llm-v1-worker-request-v1", "role": role,
               "run_id": "station-stego", "activation_sha256": act.sha256,
               "campaign_manifest_sha256": sha((campaign / "campaign-manifest.json").read_bytes()),
               "key_hex": bytes(range(32)).hex(),
               "payload_hex": hashlib.sha256(b"seeded-pair-llm-v1:payload:station").digest()[:16].hex()}
    endpoint = campaign / "station-stego/sender"; mkdir_durable(endpoint)
    return campaign, endpoint, request


def test_full_four_cases_and_two_isolated_receivers(tmp_path):
    act, loader = fixture(tmp_path)
    pids, requests = [], []
    def worker(root, apath, asha, campaign, endpoint, request, timeout):
        record = json.loads((root / request).read_bytes()); requests.append(record)
        env = dict(os.environ); env["PYTHONPATH"] = str(REPO / "src")
        proc = subprocess.Popen([sys.executable, str(REPO / "tests/framed_interval_llm_fake_worker.py"),
            str(root), str(root / campaign), str(root / endpoint), str(root / request)],
            env=env, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(timeout=60)
        assert proc.returncode == 0, (stdout, stderr)
        pids.append(proc.pid)
        return {"status": "exited", "exit_code": proc.returncode, "pid": proc.pid}
    result = run_campaign(tmp_path, "fake", act.sha256, "results/framed-interval-llm-v1/fake-success", INTENT,
        activation_loader=loader, worker_runner=worker, expected_kind="synthetic")
    assert len(pids) == len(set(pids)) == 6
    assert len(result["cases"]) == 4
    assert result["authenticated_payload_deliveries"] == result["authenticated_delivery_denominator"] == 2
    assert result["real_model_success_claimed"] is False
    assert sum(r["role"] == "receive" for r in requests) == 2
    assert all("payload_hex" not in r for r in requests if r["role"] == "receive")
    for case in result["cases"]:
        assert case["sender"]["reconciliation"]["known_committed_tokens"] == 512
        assert case["sender"]["worker"]["core"]["committed_trace_matches_returned_prefix"]
        if case["receiver"]:
            assert case["receiver"]["worker"]["core"]["processed_tokens"] == 512
            assert case["receiver"]["exact_expected_payload"]
    with pytest.raises(FileExistsError):
        run_campaign(tmp_path, "fake", act.sha256, "results/framed-interval-llm-v1/fake-success", INTENT,
            activation_loader=loader, worker_runner=worker, expected_kind="synthetic")
    assert len(pids) == 6


def test_all_four_process_failures_retain_two_blocked_denominators(tmp_path):
    act, loader = fixture(tmp_path)
    calls = []
    def fail(*args):
        calls.append(args[4]); raise RuntimeError("synthetic spawn failure")
    result = run_campaign(tmp_path, "fake", act.sha256, "results/framed-interval-llm-v1/fake-failure", INTENT,
        activation_loader=loader, worker_runner=fail, expected_kind="synthetic")
    assert len(calls) == 4 and len(result["cases"]) == 4
    assert result["authenticated_payload_deliveries"] == 0
    assert result["authenticated_delivery_denominator"] == 2
    assert all(case["sender"]["process"]["status"] == "spawn_or_wait_error" for case in result["cases"])
    assert [case["receiver"]["worker_status"] for case in result["cases"] if case["receiver"]] == ["dependency_blocked"]*2


def test_native_ack_loss_preserves_prefix_and_unknown_no_retry(tmp_path):
    act, _ = fixture(tmp_path)
    campaign, directory, request = endpoint_request(tmp_path, act)
    calls = []
    class FaultService(FakeService):
        def prepare(self):
            def step(token):
                calls.append(token)
                if len(calls) == 3: raise OSError("synthetic native acknowledgement lost")
            return fake_endpoint(observe_callback=step)
    result = execute_endpoint(request, act, campaign, directory, service_factory=FaultService)
    assert result["core"]["status"] == "backend_error"
    assert result["reconciliation"]["known_committed_tokens"] == 2
    assert result["reconciliation"]["unknown_native_step"]
    assert len((directory / "committed-wire.bin").read_bytes()) == 2
    with pytest.raises(FileExistsError):
        execute_endpoint(request, act, campaign, directory, service_factory=FaultService)
    assert len(calls) == 3


def test_zero_observation_refusal_is_saved_and_replayed_as_incomplete(tmp_path):
    act, _ = fixture(tmp_path)
    campaign, directory, request = endpoint_request(tmp_path, act)
    class EmptyService(FakeService):
        def prepare(self): return fake_endpoint("empty_support")
    result = execute_endpoint(request, act, campaign, directory, service_factory=EmptyService)
    assert result["core"]["status"] == "support_refusal"
    assert result["reconciliation"]["wire_available"]
    assert result["reconciliation"]["known_committed_tokens"] == 0
    assert (directory / "committed-wire.bin").read_bytes() == b""
    receiver = directory.parent / "receiver"; mkdir_durable(receiver)
    del request["payload_hex"]
    request.update(role="receive", wire={"path": (directory / "committed-wire.bin").relative_to(tmp_path).as_posix(),
        "sha256": sha(b"")}, envelope={"path": (directory / "envelope.json").relative_to(tmp_path).as_posix(),
        "sha256": sha((directory / "envelope.json").read_bytes())})
    decoded = execute_endpoint(request, act, campaign, receiver, service_factory=EmptyService)
    assert decoded["core"]["status"] == "incomplete_wire"
    assert decoded["core"]["payload_hex"] is None


def test_close_failure_keeps_journal_and_is_process_failure(tmp_path):
    act, _ = fixture(tmp_path)
    campaign, directory, request = endpoint_request(tmp_path, act)
    class CloseError(FakeService):
        def prepare(self): return fake_endpoint("empty_support")
        def close(self): raise OSError("synthetic close")
    result = execute_endpoint(request, act, campaign, directory, service_factory=CloseError)
    assert result["status"] == "worker_exception" and result["close_errors"]
    assert reconcile(directory)["known_committed_tokens"] == 0


def test_core_none_reconciliation_and_bad_auth_record(tmp_path):
    act, _ = fixture(tmp_path)
    campaign, directory, request = endpoint_request(tmp_path, act)
    class Error(FakeService):
        def prepare(self): raise OSError("synthetic before journal")
    result = execute_endpoint(request, act, campaign, directory, service_factory=Error)
    assert result["core"] is None and not result["reconciliation"]["wire_available"]
    bad = {"run_id": "station-stego", "role": "receive", "execution_kind": "synthetic",
           "activation_sha256": act.sha256, "status": "returned", "core": {
               "status": "authenticated_delivery", "authenticated_delivery": True,
               "payload_hex": "", "processed_tokens": 511, "wire_tokens": 512}}
    atomic_json(directory / "worker-result.json", bad, replace=True)
    with pytest.raises(ValueError, match="complete fixed512"):
        _collect(directory, {"status": "exited", "exit_code": 0}, "receive", "synthetic", "station-stego", act.sha256)


@pytest.mark.parametrize("change", ["payload", "key", "run", "manifest", "extra"])
def test_bad_request_refuses_before_service(tmp_path, change):
    act, _ = fixture(tmp_path)
    campaign, directory, request = endpoint_request(tmp_path, act)
    if change == "payload": request["payload_hex"] = "00"
    if change == "key": request["key_hex"] = "00"*32
    if change == "run": request["run_id"] = "extra"
    if change == "manifest": request["campaign_manifest_sha256"] = "0"*64
    if change == "extra": request["sender_trace"] = []
    def forbidden(*args): raise AssertionError("service reached on invalid request")
    with pytest.raises(ValueError): execute_endpoint(request, act, campaign, directory, service_factory=forbidden)
    assert not (directory / "execution-started.json").exists()


def test_production_cli_disabled_activation_never_starts_native(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fixed512_cli_fake_import", REPO / "experiments/framed-interval-llm-v1/run_fixed512.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    activation = {"enabled": False}
    (tmp_path / "inactive.json").write_text(json.dumps(activation))
    raw = (tmp_path / "inactive.json").read_bytes()
    with pytest.raises(ValueError, match="complete enabled"):
        cli.main(["run", "--activation", "inactive.json", "--activation-sha256", sha(raw),
                  "--intent", INTENT, "--output", "results/framed-interval-llm-v1/forbidden"])
    assert not (tmp_path / "results").exists()
