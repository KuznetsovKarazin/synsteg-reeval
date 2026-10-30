"""Fake backends and filesystem fault tests; no native/tokenizer/model execution."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from synsteg_audit import framed_interval_bounded_v1 as bounded
from synsteg_audit import framed_interval_llm_v1 as fixed
from synsteg_audit import framed_interval_bounded_v1_journal as durable
from synsteg_audit import framed_interval_bounded_v1_orchestration as orch
from synsteg_audit import framed_interval_bounded_v1_qwen as binding
from synsteg_audit import framed_interval_bounded_v1_activation as activation
from synsteg_audit.canonical_channel_v2 import CanonicalGate, ContractError, Prefix, PublicCodec

ROOT = Path(__file__).resolve().parents[1]
KEY, PAYLOAD = bytes(range(32)), b"0123456789abcdef"


def endpoint(n=8, *, fail_observe=None):
    state = {"observed": [], "started": 0, "distributions": 0}
    codec = PublicCodec(lambda text: tuple(ord(c)-97 for c in text),
        lambda ids: bytes(t+97 for t in ids), frozenset(), "1"*64)
    def dist(ids):
        state["distributions"] += 1
        return bounded.Distribution(tuple((i, 1) for i in range(n)))
    def start(ids): state["started"] += 1
    def observe(token):
        state["observed"].append(token)
        if len(state["observed"]) == fail_observe:
            raise OSError("injected ambiguous native STEP outcome")
    ep = bounded.Endpoint(CanonicalGate(codec), resource_id="synthetic-ascii",
        resource_digest="2"*64, provider_digest="3"*64,
        distribution_callback=dist, start_callback=start, observe_callback=observe,
        context_tokens=2048)
    return ep, state


def env(ep):
    return bounded.make_envelope(ep, bytes(12), arm="stego", prompt_id="fake", protocol_digest="4"*64)


def test_bounded_journal_uses_new_domain_and_exact_returned_transitions(tmp_path):
    ep, state = endpoint()
    envelope = env(ep)
    wrapped = durable.DurableEndpoint(ep, envelope, KEY, tmp_path/"sender")
    assert wrapped.order_key == bounded.derive_keys(KEY, envelope)[1]
    assert wrapped.order_key != fixed.derive_keys(KEY, envelope)[1]
    generated = bounded.generate(KEY, envelope, wrapped, payload=PAYLOAD)
    wrapped.close()
    recovered = durable.recover_journal(tmp_path/"sender/token-journal.jsonl")
    assert generated.emitted_tokens == generated.first_threshold_token == 91
    assert recovered["committed_wire"] == generated.wire
    assert recovered["committed_traces"] == list(generated.traces)
    assert not recovered["unknown_native_step"]
    receiver, _ = endpoint()
    recv = durable.DurableEndpoint(receiver, envelope, KEY, tmp_path/"receiver")
    result = bounded.receive(generated.wire, KEY, envelope, recv)
    recv.close()
    rjournal = durable.recover_journal(tmp_path/"receiver/token-journal.jsonl")
    assert result.payload == PAYLOAD and result.authenticated_delivery
    assert rjournal["committed_traces"] == recovered["committed_traces"]


def test_ambiguous_step_commits_only_known_prefix_and_never_retries(tmp_path):
    ep, state = endpoint(fail_observe=3)
    envelope = env(ep); obj = durable.DurableEndpoint(ep, envelope, KEY, tmp_path/"endpoint")
    result = bounded.generate(KEY, envelope, obj, payload=PAYLOAD)
    assert result.status == "backend_error" and result.emitted_tokens == 2
    assert len(state["observed"]) == 3 and state["distributions"] == 3
    saved = durable.recover_journal(obj.journal.path)
    assert len(saved["committed_token_ids"]) == 2 and saved["unknown_native_step"]
    with pytest.raises(ContractError): obj.distribution(Prefix())
    assert state["distributions"] == 3
    with pytest.raises(ContractError): obj.journal.commit()
    obj.close()
    with pytest.raises(FileExistsError): durable.DurableEndpoint(endpoint()[0], envelope, KEY, tmp_path/"endpoint")


@pytest.mark.parametrize("where", ["prepare", "commit"])
def test_journal_io_fault_preserves_unknown_and_does_not_retry(tmp_path, where):
    ep, state = endpoint(); envelope = env(ep)
    obj = durable.DurableEndpoint(ep, envelope, KEY, tmp_path/"endpoint")
    append = obj.journal._append
    def fail(event):
        if event["kind"] == ("prepared" if where == "prepare" else "committed"):
            raise OSError("injected disk failure")
        return append(event)
    obj.journal._append = fail
    result = bounded.generate(KEY, envelope, obj, payload=PAYLOAD)
    assert result.status == "backend_error" and result.emitted_tokens == 0
    assert len(state["observed"]) == (0 if where == "prepare" else 1)
    recovered = durable.recover_journal(obj.journal.path)
    assert not recovered["committed_token_ids"]
    assert recovered["unknown_native_step"] is (where == "commit")
    with pytest.raises(ContractError): obj.distribution(Prefix())
    obj.close()


def test_durable_stop_rejects_extra_backend_distribution(tmp_path):
    ep, state = endpoint(); envelope = env(ep)
    obj = durable.DurableEndpoint(ep, envelope, KEY, tmp_path/"endpoint")
    result = bounded.generate(KEY, envelope, obj, payload=PAYLOAD)
    before = state["distributions"]
    with pytest.raises(ContractError, match="bounded stop"): obj.distribution(Prefix())
    assert before == state["distributions"] == result.emitted_tokens
    obj.close()


def test_torn_tail_is_unknown_and_interior_corruption_refuses(tmp_path):
    ep, _ = endpoint(); envelope = env(ep)
    obj = durable.DurableEndpoint(ep, envelope, KEY, tmp_path/"endpoint")
    bounded.generate(KEY, envelope, obj, payload=PAYLOAD); obj.close()
    path = obj.journal.path; raw = path.read_bytes()
    path.write_bytes(raw+b'{"partial":')
    result = durable.recover_journal(path)
    assert result["unknown_native_step"] and result["torn_last_record"]
    assert len(result["committed_token_ids"]) == 91
    path.write_bytes(raw.replace(b'"sequence":0', b'"sequence":9', 1))
    with pytest.raises(ValueError, match="chain"): durable.recover_journal(path)


def fake_activation(tmp_path):
    protocol = json.loads((ROOT/activation.PUBLIC_PATH).read_bytes())
    fixture_bytes = (ROOT/activation.FIXTURES_PATH).read_bytes()
    target = tmp_path/activation.FIXTURES_PATH
    target.parent.mkdir(parents=True); target.write_bytes(fixture_bytes)
    return SimpleNamespace(root=tmp_path, sha256="f"*64,
        public_protocol=protocol, receiver_scope=False,
        data={"sender_fixtures": {"path": activation.FIXTURES_PATH, "sha256": activation.FIXTURES_SHA}},
        preflight={"prompt_ids": {"station": [], "library": []}})


class FakeService:
    kind = "synthetic"
    calls = []
    fail = False
    def __init__(self, act, directory, deadline):
        self.directory = Path(directory); self.closed = False
    def prepare(self):
        self.calls.append(self.directory.as_posix())
        if self.fail: raise OSError("synthetic resource failure before model")
        return endpoint(16)[0]
    def close(self): self.closed = True


def fake_runner(act, calls):
    def run(root, path, sha, campaign, directory, request, timeout):
        req = json.loads((root/request).read_bytes())
        calls.append((req["run_id"], req["role"]))
        scoped = copy.copy(act); scoped.receiver_scope = req["role"] == "receive"
        result = orch.execute_endpoint(req, scoped, root/campaign, root/directory, service_factory=FakeService)
        return {"status": "exited", "exit_code": 0 if result["status"] == "returned" else 2}
    return run


@pytest.fixture(scope="module")
def full_fake_campaign(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("bounded-workers")
    act = fake_activation(tmp_path)
    calls = []; FakeService.calls = []; FakeService.fail = False
    relative = "results/framed-interval-bounded-v1/synthetic"
    ledger = orch.run_campaign(tmp_path, "fake", act.sha256, relative, "fake",
        activation_loader=lambda *a: act, worker_runner=fake_runner(act, calls), expected_kind="synthetic")
    return act, calls, ledger, tmp_path/relative


def test_entire_matrix_one_attempt_and_fresh_receivers(full_fake_campaign):
    act, calls, ledger, directory = full_fake_campaign
    assert ledger["state"] == "finished_all_planned_outcomes"
    assert ledger["planned_cases"] == 16 and len(ledger["cases"]) == 16
    assert ledger["authenticated_payload_deliveries"] == ledger["authenticated_delivery_denominator"] == 8
    assert not ledger["real_model_success_claimed"] and len(calls) == 24
    expected = []
    for run in act.public_protocol["runs"]:
        expected.append((run["run_id"], "generate"))
        if run["arm"] == "stego": expected.append((run["run_id"], "receive"))
    assert calls == expected and len(FakeService.calls) == 24
    for case in ledger["cases"]:
        sender = case["sender"]
        assert sender["worker"]["core"]["emitted_tokens"] == 68
        if case["receiver"]:
            request = json.loads((directory/case["run_id"]/"receiver/request.json").read_bytes())
            assert "payload_hex" not in request and set(request) == {
                "schema", "role", "run_id", "activation_sha256",
                "campaign_manifest_sha256", "key_hex", "wire", "envelope"}
            assert case["receiver"]["exact_expected_payload"]
    with pytest.raises(FileExistsError):
        orch.run_campaign(act.root, "fake", act.sha256, directory.relative_to(act.root).as_posix(), "fake",
            activation_loader=lambda *a: act, worker_runner=fake_runner(act, calls), expected_kind="synthetic")
    assert len(calls) == 24


def test_all_resource_failures_and_blocked_receivers_remain_denominator(tmp_path):
    act = fake_activation(tmp_path); calls = []; FakeService.fail = True
    try:
        ledger = orch.run_campaign(tmp_path, "fake", act.sha256,
            "results/framed-interval-bounded-v1/failures", "fake",
            activation_loader=lambda *a: act, worker_runner=fake_runner(act, calls), expected_kind="synthetic")
    finally: FakeService.fail = False
    assert len(calls) == 16 and ledger["authenticated_payload_deliveries"] == 0
    assert ledger["authenticated_delivery_denominator"] == 8
    assert all(c["sender"]["worker_status"] == "worker_exception" for c in ledger["cases"])
    assert all(c["receiver"] is None or c["receiver"]["worker_status"] == "dependency_blocked" for c in ledger["cases"])


@pytest.mark.parametrize("mutation", ["private-wire", "sender-trace", "payload-field", "wrong-manifest"])
def test_receiver_request_scope_rejected_before_private_reads(full_fake_campaign, monkeypatch, mutation):
    act, _, _, campaign = full_fake_campaign
    scoped = copy.copy(act); scoped.receiver_scope = True
    request = json.loads((campaign/"station-c00-stego/receiver/request.json").read_bytes())
    if mutation == "private-wire": request["wire"]["path"] = activation.FIXTURES_PATH
    elif mutation == "sender-trace": request["envelope"]["path"] = "results/private-trace.json"
    elif mutation == "payload-field": request["payload_hex"] = "00"*16
    else: request["campaign_manifest_sha256"] = "0"*64
    monkeypatch.setattr(orch, "check_fixtures", lambda *a: pytest.fail("receiver opened fixtures"))
    with pytest.raises(ValueError): orch.validate_request(request, scoped, campaign)


@pytest.mark.parametrize("mutation", ["resolved", "threshold", "noncommitted", "process", "wire-hash", "torn"])
def test_delivery_acceptance_rejects_incomplete_or_inconsistent_evidence(full_fake_campaign, mutation, tmp_path):
    act, _, ledger, original = full_fake_campaign
    campaign = tmp_path / "copied"
    shutil.copytree(original, campaign)
    directory = campaign/"station-c00-stego/receiver"
    worker_path = directory/"worker-result.json"; worker = json.loads(worker_path.read_bytes())
    process = {"status": "exited", "exit_code": 0}
    if mutation == "resolved": worker["core"]["resolved_bits"] += 1
    elif mutation == "threshold": worker["core"]["first_threshold_token"] -= 1
    elif mutation == "noncommitted": worker["core"]["noncommitted_trace_suffix"] = [{"x": 1}]
    elif mutation == "process": process["exit_code"] = 2
    elif mutation == "wire-hash":
        reqpath = directory/"request.json"; request = json.loads(reqpath.read_bytes())
        request["wire"]["sha256"] = "0"*64; reqpath.write_text(json.dumps(request))
    else:
        with (directory/"token-journal.jsonl").open("ab") as stream: stream.write(b'{"torn"')
    worker_path.write_text(json.dumps(worker))
    try:
        result = orch._collect(directory, process, "receive", "synthetic", "station-c00-stego", act.sha256)
        assert not result["authenticated_delivery"]
    except ValueError:
        assert mutation == "threshold"


def test_native_binding_inert_and_stronger_context_guard_precedes_launch(tmp_path, monkeypatch):
    act = SimpleNamespace(data={"source_readback": {}, "ready": {"context_tokens": 2048}},
        preflight={"prompt_ids": {"station": list(range(1300)), "library": []}})
    calls = []
    monkeypatch.setattr(binding.ProcNativeService, "prepare", lambda self: calls.append("native"))
    service = binding.BoundedNativeService(act, tmp_path, 999)
    assert not calls
    with pytest.raises(ValueError, match="before launch"): service.prepare()
    assert not calls


def test_cli_worker_loads_receiver_scope_and_never_retries_existing(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("bounded_cli_test", ROOT/"experiments/framed-interval-bounded-v1/run_bounded.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    relative = "results/framed-interval-bounded-v1/fake"
    endpoint_path = relative+"/station-c00-stego/receiver"
    directory = tmp_path/endpoint_path; directory.mkdir(parents=True)
    (directory/"request.json").write_text(json.dumps({"role": "receive", "run_id": "station-c00-stego"}))
    args = SimpleNamespace(campaign=relative, endpoint=endpoint_path, request=endpoint_path+"/request.json",
        activation="activation", activation_sha256="a"*64, intent="intent")
    scopes = []
    monkeypatch.setattr(cli, "load_activation", lambda *a, **k: scopes.append(k["receiver_scope"]) or object())
    monkeypatch.setattr(cli, "execute_endpoint", lambda *a: {"status": "returned"})
    assert cli.worker(args) == 0 and scopes == [True]
    (directory/"execution-started.json").write_text("{}")
    with pytest.raises(ValueError, match="retry forbidden"): cli.worker(args)
    assert scopes == [True]
