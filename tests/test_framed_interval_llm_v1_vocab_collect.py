"""Synthetic service tests for the no-forward collector controller.

No native process, real tokenizer, model file or network request is used.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from synsteg_audit import framed_interval_llm_v1_preflight as preflight

ROOT = Path(__file__).resolve().parents[1]


def input_data():
    plan_bytes = (ROOT / "experiments/framed-interval-llm-v1/tokenizer-preflight-proposal.json").read_bytes()
    protocol = json.loads((ROOT / "experiments/framed-interval-llm-v1/receiver-public-protocol.json").read_bytes())
    resources = {"model_sha256": preflight.MODEL_SHA256, "runtime_commit": preflight.RUNTIME_COMMIT,
        "archive_sha256": preflight.ARCHIVE_SHA256, "cpu_backend_sha256": preflight.CPU_SHA256,
        "resource_identity_sha256": "a"*64,
        "native_libraries_sha256": {"runtime/llama-b11791/libllama.so": "b"*64}}
    return plan_bytes, protocol, resources


class SyntheticService:
    """A small explicit mapping, not a linguistic tokenizer or native binding."""
    def __init__(self, protocol):
        self.ready = {"vocabulary_size": 56178, "masked_tokens": 3, "context_tokens": 2048}
        self.calls = []
        self.fixture_ids = {text: [100+i] for i, (_, text) in enumerate(preflight.TEXT_FIXTURES) if text}
        self.fixture_ids[""] = []
        self.fixture_ids["**"] = [334]
        self.fixture_ids["**\n\n"] = [56177]
        self.rendered = {tuple(ids): text.encode() for text, ids in self.fixture_ids.items()}
        self.rendered[(334, 271)] = b"**\n\n"
        self.prompt_ids = {preflight.chatml_prompt(protocol, name): [0, 400+i, 1]
                           for i, name in enumerate(("station", "library"))}

    def inventory(self):
        self.calls.append(("inventory",))
        return [{"token_id": i, "is_control": i == 0, "is_eog": i == 1, "piece_empty": i in {0, 2}}
                for i in range(self.ready["vocabulary_size"])]

    def tokenize(self, text, add_special=False, parse_special=False):
        assert isinstance(text, str)
        assert type(add_special) is bool and type(parse_special) is bool
        self.calls.append(("tokenize", text, add_special, parse_special))
        if add_special or parse_special:
            assert add_special and parse_special
            return list(self.prompt_ids[text])
        return list(self.fixture_ids[text])

    def render(self, ids):
        self.calls.append(("render", tuple(ids)))
        return self.rendered[tuple(ids)]


def collect(service=None, on_record=None):
    from synsteg_audit.framed_interval_llm_v1_vocab_collect import collect_evidence
    plan_bytes, protocol, resources = input_data()
    if service is None: service = SyntheticService(protocol)
    return collect_evidence(service, plan_bytes, protocol, resources, "c"*64,
                            "results/synthetic/inventory.json", on_record=on_record)


def test_complete_synthetic_service_preserves_all_fifteen_cases_and_two_prompts():
    plan_bytes, protocol, resources = input_data()
    service = SyntheticService(protocol)
    receipt, inventory_bytes = collect(service)
    assert len(receipt["cases"]) == 15
    assert len(json.loads(inventory_bytes)) == 56178
    assert set(receipt["prompts"]) == {"station", "library"}
    assert receipt["cases"][-1]["gate_reasons"] == ["accepted", "retokenization_mismatch"]
    assert sum(row.get("status") == "invalid_utf8" for row in receipt["cases"]) == 3
    assert receipt["collector"]["complete_native_id_scan"] is True
    assert receipt["guards"] == {"forward_calls": 0, "prompt_calls": 0, "step_calls": 0,
        "distribution_calls": 0, "paid_calls": 0, "closed_test_accessed": False, "model_weights_loaded": False}
    assert sum(call[0] == "inventory" for call in service.calls) == 1
    prompt_calls = [call for call in service.calls if call[0] == "tokenize" and call[2:]==(True, True)]
    assert len(prompt_calls) == 4
    assert all(sum(call[1] == text for call in prompt_calls) == 2 for text in service.prompt_ids)


def test_invalid_utf8_inputs_never_reach_synthetic_tokenizer():
    _, protocol, _ = input_data()
    service = SyntheticService(protocol)
    receipt, _ = collect(service)
    rows = [row for row in receipt["cases"] if row["id"].startswith("invalid_utf8")]
    assert len(rows) == 3 and all(row["tokenizer_calls"] == 0 for row in rows)
    assert all("\ufffd" not in call[1] for call in service.calls if call[0] == "tokenize")


def test_literal_chatml_uses_output_flags_and_prompts_use_special_flags():
    _, protocol, _ = input_data()
    service = SyntheticService(protocol)
    collect(service)
    literal = dict(preflight.TEXT_FIXTURES)["chatml_literal"]
    matching = [call for call in service.calls if call[0] == "tokenize" and call[1] == literal]
    assert matching and all(call[2:] == (False, False) for call in matching)
    assert all(call[2:] == (True, True) for call in service.calls if call[0] == "tokenize" and call[1] in service.prompt_ids)


def test_incremental_callbacks_contain_inventory_then_all_complete_cases_and_prompts():
    events = []
    receipt, inventory = collect(on_record=lambda kind, name, value: events.append((kind, name, copy.deepcopy(value))))
    completed = [(kind, name) for kind, name, _ in events if kind in {"inventory", "case", "prompt"}]
    assert completed == ([('inventory', 'inventory')]
        + [("case", spec["id"]) for spec in preflight.fixed_fixtures()]
        + [("prompt", name) for name in ("station", "library")])
    assert next(value for kind, _, value in events if kind == "inventory") == inventory
    assert [value for kind, _, value in events if kind == "case"] == receipt["cases"]


@pytest.mark.parametrize("where", ["inventory", "tokenize_ascii", "render_ascii", "prompt_station"])
def test_backend_failure_preserves_prior_callbacks_and_is_never_retried(where):
    _, protocol, _ = input_data()
    events, triggers = [], []
    class FailureService(SyntheticService):
        def inventory(self):
            if where == "inventory":
                triggers.append(where); raise OSError("synthetic inventory fault")
            return super().inventory()
        def tokenize(self, text, add_special=False, parse_special=False):
            if ((where == "tokenize_ascii" and text == "Hello world.")
                    or (where == "prompt_station" and text == preflight.chatml_prompt(protocol, "station"))):
                triggers.append(where); raise OSError("synthetic tokenizer fault")
            return super().tokenize(text, add_special, parse_special)
        def render(self, ids):
            if where == "render_ascii" and tuple(ids) == tuple(self.fixture_ids["Hello world."]):
                triggers.append(where); raise OSError("synthetic render fault")
            return super().render(ids)
    service = FailureService(protocol)
    with pytest.raises(OSError, match="synthetic"):
        collect(service, lambda kind, name, value: events.append((kind, name, copy.deepcopy(value))))
    assert triggers == [where]
    completed = [(kind, name) for kind, name, _ in events if kind in {"inventory", "case", "prompt"}]
    if where == "inventory": assert completed == []
    elif where == "prompt_station":
        assert len(completed) == 16
        assert completed[-1] == ("case", "qwen_known_merge")
    else:
        assert completed == [("inventory", "inventory"), ("case", "empty")]
    assert not any(kind == "prompt" for kind, _, _ in events)


def test_publication_failure_does_not_repeat_service_or_recollect():
    _, protocol, _ = input_data()
    service = SyntheticService(protocol)
    events = []
    def fail_store(kind, name, value):
        events.append((kind, name))
        if kind == "case" and name == "empty": raise OSError("synthetic persistent store fault")
    with pytest.raises(OSError, match="persistent store"):
        collect(service, fail_store)
    assert sum(call[0] == "inventory" for call in service.calls) == 1
    assert not any(call[0] == "tokenize" and call[1] == "Hello world." for call in service.calls)
    assert events[-1] == ("case", "empty")


def test_partial_utf8_prefix_is_recorded_without_replacement_tokenization():
    _, protocol, _ = input_data()
    service = SyntheticService(protocol)
    service.fixture_ids["😀"] = [600, 601]
    service.rendered[(600,)] = b"\xf0"
    service.rendered[(600, 601)] = "😀".encode()
    receipt, _ = collect(service)
    row = next(row for row in receipt["cases"] if row["id"] == "emoji")
    assert row["gate_reasons"] == ["invalid_utf8", "invalid_prefix"]
    assert row["prefixes"][1]["reencoded_ids"] is None
    assert row["prefixes"][1]["strict_utf8"] is False
    assert all("\ufffd" not in call[1] for call in service.calls if call[0] == "tokenize")


def test_incomplete_inventory_stops_before_any_case_backend_call():
    _, protocol, _ = input_data()
    class Partial(SyntheticService):
        def inventory(self): return super().inventory()[:-1]
    service = Partial(protocol)
    emitted = []
    with pytest.raises(preflight.PreflightError, match="every native token"):
        collect(service, lambda *args: emitted.append(args))
    assert service.calls == [("inventory",)]
    assert [(kind, name) for kind, name, _ in emitted] == [("inventory_started", "inventory")]


def test_final_consistency_failure_keeps_all_collected_outcomes():
    _, protocol, _ = input_data()
    service = SyntheticService(protocol)
    service.fixture_ids["**\n\n"] = [334, 271]  # Deliberately contradicts the known Qwen merge fixture.
    emitted = []
    with pytest.raises(preflight.PreflightError, match="known Qwen merge"):
        collect(service, lambda kind, name, value: emitted.append((kind, name, copy.deepcopy(value))))
    assert sum(kind == "case" for kind, _, _ in emitted) == 15
    assert sum(kind == "prompt" for kind, _, _ in emitted) == 2
    assert sum(call[0] == "inventory" for call in service.calls) == 1


def test_bad_fixed_list_rejected_before_any_service_method():
    from synsteg_audit.framed_interval_llm_v1_vocab_collect import collect_evidence
    raw, protocol, resources = input_data()
    plan = json.loads(raw); plan["fixtures"].pop()
    service = SyntheticService(protocol)
    with pytest.raises(ValueError, match="exact fixed15"):
        collect_evidence(service, preflight.canonical_json(plan), protocol, resources, "c"*64,
                         "results/synthetic/inventory.json")
    assert service.calls == []


def test_collection_controller_preserves_full_denominators_on_fake_transport_failure(tmp_path):
    from synsteg_audit import framed_interval_llm_v1_vocab_collect as collector
    plan, protocol, _ = input_data()
    activation = {"binary": {"path": "fake-vocab-bridge"}, "timeout_seconds": 120}
    report = {"frozen_resource_identity_sha256": "a"*64,
              "verified_file_identities": {"runtime/llama-b11791/libllama.so": {"sha256": "b"*64}}}
    calls = []
    def fail_transport(*args):
        calls.append(args); raise OSError("synthetic pre-READY failure")
    result = collector.run_collection(tmp_path, "fake-activation", "d"*64,
        "results/framed-interval-llm-v1-tokenizer-preflight/fake", collector.INTENT,
        activation_loader=lambda *args: (activation, report, plan, protocol, "c"*64),
        transport_factory=fail_transport,
        resident_verifier=lambda *args: (_ for _ in ()).throw(AssertionError("resident call reached")))
    assert len(calls) == 1 and result["state"] == "failed_no_retry"
    assert len(result["fixtures"]) == result["planned_fixtures"] == 15
    assert len(result["prompts"]) == result["planned_prompts"] == 2
    assert all(row["status"] == "not_attempted_after_failure" for row in result["fixtures"])
    assert result["retry_permitted"] is False
    assert not (tmp_path / "results/framed-interval-llm-v1-tokenizer-preflight/fake/receipt.json").exists()
    with pytest.raises(FileExistsError):
        collector.run_collection(tmp_path, "fake-activation", "d"*64,
            "results/framed-interval-llm-v1-tokenizer-preflight/fake", collector.INTENT,
            activation_loader=lambda *args: (activation, report, plan, protocol, "c"*64),
            transport_factory=fail_transport, resident_verifier=lambda *args: None)
    assert len(calls) == 1


@pytest.mark.parametrize("failed_stage", ["ascii", "station"])
def test_collection_controller_distinguishes_started_failure_from_unattempted(tmp_path, monkeypatch, failed_stage):
    from synsteg_audit import framed_interval_llm_v1_vocab_collect as collector
    plan, protocol, _ = input_data()
    activation = {"binary": {"path": "fake-vocab-bridge"}, "timeout_seconds": 120}
    report = {"frozen_resource_identity_sha256": "a"*64,
              "verified_file_identities": {"runtime/llama-b11791/libllama.so": {"sha256": "b"*64}}}
    failures, closes = [], []
    class FailureService(SyntheticService):
        commands = []
        def tokenize(self, text, add_special=False, parse_special=False):
            target = "Hello world." if failed_stage == "ascii" else preflight.chatml_prompt(protocol, "station")
            if text == target:
                failures.append(failed_stage); raise OSError("synthetic active-stage failure")
            return super().tokenize(text, add_special, parse_special)
    service = FailureService(protocol)
    monkeypatch.setattr(collector, "VocabClient", lambda transport: service)
    class FakeTransport:
        process = SimpleNamespace(pid=1234567)
        def __init__(self, *args): pass
        def close(self): closes.append(True)
    result = collector.run_collection(tmp_path, "fake-activation", "d"*64,
        "results/framed-interval-llm-v1-tokenizer-preflight/partial", collector.INTENT,
        activation_loader=lambda *args: (activation, report, plan, protocol, "c"*64),
        transport_factory=FakeTransport, resident_verifier=lambda *args: {"status": "synthetic"})
    assert failures == [failed_stage] and closes == [True]
    assert result["state"] == "failed_no_retry"
    assert len(result["fixtures"]) == 15 and len(result["prompts"]) == 2
    statuses = {row["id"]: row["status"] for row in result["fixtures"]}
    if failed_stage == "ascii":
        assert statuses["empty"] == "observed"
        assert statuses["ascii"] == "failed_unknown"
        assert all(value == "not_attempted_after_failure" for key, value in statuses.items() if key not in {"empty", "ascii"})
        assert all(row["status"] == "not_attempted_after_failure" for row in result["prompts"].values())
    else:
        assert all(value in {"observed", "invalid_utf8"} for value in statuses.values())
        assert result["prompts"]["station"]["status"] == "failed_unknown"
        assert result["prompts"]["library"]["status"] == "not_attempted_after_failure"
    output = tmp_path / "results/framed-interval-llm-v1-tokenizer-preflight/partial"
    assert (output / "inventory.json").is_file()
    assert (output / "case-empty.json").is_file()
    assert not (output / "receipt.json").exists()


@pytest.mark.parametrize("failure", ["finish", "close", None])
def test_passed_receipt_published_only_after_finish_and_close(tmp_path, monkeypatch, failure):
    from synsteg_audit import framed_interval_llm_v1_vocab_collect as collector
    plan, protocol, _ = input_data()
    activation = {"binary": {"path": "fake-vocab-bridge"}, "timeout_seconds": 120}
    report = {"frozen_resource_identity_sha256": "a"*64,
              "verified_file_identities": {"runtime/llama-b11791/libllama.so": {"sha256": "b"*64}}}
    output = tmp_path / "results/framed-interval-llm-v1-tokenizer-preflight/close"
    order = []
    service = SyntheticService(protocol)
    service.commands = []
    service.quit = lambda: order.append("quit")
    monkeypatch.setattr(collector, "VocabClient", lambda transport: service)
    class FakeTransport:
        process = SimpleNamespace(pid=1234567)
        def __init__(self, *args): pass
        def finish(self):
            order.append("finish")
            assert not (output / "receipt.json").exists()
            if failure == "finish": raise OSError("synthetic nonzero native exit")
        def close(self):
            order.append("close")
            assert not (output / "receipt.json").exists()
            if failure == "close": raise OSError("synthetic close failure")
    result = collector.run_collection(tmp_path, "fake", "d"*64,
        output.relative_to(tmp_path).as_posix(), collector.INTENT,
        activation_loader=lambda *args: (activation, report, plan, protocol, "c"*64),
        transport_factory=FakeTransport, resident_verifier=lambda *args: {"status": "synthetic"})
    assert order == ["quit", "finish", "close"]
    assert (output / "receipt.json").exists() is (failure is None)
    assert result["state"] == ("passed" if failure is None else "failed_no_retry")
