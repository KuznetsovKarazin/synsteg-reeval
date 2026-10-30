import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys

import pytest

from synsteg_audit.budget_ledger import (BudgetError, CampaignLedger, UNIT,
    atomic_json, money, token_cost)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_budget_generation as runner

KEY = "test-private-credential-budget-12345"
MODEL = "claude-haiku-5-5"


@pytest.fixture
def prepared(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    rows = [{"record_id": "record-" + str(i), "source_group_id": "source-" + str(i),
             "split": "development", "topic": "science", "genre": "explanation",
             "target_words": 250, "prompt": "Explain a community project " + str(i)} for i in range(3)]
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    pricing = tmp_path / "pricing.json"
    pricing.write_text(json.dumps({"schema": "synsteg-generation-pricing-v1",
        "source_url": "https://www.anthropic.com/pricing", "checked_at": "2026-10-23",
        "models": {MODEL: {"input_usd_per_million": "0.5", "output_usd_per_million": "2.5",
        "input_context_tokens": 1_000_000, "max_output_tokens": 8192}}}))
    return {"prompts": prompts, "output": tmp_path / "output", "provider": "claude",
            "model": MODEL, "pricing": pricing, "campaign_ledger": tmp_path / "campaign",
            "max_records": 3, "max_output_tokens": 100, "budget_usd": "20",
            "environ": {"ANTHROPIC_API_KEY": KEY},
            "preflight_sender": lambda model, key: {"status": "model_access_verified_no_generation"}}


def good_response(body, key):
    return {"status": "http_success", "credential_redacted": False,
            "retained_response_sha256": "a" * 64,
            "retained_response": {"id": "msg_test", "type": "message", "role": "assistant",
            "model": body["model"], "usage": {"input_tokens": 80, "output_tokens": 90},
            "stop_reason": "end_turn", "content": [{"type": "text", "text": "A useful cover text."}]}}


def test_preview_has_no_files_key_read_or_requests(prepared):
    class NoEnvironment:
        def get(self, *args):
            raise AssertionError("Preview read credential")
    prepared["environ"] = NoEnvironment()
    code, report = runner.run(**prepared, sender=lambda *a: pytest.fail("request"))
    assert code == 0 and report["status"] == "preview_only_no_requests"
    assert not prepared["output"].exists() and not prepared["campaign_ledger"].exists()
    assert report["plan"]["reservation_per_attempt_usd"] == 0.50025


def test_success_resumes_without_network_and_preserves_planned_denominator(prepared):
    calls = []
    def send(body, key):
        calls.append(body)
        return good_response(body, key)
    code, report = runner.run(**prepared, execute=True, sender=send)
    assert code == 0 and report["completed"] == 3 and len(calls) == 3
    assert report["ledger"]["accounted_or_reserved_usd"] == pytest.approx(3 * 0.000265)
    run = json.loads((prepared["output"] / "run.json").read_text())
    assert len(run["selected_records"]) == run["selected_records_count"] == 3
    def forbidden(*args):
        pytest.fail("Resume contacted provider")
    prepared["preflight_sender"] = forbidden
    code, resumed = runner.run(**prepared, execute=True, sender=forbidden)
    assert code == 0 and resumed["resumed_completed"] == 3
    assert resumed["requests_attempted_this_invocation"] == 0
    assert len(list((prepared["output"] / "attempts").glob("*.json"))) == 3
    for path in (prepared["output"] / "attempts").glob("*.json"):
        assert json.loads(path.read_text())["response"]["retained_response"]["id"] == "msg_test"


def test_operational_batches_resume_same_full_plan(prepared):
    calls = []
    def send(body, key):
        calls.append(body)
        return good_response(body, key)
    code, first = runner.run(**prepared, execute=True, sender=send, max_new_records=1)
    assert code == 0 and first["status"] == "batch_completed"
    assert first["requests_attempted_this_invocation"] == 1 and first["completed"] == 1
    plan = (prepared["output"] / "run.json").read_bytes()
    assert json.loads(plan)["selected_records_count"] == 3
    assert "max_new_records" not in json.loads(plan)
    code, second = runner.run(**prepared, execute=True, sender=send, max_new_records=1)
    assert code == 0 and second["status"] == "batch_completed"
    assert second["requests_attempted_this_invocation"] == 1 and second["resumed_completed"] == 1
    assert second["completed"] == 2 and len(calls) == 2
    assert (prepared["output"] / "run.json").read_bytes() == plan
    code, final = runner.run(**prepared, execute=True, sender=send)
    assert code == 0 and final["status"] == "selected_records_completed"
    assert final["requests_attempted_this_invocation"] == 1 and final["resumed_completed"] == 2
    assert final["completed"] == 3 and len(calls) == 3
    assert (prepared["output"] / "run.json").read_bytes() == plan


def test_insufficient_reservation_sends_no_paid_request(prepared):
    prepared["budget_usd"] = "0.5"
    code, report = runner.run(**prepared, execute=True, sender=lambda *a: pytest.fail("paid request"))
    assert code == 3 and report["status"] == "stopped_before_request"
    assert report["ledger"]["attempt_count"] == 0
    assert json.loads((prepared["output"] / "run.json").read_text())["selected_records_count"] == 3


def test_actual_usage_releases_reservation_without_floating_point_overrun(prepared):
    prepared["budget_usd"] = "0.5006"
    code, report = runner.run(**prepared, execute=True, sender=good_response)
    assert code == 3 and report["completed"] == 2
    assert report["ledger"]["accounted_or_reserved_usd"] == 0.00053
    assert report["requests_attempted_this_invocation"] == 2


def test_unknown_transport_keeps_full_reservation_and_never_retries(prepared):
    prepared["max_new_records"] = 1
    calls = []
    def failure(body, key):
        calls.append(body)
        return {"status": "transport_error_outcome_unknown"}
    code, report = runner.run(**prepared, execute=True, sender=failure)
    assert code == 3 and report["ledger"]["accounted_or_reserved_usd"] == 0.50025
    code, resumed = runner.run(**prepared, execute=True, sender=failure)
    assert code == 3 and resumed["requests_attempted_this_invocation"] == 0
    assert len(calls) == 1
    # A new run directory does not create permission to repeat the paid call.
    prepared["output"] = prepared["output"].with_name("other-output")
    code, other = runner.run(**prepared, execute=True, sender=failure)
    assert code == 3 and len(calls) == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1601])
def test_invalid_batch_limit_refused(prepared, limit):
    with pytest.raises(BudgetError, match="max-new-records"):
        runner.run(**prepared, max_new_records=limit)


def test_keyboard_interrupt_stays_reserved_and_resume_does_not_send(prepared):
    def interrupted(*args):
        raise KeyboardInterrupt()
    code, report = runner.run(**prepared, execute=True, sender=interrupted)
    assert code == 130 and report["ledger"]["accounted_or_reserved_usd"] == 0.50025
    code, report = runner.run(**prepared, execute=True, sender=lambda *a: pytest.fail("retry"))
    assert code == 3 and report["requests_attempted_this_invocation"] == 0


def test_inaccessible_model_never_reserves_or_generates(prepared):
    prepared["preflight_sender"] = lambda *a: {"status": "http_error", "http_status": 404}
    code, report = runner.run(**prepared, execute=True, sender=lambda *a: pytest.fail("paid request"))
    assert code == 4 and report["status"] == "model_access_unverified_no_generation"
    assert report["ledger"]["attempt_count"] == 0


def test_fail_closed_unknown_usage_category(prepared):
    def send(body, key):
        response = good_response(body, key)
        response["retained_response"]["usage"]["cache_creation_input_tokens"] = 10
        return response
    code, report = runner.run(**prepared, execute=True, sender=send)
    assert code == 3 and report["ledger"]["unknown_or_failed_attempts"] == 1
    assert report["ledger"]["accounted_or_reserved_usd"] == 0.50025


def test_zero_cache_categories_and_standard_tier_are_valid(prepared):
    def send(body, key):
        response = good_response(body, key)
        response["retained_response"]["usage"].update(cache_creation={"ephemeral_5m_input_tokens": 0},
            cache_read_input_tokens=0, service_tier="standard")
        return response
    assert runner.run(**prepared, execute=True, sender=send)[0] == 0


def test_wrong_returned_model_stops_without_retry(prepared):
    def send(body, key):
        response = good_response(body, key)
        response["retained_response"]["model"] = "different-model"
        return response
    code, report = runner.run(**prepared, execute=True, sender=send)
    assert code == 3 and report["requests_attempted_this_invocation"] == 1


def test_changed_run_configuration_refused(prepared):
    runner.run(**prepared, execute=True, sender=good_response)
    prepared["max_output_tokens"] = 101
    with pytest.raises(BudgetError, match="Existing run differs"):
        runner.run(**prepared, execute=True, sender=lambda *a: pytest.fail("request"))


@pytest.mark.parametrize("changed", ["run_claude_smoke.py", "check_claude_access.py"])
def test_changed_transport_code_refuses_resume_before_network(prepared, monkeypatch, changed):
    runner.run(**prepared, execute=True, sender=good_response)
    original = runner.file_hash
    monkeypatch.setattr(runner, "file_hash", lambda path: "f" * 64 if Path(path).name == changed else original(path))
    with pytest.raises(BudgetError, match="Existing run differs"):
        runner.run(**prepared, execute=True, sender=lambda *a: pytest.fail("request"))


def test_non_development_requires_freeze_even_for_preview(prepared):
    content = prepared["prompts"].read_text().replace('"development"', '"test"')
    prepared["prompts"].write_text(content)
    prepared["split"] = "test"
    with pytest.raises(BudgetError, match="frozen manifest"):
        runner.run(**prepared)


def test_duplicate_source_groups_rejected(prepared):
    rows = [json.loads(line) for line in prepared["prompts"].read_text().splitlines()]
    rows[1]["source_group_id"] = rows[0]["source_group_id"]
    prepared["prompts"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(BudgetError, match="Duplicate"):
        runner.run(**prepared)


def test_credential_prompt_and_exception_text_never_persisted(prepared):
    def error(*args):
        raise RuntimeError("header x-api-key: " + KEY)
    code, report = runner.run(**prepared, execute=True, sender=error)
    assert code == 3
    for path in prepared["output"].rglob("*.json"):
        assert KEY not in path.read_text()
    assert KEY not in (prepared["campaign_ledger"] / "ledger.json").read_text()


def test_key_in_prompt_rejected_before_files(prepared):
    prepared["prompts"].write_text(prepared["prompts"].read_text().replace("Explain", KEY))
    with pytest.raises(BudgetError, match="Credential"):
        runner.run(**prepared, execute=True, sender=good_response)
    assert not prepared["output"].exists() and not prepared["campaign_ledger"].exists()


def _contend_lock(path, queue):
    try:
        with CampaignLedger(path).locked():
            queue.put("acquired")
    except BudgetError:
        queue.put("blocked")


def test_cross_process_lock_serializes_campaign(tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    ledger = CampaignLedger(tmp_path / "ledger")
    with ledger.locked():
        child = context.Process(target=_contend_lock, args=(ledger.directory, queue))
        child.start()
        child.join(10)
        assert child.exitcode == 0 and queue.get(timeout=1) == "blocked"
    with ledger.locked():
        pass


def test_crash_reservation_blocks_later_distinct_record(tmp_path):
    ledger = CampaignLedger(tmp_path)
    with ledger.locked():
        ledger.reserve("one", "fingerprint-one", money("1"), {})
        with pytest.raises(BudgetError, match="failed/unknown"):
            ledger.reserve("two", "fingerprint-two", money("1"), {})
        same, created = ledger.reserve("one", "fingerprint-one", money("1"), {})
        assert not created and same["status"] == "reserved_outcome_unknown"


def test_bound_violation_persists_real_amount_and_blocks_campaign(tmp_path):
    ledger = CampaignLedger(tmp_path)
    with ledger.locked():
        ledger.reserve("one", "fp", money("1"), {})
        with pytest.raises(BudgetError, match="exceeded"):
            ledger.finish("one", {"text": "test"}, money("21"))
        assert ledger.summary()["accounted_or_reserved_usd"] == 21
        with pytest.raises(BudgetError, match="blocked"):
            ledger.reserve("two", "fp2", 0, {})


@pytest.mark.parametrize("budget", ["20.00001", "0", "NaN", "Infinity", "-1"])
def test_budget_invalid_or_above_twenty_refused(tmp_path, budget):
    with pytest.raises(BudgetError):
        CampaignLedger(tmp_path, budget)


def test_nanodollar_cost_rounds_up_without_float_loss():
    assert token_cost(1, 0, "0.0001", "1") == 1
    assert money("0.1") + money("0.2") == money("0.3")


def test_local_generator_result_and_cached_revision_contract(prepared):
    prepared.update(provider="local", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision=runner.PINNED_LOCAL["TinyLlama/TinyLlama-1.1B-Chat-v1.0"], pricing=None,
        campaign_ledger=None, max_output_tokens=1024,
        runtime_reader=lambda: {"schema": "synthetic-test-runtime", "packages": {"torch": "fixture"}})
    calls = []
    def generate(row, limit):
        calls.append(row["record_id"])
        return {"cover_text": "Local cover.", "stop_reason": "end_turn", "hit_output_limit": False,
                "usage": {"input_tokens": 100, "output_tokens": 50}, "seed": 1}
    code, report = runner.run(**prepared, execute=True, local_generator=generate)
    assert code == 0 and len(calls) == 3
    assert report["ledger"]["accounted_or_reserved_usd"] == 0
    code, report = runner.run(**prepared, execute=True, local_generator=lambda *a: pytest.fail("repeat"))
    assert code == 0 and report["resumed_completed"] == 3


def test_local_runtime_change_refuses_resume_before_generator_or_reservation(prepared):
    prepared.update(provider="local", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision=runner.PINNED_LOCAL["TinyLlama/TinyLlama-1.1B-Chat-v1.0"], pricing=None,
        campaign_ledger=None, max_output_tokens=1024)
    environment = {"schema": "synthetic-test-runtime", "python": "fixture-python",
                   "platform": "fixture-platform", "packages": {"torch": "1.0"}}
    prepared["runtime_reader"] = lambda: environment
    def generate(*args):
        return {"cover_text": "Local text", "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 2}}
    runner.run(**prepared, execute=True, local_generator=generate)
    ledger_file = prepared["output"] / "local_ledger" / "ledger.json"
    before = ledger_file.read_bytes()
    saved = json.loads((prepared["output"] / "runtime.json").read_text())
    assert saved == environment
    environment["packages"]["torch"] = "2.0"
    with pytest.raises(BudgetError, match="runtime differs"):
        runner.run(**prepared, execute=True, local_generator=lambda *a: pytest.fail("generator called"))
    assert ledger_file.read_bytes() == before
    assert json.loads((prepared["output"] / "runtime.json").read_text()) == saved


def test_local_preview_does_not_require_runtime_dependencies(prepared):
    prepared.update(provider="local", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision=runner.PINNED_LOCAL["TinyLlama/TinyLlama-1.1B-Chat-v1.0"], pricing=None,
        campaign_ledger=None, max_output_tokens=1024,
        runtime_reader=lambda: pytest.fail("Preview read local dependency versions"))
    code, report = runner.run(**prepared)
    assert code == 0 and "runtime" not in report["plan"]
    assert not prepared["output"].exists()


def test_missing_local_dependency_fails_before_reservation(prepared, monkeypatch):
    prepared.update(provider="local", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision=runner.PINNED_LOCAL["TinyLlama/TinyLlama-1.1B-Chat-v1.0"], pricing=None,
        campaign_ledger=None, max_output_tokens=1024)
    def missing(name):
        raise runner.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(runner.importlib.metadata, "version", missing)
    with pytest.raises(BudgetError, match="dependencies are missing"):
        runner.run(**prepared, execute=True, local_generator=lambda *a: pytest.fail("generator called"))
    assert not (prepared["output"] / "local_ledger" / "ledger.json").exists()
    assert not (prepared["output"] / "runtime.json").exists()
