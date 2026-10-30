"""No model, network or production generations: contract and process cleanup tests."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_qwen15b_gguf_v1 as prep
import run_qwen15b_gguf_v1 as run


def parent_inputs():
    return (prep.read(ROOT / prep.PARENT / 'plan.json')['records'],
            prep.rows(ROOT / prep.PARENT / 'prompts.jsonl'), prep.rows(ROOT / prep.SOURCE))


def fixture():
    files = prep.build(ROOT)
    plan = json.loads(files['plan.json'])['records']
    protocol = json.loads(files['protocol.json'])
    prompt = json.loads(files['prompts.jsonl'].splitlines()[0])
    row = next(r for r in plan if r['planned_action'] == 'generate_once')
    return row, prompt, protocol


class FakeClient:
    def __init__(self, directory, prompt, count=100, error=None, content='A supported summary.'):
        self.directory, self.count, self.error, self.content = directory, count, error, content
        self.calls = []
        self.rendered = ''.join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'
                                for m in prompt['messages']) + '<|im_start|>assistant\n'

    def request(self, endpoint, payload, timeout, label):
        self.calls.append((endpoint, payload))
        if endpoint == '/apply-template':
            return {'prompt': self.rendered}
        if endpoint == '/tokenize':
            return {'tokens': [1] * self.count}
        marker = prep.read(self.directory / 'attempt.json')
        assert marker['generation_request_attempted'] is True
        assert marker['status'] == 'generation_request_started'
        if self.error:
            raise self.error
        return {'content': self.content, 'tokens_predicted': 4, 'tokens_evaluated': self.count, 'tokens': [1, 2, 3, 4],
                'stop_type': 'eos', 'stopped_eos': True, 'truncated': False}


def test_frozen_selection_and_full_denominator():
    files = prep.build(ROOT)
    plan = json.loads(files['plan.json'])['records']
    assert [r['plan_index'] for r in plan if r['planned_action'] == 'generate_once'] == [2, 5, 6, 7, 8, 9, 10, 11]
    assert run.summarize(run.initial_ledger(plan)) == {
        'planned_source_groups': 20, 'selected_generation_denominator': 8, 'source_gates': 5,
        'not_selected': 7, 'status_counts_all20': {'abstain_source_gate': 5, 'not_started': 8, 'not_selected_fixed_eight_budget': 7},
        'generation_requests_attempted': 0, 'available_outputs_including_empty': 0,
        'within_soft_word_band': 0, 'factual_quality_reviewed': 0, 'test_data_used': False, 'paid_calls': 0}


@pytest.mark.parametrize('change', ['test_split', 'order', 'duplicate', 'source_text', 'prompt_text'])
def test_reject_parent_tampering(change):
    plan, prompts, windows = parent_inputs()
    if change == 'test_split': plan[0]['split'] = 'test'
    elif change == 'order': plan[0], plan[1] = plan[1], plan[0]
    elif change == 'duplicate': plan[1]['record_id'] = plan[0]['record_id']
    elif change == 'source_text': windows[0]['source_window'] += ' invented fact'
    else: prompts[0]['prompt'] += ' fabricate a date'
    with pytest.raises(ValueError):
        prep.validate_parent(plan, prompts, windows)


def test_token_gate_reserves_whole_output_and_does_not_replace(tmp_path):
    row, prompt, protocol = fixture()
    limit = protocol['context_tokens'] - protocol['max_output_tokens'] - protocol['reserved_tokens']
    assert run.token_budget_ok(limit, protocol)
    assert not run.token_budget_ok(limit + 1, protocol)
    client = FakeClient(tmp_path, prompt, count=limit + 1)
    result = run.perform_request(row, prompt, protocol, client, tmp_path)
    assert result['status'] == 'context_budget_reject'
    assert not result['generation_request_attempted']
    assert [c[0] for c in client.calls] == ['/apply-template', '/tokenize']


@pytest.mark.parametrize('error,status', [(TimeoutError('synthetic timeout'), 'timeout'),
                                          (RuntimeError('synthetic failure'), 'request_error')])
def test_completion_error_is_logged_once_without_retry(tmp_path, error, status):
    row, prompt, protocol = fixture()
    client = FakeClient(tmp_path, prompt, error=error)
    result = run.perform_request(row, prompt, protocol, client, tmp_path)
    assert result['status'] == status and result['generation_request_attempted']
    assert len([c for c in client.calls if c[0] == '/completion']) == 1
    assert prep.read(tmp_path / 'attempt.json') == result


def test_completed_request_uses_exact_tokens_and_pinned_sampling(tmp_path):
    row, prompt, protocol = fixture()
    client = FakeClient(tmp_path, prompt)
    result = run.perform_request(row, prompt, protocol, client, tmp_path)
    payload = client.calls[-1][1]
    assert result['status'] == 'generated'
    assert payload['prompt'] == [1] * 100 and payload['n_predict'] == 1536
    assert payload['seed'] == 20261026 and payload['cache_prompt'] is False
    assert payload['samplers'] == ['penalties', 'temperature', 'top_k', 'top_p']
    assert result['factual_quality'] == 'not_reviewed'


def test_empty_and_truncated_outputs_are_not_silent_success(tmp_path):
    row, prompt, protocol = fixture()
    result = run.perform_request(row, prompt, protocol, FakeClient(tmp_path, prompt, content=' \n'), tmp_path)
    assert result['status'] == 'empty_output' and result['output_words'] == 0
    assert not result['within_soft_word_band']


def test_unexpected_template_fails_before_generation(tmp_path):
    row, prompt, protocol = fixture()
    client = FakeClient(tmp_path, prompt)
    client.rendered = 'different system<|im_start|>assistant\n'
    result = run.perform_request(row, prompt, protocol, client, tmp_path)
    assert result['status'] == 'request_error' and not result['generation_request_attempted']
    assert len(client.calls) == 1


def test_expired_record_deadline_makes_no_http_call(tmp_path):
    row, prompt, protocol = fixture()
    client = FakeClient(tmp_path, prompt)
    result = run.perform_request(row, prompt, protocol, client, tmp_path, deadline=0)
    assert result['status'] == 'timeout' and not client.calls


def test_owned_process_group_is_killed():
    # This sleeps without compute; verifies real cleanup, not model behavior.
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        run.terminate_group(process)
        assert process.poll() is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_hard_wall_deadline_interrupts_work_independently_of_socket():
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        with run.hard_deadline(.03):
            time.sleep(.2)
    assert time.monotonic() - started < .15


def test_late_completion_cannot_be_marked_generated(tmp_path):
    row, prompt, protocol = fixture()
    protocol['runtime']['per_record_timeout_seconds'] = .02
    class SlowClient(FakeClient):
        def request(self, endpoint, payload, timeout, label):
            if endpoint == '/completion': time.sleep(.04)
            return super().request(endpoint, payload, timeout, label)
    result = run.perform_request(row, prompt, protocol, SlowClient(tmp_path, prompt), tmp_path)
    assert result['status'] == 'timeout' and result['generation_request_attempted']


def test_missing_native_metadata_is_explicit_uncertainty():
    row, _, _ = fixture()
    result = run.completion_summary({'content': 'Text without metadata.'}, row)
    assert result['status'] == 'response_schema_uncertain' and result['schema_issues']


def test_redirects_forbidden_without_network():
    with pytest.raises(ValueError):
        run.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com')


def test_sigterm_runs_cleanup_and_restores_handler(monkeypatch):
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted_execution(*args):
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            run.terminate_group(process)
    monkeypatch.setattr(run, '_execute', interrupted_execution)
    with pytest.raises(KeyboardInterrupt):
        run.execute(ROOT, Path('unused'), 'a' * 40)
    assert process.poll() is not None
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize('tokens', [None, '1 2', [1, -1], [True, 2], [1, 1.5]])
def test_bad_returned_tokens_are_explicit_uncertainty(tokens):
    row, _, _ = fixture()
    response = {'content': 'Text.', 'tokens_predicted': 2, 'tokens_evaluated': 100,
                'truncated': False, 'stop_type': 'eos', 'tokens': tokens}
    result = run.completion_summary(response, row)
    assert result['status'] == 'response_schema_uncertain'
    assert 'missing_or_invalid_returned_tokens' in result['schema_issues']


def test_contradictory_native_stop_fields_are_uncertain():
    row, _, _ = fixture()
    response = {'content': 'Text.', 'tokens_predicted': 2, 'tokens_evaluated': 100,
                'truncated': False, 'stop_type': 'eos', 'stopped_limit': True, 'tokens': [1, 2]}
    result = run.completion_summary(response, row)
    assert result['status'] == 'response_schema_uncertain'
    assert 'contradictory_stop_metadata' in result['schema_issues']
