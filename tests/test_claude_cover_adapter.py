"""No network: wire serialization, failure retention and credential boundaries."""
import importlib.util
import io
import json
from pathlib import Path
import urllib.error

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('claude_cover_adapter_test', ROOT / 'scripts/claude_cover_adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class Reply(io.BytesIO):
    def __init__(self, value, status=200):
        super().__init__(json.dumps(value).encode())
        self.status = status
        self.headers = {'request-id': 'request-fixture'}


def message(model='claude-sonnet-5-5', stop='end_turn'):
    return {'id': 'msg_fixture', 'type': 'message', 'role': 'assistant', 'model': model,
            'content': [{'type': 'thinking', 'thinking': 'fixture not visible'},
                        {'type': 'text', 'text': 'A clear '}, {'type': 'text', 'text': 'cover.'}],
            'usage': {'input_tokens': 20, 'output_tokens': 12}, 'stop_reason': stop}


def test_request_has_only_supported_fields():
    request = adapter.build_request('claude-sonnet-5-5', 'Describe a garden.', 384)
    assert set(request) == {'model', 'max_tokens', 'system', 'messages'}
    assert not {'seed', 'temperature', 'top_p', 'top_k'} & request.keys()
    assert request['messages'] == [{'role': 'user', 'content': 'Describe a garden.'}]


@pytest.mark.parametrize('model', ['claude-sonnet-latest', 'claude-sonnet-4-5', '', 'arbitrary/model'])
def test_unknown_or_alias_model_rejected(model):
    with pytest.raises(ValueError):
        adapter.build_request(model, 'A cover.')


def test_transport_serializes_exact_endpoint_and_version_without_key_persistence():
    calls = []
    key = 'fixture-secret-key'
    def opener(req, timeout):
        calls.append(req)
        assert req.full_url == 'https://api.anthropic.com/v1/messages'
        assert req.get_header('Anthropic-version') == '2026-10-21'
        assert req.get_header('X-api-key') == key
        assert 'seed' not in json.loads(req.data)
        return Reply(message())
    reply = adapter.transport('POST', '/v1/messages', adapter.build_request('claude-sonnet-5-5', 'A cover.'), key, opener=opener)
    assert len(calls) == 1
    assert key not in json.dumps(reply)
    cover = adapter.extract_cover(reply, 'claude-sonnet-5-5')
    assert cover['text'] == 'A clear cover.'
    assert cover['metadata']['response_id'] == 'msg_fixture'
    assert cover['metadata']['usage']['output_tokens'] == 12


def test_missing_key_never_calls_network_and_retains_readiness_failure(tmp_path):
    def forbidden(*args):
        raise AssertionError('network must not be called')
    path = tmp_path / 'missing.json'
    assert adapter.run('preflight', 'claude-sonnet-5-5', path, environ={}, sender=forbidden) == 2
    record = json.loads(path.read_text())
    assert record['status'] == 'unavailable_missing_api_key'
    assert record['generation_request_sent'] is False


def test_dry_run_checks_presence_without_network(tmp_path):
    def forbidden(*args):
        raise AssertionError('network must not be called')
    path = tmp_path / 'dry.json'
    assert adapter.run('dry-run', 'claude-sonnet-5-5', path, 'A garden.', environ={'ANTHROPIC_API_KEY': 'fixture'}, sender=forbidden) == 0
    record = json.loads(path.read_text())
    assert record['status'] == 'request_prepared_not_sent'
    assert record['generation_request_sent'] is False


def test_preflight_is_get_only_and_checks_returned_model(tmp_path):
    calls = []
    def sender(method, path, body, key):
        calls.append((method, path, body))
        return {'status': 'http_success', 'raw_response_json': {'id': 'claude-sonnet-5-5'}}
    path = tmp_path / 'preflight.json'
    assert adapter.run('preflight', 'claude-sonnet-5-5', path, environ={'ANTHROPIC_API_KEY': 'fixture'}, sender=sender) == 0
    assert calls == [('GET', '/v1/models/claude-sonnet-5-5', None)]
    assert json.loads(path.read_text())['generation_request_sent'] is False


def test_generation_requires_explicit_flag(tmp_path):
    def forbidden(*args):
        raise AssertionError('network must not be called')
    path = tmp_path / 'blocked.json'
    assert adapter.run('generate', 'claude-sonnet-5-5', path, 'A cover.', environ={'ANTHROPIC_API_KEY': 'fixture'}, sender=forbidden) == 2
    assert json.loads(path.read_text())['status'] == 'explicit_execution_flag_required'


def test_existing_attempt_is_never_overwritten(tmp_path):
    path = tmp_path / 'attempt.json'
    path.write_text('previous interrupted attempt')
    with pytest.raises(FileExistsError):
        adapter.run('preflight', 'claude-sonnet-5-5', path, environ={})
    assert path.read_text() == 'previous interrupted attempt'


def test_http_failure_retained_once_no_retry_and_echoed_secret_redacted():
    calls = []
    def opener(req, timeout):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, 401, 'unauthorized', {},
            io.BytesIO(b'{"error":{"message":"fixture-secret-key denied"}}'))
    reply = adapter.transport('POST', '/v1/messages', {}, 'fixture-secret-key', opener=opener)
    assert len(calls) == 1
    assert reply['http_status'] == 401
    assert reply['status'] == 'http_error'
    assert reply['credential_redacted_from_response'] is True
    assert 'fixture-secret-key' not in json.dumps(reply)


def test_transport_error_does_not_copy_exception_secrets():
    def opener(req, timeout):
        raise urllib.error.URLError('fixture-secret-key should not appear')
    reply = adapter.transport('POST', '/v1/messages', {}, 'fixture-secret-key', opener=opener)
    assert reply['status'] == 'transport_error'
    assert 'fixture-secret-key' not in json.dumps(reply)


def test_prompt_cannot_persist_configured_credential(tmp_path):
    path = tmp_path / 'leak.json'
    with pytest.raises(adapter.AdapterError):
        adapter.run('dry-run', 'claude-sonnet-5-5', path, 'Secret fixture-secret-key.',
                    environ={'ANTHROPIC_API_KEY': 'fixture-secret-key'})
    assert not path.exists()


@pytest.mark.parametrize('invalid_key', ['bad\nkey', 123])
def test_invalid_header_is_not_reported_as_sent(tmp_path, invalid_key):
    path = tmp_path / 'invalid.json'
    assert adapter.run('preflight', 'claude-sonnet-5-5', path,
                       environ={'ANTHROPIC_API_KEY': invalid_key}) == 2
    data = json.loads(path.read_text())
    assert data['status'] == 'unavailable_invalid_api_key'
    assert data['generation_request_sent'] is False


def test_redirects_are_not_followed():
    request = adapter.urllib.request.Request('https://api.anthropic.com/v1/messages')
    assert adapter._NoRedirect().redirect_request(request, None, 302, 'redirect', {},
                                                   'https://untrusted.invalid') is None


def test_max_token_stop_remains_visible_not_silent_success_rewrite():
    reply = {'status': 'http_success', 'raw_response_json': message(stop='max_tokens'), 'response_sha256': 'hash'}
    cover = adapter.extract_cover(reply, 'claude-sonnet-5-5')
    assert cover['metadata']['hit_output_limit'] is True
    assert cover['metadata']['stop_reason'] == 'max_tokens'


@pytest.mark.parametrize('mutation', ['model', 'id', 'usage', 'text'])
def test_invalid_provider_response_rejected(mutation):
    m = message()
    if mutation == 'model': m['model'] = 'different-model'
    if mutation == 'id': del m['id']
    if mutation == 'usage': m['usage']['output_tokens'] = -1
    if mutation == 'text': m['content'] = [{'type': 'thinking', 'thinking': 'hidden'}]
    with pytest.raises(adapter.AdapterError):
        adapter.extract_cover({'status': 'http_success', 'raw_response_json': m}, 'claude-sonnet-5-5')
