#!/usr/bin/env python3
"""Explicit, single-attempt Claude cover generation and no-generation readiness.

No retries, hidden sampling seed, SDK defaults, browser-session inference, or
credential persistence. `dry-run` and `preflight` never send a generation POST.
`generate` requires both --execute and ANTHROPIC_API_KEY. Nothing runs on import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable
import urllib.error
import urllib.request

API_ROOT = 'https://api.anthropic.com'
API_VERSION = '2026-10-21'
MODEL_IDS = ('claude-sonnet-4-6', 'claude-sonnet-5-5', 'claude-haiku-5-5')
SYSTEM = 'Write clear English prose. Return only the requested text, without a heading or commentary.'
SCHEMA = 'synsteg-claude-cover-adapter-v1'


class AdapterError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward x-api-key to a redirect target, even another hostname.
        return None


def build_request(model: str, prompt: str, max_tokens: int = 1024) -> dict:
    if model not in MODEL_IDS:
        raise ValueError('Use an explicitly documented exact model ID: ' + ', '.join(MODEL_IDS))
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000:
        raise ValueError('Prompt must contain 1 to 20000 characters')
    if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
        raise ValueError('max_tokens must be an integer in [1, 8192]')
    # API 2026-10-21 removes temperature, top_p and top_k. No seed is documented.
    return {'model': model, 'max_tokens': max_tokens, 'system': SYSTEM,
            'messages': [{'role': 'user', 'content': prompt}]}


def _headers(key: str) -> dict:
    if not isinstance(key, str) or not key or '\n' in key or '\r' in key:
        raise AdapterError('ANTHROPIC_API_KEY is missing or invalid; no request sent')
    return {'x-api-key': key, 'anthropic-version': API_VERSION,
            'content-type': 'application/json', 'accept': 'application/json'}


def transport(method: str, path: str, body: dict | None, key: str,
              timeout: float = 120.0, opener: Callable | None = None) -> dict:
    if path != '/v1/messages' and path not in {'/v1/models/' + m for m in MODEL_IDS}:
        raise ValueError('Endpoint is outside the adapter allowlist')
    if method not in ('GET', 'POST') or (method == 'POST') != (path == '/v1/messages'):
        raise ValueError('Method/endpoint mismatch')
    if not 0 < timeout <= 300:
        raise ValueError('Timeout must be in (0, 300] seconds')
    data = None if body is None else json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    req = urllib.request.Request(API_ROOT + path, data=data, headers=_headers(key), method=method)
    if opener is None:
        opener = urllib.request.build_opener(_NoRedirect()).open
    started = time.time()
    try:
        response = opener(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Exception strings can include transport internals; persist only class.
        return {'status': 'transport_error', 'exception_type': type(exc).__name__,
                'started_at_unix': started, 'finished_at_unix': time.time(),
                'retry_performed': False, 'generation_outcome_known': False}
    with response:
        raw = response.read()
        status = response.status if hasattr(response, 'status') else response.code
        request_id = response.headers.get('request-id')
        if isinstance(request_id, str):
            request_id = request_id.replace(key, '[REDACTED_CREDENTIAL]')
    text = raw.decode('utf-8', errors='replace')
    secret_echoed = key in text
    if secret_echoed:
        text = text.replace(key, '[REDACTED_CREDENTIAL]')
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    return {'status': 'http_success' if 200 <= status < 300 else 'http_error',
            'http_status': status, 'request_id': request_id,
            'raw_response_text': text, 'raw_response_json': parsed,
            'response_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'credential_redacted_from_response': secret_echoed,
            'hash_scope': 'retained UTF-8 response; redacted first if credential unexpectedly echoed',
            'started_at_unix': started, 'finished_at_unix': time.time(),
            'retry_performed': False}


def extract_cover(reply: dict, expected_model: str) -> dict:
    if reply.get('status') != 'http_success':
        raise AdapterError('API call did not succeed; retain the failure record')
    raw = reply.get('raw_response_json')
    if not isinstance(raw, dict) or raw.get('type') != 'message' or raw.get('role') != 'assistant':
        raise AdapterError('Malformed message response')
    if raw.get('model') != expected_model:
        raise AdapterError('Returned model does not match the requested pinned ID')
    if not isinstance(raw.get('id'), str) or not raw['id']:
        raise AdapterError('Missing response ID')
    usage = raw.get('usage')
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                        for k in ('input_tokens', 'output_tokens')):
        raise AdapterError('Missing or invalid token usage')
    if not isinstance(raw.get('content'), list):
        raise AdapterError('Missing content blocks')
    pieces = []
    for block in raw['content']:
        if not isinstance(block, dict):
            raise AdapterError('Malformed content block')
        if block.get('type') == 'text':
            if not isinstance(block.get('text'), str):
                raise AdapterError('Malformed text block')
            pieces.append(block['text'])
    text = ''.join(pieces).strip()
    if not text:
        raise AdapterError('No visible cover text; do not silently regenerate')
    reason = raw.get('stop_reason')
    if not isinstance(reason, str) or not reason:
        raise AdapterError('Missing stop reason')
    return {'text': text, 'metadata': {'provider': 'anthropic', 'api_version': API_VERSION,
            'requested_model': expected_model, 'returned_model': raw['model'],
            'response_id': raw['id'], 'request_id': reply.get('request_id'),
            'usage': usage, 'stop_reason': reason, 'stop_sequence': raw.get('stop_sequence'),
            'hit_output_limit': reason == 'max_tokens', 'refusal': reason == 'refusal',
            'seed_supported': False, 'sampling_parameters_sent': False,
            'raw_response_sha256': reply['response_sha256']}}


def _atomic_replace(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', delete=False) as f:
        name = f.name
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
    os.replace(name, path)


def run(mode: str, model: str, output: Path, prompt: str | None = None,
        max_tokens: int = 1024, execute: bool = False, environ=None,
        sender: Callable = transport) -> int:
    if mode not in ('dry-run', 'preflight', 'generate'):
        raise ValueError('Unknown mode')
    if model not in MODEL_IDS:
        raise ValueError('Unknown or unpinned model ID')
    env = os.environ if environ is None else environ
    key = env.get('ANTHROPIC_API_KEY', '')
    body = None if mode == 'preflight' else build_request(model, prompt, max_tokens)
    if isinstance(key, str) and key and key in json.dumps(body, ensure_ascii=False):
        raise AdapterError('The prompt contains the configured credential; no request or result file created')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {'schema': SCHEMA, 'mode': mode, 'status': 'started',
              'created_at_unix': time.time(), 'api_version': API_VERSION,
              'requested_model': model, 'request_body': body,
              'api_key_present': bool(key), 'generation_request_sent': False,
              'model_list_preflight_is_not_inference': True,
              'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'no_retries': True}
    # Exclusive creation preserves any earlier attempt, including interruptions.
    with output.open('x', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write('\n')
    code = 0
    if not key:
        record.update(status='unavailable_missing_api_key',
                      detail='No API request sent. Set ANTHROPIC_API_KEY in the execution environment; do not put it in prompts or result files.')
        code = 2
    elif not isinstance(key, str) or '\n' in key or '\r' in key:
        record.update(status='unavailable_invalid_api_key', detail='Invalid credential header; no API request sent.')
        code = 2
    elif mode == 'dry-run':
        record.update(status='request_prepared_not_sent',
                      detail='Credential presence only; validity, credit and model access remain unverified.')
    elif mode == 'generate' and not execute:
        record.update(status='explicit_execution_flag_required', detail='No API request sent; --execute is required.')
        code = 2
    else:
        path = '/v1/models/' + model if mode == 'preflight' else '/v1/messages'
        method = 'GET' if mode == 'preflight' else 'POST'
        record.update(endpoint=API_ROOT + path, method=method,
                      generation_request_sent=mode == 'generate')
        _atomic_replace(output, record)
        reply = sender(method, path, body, key)
        record['response'] = reply
        if mode == 'preflight':
            returned = reply.get('raw_response_json')
            okay = reply.get('status') == 'http_success' and isinstance(returned, dict) and returned.get('id') == model
            record['status'] = 'model_access_verified_no_generation' if okay else 'model_access_not_verified'
            code = 0 if okay else 3
        else:
            try:
                record['cover'] = extract_cover(reply, model)
                record['status'] = 'generation_recorded'
            except AdapterError as exc:
                record.update(status='generation_failed_or_invalid', detail=str(exc))
                code = 3
    record.update(finished_at_unix=time.time(), exit_code=code)
    _atomic_replace(output, record)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('dry-run', 'preflight', 'generate'))
    parser.add_argument('--model', required=True, choices=MODEL_IDS)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompt-file', type=Path)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    prompt = args.prompt_file.read_text(encoding='utf-8') if args.prompt_file else args.prompt
    if args.mode != 'preflight' and not prompt:
        parser.error('--prompt or --prompt-file is required outside preflight')
    try:
        code = run(args.mode, args.model, args.output, prompt, args.max_tokens, args.execute)
    except (ValueError, AdapterError, FileExistsError) as exc:
        # Our own errors never contain the key or HTTP request headers.
        print(json.dumps({'status': 'not_executed', 'error': str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({'output': str(args.output), 'exit_code': code,
                      'mode': args.mode, 'model': args.model}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
