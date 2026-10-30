#!/usr/bin/env python3
"""One completion per frozen selected record, with local CPU inference only.

No downloader, retry, alternative source, prompt truncation or automatic resume.
Every server is owned by this process and killed as a process group on exit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

import prepare_qwen15b_gguf_v1 as prep


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(prep.formatted(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_resources(root, model):
    resources = prep.read(root / prep.OUT / 'resources.json')
    wanted = resources['model']
    if not model.is_file() or model.stat().st_size != wanted['bytes'] or file_hash(model) != wanted['sha256']:
        raise ValueError('Pinned model file missing or SHA/size differs')
    runtime = prep.read(root / prep.OUT / 'preflight/runtime-preflight.json')['runtime']
    if runtime['commit'] != resources['runtime']['commit'] or runtime['archive_sha256'] != prep.RUNTIME_HASH:
        raise ValueError('Runtime manifest pin differs')
    for entry in runtime['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']:
            raise ValueError('Runtime file mismatch: ' + entry['path'])
    server = (root / runtime['server_path']).resolve()
    if not server.is_relative_to(root) or runtime['server_path'] not in {f['path'] for f in runtime['files']}:
        raise ValueError('Server path escapes root or is absent from hashed runtime files')
    return server


def terminate_group(process):
    """Terminate all children of our new session, including on request timeout."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    finally:
        # Parent may exit before a descendant; also kill if wait was interrupted.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


@contextmanager
def hard_deadline(seconds):
    """Linux wall clock deadline, independent of HTTP progress/socket timeouts."""
    def expired(signum, frame):
        raise TimeoutError('Hard per-record wall deadline exceeded')
    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        raise ValueError('Local inference HTTP redirects are forbidden')


class LocalServer:
    def __init__(self, binary, model, directory, settings):
        self.binary, self.model, self.directory, self.settings = binary, model, directory, settings
        self.process = self.log = None
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); self.port = sock.getsockname()[1]
        self.url = 'http://127.0.0.1:' + str(self.port)
        # Never use environment HTTP proxies for localhost requests.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def command(self):
        return [str(self.binary), '-m', str(self.model), '--host', '127.0.0.1', '--port', str(self.port),
                '--threads', '6', '--threads-batch', '6', '--parallel', '1', '--ctx-size', '8192',
                '--gpu-layers', '0', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
                '--flash-attn', 'off', '--no-context-shift', '--no-warmup']

    def __enter__(self):
        command = self.command()
        write_json(self.directory / 'server-command.json', command)
        self.log = (self.directory / 'server.log').open('wb')
        # Remove ambient model/backend overrides; no credential inspection.
        env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_', 'GGML_'))}
        env['LD_LIBRARY_PATH'] = str(self.binary.parent)
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True, env=env)
        deadline = time.monotonic() + self.settings['startup_timeout_seconds']
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Local server exited during startup')
                try:
                    with self.opener.open(self.url + '/health', timeout=1) as response:
                        if response.status == 200:
                            return self
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(.1)
            raise TimeoutError('Server startup timeout')
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *unused):
        if self.process is not None:
            terminate_group(self.process)
        if self.log is not None:
            self.log.close()

    def request(self, endpoint, payload, timeout, label):
        raw_request = prep.formatted(payload)
        (self.directory / (label + '.request.json')).write_bytes(raw_request)
        request = urllib.request.Request(self.url + endpoint, data=raw_request,
                                         headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw_response = response.read()
            (self.directory / (label + '.response.json')).write_bytes(raw_response)
            value = json.loads(raw_response)
            if not isinstance(value, dict) or 'error' in value:
                raise ValueError('Invalid/error local server response')
            return value
        except urllib.error.HTTPError as error:
            (self.directory / (label + '.error-response.txt')).write_bytes(error.read())
            raise


def token_budget_ok(count, protocol):
    return (type(count) is int and count > 0 and
            count + protocol['max_output_tokens'] + protocol['reserved_tokens'] <= protocol['context_tokens'])


def completion_summary(response, row):
    text = response.get('content')
    if not isinstance(text, str):
        raise ValueError('Completion has no string content')
    words = len(text.split())
    lower, upper = row['preferred_word_band']
    schema_issues = []
    for field in ('tokens_predicted', 'tokens_evaluated'):
        if type(response.get(field)) is not int or response[field] < 0:
            schema_issues.append('missing_or_invalid_' + field)
    returned_tokens = response.get('tokens')
    if not isinstance(returned_tokens, list) or any(type(t) is not int or t < 0 for t in returned_tokens):
        schema_issues.append('missing_or_invalid_returned_tokens')
    if type(response.get('truncated')) is not bool:
        schema_issues.append('missing_or_invalid_truncated')
    if not (response.get('stop_type') in ('eos', 'limit', 'word') or
            all(type(response.get(k)) is bool for k in ('stopped_eos', 'stopped_limit', 'stopped_word'))):
        schema_issues.append('unknown_stop_metadata')
    legacy_true = [name for name in ('eos', 'limit', 'word') if response.get('stopped_' + name) is True]
    if len(legacy_true) > 1 or (legacy_true and response.get('stop_type') in ('eos', 'limit', 'word')
                               and legacy_true != [response['stop_type']]):
        schema_issues.append('contradictory_stop_metadata')
    return {'status': ('response_schema_uncertain' if schema_issues else
                      'generated' if text.strip() else 'empty_output'), 'schema_issues': schema_issues,
            'text': text, 'text_sha256': prep.digest(text.encode()), 'output_words': words,
            'within_soft_word_band': lower <= words <= upper, 'factual_quality': 'not_reviewed',
            'stopped_eos': response.get('stopped_eos'), 'stopped_limit': response.get('stopped_limit'),
            'stopped_word': response.get('stopped_word'), 'stop_type': response.get('stop_type'),
            'tokens_predicted': response.get('tokens_predicted'), 'tokens_evaluated': response.get('tokens_evaluated'),
            'truncated': response.get('truncated'), 'timings': response.get('timings')}


def perform_request(row, prompt, protocol, client, directory, deadline=None):
    """The only completion call; pure tests replace client with a deterministic fake."""
    result = {'record_id': row['record_id'], 'plan_index': row['plan_index'],
              'status': 'started', 'generation_request_attempted': False, 'started_at': now()}
    write_json(directory / 'attempt.json', result)
    started = time.monotonic()
    deadline = deadline if deadline is not None else started + protocol['runtime']['per_record_timeout_seconds']

    def request(endpoint, payload, label):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Record deadline exceeded')
        value = client.request(endpoint, payload, remaining, label)
        if time.monotonic() > deadline:
            raise TimeoutError('Record deadline exceeded after response')
        return value

    try:
        rendered = request('/apply-template', {'messages': prompt['messages']}, 'apply-template')['prompt']
        if not isinstance(rendered, str) or not rendered.endswith('<|im_start|>assistant\n'):
            raise ValueError('Unexpected ChatML assistant prefix')
        # Fixed system/user messages only; compare the complete rendered template.
        expected = ''.join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'
                           for m in prompt['messages']) + '<|im_start|>assistant\n'
        if rendered != expected:
            raise ValueError('Rendered template differs from fixed no-tools Qwen ChatML')
        (directory / 'rendered-prompt.txt').write_text(rendered, encoding='utf-8')
        tokenized = request('/tokenize', {'content': rendered, 'add_special': True, 'parse_special': True}, 'tokenize')
        tokens = tokenized['tokens']
        if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('Invalid token IDs')
        result.update(prompt_tokens=len(tokens), rendered_prompt_sha256=prep.digest(rendered.encode()))
        if not token_budget_ok(len(tokens), protocol):
            result['status'] = 'context_budget_reject'
        else:
            payload = {k: v for k, v in protocol['sampling'].items() if k != 'seed_rule'}
            payload.update(prompt=tokens, n_predict=protocol['max_output_tokens'], seed=row['sampling_seed'],
                           cache_prompt=False, return_tokens=True, stream=False, ignore_eos=False,
                           stop=[], id_slot=0, timings_per_token=True)
            result.update(status='generation_request_started', generation_request_attempted=True,
                          generation_request_started_at=now())
            # Durable marker comes before sending; crash means unknown, never safe-to-retry.
            write_json(directory / 'attempt.json', result)
            response = request('/completion', payload, 'completion')
            result.update(completion_summary(response, row))
            (directory / 'cover.txt').write_text(result['text'], encoding='utf-8')
            if response.get('truncated') is True:
                result['status'] = 'protocol_violation_truncated'
    except Exception as error:
        result.update(status='timeout' if isinstance(error, (TimeoutError, socket.timeout)) else 'request_error',
                      error_type=type(error).__name__, error=str(error))
    result.update(finished_at=now(), wall_seconds=time.monotonic() - started)
    write_json(directory / 'attempt.json', result)
    return result


def initial_ledger(plan):
    return [{**row, 'status': 'not_started' if row['planned_action'] == 'generate_once' else row['planned_action'],
             'generation_request_attempted': False} for row in plan]


def summarize(ledger):
    selected = [r for r in ledger if r['planned_action'] == 'generate_once']
    available = [r for r in selected if isinstance(r.get('text'), str)]
    return {'planned_source_groups': len(ledger), 'selected_generation_denominator': len(selected),
            'source_gates': sum(r['planned_action'] == 'abstain_source_gate' for r in ledger),
            'not_selected': sum(r['planned_action'] == 'not_selected_fixed_eight_budget' for r in ledger),
            'status_counts_all20': dict(Counter(r['status'] for r in ledger)),
            'generation_requests_attempted': sum(r['generation_request_attempted'] for r in selected),
            'available_outputs_including_empty': len(available),
            'within_soft_word_band': sum(r.get('within_soft_word_band', False) for r in selected),
            'factual_quality_reviewed': 0, 'test_data_used': False, 'paid_calls': 0}


def _execute(root, model, freeze_commit):
    if not re.fullmatch('[0-9a-f]{40}', freeze_commit):
        raise ValueError('Explicit full Git freeze commit required')
    prep.verify(root)
    binary = verify_resources(root, model)
    output = root / prep.OUT / 'run'
    output.mkdir(exist_ok=False)  # Never overwrite/retry an existing run.
    protocol = prep.read(root / prep.OUT / 'protocol.json')
    plan = prep.read(root / prep.OUT / 'plan.json')['records']
    prompts = {p['record_id']: p for p in prep.rows(root / prep.OUT / 'prompts.jsonl')}
    ledger = initial_ledger(plan)
    write_json(output / 'execution.json', {'started_at': now(), 'freeze_commit': freeze_commit,
        'freeze_binding': 'Caller-verified remote Git commit; runner verifies local frozen hashes',
        'protocol_sha256': file_hash(root / prep.OUT / 'protocol.json'), 'model_sha256': file_hash(model),
        'server_sha256': file_hash(binary), 'platform': platform.platform(), 'cpu_count': os.cpu_count()})
    write_json(output / 'ledger.json', ledger)
    for row in ledger:
        if row['planned_action'] != 'generate_once':
            continue
        directory = output / 'attempts' / ('%02d' % row['plan_index'])
        directory.mkdir(parents=True, exist_ok=False)
        row.update(status='server_starting', started_at=now())
        write_json(output / 'ledger.json', ledger)
        deadline = time.monotonic() + protocol['runtime']['per_record_timeout_seconds']
        try:
            prep.verify(root)
            verify_resources(root, model)
            with hard_deadline(max(.001, deadline - time.monotonic())):
                with LocalServer(binary, model, directory, protocol['runtime']) as client:
                    row.update(perform_request(row, prompts[row['record_id']], protocol, client, directory, deadline))
        except Exception as error:
            row.update(status='timeout' if isinstance(error, TimeoutError) else 'server_error',
                       error_type=type(error).__name__, error=str(error), finished_at=now())
            write_json(directory / 'attempt.json', row)
        write_json(output / 'ledger.json', ledger)
        write_json(output / 'summary.json', summarize(ledger))
    prep.verify(root)
    verify_resources(root, model)
    write_json(output / 'summary.json', {**summarize(ledger), 'finished_at': now(), 'post_run_hashes_verified': True})
    return summarize(ledger)


def execute(root, model, freeze_commit):
    def interrupted(signum, frame):
        # BaseException deliberately bypasses row-level Exception handlers;
        # finally kills the server, while the started marker preserves uncertainty.
        raise KeyboardInterrupt('Termination signal: owned server cleanup required')
    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        return _execute(root, model, freeze_commit)
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('verify', 'execute'))
    parser.add_argument('--root', type=Path, default=prep.ROOT)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--freeze-commit')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == 'verify':
        result = prep.verify(root)
    else:
        if not args.model or not args.freeze_commit:
            parser.error('execute requires --model and --freeze-commit; no implicit download or inference')
        result = execute(root, args.model.resolve(), args.freeze_commit)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
