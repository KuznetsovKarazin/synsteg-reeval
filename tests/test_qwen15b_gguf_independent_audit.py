"""Independent synthetic pre-inference checks; never load or generate a model."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_qwen15b_gguf_v1 as prep
import run_qwen15b_gguf_v1 as runner


def inputs():
    directory = ROOT / prep.OUT
    protocol = prep.read(directory / 'protocol.json')
    plan = prep.read(directory / 'plan.json')['records']
    prompt = prep.rows(directory / 'prompts.jsonl')[0]
    row = next(r for r in plan if r['record_id'] == prompt['record_id'])
    return row, prompt, protocol


def native_reply(**changes):
    return dict({'content': 'Synthetic response only.', 'tokens': [100, 101, 102],
                 'tokens_predicted': 3, 'tokens_evaluated': 5, 'truncated': False,
                 'stop_type': 'eos', 'stopped_eos': True, 'stopped_limit': False,
                 'stopped_word': False}, **changes)


class FakeClient:
    def __init__(self, prompt, reply):
        self.prompt, self.reply, self.calls = prompt, reply, []

    def request(self, endpoint, payload, timeout, label):
        self.calls.append((endpoint, payload))
        if endpoint == '/apply-template':
            rendered = ''.join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'
                               for m in self.prompt['messages']) + '<|im_start|>assistant\n'
            return {'prompt': rendered}
        if endpoint == '/tokenize':
            return {'tokens': [151644, 1, 2, 3, 151645]}
        return self.reply


def test_exact_token_id_request_and_true_truncation_not_clean_success(tmp_path):
    row, prompt, protocol = inputs()
    client = FakeClient(prompt, native_reply(truncated=True, stop_type='limit', stopped_eos=False, stopped_limit=True))
    result = runner.perform_request(row, prompt, protocol, client, tmp_path)
    request = client.calls[-1][1]
    assert request['prompt'] == [151644, 1, 2, 3, 151645]
    assert request['return_tokens'] is True and request['n_predict'] == 1536
    assert result['status'] == 'protocol_violation_truncated'
    assert result['text'] == 'Synthetic response only.'
    assert result['generation_request_attempted']


@pytest.mark.parametrize('tokens', [None, [True], [-1], ['42']])
def test_missing_or_malformed_return_token_array_is_uncertain(tokens):
    row, _, _ = inputs()
    reply = native_reply(tokens=tokens)
    if tokens is None:
        del reply['tokens']
    result = runner.completion_summary(reply, row)
    assert result['status'] == 'response_schema_uncertain'
    assert result['schema_issues']


def test_valid_empty_and_limit_native_metadata_are_distinct():
    row, _, _ = inputs()
    empty = runner.completion_summary(native_reply(content='', tokens=[], tokens_predicted=0), row)
    assert empty['status'] == 'empty_output'
    limited = runner.completion_summary(native_reply(stop_type='limit', stopped_eos=False, stopped_limit=True), row)
    assert limited['status'] == 'generated' and limited['stopped_limit'] is True
    assert limited['factual_quality'] == 'not_reviewed'


def stage_fake_execution(tmp_path, monkeypatch):
    root = tmp_path / 'synthetic'; directory = root / prep.OUT
    directory.mkdir(parents=True)
    for name in ('plan.json', 'protocol.json', 'prompts.jsonl'):
        (directory / name).write_bytes((ROOT / prep.OUT / name).read_bytes())
    model = root / 'synthetic-model'; model.write_bytes(b'not model weights')
    binary = root / 'synthetic-binary'; binary.write_bytes(b'not an executable')
    monkeypatch.setattr(prep, 'verify', lambda root: {'status': 'synthetic-only'})
    monkeypatch.setattr(runner, 'verify_resources', lambda root, model: binary)
    class FakeServer:
        completion_calls = 0
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def request(self, endpoint, payload, timeout, label):
            if endpoint == '/apply-template':
                self.prompt = {'messages': payload['messages']}
                self.client = FakeClient(self.prompt, native_reply())
            if endpoint == '/completion':
                type(self).completion_calls += 1
            return self.client.request(endpoint, payload, timeout, label)
    monkeypatch.setattr(runner, 'LocalServer', FakeServer)
    return root, directory, model, FakeServer


def test_completed_run_is_exclusive_and_all_twenty_rows_remain(tmp_path, monkeypatch):
    root, directory, model, fake = stage_fake_execution(tmp_path, monkeypatch)
    result = runner.execute(root, model, '1' * 40)
    assert fake.completion_calls == 8
    assert result['planned_source_groups'] == 20 and result['selected_generation_denominator'] == 8
    assert result['source_gates'] == 5 and result['not_selected'] == 7
    assert len(prep.read(directory / 'run/ledger.json')) == 20
    before = (directory / 'run/ledger.json').read_bytes()
    with pytest.raises(FileExistsError):
        runner.execute(root, model, '1' * 40)
    assert fake.completion_calls == 8 and (directory / 'run/ledger.json').read_bytes() == before


def test_partial_run_is_never_resumed_or_retried(tmp_path, monkeypatch):
    root, directory, model, fake = stage_fake_execution(tmp_path, monkeypatch)
    (directory / 'run').mkdir()
    (directory / 'run/started-marker.json').write_text('{"outcome":"unknown"}\n')
    with pytest.raises(FileExistsError):
        runner.execute(root, model, '1' * 40)
    assert fake.completion_calls == 0
