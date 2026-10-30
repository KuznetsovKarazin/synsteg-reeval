"""Public-spec interface and sender nonce controls; all synthetic, no NN."""
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from synsteg_audit import framed_interval_public_v1 as public
from synsteg_audit import framed_interval_bounded_v1 as old
from synsteg_audit.canonical_channel_v2 import ContractError
from synsteg_audit.framed_interval_public_v1_cli import synthetic_endpoint
from synsteg_audit.framed_interval_public_v1_nonce import NonceAlreadyReserved, reserve_nonce

KEY = bytes(range(32))
ROOT = Path(__file__).resolve().parents[1]


def setup(nonce=bytes(12)):
    endpoint = synthetic_endpoint()
    spec = public.make_public_spec(endpoint)
    context = public.make_public_context((), context_id="all-payloads-share-this-context")
    envelope = public.make_envelope(spec, context, nonce)
    return endpoint, spec, context, envelope


@pytest.mark.parametrize("payload", [b"", b"a", b"completely new", bytes(range(16))])
def test_same_spec_and_context_for_arbitrary_short_payloads(payload):
    ep, spec, context, env = setup(len(payload).to_bytes(12,"big"))
    baseline = setup()[1:3]
    assert (spec, context) == baseline
    sent = public.generate(KEY, env, ep, public_spec=spec, public_context=context, payload=payload)
    received = public.receive(sent.wire, KEY, env, synthetic_endpoint(), public_spec=spec, public_context=context)
    assert sent.status == "threshold_pending_receiver" and sent.emitted_tokens == 91
    assert received.authenticated_delivery and received.payload == payload
    assert set(env) == public.ENVELOPE_FIELDS
    assert "protocol_sha256" not in env and "experiment_id" not in env
    assert public.DOMAIN != old.DOMAIN


@pytest.mark.parametrize("location", ["spec", "context", "envelope"])
def test_experiment_provenance_extension_rejected(location):
    ep, spec, context, env = setup()
    {"spec":spec,"context":context,"envelope":env}[location]["sender_fixtures_sha256"] = "a"*64
    with pytest.raises(ContractError):
        public.generate(KEY, env, ep, public_spec=spec, public_context=context, payload=b"a")
    assert not ep._started


def test_false_numeric_limit_is_rejected_before_backend():
    ep, spec, context, env = setup()
    spec["limits"]["length_bytes"] = True
    with pytest.raises(ContractError):
        public.generate(KEY, env, ep, public_spec=spec, public_context=context, payload=b"a")
    assert not ep._started


def test_context_mismatch_and_different_valid_spec_rejected():
    ep, spec, context, env = setup()
    context["context_id"] = "changed"
    with pytest.raises(ContractError, match="digest mismatch"):
        public.generate(KEY, env, ep, public_spec=spec, public_context=context, payload=b"a")
    context["context_id"] = "all-payloads-share-this-context"
    spec["limits"]["max_wire_bytes"] -= 1
    with pytest.raises(ContractError, match="digest mismatch"):
        public.generate(KEY, env, ep, public_spec=spec, public_context=context, payload=b"a")
    assert not ep._started


def test_bad_resources_are_rejected_not_trusted_from_envelope():
    ep, spec, context, _ = setup()
    spec["resources"]["provider_sha256"] = "0"*64
    env = public.make_envelope(spec, context, bytes(12))
    with pytest.raises(ContractError, match="endpoint differs"):
        public.generate(KEY, env, ep, public_spec=spec, public_context=context, payload=b"a")
    assert not ep._started


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":{"a":1,"a":2}}', b'{"x":NaN}', b'\xff'])
def test_ambiguous_or_invalid_json_is_rejected(raw):
    with pytest.raises(ContractError): public.load_public_json(raw)


def test_reservation_independent_of_context_and_no_key_saved(tmp_path):
    ledger = tmp_path / "ledger"
    first = reserve_nonce(ledger, KEY, bytes(12))
    assert first["state"] == "burned_before_generation"
    assert first["key_saved"] is False
    assert KEY.hex() not in json.dumps(first)
    assert KEY.hex() not in next(ledger.iterdir()).name
    with pytest.raises(NonceAlreadyReserved): reserve_nonce(ledger, KEY, bytes(12))
    reserve_nonce(ledger, bytes(reversed(KEY)), bytes(12))
    reserve_nonce(ledger, KEY, b"\1"*12)
    assert len(list(ledger.iterdir())) == 3


def test_ledger_survives_sender_failure_and_reservation_symlink(tmp_path):
    ledger = tmp_path / "ledger"; nonce = bytes(12)
    reserve_nonce(ledger, KEY, nonce)
    ep, spec, context, env = setup(nonce)
    ep._observe = lambda token: (_ for _ in ()).throw(OSError("unknown native step"))
    failed = public.generate(KEY, env, ep, public_spec=spec, public_context=context, payload=b"a")
    assert failed.status == "backend_error" and failed.emitted_tokens == 0
    with pytest.raises(NonceAlreadyReserved): reserve_nonce(ledger, KEY, nonce)
    alias = tmp_path / "alias"; alias.symlink_to(ledger, target_is_directory=True)
    with pytest.raises(ContractError): reserve_nonce(alias, KEY, b"\2"*12)


def test_ledger_file_creation_is_exclusive_across_processes(tmp_path):
    ledger = tmp_path / "ledger"
    code = "from synsteg_audit.framed_interval_public_v1_nonce import reserve_nonce; import sys; reserve_nonce(sys.argv[1],bytes(range(32)),bytes(12))"
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
    workers = [subprocess.Popen([sys.executable,"-c",code,str(ledger)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(2)]
    exits = []
    for p in workers:
        p.communicate(timeout=10); exits.append(p.returncode)
    assert sorted(exits) == [0,1] and len(list(ledger.iterdir())) == 1


def test_metadata_cost_counts_context_separately_without_double_nonce():
    _, spec, context, env = setup()
    cost = public.metadata_costs(env,spec,context,wire_bytes=91,payload_bytes=16)
    assert cost["per_message_with_pre_shared_spec_context_bytes"] == 91+len(public.canonical_json(env))
    assert cost["first_message_without_pre_shared_spec_context_bytes"] == sum([91,len(public.canonical_json(env)),len(public.canonical_json(spec)),len(public.canonical_json(context))])
    assert cost["model_runtime_distribution_bytes"] is None and not cost["key_in_transport_totals"]


def test_backend_cannot_mutate_caller_envelope_after_validation():
    ep, spec, context, env = setup()
    original = copy.deepcopy((spec,context,env))
    ep._start = lambda _: (env.update(nonce_hex="ff"*12), context.update(context_id="changed"), spec["limits"].update(max_wire_bytes=1))
    sent = public.generate(KEY, env, ep, public_spec=spec, public_context=context, payload=b"callback mutate")
    spec,context,env = original
    got = public.receive(sent.wire,KEY,env,synthetic_endpoint(),public_spec=spec,public_context=context)
    assert got.authenticated_delivery and got.payload == b"callback mutate"


def test_fresh_receiver_directory_has_no_sender_or_experiment_files(tmp_path):
    # Copy only runtime source dependencies and receiver inputs; expected payload
    # stays in this parent test process, never in the receiver directory/arguments.
    ep,spec,context,env = setup()
    expected = bytes(range(16))
    sent = public.generate(KEY,env,ep,public_spec=spec,public_context=context,payload=expected)
    isolated=tmp_path/'isolated'; isolated.mkdir(); package=isolated/'src'/'synsteg_audit';package.mkdir(parents=True)
    modules=['__init__.py','canonical_channel_v2.py','integer_interval_channel_v1.py','framed_interval_llm_v1.py',
             'framed_interval_public_v1.py','framed_interval_public_v1_nonce.py','framed_interval_public_v1_cli.py']
    for name in modules: shutil.copyfile(ROOT/'src'/'synsteg_audit'/name,package/name)
    for name,obj in [('spec.json',spec),('context.json',context),('envelope.json',env)]:
        (isolated/name).write_bytes(public.canonical_json(obj))
    (isolated/'key.bin').write_bytes(KEY);(isolated/'wire.bin').write_bytes(sent.wire)
    command=[sys.executable,'-m','synsteg_audit.framed_interval_public_v1_cli','receive','--spec','spec.json','--context','context.json',
             '--envelope','envelope.json','--wire','wire.bin','--key-file','key.bin','--output','received']
    process=subprocess.run(command,cwd=isolated,env=dict(os.environ,PYTHONPATH=str(isolated/'src'),PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=10)
    assert process.returncode == 0,process.stderr
    assert (isolated/'received'/'payload.bin').read_bytes() == expected
    assert not any('fixture' in p.name or 'activation' in p.name or 'freeze' in p.name or 'readback' in p.name for p in isolated.rglob('*'))
    report=json.loads((isolated/'received'/'result.json').read_text())
    assert report['authenticated_delivery'] and report['execution_kind']=='synthetic_no_model'
