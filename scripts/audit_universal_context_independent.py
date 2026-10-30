#!/usr/bin/env python3
"""Independent synthetic adversarial probe; no corpus evaluation or generation."""
from __future__ import annotations

from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit import shared_dictionary as base
from synsteg_audit import universal_context as universal


def execute():
    dictionary = base.PublicDictionary.from_mapping({"article": ["an", "the"],
        "color": ["amber", "ochre"], "risk": ["danger", "hazard"]}, "audit-fixture")
    reference = base.PublicDictionary.from_mapping({"article": ["an", "the"],
        "color": ["amber", "ochre"]}, "audit-reference")
    policy = universal.UniversalPolicy.from_dictionaries(dictionary, reference)
    checks = 0
    for radius in (0, 1, 2):
        candidate = replace(policy, radius=radius)
        decisions = []
        for article, color, risk in product(("an", "the"), ("amber", "ochre"), ("danger", "hazard")):
            analysis = universal.analyze_text(f"{article} {color} {risk}", dictionary, candidate)
            decisions.append((analysis.eligible_token_indices, analysis.protected_token_indices,
                              analysis.skeleton_sha256, analysis.slot_decisions))
            assert analysis.eligible_token_indices == (0,)
            checks += 1
        assert all(decision == decisions[0] for decision in decisions)
        checks += 1
    for cap in (1, 2, 4, 8, 4096):
        candidate = replace(policy, orbit_cap=cap)
        signatures = []
        for article, color, risk in product(("an", "the"), ("amber", "ochre"), ("danger", "hazard")):
            analysis = universal.analyze_text(f"{article} {color} {risk}", dictionary, candidate)
            signatures.append((analysis.eligible_token_indices, analysis.protected_token_indices,
                               analysis.slot_decisions, analysis.skeleton_sha256))
            checks += 1
        assert len(set(signatures)) == 1
        checks += 1
    cover = "a amber " + "amber " * 180 + "danger"
    key = bytes(range(32))
    for framing in ("aes_siv", "plain_length16"):
        protocol = universal.UniversalProtocol(framing=framing)
        encoded = universal.encode(cover, b"x", key, dictionary, policy, protocol)
        assert universal.decode(encoded.stego_text, key, dictionary, policy, protocol) == b"x"
        checks += 1
        assert encoded.stego_text.startswith("a amber ") and encoded.stego_text.endswith("danger")
        checks += 1
        assert universal.analyze_text(encoded.stego_text, dictionary, policy, protocol) == universal.analyze_text(cover, dictionary, policy, protocol)
        checks += 1
        if framing == "aes_siv":
            changes = ((encoded.stego_text[:-6] + "hazard", key, policy),
                       (encoded.stego_text, bytes(reversed(range(32))), policy),
                       (encoded.stego_text, key, replace(policy, version="audit-changed")))
            for text, altered_key, altered_policy in changes:
                try:
                    universal.decode(text, altered_key, dictionary, altered_policy, protocol)
                except base.CodecError:
                    checks += 1
                else:
                    raise AssertionError("AES alteration unexpectedly accepted")
    request = {"key_hex": key.hex(), "stego_text": encoded.stego_text,
               "public_dictionary": dictionary.to_dict(), "public_policy": policy.to_dict(),
               "public_protocol": protocol.to_dict()}
    command = [sys.executable, str(ROOT / "src/synsteg_audit/universal_context.py"), "receiver"]
    completed = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True, cwd="/tmp", timeout=60)
    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    assert response["payload_hex"] == "78" and all(response["io_guard_self_checks"].values())
    checks += 1
    request["cover_text"] = cover
    completed = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True, cwd="/tmp", timeout=60)
    assert completed.returncode != 0
    checks += 1
    for text in ("\ue000", "\ue001", "a \ue000uc1:1:616d626572\ue001"):
        for call in (lambda: universal.analyze_text(text, dictionary, policy),
                     lambda: universal.encode(text, b"", bytes(32), dictionary, policy),
                     lambda: universal.decode(text, bytes(32), dictionary, policy)):
            try:
                call()
            except universal.UniversalContextError:
                checks += 1
            else:
                raise AssertionError("Reserved marker accepted")
    assert checks == 92
    return {"schema": "synsteg-universal-independent-probe-v1", "status": "passed",
            "counted_check_units": checks, "count_is_not_pytest_test_count": True,
            "mutable_article_neighbor_variants": 24, "cap_boundary_alias_variants": 40,
            "aes_wrongkey_policy_protectedalias_rejected": 3, "fresh_receiver": response,
            "radius0_article_rejection": True, "new_model_calls": 0, "closed_test_opened": False,
            "scope": "Synthetic mechanics and receiver contract; not natural-language validity or OS sandboxing"}


if __name__ == "__main__":
    print(json.dumps(execute(), sort_keys=True, indent=2))
