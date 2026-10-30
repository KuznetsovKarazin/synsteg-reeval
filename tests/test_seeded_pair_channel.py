from copy import deepcopy
import itertools

from synsteg_audit.seeded_pair_channel import (
    DEMO_KEY, PROFILES, decode, encode, exact_output_law, pair, weighted_choice,
)


def test_exact_laws_all_messages_and_profiles():
    for profile in PROFILES:
        for bits in itertools.product("01", repeat=4):
            evidence = exact_output_law(profile, "".join(bits), 4)
            assert evidence["total_variation"] == "0"
            assert evidence["probability_mass"] == "1"


def test_unbiased_integer_sampling_rejects_out_of_range_tail():
    draws = iter([(1 << 256) - 1, 1])
    assert weighted_choice({"symbols": "AB", "initial": [1, 2]}, "", lambda: next(draws)) == "B"


def test_receiver_contract_repeat_nonce_and_authentication():
    args = (b"seed", DEMO_KEY, bytes(16), "uniform4", 512)
    first = encode(*args)
    assert first == encode(*args)
    assert first["frame_completed"]
    assert decode(first["envelope"], DEMO_KEY)["payload_hex"] == b"seed".hex()
    assert encode(b"seed", DEMO_KEY, bytes([1]) * 16, "uniform4", 512)["envelope"] != first["envelope"]
    assert decode(first["envelope"], bytes([7]) * 32)["status"] != "ok"
    changed = deepcopy(first["envelope"])
    changed["nonce_hex"] = "01" * 16
    assert decode(changed, DEMO_KEY)["status"] != "ok"
    changed = deepcopy(first["envelope"])
    changed["stego_text"] = "Z" + changed["stego_text"][1:]
    assert decode(changed, DEMO_KEY)["status"] == "invalid_text"
    drift = {"symbols": "ABCD", "initial": [4, 3, 2, 1]}
    assert decode(first["envelope"], DEMO_KEY, drift)["status"] != "ok"


def test_constant_channel_and_short_frame_are_retained():
    result = encode(b"x", DEMO_KEY, bytes(16), "constant1", 64)
    assert result["capacity_bits"] == 0
    assert len(result["envelope"]["stego_text"].split()) == 64
    assert not result["frame_completed"]
    assert decode(result["envelope"], DEMO_KEY)["status"] == "insufficient_capacity"
    short = encode(b"x", DEMO_KEY, bytes(16), "uniform4", 8)
    assert not short["frame_completed"]
    assert decode(short["envelope"], DEMO_KEY)["status"] == "insufficient_capacity"


def test_context_history_receiver_and_padding():
    result = encode(b"", DEMO_KEY, bytes(16), "context4", 512)
    decoded = decode(result["envelope"], DEMO_KEY)
    assert decoded["status"] == "ok"
    assert decoded["payload_hex"] == ""
    assert decoded["decoded_bits"] == result["capacity_bits"]
    assert result["capacity_bits"] > result["frame_bits"]
    assert result["samples_drawn"] == 1024


def test_authentication_covers_frame_not_padding():
    # Explicit counterexample prevents overstating the scope of the frame MAC.
    for fixture_index in range(16):
        nonce = fixture_index.to_bytes(16, "big")
        result = encode(b"", DEMO_KEY, nonce, "uniform4", 512)
        text = result["envelope"]["stego_text"].replace(" ", "")
        a, b = pair(DEMO_KEY, nonce.hex(), "uniform4", text[:-1])
        if a != b and result["capacity_bits"] > result["frame_bits"]:
            mutated = dict(result["envelope"])
            mutated["stego_text"] = " ".join(text[:-1] + (b if text[-1] == a else a))
            assert decode(mutated, DEMO_KEY)["status"] == "ok"
            return
    raise AssertionError("deterministic padding counterexample fixture missing")
