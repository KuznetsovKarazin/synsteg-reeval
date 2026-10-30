import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("receiver", BASE / "receiver_proof.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


def receive(key, stego):
    with tempfile.TemporaryDirectory(prefix="synsteg-receiver-empty-") as directory:
        run = subprocess.run([sys.executable, "-I", "-S", str(BASE / "receiver_proof.py")],
                             input=json.dumps({"key_hex": key, "stego_text": stego}),
                             text=True, capture_output=True, cwd=directory,
                             env={"PYTHONHASHSEED": "0"}, check=True)
    response = json.loads(run.stdout)
    assert response["receiver_io_guard_checks"] == {
        "open": True, "socket.__new__": True, "subprocess.Popen": True,
    }
    return response["payload_bits"]


key = hashlib.sha256(b"PUBLIC FIXTURE KEY, NOT A PRODUCTION SECRET").hexdigest()
cover = "London reports: " + "\t ".join(["small,", "fast.", "calm!"] * 110)
cases = []
for size in [0, 1, 7, 32, 64, 127]:
    bits = bin(int.from_bytes(hashlib.sha256(str(size).encode()).digest(), "big"))[2:].zfill(256)[:size]
    nonce = hashlib.sha256(f"PUBLIC FIXTURE NONCE {size}".encode()).digest()[:16].hex()
    texts = []
    for choice in [0, 1]:
        stego = codec.encode(cover, bits, key, nonce, choice)
        assert receive(key, stego) == bits
        assert codec.encode(cover, bits, key, nonce, choice) == stego
        texts.append(stego)
    assert texts[0] != texts[1]
    cases.append({"payload_bits": size, "sender_choices_tested": 2, "roundtrips_exact": True})

bits = "10110010" * 8
nonce = "ab" * 16
stego = codec.encode(cover, bits, key, nonce)
wrong_key_bits = receive("00" * 32, stego)
assert wrong_key_bits != bits
available = codec.slots(stego)
index, start, end, class_id, token = available[codec.HEADER_BITS]
grouping = codec.groups(bytes.fromhex(key), nonce, index, class_id)
opposite = grouping[1 - int(bits[0])][0]
tampered = stego[:start] + opposite + stego[end:]
recovered_tampered = receive(key, tampered)
assert recovered_tampered != bits
assert len(recovered_tampered) == len(bits)

sys.path.insert(0, str(REPO / "src"))
from synsteg_audit.fixed_codebook import freeze_codebook, encode_bits, decode_bits
old_cover = " ".join(["small"] * 128)
options = [["tiny", "little", "compact"]] * 128
book = freeze_codebook(old_cover, options, seed="private-seed-example", min_group_size=1)
old_bits = "10110010" * 4
old_stego = encode_bits(book, old_bits, framing="external_length").text
assert decode_bits(book, old_stego, framing="external_length", expected_bits=len(old_bits)) == old_bits
rebuilt = freeze_codebook(old_stego, options, seed="private-seed-example", min_group_size=1)
old_decoded = decode_bits(rebuilt, old_stego, framing="external_length", expected_bits=len(old_bits))
assert old_decoded != old_bits

result = {
    "scope": "Functional synchronization toy only; no semantic, confidentiality, authentication, or steganographic-security claim",
    "dictionary_origin": "Manually specified toy classes; NOT AI-generated",
    "cover_origin": "Synthetic repeated words with punctuation and whitespace; NOT natural-text feasibility data",
    "keys_and_nonces": "Public deterministic fixtures only, NOT production secrets or a production nonce policy",
    "shared_public_assets": ["receiver code/protocol", "global disjoint dictionary"],
    "receiver_input_keys": ["key_hex", "stego_text"],
    "receiver_cover_or_codebook_provided": False,
    "receiver_length_or_nonce_provided_outside_text": False,
    "receiver_process": "fresh Python -I -S subprocess, empty cwd and minimal environment; external file/socket/subprocess I/O blocked by audit hook after public imports",
    "receiver_io_guard_policy_self_checks_passed": True,
    "receiver_isolation_limit": "Python audit-hook guard for this inspected toy program, not an operating-system security sandbox",
    "header_overhead_bits": codec.HEADER_BITS,
    "cases": cases,
    "roundtrips_passed": 12,
    "sender_choice_independence_verified": True,
    "repeated_encoding_same_inputs_exact": True,
    "wrong_key_not_detected_but_payload_wrong": True,
    "payload_corruption_not_detected": True,
    "production_requirements_missing": ["AEAD authentication/encryption", "vetted context-invariant synonym classes", "nonce lifecycle", "channel error policy", "blind detection/quality evaluation"],
    "current_codec_counterexample": {
        "scope": "Separate synthetic 32-bit unit counterexample; not an original v3 closed-test attempt or reevaluation",
        "source_path": "src/synsteg_audit/fixed_codebook.py",
        "source_sha256": hashlib.sha256((REPO / "src/synsteg_audit/fixed_codebook.py").read_bytes()).hexdigest(),
        "source_freeze_commit": "674b370c79093f8218adeed09a07bf16f250e143",
        "seed_fixture": "private-seed-example (public literal fixture, despite its name)",
        "original_book_roundtrip_exact": True,
        "same_seed_and_candidate_sets": True,
        "original_cover_hash": book.cover_sha256,
        "stego_cover_hash": rebuilt.cover_sha256,
        "payload_bits": old_bits,
        "recovered_bits": old_decoded,
        "bit_errors": sum(a != b for a, b in zip(old_bits, old_decoded)),
        "external_length_decode_raised": False,
    },
    "proof_files_sha256": {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest()
                           for name in ["receiver_proof.py", "run_proof.py"]},
}
(BASE / "proof_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
