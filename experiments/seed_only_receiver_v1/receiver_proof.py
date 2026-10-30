"""Toy synchronization proof: no models, semantic or steganographic security claim.

The key is the sole SECRET input. A versioned public dictionary is installed in
this file; a public 128-bit nonce and 32-bit payload length are carried in text.
Payload encryption/authentication and channel error correction are absent.
"""
import hashlib
import hmac
import json
import re
import sys

VERSION = "toy-global-class-prf-v1"
CLASSES = {
    "size": ("compact", "little", "small", "tiny"),
    "speed": ("fast", "quick", "rapid", "swift"),
    "calm": ("calm", "placid", "serene", "tranquil"),
}
REVERSE = {word: class_id for class_id, words in CLASSES.items() for word in words}
assert len(REVERSE) == sum(map(len, CLASSES.values()))
DICT_HASH = hashlib.sha256(json.dumps(CLASSES, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
TOKEN = re.compile(r"([^A-Za-z0-9_]*)([a-z]+)([^A-Za-z0-9_]*)\Z")
HEADER_BITS = 128 + 32


def slots(text):
    result = []
    for index, match in enumerate(re.finditer(r"\S+", text)):
        lexical = TOKEN.fullmatch(match.group())
        if lexical and lexical.group(2) in REVERSE:
            core = lexical.group(2)
            result.append((index, match.start() + len(lexical.group(1)),
                           match.start() + len(lexical.group(1)) + len(core),
                           REVERSE[core], core))
    return result


def groups(key, nonce, index, class_id):
    def rank(word):
        context = json.dumps([VERSION, DICT_HASH, nonce, index, class_id, word],
                             separators=(",", ":")).encode()
        return hmac.new(key, context, hashlib.sha256).digest(), word
    order = sorted(CLASSES[class_id], key=rank)
    return order[::2], order[1::2]


def bit_for(token, grouping):
    return "0" if token in grouping[0] else "1"


def encode(cover, payload, key_hex, nonce_hex, choice=0):
    if set(payload) - {"0", "1"}:
        raise ValueError("binary payload required")
    key, nonce = bytes.fromhex(key_hex), bytes.fromhex(nonce_hex)
    if len(key) != 32 or len(nonce) != 16:
        raise ValueError("32-byte key and 16-byte nonce required")
    available = slots(cover)
    header = f"{int.from_bytes(nonce, 'big'):0128b}{len(payload):032b}"
    frame = header + payload
    if len(frame) > len(available):
        raise ValueError("insufficient capacity; no partial output")
    edits = []
    for offset, (index, start, end, class_id, _) in enumerate(available[:len(frame)]):
        grouping = (CLASSES[class_id][::2], CLASSES[class_id][1::2]) if offset < HEADER_BITS else groups(key, nonce_hex, index, class_id)
        group = grouping[int(frame[offset])]
        edits.append((start, end, group[choice % len(group)]))
    result = cover
    for start, end, word in reversed(edits):
        result = result[:start] + word + result[end:]
    return result


def decode(stego, key_hex):
    key = bytes.fromhex(key_hex)
    if len(key) != 32:
        raise ValueError("32-byte key required")
    available = slots(stego)
    if len(available) < HEADER_BITS:
        raise ValueError("incomplete header")
    header = "".join(bit_for(token, (CLASSES[class_id][::2], CLASSES[class_id][1::2]))
                     for _, _, _, class_id, token in available[:HEADER_BITS])
    nonce = int(header[:128], 2).to_bytes(16, "big").hex()
    length = int(header[128:], 2)
    if length > len(available) - HEADER_BITS:
        raise ValueError("incomplete payload")
    return "".join(bit_for(token, groups(key, nonce, index, class_id))
                   for index, _, _, class_id, token in available[HEADER_BITS:HEADER_BITS + length])


if __name__ == "__main__":
    def deny_external_io(event, args):
        if event == "open" or event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("receiver external I/O blocked after public-code imports")
    sys.addaudithook(deny_external_io)
    guard_checks = {}
    for event in ("open", "socket.__new__", "subprocess.Popen"):
        try:
            sys.audit(event)
        except RuntimeError:
            guard_checks[event] = True
        else:
            raise AssertionError("receiver I/O audit policy self-check failed")
    incoming = json.load(sys.stdin)
    if set(incoming) != {"key_hex", "stego_text"}:
        raise ValueError("receiver input allowlist violated")
    print(json.dumps({"payload_bits": decode(incoming["stego_text"], incoming["key_hex"]),
                      "receiver_io_guard_checks": guard_checks}))
