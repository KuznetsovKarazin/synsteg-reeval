"""Pinned finite-budget LLM channel mechanics; no linguistic/security claim."""
from __future__ import annotations
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import time

VERSION = "seeded-pair-llm-v1"
TOTAL = 1 << 24
DEMO_KEY = bytes(range(32))


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def subkey(key, label):
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("key must be exactly32bytes")
    return hmac.new(key, (VERSION + ":" + label).encode(), hashlib.sha256).digest()


def quantize(logits, temperature=.8, total=TOTAL):
    """Exact integer categorical law derived from pinned float32 top-k logits."""
    if not logits or len({int(t) for t, _ in logits}) != len(logits):
        raise ValueError("empty/duplicate logits")
    decoded = [(int(token), raw, struct.unpack(">f", bytes.fromhex(raw))[0])
               for token, raw in logits]
    if any(len(raw) != 8 or not math.isfinite(value) for _, raw, value in decoded):
        raise ValueError("nonfinite/invalid float32")
    if temperature <= 0 or type(total) is not int or total < 1:
        raise ValueError("invalid quantization parameters")
    decoded.sort(key=lambda x: (-x[2], x[0]))
    maximum = decoded[0][2]
    masses = [math.exp((value - maximum) / temperature) for _, _, value in decoded]
    denominator = math.fsum(masses)
    scaled = [m / denominator * total for m in masses]
    weights = [math.floor(x) for x in scaled]
    remaining = total - sum(weights)
    if not 0 <= remaining <= len(weights):
        raise ValueError("floating quantization residual outside expected bounds")
    order = sorted(range(len(weights)), key=lambda i: (-(scaled[i] - weights[i]), decoded[i][0]))
    for i in order[:remaining]:
        weights[i] += 1
    return [{"token_id": token, "logit_hex": raw, "weight": weight}
            for (token, raw, _), weight in zip(decoded, weights)]


def draw(weights, key, context, domain, branch):
    total = sum(row["weight"] for row in weights)
    if total <= 0:
        raise ValueError("zero distribution")
    bound = (1 << 256) - (1 << 256) % total
    k = subkey(key, domain)
    counter = 0
    while True:
        raw = hmac.new(k, canonical([context, branch, counter]), hashlib.sha256).digest()
        counter += 1
        value = int.from_bytes(raw, "big")
        if value < bound:
            residue = value % total
            for row in weights:
                if residue < row["weight"]:
                    return row["token_id"], counter
                residue -= row["weight"]


def context_for(envelope, step, prompt_ids, generated_ids):
    return [VERSION, envelope["protocol_sha256"], envelope["resources_sha256"],
            envelope["prompt_id"], envelope["nonce_hex"], step,
            digest(prompt_ids + generated_ids)]


def frame_bits(payload, key, envelope):
    if not isinstance(payload, bytes) or len(payload) * 8 > 65535:
        raise ValueError("invalid payload")
    body = (len(payload) * 8).to_bytes(2, "big") + payload
    tag = hmac.new(subkey(key, "frame-auth"), canonical(envelope) + b"\0" + body,
                   hashlib.sha256).digest()[:16]
    return "".join(f"{x:08b}" for x in body + tag)


def read_frame(bits, key, envelope):
    if len(bits) < 16:
        return {"status": "insufficient_capacity", "decoded_bits": len(bits)}
    length = int(bits[:16], 2)
    if length % 8:
        return {"status": "invalid_frame", "decoded_bits": len(bits)}
    required = 16 + length + 128
    if len(bits) < required:
        return {"status": "insufficient_capacity", "decoded_bits": len(bits),
                "required_frame_bits": required}
    raw = int(bits[:required], 2).to_bytes(required // 8, "big")
    tag = hmac.new(subkey(key, "frame-auth"), canonical(envelope) + b"\0" + raw[:-16],
                   hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(tag, raw[-16:]):
        return {"status": "authentication_failed", "decoded_bits": len(bits)}
    return {"status": "ok", "payload_hex": raw[2:-16].hex(),
            "decoded_bits": len(bits), "required_frame_bits": required}


def padding_bit(key, envelope, index):
    value = hmac.new(subkey(key, "padding"),
                     canonical([envelope, index // 256]), hashlib.sha256).digest()
    return (value[(index % 256) // 8] >> (7 - index % 8)) & 1


def prompt_text(protocol, prompt_id):
    prompt = next(p for p in protocol["prompts"] if p["id"] == prompt_id)
    return ("<|im_start|>system\n" + protocol["system_prompt"] + "<|im_end|>\n"
            "<|im_start|>user\n" + prompt["user"] + "<|im_end|>\n"
            "<|im_start|>assistant\n")


class NativeBridge:
    """One owned native process, exact line protocol, hard endpoint deadline."""
    def __init__(self, binary, model, cpu_backend, log_path, deadline):
        self.deadline = deadline
        self.buffer = b""
        self.log = Path(log_path).open("xb")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_"))}
        env["LD_LIBRARY_PATH"] = str(Path(cpu_backend).parent)
        self.process = subprocess.Popen(
            [str(binary), str(model), str(cpu_backend)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, env=env)
        try:
            line = self._line().split()
            if len(line) != 4 or line[0] != "READY":
                raise ValueError("invalid native startup")
            self.vocabulary_size, self.masked_tokens, self.context_tokens = map(int, line[1:])
        except BaseException:
            self.close()
            raise

    def _line(self):
        while b"\n" not in self.buffer:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("native endpoint deadline")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError("native endpoint deadline")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise RuntimeError("native process ended before response")
            self.buffer += block
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("ascii")

    def command(self, line):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("native endpoint deadline")
        self.process.stdin.write((line + "\n").encode("ascii"))
        self.process.stdin.flush()
        return self._line()

    def tokenize(self, text, add_special=False, parse_special=False):
        raw = text.encode("utf-8").hex() or "-"
        fields = self.command(f"TOKENIZE {int(add_special)} {int(parse_special)} {raw}").split()
        if fields[0] != "TOK" or len(fields) != int(fields[1]) + 2:
            raise ValueError("invalid tokenizer response")
        return list(map(int, fields[2:]))

    def render(self, ids):
        fields = self.command("TEXT " + str(len(ids)) + " " + " ".join(map(str, ids))).split()
        if len(fields) != 2 or fields[0] != "TEXT":
            raise ValueError("invalid renderer response")
        return b"" if fields[1] == "-" else bytes.fromhex(fields[1])

    def start(self, ids):
        response = self.command("PROMPT " + str(len(ids)) + " " + " ".join(map(str, ids)))
        if response != f"OK {len(ids)}":
            raise ValueError("prompt evaluation count mismatch")
        self.evaluated = len(ids)

    def step(self, token):
        self.evaluated += 1
        if self.command(f"STEP {token}") != f"OK {self.evaluated}":
            raise ValueError("step evaluation count mismatch")

    def distribution(self):
        fields = self.command("DIST").split()
        if fields[0] != "DIST" or int(fields[1]) != self.vocabulary_size or int(fields[2]) != self.masked_tokens:
            raise ValueError("native distribution metadata drift")
        count = int(fields[3])
        if count != 50 or len(fields) != 4 + 2 * count:
            raise ValueError("unexpected native top-k")
        return [(int(fields[i]), fields[i+1]) for i in range(4, len(fields), 2)]

    def close(self):
        if getattr(self, "process", None) is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
        if getattr(self, "log", None) is not None:
            self.log.close()


def _trace_write(stream, row):
    if stream is not None:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


def generate(bridge, protocol, envelope, key, payload=None, *, trace=None, raw_stream=None):
    """Generate all N tokens, even when the full frame cannot be delivered."""
    arm = envelope["arm"]
    if arm not in ("stego", "ordinary"):
        raise ValueError("invalid arm")
    prompt_ids = bridge.tokenize(prompt_text(protocol, envelope["prompt_id"]), True, True)
    if len(prompt_ids) + envelope["units"] > 2048:
        raise ValueError("context budget exceeded")
    bridge.start(prompt_ids)
    frame = frame_bits(payload, key, envelope) if arm == "stego" else ""
    ids, raw, capacity, samples = [], b"", 0, 0
    control_key = hashlib.sha256((VERSION + ":ordinary-control").encode()).digest()
    for i in range(envelope["units"]):
        weights = quantize(bridge.distribution(), protocol["sampler"]["temperature"])
        context = context_for(envelope, i, prompt_ids, ids)
        if arm == "stego":
            a, ca = draw(weights, key, context, "sample", 0)
            b, cb = draw(weights, key, context, "sample", 1)
            offset = capacity
            if a == b:
                chosen, carried = a, None
            else:
                carried = int(frame[capacity]) if capacity < len(frame) else padding_bit(key, envelope, capacity - len(frame))
                chosen = (a, b)[carried]
                capacity += 1
            candidates, counters = [a, b], [ca, cb]
        else:
            chosen, count = draw(weights, control_key, context, "ordinary-sample", 0)
            candidates, counters, carried, offset = [chosen], [count], None, None
        samples += len(candidates)
        fragment = bridge.render([chosen])
        raw += fragment
        if raw_stream is not None:
            raw_stream.write(fragment)
            raw_stream.flush()
        row = {"step": i, "prefix_sha256": context[-1], "weights": weights,
               "weights_sha256": digest(weights), "candidates": candidates,
               "draw_counters": counters, "selected_token": chosen,
               "payload_cursor_before": offset, "carried_bit": carried,
               "capacity_bits_after": capacity if arm == "stego" else None,
               "token_bytes_hex": fragment.hex(),
               "raw_distinct_pair_probability": 1 - sum(w["weight"] ** 2 for w in weights) / TOTAL**2}
        _trace_write(trace, row)
        ids.append(chosen)
        bridge.step(chosen)
    result = {"status": "generated", "arm": arm, "envelope": envelope,
              "prompt_tokens": len(prompt_ids), "prompt_token_ids": prompt_ids, "generated_token_ids": ids,
              "generated_tokens": len(ids), "capacity_bits": capacity if arm == "stego" else None,
              "frame_bits": len(frame) if arm == "stego" else None,
              "frame_completed": capacity >= len(frame) if arm == "stego" else None,
              "sender_categorical_draws": samples, "raw_bytes": len(raw),
              "raw_sha256": hashlib.sha256(raw).hexdigest()}
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        result.update(status="invalid_utf8", text_roundtrip_exact=False, stego_text=None)
        return result
    retokenized = bridge.tokenize(text, False, False)
    result.update(stego_text=text, canonical_text_token_ids=retokenized,
                  text_roundtrip_exact=retokenized == ids)
    if retokenized != ids:
        result["status"] = "tokenization_mismatch"
    elif arm == "stego" and capacity < len(frame):
        result["status"] = "insufficient_capacity"
    return result


def receive(bridge, protocol, envelope, key, stego_text, *, trace=None):
    """Receiver obtains text only, never sender token IDs or sender traces."""
    ids = bridge.tokenize(stego_text, False, False)
    prompt_ids = bridge.tokenize(prompt_text(protocol, envelope["prompt_id"]), True, True)
    result = {"retokenized_token_ids": ids, "retokenized_tokens": len(ids),
              "prompt_tokens": len(prompt_ids), "prompt_token_ids": prompt_ids}
    if len(ids) != envelope["units"]:
        return {**result, "status": "token_count_mismatch"}
    if len(prompt_ids) + len(ids) > 2048:
        return {**result, "status": "context_budget_reject"}
    bridge.start(prompt_ids)
    observed, bits = [], []
    for i, token in enumerate(ids):
        weights = quantize(bridge.distribution(), protocol["sampler"]["temperature"])
        context = context_for(envelope, i, prompt_ids, observed)
        a, ca = draw(weights, key, context, "sample", 0)
        b, cb = draw(weights, key, context, "sample", 1)
        if token not in (a, b):
            _trace_write(trace, {"step": i, "prefix_sha256": context[-1], "weights": weights,
                                 "weights_sha256": digest(weights), "candidates": [a, b],
                                 "draw_counters": [ca, cb], "observed_token": token,
                                 "status": "unrecognized_token"})
            return {**result, "status": "unrecognized_token", "failure_step": i, "decoded_bits": len(bits)}
        bit = None if a == b else int(token == b)
        if bit is not None:
            bits.append(str(bit))
        _trace_write(trace, {"step": i, "prefix_sha256": context[-1], "weights": weights,
                             "weights_sha256": digest(weights), "candidates": [a, b],
                             "draw_counters": [ca, cb], "observed_token": token,
                             "decoded_bit": bit, "decoded_bits_after": len(bits), "status": "recognized"})
        observed.append(token)
        bridge.step(token)
    return {**result, **read_frame("".join(bits), key, envelope)}
