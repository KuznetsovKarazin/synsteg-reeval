"""Exercise the production VocabClient parser against fake line transcripts.

No subprocess or C++ binary is constructed. The transcripts are invented unit
data and do not attest real model resources, compilation, or native tokenization.
"""
from collections import deque

import pytest

from synsteg_audit.framed_interval_llm_v1_vocab_collect import MAX_LINE, VocabClient


class ScriptedTransport:
    def __init__(self, lines, *, send_error=None):
        self.lines = deque(lines)
        self.sent = []
        self.send_error = send_error
        self.reads = 0

    def readline(self):
        self.reads += 1
        if not self.lines:
            raise EOFError("synthetic truncated transcript")
        value = self.lines.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    def send(self, line):
        self.sent.append(line)
        if self.send_error:
            raise self.send_error


def client(*responses):
    transport = ScriptedTransport(["READY_VOCAB 6 3 2048", *responses])
    return VocabClient(transport), transport


def inventory_lines():
    # Fake controls 0/1, ordinary byte pieces 2/3/4, silent ordinary token 5.
    return ["INVENTORY 6", "ROW 0 1 0 1", "ROW 1 0 1 0", "ROW 2 0 0 0",
        "ROW 3 0 0 0", "ROW 4 0 0 0", "ROW 5 0 0 1", "END_INVENTORY 6 3"]


def assert_terminal_without_retry(service, transport):
    assert service.terminal is True
    before = list(transport.sent)
    with pytest.raises(ValueError, match="terminal"):
        service.render([])
    service.quit()
    assert transport.sent == before


def test_fake_line_session_preserves_complete_inventory_flags_raw_fragments_and_quit():
    service, transport = client(*inventory_lines(), "TOK 0", "TOK 2 3 4", "TEXT c3",
        "TEXT c3a9", "TOK 3 0 2 1", "TEXT -", "BYE")
    assert service.ready == {"vocabulary_size": 6, "masked_tokens": 3, "context_tokens": 2048}
    rows = service.inventory()
    assert [row["token_id"] for row in rows] == list(range(6))
    assert rows[0] == {"token_id": 0, "is_control": True, "is_eog": False, "piece_empty": True}
    assert rows[5] == {"token_id": 5, "is_control": False, "is_eog": False, "piece_empty": True}
    assert service.tokenize("") == []
    assert service.tokenize("é") == [3, 4]
    assert service.render([3]) == b"\xc3"  # Invalid standalone UTF8 is preserved by TEXT.
    assert service.render((3, 4)) == "é".encode()
    assert service.tokenize("a", True, True) == [0, 2, 1]
    assert service.render([]) == b""
    service.quit()
    assert transport.sent == ["INVENTORY", "TOKENIZE 0 0 -", "TOKENIZE 0 0 c3a9",
        "TEXT 1 3", "TEXT 2 3 4", "TOKENIZE 1 1 61", "TEXT 0", "QUIT"]
    assert service.commands == transport.sent
    assert not transport.lines
    assert_terminal_without_retry(service, transport)


@pytest.mark.parametrize("line", [
    "READY 6 3 2048", "READY_VOCAB 6 3", "READY_VOCAB 6 3 2048 extra",
    "READY_VOCAB 06 3 2048", "READY_VOCAB +6 3 2048", "READY_VOCAB 6 -1 2048",
    "READY_VOCAB 6 0 2048", "READY_VOCAB 6 6 2048", "READY_VOCAB 6 3 1024",
    "READY_VOCAB 2000001 3 2048", "READY_VOCAB 6 3 2048 ", " READY_VOCAB 6 3 2048",
    "READY_VOCAB 6  3 2048", "READY_VOCAB\t6 3 2048", "READY_VOCAB 6 3 2048\r",
    "READY_VOCAB 6 3 2048\n", "READY_VOCAB é 3 2048", b"READY_VOCAB 6 3 2048", "",
    "X" * (MAX_LINE + 1),
])
def test_malformed_startup_is_rejected_without_sending_any_command(line):
    transport = ScriptedTransport([line])
    with pytest.raises(ValueError):
        VocabClient(transport)
    assert transport.sent == []


@pytest.mark.parametrize("index, replacement", [
    (0, "INVENTORY 5"), (0, "INVENTORY 6 extra"),
    (1, "ROW 1 1 0 1"), (1, "ROW 00 1 0 1"),
    (2, "ROW 0 0 1 0"), (2, "ROW 1 true 1 0"),
    (3, "ROW 2 0 2 0"), (3, "ROW 2 0 0"), (3, "ROW 2 0 0 0 extra"),
    (3, "ROW 2 1 0 0"),  # Recorded mask then disagrees with the actual complete scan.
    (7, "END_INVENTORY 6 2"), (7, "END_INVENTORY 5 3"),
])
def test_malformed_or_inconsistent_inventory_is_terminal(index, replacement):
    lines = inventory_lines()
    lines[index] = replacement
    service, transport = client(*lines)
    with pytest.raises(ValueError):
        service.inventory()
    assert transport.sent == ["INVENTORY"]
    assert_terminal_without_retry(service, transport)


@pytest.mark.parametrize("keep", [0, 1, 3, 7])
def test_truncated_inventory_is_preserved_as_terminal_failure_without_retry(keep):
    service, transport = client(*inventory_lines()[:keep])
    with pytest.raises(EOFError):
        service.inventory()
    assert_terminal_without_retry(service, transport)


def test_complete_inventory_cannot_be_requested_twice():
    service, transport = client(*inventory_lines())
    service.inventory()
    with pytest.raises(ValueError, match="already attempted"):
        service.inventory()
    assert transport.sent == ["INVENTORY"]


@pytest.mark.parametrize("response", [
    "TOK", "TEXT 61", "TOK 1", "TOK 0 2", "TOK 2 2", "TOK 01 2", "TOK 1 02",
    "TOK 1 -1", "TOK 1 6", "TOK 1 2 extra", "TOK 1 2 ", "TOK 1 2\t", "TOK 1 2\n",
    "TOK 2049 " + " ".join(["2"] * 2049),
])
def test_invalid_token_response_is_terminal(response):
    service, transport = client(response)
    with pytest.raises(ValueError):
        service.tokenize("a")
    assert transport.sent == ["TOKENIZE 0 0 61"]
    assert_terminal_without_retry(service, transport)


def test_token_response_at_public_2048_token_bound_is_accepted():
    service, transport = client("TOK 2048 " + " ".join(["2"] * 2048))
    assert service.tokenize("a") == [2] * 2048
    assert transport.sent == ["TOKENIZE 0 0 61"]


@pytest.mark.parametrize("text, add, parse", [
    (b"a", False, False), ("a", 0, False), ("a", False, 1),
    ("\ud800", False, False), ("a" * 65537, False, False), ("é" * 32769, False, False),
])
def test_invalid_tokenize_arguments_never_reach_fake_transport(text, add, parse):
    service, transport = client()
    with pytest.raises((ValueError, UnicodeError)):
        service.tokenize(text, add, parse)
    assert transport.sent == []
    assert service.terminal is False


@pytest.mark.parametrize("response", [
    "TEXT", "TEXT 6", "TEXT 6A", "TEXT gg", "TEXT --", "TEXT - extra", "TEXT 61 extra",
    "TEXT 61 ", "TEXT 61\r", "TEXT 00" + "61" * 65536,
])
def test_invalid_or_oversized_raw_text_response_is_terminal(response):
    service, transport = client(response)
    with pytest.raises(ValueError):
        service.render([2])
    assert_terminal_without_retry(service, transport)


def test_text_byte_cap_and_non_utf8_output_are_distinct_contracts():
    service, transport = client("TEXT " + "ff" * 65536)
    assert service.render([2]) == b"\xff" * 65536
    assert service.terminal is False
    assert transport.sent == ["TEXT 1 2"]


@pytest.mark.parametrize("ids", [None, "2", {2}, [True], [2.0], [-1], [6], [2] * 2049])
def test_invalid_render_arguments_never_reach_fake_transport(ids):
    service, transport = client()
    with pytest.raises(ValueError):
        service.render(ids)
    assert transport.sent == []
    assert service.terminal is False


@pytest.mark.parametrize("operation", ["tokenize", "render", "inventory"])
def test_transport_send_error_is_terminal_for_every_command(operation):
    transport = ScriptedTransport(["READY_VOCAB 6 3 2048"], send_error=OSError("synthetic failed write"))
    service = VocabClient(transport)
    with pytest.raises(OSError, match="synthetic failed write"):
        {"tokenize": lambda: service.tokenize("a"), "render": lambda: service.render([]),
         "inventory": service.inventory}[operation]()
    assert len(transport.sent) == 1
    assert_terminal_without_retry(service, transport)


@pytest.mark.parametrize("response", ["BYE ", "BYE extra", "READY_VOCAB 6 3 2048", EOFError("synthetic EOF")])
def test_malformed_close_is_terminal_and_never_retried(response):
    service, transport = client(response)
    with pytest.raises((ValueError, EOFError)):
        service.quit()
    assert transport.sent == ["QUIT"]
    assert_terminal_without_retry(service, transport)
