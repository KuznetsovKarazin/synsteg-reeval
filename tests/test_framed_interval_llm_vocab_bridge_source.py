"""Static guards for the uncompiled vocab-only bridge; not native execution.

These tests check the reviewed source contract. They do not establish C++
compilability, ABI compatibility, actual GGUF loading behavior or tokenizer output.
Native build/preflight evidence must be collected separately after source freeze.
"""
from pathlib import Path
import re

import pytest


SOURCE_PATH = Path(__file__).resolve().parents[1] / "scripts/framed_interval_llm_vocab_bridge.cpp"
SOURCE = SOURCE_PATH.read_text()
CODE = re.sub(r"//[^\n]*|/\*.*?\*/", "", SOURCE, flags=re.DOTALL)


def test_native_call_surface_contains_vocabulary_and_lifecycle_only():
    calls = set(re.findall(r"\b(llama_[a-zA-Z0-9_]+)\s*\(", CODE))
    assert calls == {
        "llama_backend_init", "llama_backend_free", "llama_model_default_params",
        "llama_model_load_from_file", "llama_model_free", "llama_model_get_vocab",
        "llama_vocab_n_tokens", "llama_vocab_is_control", "llama_vocab_is_eog",
        "llama_token_to_piece", "llama_tokenize",
    }
    assert not re.search(r"\b(?:ggml_backend_load|system|popen|exec[lvpe]*|fork|socket)\s*\(", CODE)
    assert not re.search(r"\b(?:llama_context|llama_batch|llama_sampler)\b", CODE)


def test_vocab_only_and_zero_gpu_layers_are_set_before_the_only_model_load():
    assert len(re.findall(r"\bllama_model_load_from_file\s*\(", CODE)) == 1
    assert re.findall(r"mp\.vocab_only\s*=\s*([^;]+);", CODE) == ["true"]
    assert re.findall(r"mp\.n_gpu_layers\s*=\s*([^;]+);", CODE) == ["0"]
    assert CODE.index("mp.vocab_only = true;") < CODE.index("llama_model_load_from_file(argv[1], mp)")
    assert CODE.index("mp.n_gpu_layers = 0;") < CODE.index("llama_model_load_from_file(argv[1], mp)")
    assert "argc != 2" in CODE


def test_command_dispatch_has_exactly_four_non_inference_commands():
    assert re.findall(r'args\[0\]\s*==\s*"([A-Z_]+)"', CODE) == ["QUIT", "INVENTORY", "TOKENIZE", "TEXT"]
    assert 'fail("unknown command")' in CODE
    assert 'fail("QUIT takes no arguments")' in CODE
    assert 'fail("INVENTORY takes no arguments")' in CODE
    assert 'fail("TOKENIZE requires two flags and hex bytes")' in CODE
    assert 'fail("TEXT exact token count required")' in CODE


def test_renderer_preserves_old_raw_piece_concatenation_and_does_not_detokenize():
    calls = re.findall(r"llama_token_to_piece\([^;]+;", CODE)
    assert len(calls) == 2
    assert all(re.search(r",\s*0,\s*false\);$", call) for call in calls)
    assert "raw += next;" in CODE
    assert "llama_detokenize" not in CODE
    text_branch = CODE[CODE.index('args[0] == "TEXT"'):]
    assert "strict_utf8(" not in text_branch


@pytest.mark.parametrize("name, expected", [
    ("MAX_TEXT_BYTES", 65536), ("MAX_LINE_BYTES", 131104),
    ("MAX_TOKEN_IDS", 2048), ("MAX_VOCABULARY", 2000000), ("DECLARED_CONTEXT_TOKENS", 2048),
])
def test_public_parser_caps_remain_explicit(name, expected):
    match = re.search(r"constexpr\s+[^;]+\b" + name + r"\s*=\s*(\d+);", CODE)
    assert match and int(match.group(1)) == expected


def test_input_line_is_bounded_before_append_and_rejects_partial_eof():
    function = CODE[CODE.index("std::string read_line("):CODE.index("std::vector<std::string> fields(")]
    assert "input.get()" in function
    assert function.index("line.size() >= MAX_LINE_BYTES") < function.index("line.push_back(")
    assert "ch < 0x20 || ch > 0x7e" in function
    assert 'fail("EOF before complete command")' in function
    assert not re.search(r"\b(?:getline|atoi|strtol|stoi|istringstream)\b", CODE)


def test_tokenize_is_called_only_after_strict_utf8_and_uses_explicit_flags():
    function = CODE[CODE.index("std::vector<llama_token> tokenize("):CODE.index("int main(")]
    assert function.index("strict_utf8(text);") < function.index("llama_tokenize(")
    assert "add_special, parse_special)" in function
    assert "std::vector<llama_token> ids(MAX_TOKEN_IDS)" in function
    assert "size < 0" in function
    assert "id < 0 || id >= vocabulary_size" in function


def test_complete_native_inventory_and_mask_are_computed_before_ready():
    begin = CODE.index("for (int32_t id = 0; id < vocabulary_size; ++id)")
    ready = CODE.index('"READY_VOCAB "')
    scan = CODE[begin:ready]
    assert "llama_vocab_is_control(vocab, id)" in scan
    assert "llama_vocab_is_eog(vocab, id)" in scan
    assert "piece(vocab, id).empty()" in scan
    assert "masked += row.control || row.eog || row.empty;" in scan
    assert '"END_INVENTORY "' in CODE


def test_piece_buffer_failure_handles_int32_min_without_signed_overflow():
    assert "-static_cast<int64_t>(size)" in CODE
    assert "needed > static_cast<int64_t>(MAX_TEXT_BYTES)" in CODE
    assert "next.size() > MAX_TEXT_BYTES - raw.size()" in CODE


def test_error_path_is_terminal_and_does_not_emit_a_success_response():
    catch = CODE[CODE.index("} catch (const std::exception &error)"):]
    assert 'std::cerr << "VOCAB_BRIDGE_ERROR "' in catch
    assert "std::cout" not in catch
    assert "return 1;" in catch
    assert "if (model) llama_model_free(model);" in catch
    assert "if (backend_initialized) llama_backend_free();" in catch
