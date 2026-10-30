// Vocab-only preflight bridge, targeting llama.cpp ea9e3446b5c84a7f972cbbdea9df615fc4ff9159.
// Source-only until a separately reviewed build and native preflight are recorded.
// No context, inference, generation, dynamic backend loading, or network API.
// Output rendering intentionally preserves the frozen bridge's cumulative
// token_to_piece(..., lstrip=0, special=false) byte concatenation.
#include "llama.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t MAX_TEXT_BYTES = 65536;
constexpr std::size_t MAX_LINE_BYTES = 131104;
constexpr std::size_t MAX_TOKEN_IDS = 2048;
constexpr int32_t MAX_VOCABULARY = 2000000;
constexpr int32_t DECLARED_CONTEXT_TOKENS = 2048;

[[noreturn]] void fail(const char *message) {
    throw std::runtime_error(message);
}

// Read one bounded ASCII line without getline's unbounded preallocation.
// A partial line at EOF, CRLF, tabs, NUL and non-ASCII bytes are refusals.
std::string read_line(std::istream &input) {
    std::string line;
    while (true) {
        const int ch = input.get();
        if (ch == std::char_traits<char>::eof()) fail("EOF before complete command");
        if (ch == '\n') break;
        if (ch < 0x20 || ch > 0x7e) fail("command must use printable ASCII and LF");
        if (line.size() >= MAX_LINE_BYTES) fail("command line byte cap exceeded");
        line.push_back(static_cast<char>(ch));
    }
    if (line.empty() || line.front() == ' ' || line.back() == ' ')
        fail("empty command or edge whitespace");
    return line;
}

std::vector<std::string> fields(const std::string &line) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    for (std::size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ' ') {
            if (begin == i) fail("single-space fields required");
            result.push_back(line.substr(begin, i - begin));
            if (result.size() > MAX_TOKEN_IDS + 2) fail("too many command fields");
            begin = i + 1;
        }
    }
    return result;
}

uint32_t decimal(const std::string &raw, uint32_t maximum) {
    if (raw.empty() || (raw.size() > 1 && raw[0] == '0')) fail("noncanonical decimal");
    uint32_t value = 0;
    for (char ch : raw) {
        if (ch < '0' || ch > '9') fail("unsigned decimal required");
        const uint32_t digit = static_cast<uint32_t>(ch - '0');
        if (digit > maximum || value > (maximum - digit) / 10)
            fail("decimal value exceeds cap");
        value = value * 10 + digit;
    }
    return value;
}

int hex_digit(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    fail("lowercase hexadecimal required");
}

std::string unhex(const std::string &hex) {
    if (hex == "-") return {};
    if (hex.empty() || hex.size() % 2 || hex.size() / 2 > MAX_TEXT_BYTES)
        fail("hex byte count invalid");
    std::string raw;
    raw.reserve(hex.size() / 2);
    for (std::size_t i = 0; i < hex.size(); i += 2)
        raw.push_back(static_cast<char>(16 * hex_digit(hex[i]) + hex_digit(hex[i + 1])));
    return raw;
}

std::string hexbytes(const std::string &raw) {
    if (raw.empty()) return "-";
    constexpr char digits[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(raw.size() * 2);
    for (unsigned char ch : raw) {
        hex.push_back(digits[ch >> 4]);
        hex.push_back(digits[ch & 15]);
    }
    return hex;
}

// Reject overlong encodings, surrogates, out-of-range scalars and partial input.
// TEXT deliberately does not apply this check: token pieces may be raw fragments.
void strict_utf8(const std::string &raw) {
    for (std::size_t i = 0; i < raw.size();) {
        const auto lead = static_cast<unsigned char>(raw[i++]);
        if (lead <= 0x7f) continue;
        uint32_t scalar = 0;
        uint32_t minimum = 0;
        std::size_t trailing = 0;
        if (lead >= 0xc2 && lead <= 0xdf) { scalar = lead & 0x1f; minimum = 0x80; trailing = 1; }
        else if (lead >= 0xe0 && lead <= 0xef) { scalar = lead & 0x0f; minimum = 0x800; trailing = 2; }
        else if (lead >= 0xf0 && lead <= 0xf4) { scalar = lead & 0x07; minimum = 0x10000; trailing = 3; }
        else fail("invalid UTF8 lead");
        if (trailing > raw.size() - i) fail("partial UTF8 input");
        for (std::size_t j = 0; j < trailing; ++j) {
            const auto ch = static_cast<unsigned char>(raw[i++]);
            if ((ch & 0xc0) != 0x80) fail("invalid UTF8 continuation");
            scalar = (scalar << 6) | (ch & 0x3f);
        }
        if (scalar < minimum || scalar > 0x10ffff || (scalar >= 0xd800 && scalar <= 0xdfff))
            fail("invalid UTF8 scalar");
    }
}

std::string piece(const llama_vocab *vocab, llama_token id) {
    std::vector<char> buffer(64);
    int32_t size = llama_token_to_piece(vocab, id, buffer.data(),
        static_cast<int32_t>(buffer.size()), 0, false);
    if (size < 0) {
        const int64_t needed = -static_cast<int64_t>(size);
        if (needed > static_cast<int64_t>(MAX_TEXT_BYTES)) fail("token piece byte cap exceeded");
        buffer.resize(static_cast<std::size_t>(needed));
        size = llama_token_to_piece(vocab, id, buffer.data(),
            static_cast<int32_t>(buffer.size()), 0, false);
    }
    if (size < 0 || static_cast<std::size_t>(size) > buffer.size()) fail("token piece failed");
    return std::string(buffer.data(), static_cast<std::size_t>(size));
}

struct InventoryRow {
    bool control;
    bool eog;
    bool empty;
};

std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text,
                                 bool add_special, bool parse_special, int32_t vocabulary_size) {
    strict_utf8(text);
    std::vector<llama_token> ids(MAX_TOKEN_IDS);
    const int32_t size = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
        ids.data(), static_cast<int32_t>(ids.size()), add_special, parse_special);
    // Any negative result, including INT32_MIN, is refused without resizing past
    // the public token cap. Successful IDs must all belong to the native vocab.
    if (size < 0 || static_cast<std::size_t>(size) > ids.size()) fail("token count cap or tokenize failure");
    ids.resize(static_cast<std::size_t>(size));
    for (llama_token id : ids)
        if (id < 0 || id >= vocabulary_size) fail("tokenizer returned invalid native ID");
    return ids;
}
}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: framed_interval_llm_vocab_bridge MODEL_GGUF\n";
        return 2;
    }
    llama_model *model = nullptr;
    bool backend_initialized = false;
    try {
        // Documented library lifecycle only; no dynamic device/backend loading.
        llama_backend_init();
        backend_initialized = true;
        auto mp = llama_model_default_params();
        mp.vocab_only = true;
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(argv[1], mp);
        if (!model) fail("vocabulary-only model load failed");
        const llama_vocab *vocab = llama_model_get_vocab(model);
        if (!vocab) fail("native vocabulary missing");
        const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
        if (vocabulary_size <= 0 || vocabulary_size > MAX_VOCABULARY) fail("native vocabulary size exceeds cap");
        std::vector<InventoryRow> inventory;
        inventory.reserve(static_cast<std::size_t>(vocabulary_size));
        int32_t masked = 0;
        for (int32_t id = 0; id < vocabulary_size; ++id) {
            InventoryRow row{llama_vocab_is_control(vocab, id), llama_vocab_is_eog(vocab, id),
                             piece(vocab, id).empty()};
            inventory.push_back(row);
            masked += row.control || row.eog || row.empty;
        }
        // 2048 is a declared comparison-contract bound, not a created context.
        std::cout << "READY_VOCAB " << vocabulary_size << ' ' << masked << ' '
                  << DECLARED_CONTEXT_TOKENS << '\n' << std::flush;
        bool finished = false;
        while (!finished) {
            const auto args = fields(read_line(std::cin));
            if (args[0] == "QUIT") {
                if (args.size() != 1) fail("QUIT takes no arguments");
                std::cout << "BYE\n" << std::flush;
                finished = true;
            } else if (args[0] == "INVENTORY") {
                if (args.size() != 1) fail("INVENTORY takes no arguments");
                std::cout << "INVENTORY " << vocabulary_size << '\n';
                for (int32_t id = 0; id < vocabulary_size; ++id) {
                    const auto &row = inventory[static_cast<std::size_t>(id)];
                    std::cout << "ROW " << id << ' ' << int(row.control) << ' '
                              << int(row.eog) << ' ' << int(row.empty) << '\n';
                }
                std::cout << "END_INVENTORY " << vocabulary_size << ' ' << masked << '\n' << std::flush;
            } else if (args[0] == "TOKENIZE") {
                if (args.size() != 4) fail("TOKENIZE requires two flags and hex bytes");
                const bool add = decimal(args[1], 1) != 0;
                const bool parse = decimal(args[2], 1) != 0;
                const auto ids = tokenize(vocab, unhex(args[3]), add, parse, vocabulary_size);
                std::cout << "TOK " << ids.size();
                for (llama_token id : ids) std::cout << ' ' << id;
                std::cout << '\n' << std::flush;
            } else if (args[0] == "TEXT") {
                if (args.size() < 2) fail("TEXT token count required");
                const auto count = decimal(args[1], static_cast<uint32_t>(MAX_TOKEN_IDS));
                if (args.size() != static_cast<std::size_t>(count) + 2) fail("TEXT exact token count required");
                std::vector<llama_token> ids;
                ids.reserve(count);
                for (std::size_t i = 0; i < count; ++i)
                    ids.push_back(static_cast<llama_token>(decimal(args[i + 2], vocabulary_size - 1)));
                std::string raw;
                for (llama_token id : ids) {
                    const auto next = piece(vocab, id);
                    if (next.size() > MAX_TEXT_BYTES - raw.size()) fail("TEXT byte cap exceeded");
                    raw += next;
                }
                std::cout << "TEXT " << hexbytes(raw) << '\n' << std::flush;
            } else {
                fail("unknown command");
            }
            if (!std::cout) fail("protocol output write failed");
        }
        llama_model_free(model);
        model = nullptr;
        llama_backend_free();
        backend_initialized = false;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "VOCAB_BRIDGE_ERROR " << error.what() << '\n';
        if (model) llama_model_free(model);
        if (backend_initialized) llama_backend_free();
        return 1;
    }
}
