#include "bpe.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace bpe {
namespace {

inline bool is_separator(Byte b) {
    return b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D || b == 0x00;
}

inline std::uint64_t hasless_any(std::uint64_t word, unsigned limit) {
    const std::uint64_t ones = ~0ULL / 255;
    return (word - ones * limit) & ~word & (ones * 128);
}
}
std::vector<Byte> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("cannot open file: " + path);
    }
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<Byte> data(static_cast<std::size_t>(size) + 1);
    if (size > 0) {
        in.read(reinterpret_cast<char*>(data.data()), size);
    }
    data.pop_back();
    return data;
}
// split_words: in place — whitespace becomes NUL, each Word is a C-string.
std::vector<Word> split_words(std::vector<Byte>& input) {
    input.push_back(Byte('\0'));
    std::vector<Word> words;
    words.reserve(input.size() / 3 + 1);
    Byte* p = input.data();
    const Byte* const end = input.data() + input.size();
    while (p < end) {
        while (p < end && is_separator(*p)) {
            ++p;
        }
        if (p == end) {
            break;
        }
        words.push_back(Word{p});

        while (p + 8 <= end) {
            std::uint64_t chunk;
            std::memcpy(&chunk, p, 8);
            const std::uint64_t hits = hasless_any(chunk, 0x21);
            if (hits != 0) {
                p += __builtin_ctzll(hits) >> 3;
                break;
            }
            p += 8;
        }

        while (p < end && !is_separator(*p)) {
            ++p;
        }
        *p = Byte('\0');
        ++p;
    }

    return words;
}

}
