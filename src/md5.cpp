#include "md5.h"
#include <cstdint>
#include <cstring>
#include <string>
namespace bpe {
namespace {

const std::uint32_t kShifts[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

const std::uint32_t kConstants[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
    0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
    0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
    0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
    0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

std::uint32_t round_function(int i, std::uint32_t b, std::uint32_t c,
                             std::uint32_t d, std::uint32_t& g) {
    if (i < 16) {
        g = static_cast<std::uint32_t>(i);
        return (b & c) | (~b & d);
    }
    if (i < 32) {
        g = static_cast<std::uint32_t>(5 * i + 1) % 16;
        return (d & b) | (~d & c);
    }
    if (i < 48) {
        g = static_cast<std::uint32_t>(3 * i + 5) % 16;
        return b ^ c ^ d;
    }
    g = static_cast<std::uint32_t>(7 * i) % 16;
    return c ^ (b | ~d);
}
std::uint32_t rotate_left(std::uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

void transform(const std::uint8_t* block, std::uint32_t& a, std::uint32_t& b,
               std::uint32_t& c, std::uint32_t& d) {
    std::uint32_t m[16];
    for (int i = 0; i < 16; ++i) {
        const int o = 4 * i;
        m[i] = static_cast<std::uint32_t>(block[o]) |
               (static_cast<std::uint32_t>(block[o + 1]) << 8) |
               (static_cast<std::uint32_t>(block[o + 2]) << 16) |
               (static_cast<std::uint32_t>(block[o + 3]) << 24);
    }
    const std::uint32_t aa = a, bb = b, cc = c, dd = d;
    for (int i = 0; i < 64; ++i) {
        std::uint32_t g;
        const std::uint32_t f = round_function(i, b, c, d, g);
        const std::uint32_t tmp = d;
        d = c;
        c = b;
        b = b + rotate_left(a + f + kConstants[i] + m[g], kShifts[i]);
        a = tmp;
    }
    a += aa;
    b += bb;
    c += cc;
    d += dd;
}

void pad_and_finish(const std::uint8_t* data, std::size_t len, std::uint32_t& a,
                    std::uint32_t& b, std::uint32_t& c, std::uint32_t& d) {
    const std::size_t rem = len % 64;
    std::uint8_t pad[64];
    if (rem > 0) {
        std::memcpy(pad, data + len - rem, rem);
    }
    pad[rem] = 0x80;
    const std::uint64_t bits = static_cast<std::uint64_t>(len) * 8;
    if (rem < 56) {
        std::memset(pad + rem + 1, 0, 56 - (rem + 1));
    } else {
        std::memset(pad + rem + 1, 0, 64 - (rem + 1));
        transform(pad, a, b, c, d);
        std::memset(pad, 0, 56);
    }
    for (int i = 0; i < 8; ++i) {
        pad[56 + i] = static_cast<std::uint8_t>(bits >> (8 * i));
    }
    transform(pad, a, b, c, d);
}
}
std::string md5_hex(const std::vector<Byte>& data) {
    std::uint32_t a = 0x67452301;
    std::uint32_t b = 0xefcdab89;
    std::uint32_t c = 0x98badcfe;
    std::uint32_t d = 0x10325476;
    const std::size_t full_blocks = data.size() / 64;
    for (std::size_t i = 0; i < full_blocks; ++i) {
        transform(data.data() + 64 * i, a, b, c, d);
    }
    pad_and_finish(data.data(), data.size(), a, b, c, d);
    static const char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(32);
    const std::uint32_t words[4] = {a, b, c, d};
    for (std::uint32_t word : words) {
        for (int i = 0; i < 4; ++i) {
            const std::uint8_t byte =
                static_cast<std::uint8_t>(word >> (8 * i));
            out.push_back(kHex[byte >> 4]);
            out.push_back(kHex[byte & 0x0f]);
        }
    }

    return out;
}

}
