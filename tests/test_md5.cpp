// Tests for the MD5 fingerprint helper.
//
// Expected digests come from the reference test suite in RFC 1321, appendix
// A.5. The multi-byte cases were re-derived locally with python3 hashlib and
// cross-checked against the RFC's published values.

#include "md5.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace {

std::vector<bpe::Byte> bytes_of(const std::string& s) {
    return std::vector<bpe::Byte>(s.begin(), s.end());
}

}  // namespace

TEST(Md5, RfcEmptyString) {
    EXPECT_EQ(bpe::md5_hex(bytes_of("")), "d41d8cd98f00b204e9800998ecf8427e");
}

TEST(Md5, RfcAbc) {
    EXPECT_EQ(bpe::md5_hex(bytes_of("abc")),
              "900150983cd24fb0d6963f7d28e17f72");
}

TEST(Md5, RfcMessageDigest) {
    EXPECT_EQ(bpe::md5_hex(bytes_of("message digest")),
              "f96b697d7cb7938d525a2f31aaf161d0");
}

TEST(Md5, RfcAlphabet) {
    EXPECT_EQ(bpe::md5_hex(bytes_of("abcdefghijklmnopqrstuvwxyz")),
              "c3fcd3d76192e4007dfb496cca67e13b");
}

TEST(Md5, RfcAlphanumeric) {
    EXPECT_EQ(
        bpe::md5_hex(bytes_of(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")),
        "d174ab98d277d9f5a5611c2c9f419d9f");
}

TEST(Md5, RfcDigits) {
    EXPECT_EQ(bpe::md5_hex(bytes_of(
                  "123456789012345678901234567890123456789012345678901234567890"
                  "12345678901234567890")),
              "57edf4a22be3c955ac49da2e2107b67a");
}

TEST(Md5, PaddingFitsSingleBlock) {
    // 55 bytes: padding (0x80 + length) fits exactly in the final block, so
    // only one extra block is processed.
    EXPECT_EQ(bpe::md5_hex(bytes_of(std::string(55, 'a'))),
              "ef1772b6dff9a122358552954ad0df65");
}

TEST(Md5, PaddingOverflowsSecondBlock) {
    // 56 bytes: 0x80 lands at offset 56, so the pad block holds the marker and
    // a second block carries the 64-bit length (the rem >= 56 branch).
    EXPECT_EQ(bpe::md5_hex(bytes_of(std::string(56, 'a'))),
              "3b0c8ac703f828b04c6c197006d17218");
}

TEST(Md5, OneFullBlock) {
    // 64 bytes: one whole block is compressed before the padding block.
    EXPECT_EQ(bpe::md5_hex(bytes_of(std::string(64, 'a'))),
              "014842d480b571495a4a0363793f7367");
}

TEST(Md5, TwoFullBlocks) {
    EXPECT_EQ(bpe::md5_hex(bytes_of(std::string(128, 'a'))),
              "e510683b3f5ffe4093d021808bc6ff70");
}

TEST(Md5, RfcMillionA) {
    EXPECT_EQ(bpe::md5_hex(bytes_of(std::string(1000000, 'a'))),
              "7707d6ae4e027c70eea2a935c2296f21");
}
