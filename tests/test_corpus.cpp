// Tests for input reading and word splitting (corpus.cpp).

#include "bpe.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// A NUL-terminated word as a std::string (stops at the first NUL byte).
std::string to_string(const bpe::Byte* s) {
    std::string out;
    for (const bpe::Byte* p = s; p != nullptr && *p != 0; ++p) {
        out.push_back(static_cast<char>(*p));
    }
    return out;
}

std::vector<std::string> to_strings(const std::vector<bpe::Word>& words) {
    std::vector<std::string> out;
    for (const bpe::Word& w : words) {
        out.push_back(to_string(w.bytes));
    }
    return out;
}

bpe::Byte byte(unsigned char c) { return static_cast<bpe::Byte>(c); }

}  // namespace

TEST(ReadFile, ReadsWholeFileAsBytes) {
    const char* path = "/tmp/bpe_readfile_test_input.txt";
    {
        std::FILE* f = std::fopen(path, "wb");
        ASSERT_TRUE(f != nullptr);
        const char content[] = "hug pug\npun\tbun \n";
        std::fwrite(content, 1, sizeof(content) - 1, f);
        std::fclose(f);
    }
    const std::vector<bpe::Byte> data = bpe::read_file(path);
    EXPECT_EQ(data, std::vector<bpe::Byte>(
                        {byte('h'), byte('u'), byte('g'), byte(' '), byte('p'),
                         byte('u'), byte('g'), byte('\n'), byte('p'), byte('u'),
                         byte('n'), byte('\t'), byte('b'), byte('u'), byte('n'),
                         byte(' '), byte('\n')}));
    std::remove(path);
}

TEST(ReadFile, EmptyFileYieldsEmptyVector) {
    const char* path = "/tmp/bpe_readfile_empty_test.txt";
    {
        std::FILE* f = std::fopen(path, "wb");
        ASSERT_TRUE(f != nullptr);
        std::fclose(f);
    }
    const std::vector<bpe::Byte> data = bpe::read_file(path);
    EXPECT_TRUE(data.empty());
    std::remove(path);
}

TEST(ReadFile, MissingFileThrows) {
    EXPECT_THROW(bpe::read_file("/tmp/bpe_no_such_file_xyz.txt"),
                 std::runtime_error);
}

TEST(SplitWords, EmptyInputYieldsNoWords) {
    std::vector<bpe::Byte> in;
    EXPECT_TRUE(bpe::split_words(in).empty());
}

TEST(SplitWords, WhitespaceOnlyYieldsNoWords) {
    std::vector<bpe::Byte> in = {byte(' '), byte('\t'), byte('\n'), byte('\r'),
                                 byte(' ')};
    EXPECT_TRUE(bpe::split_words(in).empty());
}

TEST(SplitWords, SingleWord) {
    std::vector<bpe::Byte> in = {byte('h'), byte('e'), byte('l'), byte('l'),
                                 byte('o')};
    EXPECT_EQ(to_strings(bpe::split_words(in)),
              std::vector<std::string>({"hello"}));
}

TEST(SplitWords, SeparatesOnAllWhitespaceKinds) {
    std::vector<bpe::Byte> in = {
        byte('h'), byte('u'),  byte('g'), byte(' '),  byte('p'), byte('u'),
        byte('g'), byte('\t'), byte('p'), byte('u'),  byte('n'), byte('\n'),
        byte('b'), byte('u'),  byte('n'), byte('\r'), byte('x')};
    EXPECT_EQ(to_strings(bpe::split_words(in)),
              std::vector<std::string>({"hug", "pug", "pun", "bun", "x"}));
}

TEST(SplitWords, DiscardsLeadingAndTrailingWhitespace) {
    std::vector<bpe::Byte> in = {byte(' '), byte('a'), byte(' '), byte('b'),
                                 byte('\n')};
    EXPECT_EQ(to_strings(bpe::split_words(in)),
              std::vector<std::string>({"a", "b"}));
}

TEST(SplitWords, ConsecutiveSeparatorsDoNotCreateEmptyWords) {
    std::vector<bpe::Byte> in = {byte('a'), byte(' '), byte(' '), byte('\t'),
                                 byte('b'), byte(' '), byte(' '), byte('c')};
    EXPECT_EQ(to_strings(bpe::split_words(in)),
              std::vector<std::string>({"a", "b", "c"}));
}

TEST(SplitWords, SwarScanFindsSeparatorInFirstChunk) {
    // "abcdef " : the separator lands inside the SWAR scan's first 8-byte
    // chunk (byte 6), exercising the hasless-hit path that stops the scan.
    std::vector<bpe::Byte> in = {byte('a'), byte('b'), byte('c'), byte('d'),
                                 byte('e'), byte('f'), byte(' '), byte('x')};
    EXPECT_EQ(to_strings(bpe::split_words(in)),
              std::vector<std::string>({"abcdef", "x"}));
}

TEST(SplitWords, SwarScanSkipsLongWordAcrossChunks) {
    // A 26-letter word: the SWAR scan crosses three 8-byte chunks with no hit
    // (the hasless-miss path), then the byte loop walks the tail to the
    // trailing sentinel NUL.
    std::vector<bpe::Byte> in;
    for (char c : std::string("abcdefghijklmnopqrstuvwxyz")) {
        in.push_back(byte(c));
    }
    EXPECT_EQ(to_strings(bpe::split_words(in)),
              std::vector<std::string>({"abcdefghijklmnopqrstuvwxyz"}));
}
