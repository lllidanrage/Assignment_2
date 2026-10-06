#include "bpe.h"
#include "md5.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

TEST(Smoke, FullCliOutputMatchesLockedMd5) {
    const fs::path dir =
        fs::temp_directory_path() / ("bpe_smoke_" + std::to_string(::getpid()));
    struct Cleanup {
        fs::path old;
        fs::path dir;
        ~Cleanup() {
            std::error_code ec;
            fs::current_path(old, ec);
            fs::remove_all(dir, ec);
        }
    } cleanup{fs::current_path(), dir};
    fs::create_directories(dir);
    ASSERT_TRUE(fs::is_directory(dir));
    const fs::path input = dir / "input.txt";
    {
        std::ofstream out(input);
        out << "hug hug\npug\n";
    }
    fs::current_path(dir);
    char arg0[] = "bpe";
    std::string arg1 = input.string();
    char* argv[] = {arg0, arg1.data()};
    const int rc = bpe::run_cli(2, argv);

    ASSERT_EQ(rc, 0);
    const std::vector<bpe::Byte> out_bytes =
        bpe::read_file((dir / "output.txt").string());
    EXPECT_EQ(bpe::md5_hex(out_bytes), "bfbeba8844c9ad3d8c61431bb9a131b5");
}
