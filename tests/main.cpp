// Test entry point (replaces gtest's own gtest_main.cc).
//
// The only difference from the stock gtest main is that the Abseil logger is
// initialized before the tests run, so LOG(INFO)/LOG(ERROR) calls inside the
// CLI tests (run_cli) are formatted normally and no "logging not initialized"
// banner appears in the test output.

#include "absl/log/globals.h"
#include "absl/log/initialize.h"

#include "gtest/gtest.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    return RUN_ALL_TESTS();
}
