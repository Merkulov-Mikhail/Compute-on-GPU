#include "invoker.hpp"


TEST(GPU_TESTS, Ascending) {
    GPU_tests::test("ascending");
}

TEST(GPU_TESTS, Descending) {
    GPU_tests::test("descending");
}

TEST(GPU_TESTS, Random) {
    GPU_tests::test("random");
}

