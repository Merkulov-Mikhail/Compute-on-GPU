cmake_minimum_required(VERSION 3.10)


add_executable(tests invoker.cpp)

add_definitions(-DTEST_FOLDER="${TESTS_DIR}/test_folder/")
add_definitions(-DKERNEL="${KERNELS_DIR}/bitonic_localmem.cl")
set_target_properties(tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(tests PRIVATE GTest::GTest GTest::gtest_main OpenCL::OpenCL)

include(GoogleTest)
gtest_discover_tests(tests)