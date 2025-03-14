cmake_minimum_required(VERSION 3.10)

project(Compute_On_GPU)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(KERNELS_DIR "${ROOT_DIR}/kernels")
set(TESTS_DIR "${ROOT_DIR}/tests")

find_package(OpenCL REQUIRED)
find_package(GTest REQUIRED)

include_directories(include)

add_subdirectory(tests)
add_subdirectory(sources)
add_subdirectory(benchmarking)

add_executable(main main.cpp)

set_target_properties(main PROPERTIES 
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(main PRIVATE OpenCL::OpenCL)
target_compile_definitions(main PRIVATE -DLOGGING=1)