project(benchmark)

set(CMAKE_VERBOSE_MAKEFILE ON) 

find_package(OpenCL REQUIRED)
find_package(benchmark REQUIRED)

add_executable(bench_o3 benchmark.cpp)

target_compile_options(bench_o3 PRIVATE -O3)

target_link_libraries(bench_o3 OpenCL::OpenCL benchmark::benchmark)

set(CMAKE_CXX_FLAGS -O2)
