project(benchmark)

find_package(OpenCL REQUIRED)
find_package(benchmark REQUIRED)

add_executable(bench_o2 benchmark.cpp)

target_link_libraries(bench_o2 OpenCL::OpenCL benchmark::benchmark)