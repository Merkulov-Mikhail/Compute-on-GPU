cmake_minimum_required(VERSION 3.10)

project(everything)

add_subdirectory(sources)
include_directories(include)

find_package(OpenCL REQUIRED)

add_executable(main main.cpp)
add_executable(main2 new_main.cpp)

target_link_libraries(main PRIVATE OpenCL::OpenCL)
target_link_libraries(main2 PRIVATE OpenCL::OpenCL)
