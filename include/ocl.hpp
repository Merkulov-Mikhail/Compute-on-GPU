#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "CL/opencl.hpp"
#include <cassert>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>


#define TYPE     int
#define STRINGIFY(X) #X
#define STYPE(X) STRINGIFY(X)
#ifndef LOGGING
#define LOGGING 0
#endif //LOGGING

#ifndef LSZ
#define LSZ 1024
#endif //LSZ

#define dbgs \
    if(!LOGGING){\
    }\
    else std::cout 

std::string readFile(const std::string& filename);

class Ocl {
private:
    cl::Platform     platform_;
    cl::Device       device_;
    cl::Context      context_;
    cl::CommandQueue queue_;
    std::string      code_;
    cl::Program      program_;

    using functor_fast_t_ = cl::KernelFunctor<cl::Buffer, int>;
    using functor_slow_t_ = cl::KernelFunctor<cl::Buffer, int, int>;
    using args            = cl::EnqueueArgs;

    static cl::Platform get_platform();
    static cl::Device   get_device(const cl::Platform& pl);
    static cl::Context create_context(const cl::Platform&);

public:

    Ocl(const std::string& file_name): platform_(get_platform()), device_(get_device(platform_)), context_(create_context(platform_)) {
        queue_ = cl::CommandQueue(context_, cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder);
        code_ = readFile(file_name);
        code_ = std::string("#define TYPE ") + STYPE(TYPE) + "\n" + code_; 
        dbgs << "Chosen platform: " << platform_.getInfo<CL_PLATFORM_NAME>() << "\n";
        dbgs << "Chosen device: " << device_.getInfo<CL_DEVICE_NAME>() << "\n";

        program_ = cl::Program{context_, code_};
        std::string buildOptions = "-DLSZ=" + std::to_string(LSZ);
        program_.build(buildOptions.c_str());
    }    

    cl::Event run(TYPE* input, int size);
}; // Ocl


std::string readFile(const std::string& filename) {
    std::ifstream input(filename);
    std::stringstream stream;
    stream << input.rdbuf();
    return stream.str();
}


cl::Platform Ocl::get_platform() {
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);
    cl_uint GPU_ID = 0;
    for (auto plat: platforms) {
        ::clGetDeviceIDs(plat(), CL_DEVICE_TYPE_GPU, 0, NULL, &GPU_ID);
        if (GPU_ID > 0) {
            return cl::Platform(plat());
        }
    }
    throw std::runtime_error("no GPU platform");
}


cl::Device Ocl::get_device(const cl::Platform& pl) {
    std::vector<cl::Device> devices;
    pl.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty())
        throw std::runtime_error("no GPU device");
    return devices[0];
}


cl::Context Ocl::create_context(const cl::Platform& platform) {
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<intptr_t>(platform()), 0};
    return cl::Context(CL_DEVICE_TYPE_GPU, properties);
}


cl::Event Ocl::run(TYPE* input, int size) {
    // hehe
    assert(size % 2 == 0);

    cl::size_type bufSZ = size * sizeof(TYPE);

    cl::Buffer buf_sort{context_, CL_MEM_READ_WRITE, bufSZ};
    cl::copy(queue_, input, input + size, buf_sort);

    functor_fast_t_ bitonic_fast(program_, "bitonic_fast");
    functor_slow_t_ bitonic_slow(program_, "bitonic_slow");

    cl::NDRange globalRange(size);

    cl::Event ev;

    cl_ulong wg_size = LSZ;

    for (int scale = 2; scale <= size; scale <<= 1) {
        if (scale <= wg_size) {
            ev = bitonic_fast(args{queue_, static_cast<cl::size_type>(size), static_cast<cl::size_type>(scale)}, buf_sort, scale);
            ev.wait();
        }
        else {
            for (int j = scale / 2; j > 0; j >>= 1) {
                ev = bitonic_slow(args{queue_, static_cast<cl::size_type>(size), 1}, buf_sort, j, scale);
                ev.wait();
            }
        }
    }

    cl::copy(queue_, buf_sort, input, input + size);
    return ev;
}


/*
This section contains some experimental code, that I used, while debugging bitonic kernel.
I find it too beautiful for it to get removed, so I move it here
It was really fun using external 1D buffer, writing data to it and then parsing. There were lots of funny mistakes with incorrect indexing .


cl::size_type debug_size  = size * 10;
cl::size_type debug_bufSZ = debug_size * sizeof(TYPE);
cl::Buffer debug_buffer{context_, CL_MEM_WRITE_ONLY, debug_bufSZ};
TYPE debug[debug_size];


for (int scale = 2; scale <= size; scale *= 2) {

    std::cout << "Now stage " << scale << std::endl;
    ev = cas(Args, buf_sort, debug_buffer, scale);

    cl::copy(queue_, debug_buffer, debug, debug + debug_size);
    for (int i = 0; i < debug_size; i++) {
        std::cout << debug[i] << " ";
        if (i % 5 == 4)
            std::cout << "\n";
    }
    std::cout << "\n";

    cl::copy(queue_, buf_sort, input, input + size);
    for (int i = 0; i < size; i++) 
        std::cout << input[i] << " ";
    std::cout << "\n\n";

    ev.wait();
}
*/