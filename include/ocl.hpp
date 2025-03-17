#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "CL/opencl.hpp"
#include <cassert>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>


#ifndef TYPE
#define TYPE int
#endif

#define STRINGIFY(X) #X
#define STYPE(X) STRINGIFY(X)
#ifndef LOGGING
#define LOGGING 0
#endif //LOGGING

// #ifndef LSZ
// #define LSZ 1024
// #endif //LSZ

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
    cl::Program      program_;
    cl::Kernel       kernel_fast_;
    cl::Kernel       kernel_slow_;
    cl::Buffer       buffer_;
    cl::size_type    size_;

    using functor_t_ = cl::KernelFunctor<cl::Buffer, TYPE, TYPE>;
    using args       = cl::EnqueueArgs;

    static cl::Platform get_platform();
    static cl::Device   get_device(const cl::Platform& pl);
    static cl::Context create_context(const cl::Platform&);

public:

    Ocl(const std::string& file_name): platform_(get_platform()), device_(get_device(platform_)), context_(device_),
                                       queue_(context_, device_, cl::QueueProperties::Profiling), buffer_(context_, CL_MEM_READ_WRITE, 0, nullptr) {
        std::string code = readFile(file_name);
        code = std::string("#define TYPE ") + STYPE(TYPE) + "\n" + code; 

        dbgs << "Chosen platform: " << platform_.getInfo<CL_PLATFORM_NAME>() << "\n";
        dbgs << "Chosen device: " << device_.getInfo<CL_DEVICE_NAME>() << "\n";

        program_ = cl::Program{context_, code};
        #ifdef LSZ
        std::string buildOptions = "-DLSZ=" + std::to_string(LSZ);
        #else
        std::string buildOptions = std::string("-DLSZ=") + std::to_string(device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
        #endif

        program_.build(device_, buildOptions.c_str());
        kernel_fast_ = cl::Kernel(program_, "bitonic_fast");
        kernel_slow_ = cl::Kernel(program_, "bitonic_slow");
    }

    uint64_t run();
    void writeToBuffer (TYPE* input, int size);
    void readFromBuffer(TYPE* output) const;
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
    dbgs << "No GPU platform found, switching to a default one\n";
    return cl::Platform::getDefault();
}


cl::Device Ocl::get_device(const cl::Platform& pl) {
    std::vector<cl::Device> devices;
    pl.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        dbgs << "No GPU device found, switching to a default one\n";
        return cl::Device::getDefault();
    }
    return devices[0];
}


cl::Context Ocl::create_context(const cl::Platform& platform) {
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<intptr_t>(platform()), 0};
    return cl::Context(CL_DEVICE_TYPE_GPU, properties);
}


void Ocl::writeToBuffer(TYPE* input, int size) {
    #if (LOGGING == 1)
    auto TimeStartWrite = std::chrono::high_resolution_clock::now();
    #endif

    assert(size % 2 == 0);
    size_ = size;
    cl::size_type bufSZ = size * sizeof(TYPE);
    buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, bufSZ);
    queue_.enqueueWriteBuffer(buffer_, CL_TRUE, 0, bufSZ, input);
    #if (LOGGING == 1)
    auto TimeFinWrite = std::chrono::high_resolution_clock::now();
    auto DurWrite = std::chrono::duration_cast<std::chrono::nanoseconds>(TimeFinWrite - TimeStartWrite).count();
    dbgs << "Write buffer in: " << DurWrite << "\n";
    #endif
}

// assumes, that ouput can handle at least size_ items
void Ocl::readFromBuffer(TYPE* output) const {
    #if (LOGGING == 1)
    auto TimeStartRead = std::chrono::high_resolution_clock::now();
    #endif
    cl::size_type bufSZ = size_ * sizeof(TYPE);
    queue_.enqueueReadBuffer(buffer_, CL_TRUE, 0, bufSZ, output);
    queue_.finish();
    #if (LOGGING == 1)
    auto TimeFinRead = std::chrono::high_resolution_clock::now();
    auto DurRead = std::chrono::duration_cast<std::chrono::nanoseconds>(TimeFinRead - TimeStartRead).count();
    dbgs << "Read buffer in: " << DurRead << "\n";
    #endif
}


uint64_t Ocl::run() {
    // hehe

    cl::size_type bufSZ = size_ * sizeof(TYPE);

    kernel_fast_.setArg(0, buffer_);
    kernel_slow_.setArg(0, buffer_);
    

    #ifdef LSZ
    cl_ulong wg_size = LSZ;
    #else
    cl_ulong wg_size = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    #endif

    if (size_ < wg_size)
        wg_size = size_;

    uint64_t time_spent = 0;

    cl::NDRange globalSize(size_);
    cl::NDRange localSize (wg_size);

    auto TimeStartofCycle = std::chrono::high_resolution_clock::now();

    for (int scale = 2; scale <= size_; scale <<= 1) {

        if (scale <= wg_size) {
            kernel_fast_.setArg(1, scale / 2);
            kernel_fast_.setArg(2, scale);
            queue_.enqueueNDRangeKernel(kernel_fast_, cl::NullRange, globalSize, localSize);
        }
        else {
            for (int j = scale / 2; j > 0; j >>= 1) {
                if (j <= wg_size / 2) {
                    kernel_fast_.setArg(1, j);
                    kernel_fast_.setArg(2, scale);
                    queue_.enqueueNDRangeKernel(kernel_fast_, cl::NullRange, globalSize, localSize);
                    break;
                }
                else {
                    kernel_slow_.setArg(1, j);
                    kernel_slow_.setArg(2, scale);
                    queue_.enqueueNDRangeKernel(kernel_slow_, cl::NullRange, globalSize, localSize);
                }
            }
        }
    }

    auto TimeFinofCycle = std::chrono::high_resolution_clock::now();
    auto DurCycle = std::chrono::duration_cast<std::chrono::nanoseconds>(TimeFinofCycle - TimeStartofCycle).count();
    return DurCycle;
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
    ev = cas(Args, buffer, debug_buffer, scale);

    cl::copy(queue_, debug_buffer, debug, debug + debug_size);
    for (int i = 0; i < debug_size; i++) {
        std::cout << debug[i] << " ";
        if (i % 5 == 4)
            std::cout << "\n";
    }
    std::cout << "\n";

    cl::copy(queue_, buffer, input, input + size);
    for (int i = 0; i < size; i++) 
        std::cout << input[i] << " ";
    std::cout << "\n\n";

    ev.wait();
}
*/