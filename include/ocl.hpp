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

// #ifndef MAX_WG_SIZE
// #define MAX_WG_SIZE 1024
// #endif //MAX_WG_SIZE

#define dbgs \
    if(!LOGGING){\
    }\
    else std::cout 

std::string readFile(const std::string& filename);
int max_degree_2(int x);

namespace ocl_detail {
    class Ocl_device_info {
    private:
        cl::size_type max_lcl_sz_;
        cl::size_type lcl_sz_;
        cl::size_type wg_sz_;
        cl::size_type glb_sz_;
    public:

        Ocl_device_info() {}

        Ocl_device_info(const cl::Device& dev) {
            wg_sz_  = dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            glb_sz_ = dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

            cl_int max_lcl_mem = dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();;
            max_lcl_sz_ = max_lcl_mem / sizeof(TYPE);
            lcl_sz_ = max_degree_2(max_lcl_mem / sizeof(TYPE));
            if (max_lcl_sz_ - lcl_sz_ < 1024)
                lcl_sz_ <<= 1;
            
            dbgs << "Maximum work-group size: " << wg_sz_ << "\n";
            dbgs << "Maximum global size: " << glb_sz_ << "\n";

            dbgs << "Maximum local memory size: " << max_lcl_sz_ << "\n";
            dbgs << "Modified local memory size: " << lcl_sz_ << "\n";
        }

        cl::size_type get_local_mem_size() {
            return lcl_sz_;
        }

        cl::size_type get_wg_size(int n_elems) {
            if (n_elems < wg_sz_)
                return n_elems;
            return wg_sz_;
        }
    };
}


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
    cl::size_type    n_elems_;
    ocl_detail::Ocl_device_info info_;

    using functor_t_ = cl::KernelFunctor<cl::Buffer, TYPE, TYPE>;
    using args       = cl::EnqueueArgs;

    static cl::Platform get_platform();
    static cl::Device   get_device(const cl::Platform& pl);
    static cl::Context create_context(const cl::Platform&);
public:

    Ocl(const std::string& file_name): platform_(get_platform()), device_(get_device(platform_)), context_(device_),
                                       queue_(context_, device_, cl::QueueProperties::Profiling), buffer_(context_, CL_MEM_READ_WRITE, 0, nullptr), info_(device_) {
        // Ocl(const std::string& file_name): platform_(get_platform()), device_(get_device(platform_)), context_(device_),
        //                                    queue_(context_, device_, cl::QueueProperties::Profiling), buffer_(context_, CL_MEM_READ_WRITE, 0, nullptr) {
        std::string code = readFile(file_name);

        code = std::string("#define LCL_SZ ") + std::to_string(device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n" 
             + std::string("#define TYPE ") + STYPE(TYPE) + "\n" 
             + code; 

        program_ = cl::Program{context_, code};
        program_.build();
        kernel_fast_ = cl::Kernel(program_, "bitonic_fast");
        kernel_slow_ = cl::Kernel(program_, "bitonic_slow");
    }

    uint64_t run();
    void writeToBuffer (TYPE* input, int size);
    void readFromBuffer(TYPE* output) const;
    void queue_finish() const;
}; // Ocl

//----------------------------------------------
//              Ocl methods
//----------------------------------------------
uint64_t Ocl::run() {

    cl::size_type bufSZ = n_elems_ * sizeof(TYPE);

    kernel_fast_.setArg(0, buffer_);
    kernel_slow_.setArg(0, buffer_);

    int wg_size = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    if (n_elems_ < wg_size)
        wg_size = n_elems_;

    // int wg_size = info_.get_wg_size(n_elems_);

    uint64_t time_spent = 0;

    cl::NDRange global_size(n_elems_);
    cl::NDRange local_size (wg_size);

    auto time_start_cycle = std::chrono::high_resolution_clock::now();

    for (int scale = 2; scale <= n_elems_; scale <<= 1) {

        if (scale <= wg_size) {
            kernel_fast_.setArg(1, scale / 2);
            kernel_fast_.setArg(2, scale);
            int err = queue_.enqueueNDRangeKernel(kernel_fast_, cl::NullRange, global_size, local_size);
        }
        else {
            for (int j = scale / 2; j > 0; j >>= 1) {
                if (j <= wg_size / 2) {
                    kernel_fast_.setArg(1, j);
                    kernel_fast_.setArg(2, scale);
                    queue_.enqueueNDRangeKernel(kernel_fast_, cl::NullRange, global_size, local_size);
                    break;
                }
                else {
                    kernel_slow_.setArg(1, j);
                    kernel_slow_.setArg(2, scale);
                    queue_.enqueueNDRangeKernel(kernel_slow_, cl::NullRange, global_size, local_size);
                }
            }
        }
    }

    auto time_fin_cycle = std::chrono::high_resolution_clock::now();
    auto dur_cycle = std::chrono::duration_cast<std::chrono::nanoseconds>(time_fin_cycle - time_start_cycle).count();
    return dur_cycle;
}

void Ocl::queue_finish() const {
    queue_.finish();
}

// assumes, that ouput can handle at least n_elems_ items
void Ocl::readFromBuffer(TYPE* output) const {
    auto TimeStartRead = std::chrono::high_resolution_clock::now();

    cl::size_type bufSZ = n_elems_ * sizeof(TYPE);
    queue_.enqueueReadBuffer(buffer_, CL_TRUE, 0, bufSZ, output);
    queue_.finish();

    auto TimeFinRead = std::chrono::high_resolution_clock::now();
    auto DurRead = std::chrono::duration_cast<std::chrono::nanoseconds>(TimeFinRead - TimeStartRead).count();
    dbgs << "Read buffer in:\t" << DurRead << " ns\n";
}

void Ocl::writeToBuffer(TYPE* input, int size) {
    // hehe
    assert(size % 2 == 0);

    auto TimeStartWrite = std::chrono::high_resolution_clock::now();
    
    n_elems_ = size;
    cl::size_type bufSZ = size * sizeof(TYPE);
    buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, bufSZ);
    queue_.enqueueWriteBuffer(buffer_, CL_TRUE, 0, bufSZ, input);

    auto TimeFinWrite = std::chrono::high_resolution_clock::now();
    auto DurWrite = std::chrono::duration_cast<std::chrono::nanoseconds>(TimeFinWrite - TimeStartWrite).count();
    dbgs << "Write buffer in:\t" << DurWrite << " ns\n";
}

cl::Context Ocl::create_context(const cl::Platform& platform) {
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<intptr_t>(platform()), 0};
    return cl::Context(CL_DEVICE_TYPE_GPU, properties);
}

cl::Platform Ocl::get_platform() {

    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);
    cl_uint GPU_ID = 0;
    for (auto plat: platforms) {
        ::clGetDeviceIDs(plat(), CL_DEVICE_TYPE_GPU, 0, NULL, &GPU_ID);
        if (GPU_ID > 0) {
            dbgs << "Chosen platform: " << plat.getInfo<CL_PLATFORM_NAME>() << "\n";
            return cl::Platform(plat());
        }
    }
    dbgs << "No GPU platform found, switching to a CPU one\n";
    cl_uint CPU_ID = 0;
    for (auto plat: platforms) {
        ::clGetDeviceIDs(plat(), CL_DEVICE_TYPE_ALL, 0, NULL, &CPU_ID);
        if (CPU_ID > 0) {
            dbgs << "Chosen platform: " << plat.getInfo<CL_PLATFORM_NAME>() << "\n";
            return cl::Platform(plat());
        }
    }
    dbgs << "No CPU platform found, switching to a default one (on what are you running?)\n";
    return cl::Platform::getDefault();
}

cl::Device Ocl::get_device(const cl::Platform& pl) {
    std::vector<cl::Device> devices;
    pl.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        dbgs << "No GPU device found, switching to a CPU one\n";
        pl.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.empty()) {
            dbgs << "No CPU device found, switching to a default one (on what are you running?)\n";
           return cl::Device::getDefault();
        }
    }

    dbgs << "Chosen device: " << devices[0].getInfo<CL_DEVICE_NAME>() << "\n";
    return devices[0];
}
//----------------------------------------------
//           End of Ocl methods
//----------------------------------------------

int max_degree_2(int x) {
    int ans = 1;
    while ((x >>= 1) > 0)
        ans <<= 1;
    return ans;
}

std::string readFile(const std::string& filename) {
    std::ifstream input(filename);
    std::stringstream stream;
    stream << input.rdbuf();
    return stream.str();
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