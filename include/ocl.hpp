#pragma once
#include "CL/opencl.hpp"
#include <string>
#include <sstream>
#include <fstream>


#define TYPE     int
#define STRINGIFY(X) #X
#define STYPE(X) STRINGIFY(X)


class Ocl {
private:
    cl::Platform     platform_;
    cl::Context      context_;
    cl::CommandQueue queue_;
    // cl_ulong         local_mem_size_;
    std::string      code_;
    cl::Program      program_;

    // using functor_t_ = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>;
    using functor_t_ = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>;

    static cl::Platform get_platform();
    static cl::Context create_context(const cl::Platform&);
    static std::string readFile(const std::string& filename);

public:

    Ocl(const std::string& file_name): platform_(get_platform()), context_(create_context(platform_)) {
        queue_ = cl::CommandQueue(context_, cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder);
        code_ = readFile(file_name);
        code_ = std::string("#define TYPE ") + STYPE(TYPE) + "\n" + code_; 
    }    

    cl::Event run(const TYPE* arr1, const TYPE* arr2, TYPE* arr3, int size);
};


std::string Ocl::readFile(const std::string& filename) {
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


cl::Context Ocl::create_context(const cl::Platform& platform) {
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<intptr_t>(platform()), 0};
    return cl::Context(CL_DEVICE_TYPE_GPU, properties);
}


cl::Event Ocl::run(const TYPE* arr1, const TYPE* arr2, TYPE* arr3, int size) {
    int bufSZ = size * sizeof(TYPE);

    cl::Buffer array1{context_, CL_MEM_READ_ONLY, bufSZ};
    cl::Buffer array2{context_, CL_MEM_READ_ONLY, bufSZ};
    cl::Buffer array3(context_, CL_MEM_WRITE_ONLY, bufSZ);

    cl::copy(queue_, arr1, arr1 + size, array1);
    cl::copy(queue_, arr2, arr2 + size, array2);

    cl::Program program_{context_, code_, true};

    functor_t_ cas(program_, "cas");

    cl::NDRange globalRange(size);
    cl::NDRange localRange(1);
    cl::EnqueueArgs Args(queue_, globalRange, localRange);

    cl::Event ev = cas(Args, array1, array2, array3);
    ev.wait();

    cl::copy(queue_, array3, arr3, arr3 + size);
    return ev;
}