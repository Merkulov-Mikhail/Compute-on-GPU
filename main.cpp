#include "ocl.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <numeric>
#include <chrono>
#include <random>


#define TS (1 << 22)


template <typename It> 
void rand_init(It start, It end, TYPE low, TYPE up) {
    static std::mt19937_64 mt_source;
    std::uniform_int_distribution<int> dist(low, up);
    for (It cur = start; cur != end; ++cur)
      *cur = dist(mt_source);
}


std::ostream& operator<<(std::ostream& out, std::vector<cl_int>& x) {
    for (auto i: x) 
        out << i << " ";
    return out;
}


auto main(int argc, char* argv[]) -> int {
    Ocl app(argv[1]);

    std::vector<cl_int> first_vector(TS);

    std::iota(first_vector.rbegin(), first_vector.rend(), 1);

    // std::cout << "Unsorted array:\n";
    // std::cout << first_vector << "\n";

    auto TimeStart = std::chrono::high_resolution_clock::now();
    cl::Event Evt = app.run(first_vector.data(), TS);
    auto TimeFin = std::chrono::high_resolution_clock::now();
    auto Dur = std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart).count();
    std::cout << "GPU wall time measured: " << Dur << " ms" << std::endl;
    auto GPUTimeStart = Evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto GPUTimeFin = Evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    auto GDur = (GPUTimeFin - GPUTimeStart) / 1000000; // ns -> ms
    std::cout << "GPU pure time measured: " << GDur << " ms" << std::endl;

    cl::vector<TYPE> CCPU(TS);
    rand_init(CCPU.begin(), CCPU.end(), -100000, 100000);
    TimeStart = std::chrono::high_resolution_clock::now();
    std::sort(CCPU.begin(), CCPU.end());
    TimeFin = std::chrono::high_resolution_clock::now();
    Dur = std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart).count();
    std::cout << "CPU time measured: " << Dur << " ms" << std::endl;
}