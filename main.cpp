#include "ocl.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <numeric>
#include <chrono>
#include <random>


#define TS (1 << 26)


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
    int sz = TS;
    if (argc > 2)
        sz = std::atoi(argv[2]);


    std::vector<cl_int> first_vector(sz);

    rand_init(first_vector.rbegin(), first_vector.rend(), -100000, 100000);
    std::cout << "Data size: " << sz << " elements\n";
    auto TimeStart = std::chrono::high_resolution_clock::now();    
    app.writeToBuffer(first_vector.data(), sz);

    uint64_t ev_time = app.run();
    std::cout << "GPU compute time:\t" << ev_time << " ns" << std::endl;

    app.readFromBuffer(first_vector.data());
    auto TimeEnd = std::chrono::high_resolution_clock::now();
    auto Dur = std::chrono::duration_cast<std::chrono::nanoseconds>(TimeEnd - TimeStart).count();
    std::cout << "Full GPU time:\t" << Dur << " ns" << std::endl;

    cl::vector<TYPE> CCPU(sz);
    rand_init(CCPU.begin(), CCPU.end(), -100000, 100000);

    TimeStart = std::chrono::high_resolution_clock::now();
    std::sort(CCPU.begin(), CCPU.end());
    TimeEnd = std::chrono::high_resolution_clock::now();

    Dur = std::chrono::duration_cast<std::chrono::nanoseconds>(TimeEnd - TimeStart).count();
    std::cout << "CPU time measured:\t" << Dur << " ns" << std::endl;

    if (std::is_sorted(first_vector.begin(), first_vector.end())) {
        std::cout << "sorted correctly\n";
    }
    else {
        std::cout << "sorted incorrectly\n";
    }
}
