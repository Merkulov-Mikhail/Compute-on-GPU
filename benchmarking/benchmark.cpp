#include "benchmark/benchmark.h"
#include "ocl.hpp"

#include <random>
#include <vector>
#include <cmath>


#ifndef KERNEL
#define KERNEL "../kernels/bitonic_localmem.cl"
#endif

#ifndef LOWER_BOUND
#define LOWER_BOUND (-100000)
#endif // LOWER_BOUND
#ifndef UPPER_BOUND
#define UPPER_BOUND  (100000)
#endif // UPPER_BOUND


template<typename It>
void randinit(It st, It en) {
    static std::mt19937_64 gen;
    std::uniform_int_distribution<int> distribution(LOWER_BOUND, UPPER_BOUND);
    for (It i = st; i != en; ++i) {
        *i = distribution(gen);
    }
}

static void bm_oclSort(benchmark::State& st) {
    std::vector<int> data;

    Ocl ocl{KERNEL};
    size_t size = st.range(0);
    data.resize(size);
    randinit(data.begin(), data.end());
    ocl.writeToBuffer(data.data(), size);

    for (auto _ : st) {
        ocl.run();
        // if (!std::is_sorted(data.begin(), data.end())) {
        //     st.SkipWithError("OclSort does not sort...");
        // }
    }
    ocl.readFromBuffer(data.data());

    if (!std::is_sorted(data.begin(), data.end()))
        throw std::runtime_error("Your bitonic sort does not sort");
        
}

static void bm_stdSort(benchmark::State& st) {
    std::vector<int> data;

    int size = st.range(0);
    data.resize(size);
    randinit(data.begin(), data.end());
    
    for (auto _ : st) {
        std::sort(data.begin(), data.end());

        // if (!std::is_sorted(data.begin(), data.end())) {
        //     st.SkipWithError("OclSort does not sort...");
        // }
    }
}


BENCHMARK(bm_oclSort)->Range(32, 32);
BENCHMARK(bm_stdSort)->Range(32, 32);

BENCHMARK(bm_oclSort)->Range(256, 256);
BENCHMARK(bm_stdSort)->Range(256, 256);
BENCHMARK(bm_oclSort)->Range(4096, 4096);
BENCHMARK(bm_stdSort)->Range(4096, 4096);

BENCHMARK(bm_oclSort)->Range(32768, 32768);
BENCHMARK(bm_stdSort)->Range(32768, 32768);

BENCHMARK(bm_oclSort)->Range(262144, 262144);
BENCHMARK(bm_stdSort)->Range(262144, 262144);


BENCHMARK(bm_oclSort)->Range(1048576, 1048576);
BENCHMARK(bm_stdSort)->Range(1048576, 1048576);


// BENCHMARK(bm_stdSort)->Range(8, 1 << 20);
BENCHMARK(bm_oclSort)->Range(1 << 22, 1  << 26)->Iterations(2);
BENCHMARK(bm_stdSort)->Range(1 << 22, 1  << 26)->Iterations(2);

BENCHMARK_MAIN();