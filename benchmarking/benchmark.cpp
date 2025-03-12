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

    for (auto _ : st) {
        size_t size = st.range(0);
        
        data.resize(size);
        randinit(data.begin(), data.end());

        ocl.run(data.data(), data.size());

        // if (!std::is_sorted(data.begin(), data.end())) {
        //     st.SkipWithError("OclSort does not sort...");
        // }
    }
}

static void bm_stdSort(benchmark::State& st) {
    std::vector<int> data;
    for (auto _ : st) {
        int size = st.range(0);

        data.resize(size);
        randinit(data.begin(), data.end());

        std::sort(data.begin(), data.end());

        // if (!std::is_sorted(data.begin(), data.end())) {
        //     st.SkipWithError("OclSort does not sort...");
        // }
    }
}


BENCHMARK(bm_oclSort)->Range(8, 1 << 20);
BENCHMARK(bm_stdSort)->Range(8, 1 << 20);
BENCHMARK(bm_oclSort)->Range(1 << 21, 1  << 24)->Iterations(5);
BENCHMARK(bm_stdSort)->Range(1 << 21, 1  << 24)->Iterations(5);

BENCHMARK_MAIN();