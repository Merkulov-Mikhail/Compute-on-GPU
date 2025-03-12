#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iterator>

#ifndef TEST_FOLDER
#define TEST_FOLDER "tests/test_folder/"
#endif // TEST_FOLDER

#ifndef KERNEL
#define KERNEL "kernels/bitonic_localmem.cl"
#endif // KERNEL

#include "gtest/gtest.h"
#include "ocl.hpp"


namespace GPU_tests {
    void get_right_answer(std::vector<int>& v) {
        std::sort(v.begin(), v.end());
    }

    void run_test(Ocl& app, const std::filesystem::path& filepath) {
        std::cout << "Running: " << filepath.string() << "\n"; 
        std::vector<int> array_gpu;
        std::ifstream in{filepath};

        int N;
        in >> N;

        std::istream_iterator<int> inp(in), end;

        std::copy(inp, end, std::back_inserter(array_gpu));
        
        std::vector<int> array_cpu = array_gpu;
        #ifdef DEBUG
        std::copy(array_gpu.begin(), array_gpu.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
        #endif
        app.run(array_gpu.data(), N);
        get_right_answer(array_cpu);

        #ifdef DEBUG
        std::copy(array_gpu.begin(), array_gpu.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
        std::copy(array_cpu.begin(), array_cpu.end(), std::ostream_iterator<int>(std::cout, " "));
        #endif

        int n = array_gpu.size();
        for (int i = 0; i < n; i++) 
            ASSERT_EQ(array_cpu[i], array_gpu[i]);
    }

    void test(const std::string& group_name) {

        Ocl app{KERNEL};
        std::string group_prefix = TEST_FOLDER + group_name;
        std::filesystem::directory_iterator dir{std::filesystem::path{TEST_FOLDER}};

        for (auto&& f: dir) {
            auto pth = f.path();    
            if (pth.string().starts_with(group_prefix)) {
                run_test(app, pth);
            }
        }
    }
} // GPU_tests