#include "ocl.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <numeric>


#define TS 16


auto main(int argc, char* argv[]) -> int {
    Ocl app(argv[1]);

    std::vector<cl_int> first_vector(TS);
    std::vector<cl_int> second_vector(TS);
    std::vector<cl_int> third_vector(TS);

    std::iota(first_vector.rbegin(), first_vector.rend(), 1);
    std::iota(second_vector.data(), second_vector.data() + TS, 100);

    app.run(first_vector.data(), second_vector.data(), third_vector.data(), TS);

    for (int i = 0; i < 10; i++)
        std::cout << first_vector[i] << " " << third_vector[i] << std::endl;

}