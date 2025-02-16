#include <iostream>


#define ASCENDING  1
#define DESCENDING 0


template<bool order>
inline void cas(int* data, size_t i, size_t j) {
    if (order == (data[i] > data[j])) 
        std::swap(data[i], data[j]);
}


template<bool order>
void bitonic_merge(int N, int* data) {
    if (N < 2)
        return;
    int half = N / 2;
    for (int i = 0; i < half; i++) 
        cas<order>(data, i, i + half);
    bitonic_merge<order>(half, data);
    bitonic_merge<order>(half, data + half);
}


template<bool order>
void bitonic_sort(int N, int* data) {
    if (N < 2) {
        return;
    }
    int half = N / 2;
    bitonic_sort<ASCENDING>(half, data);
    bitonic_sort<DESCENDING>(half, data + half);
    bitonic_merge<order>(N, data);
}


int main() {
    int N;
    std::cin >> N;
    int N2 = 1;
    while (N2 < N)
        N2 <<= 1;
    int* data = new int[N2];

    for (int i = 0; i < N; ++i) 
        std::cin >> data[i];

    for (int i = N; i < N2; ++i) 
        data[i] = __INT_MAX__;

    bitonic_sort<ASCENDING>(N2, data);
    for (int i = 0; i < N; ++i) {
        std::cout << data[i] << " ";
    }
}
