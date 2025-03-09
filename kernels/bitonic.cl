#define TYPE int


__kernel void cas(__global TYPE* A, __global TYPE* B, __global TYPE* C) {
    int id = get_global_id(0);
    C[id] = B[id] + A[id];
}