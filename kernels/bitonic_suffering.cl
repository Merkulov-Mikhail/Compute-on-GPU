// This kernel was used in debugging purposes

#define TYPE int


__kernel void cas(__global TYPE* A, __global TYPE* B, int stage) {
    int j, to_cmp, direction;
    TYPE tmp;
    int id = get_global_id(0);

    for (j = stage / 2; j > 0; j /= 2) {

        to_cmp = j ^ id;
        direction = (id & stage) ? 1 : 0;
        barrier(CLK_GLOBAL_MEM_FENCE);
        B[id * 5 + 4 + (j == 1) * 20] = (id < to_cmp) & (direction == (A[to_cmp] > A[id]));
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (id < to_cmp) {
            if (direction == (A[to_cmp] > A[id])) {
                tmp = A[to_cmp];
                A[to_cmp] = A[id];
                A[id] = tmp;

            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        B[id * 5 + 0 + (j == 1) * 20] = id;
        B[id * 5 + 1 + (j == 1) * 20] = A[id];
        B[id * 5 + 2 + (j == 1) * 20] = to_cmp;
        B[id * 5 + 3 + (j == 1) * 20] = A[to_cmp];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}