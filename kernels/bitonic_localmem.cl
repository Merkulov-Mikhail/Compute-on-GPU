#define TYPE int

#ifndef LSZ
#define LSZ 1024
#endif

__kernel void bitonic_fast(__global TYPE* A, int stage) {
    int to_cmp, j, direction;
    TYPE tmp;
    int id = get_global_id(0);
    int lid = get_local_id(0);
    __local TYPE lmem[LSZ];

    lmem[lid] = A[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (j = stage / 2; j > 0; j /= 2) {
        to_cmp = j ^ lid;
        direction = (id & stage) ? 1 : 0;

        if (lid < to_cmp) {
            if (direction == (lmem[to_cmp] > lmem[lid])) {
                tmp = lmem[to_cmp];
                lmem[to_cmp] = lmem[lid];
                lmem[lid] = tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    A[id] = lmem[lid];
}


__kernel void bitonic_slow(__global TYPE* A, int j, int stage) {
    int to_cmp, direction;
    TYPE tmp;
    int id = get_global_id(0);

    to_cmp = j ^ id;
    direction = (id & stage) ? 1 : 0;

    if (id < to_cmp) {
        if (direction == (A[to_cmp] > A[id])) {
            tmp = A[to_cmp];
            A[to_cmp] = A[id];
            A[id] = tmp;
        }
    }
}