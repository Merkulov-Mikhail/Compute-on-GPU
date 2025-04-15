/*
external define TYPE   required - used type

Expects work group size to be a power of 2

It is essential for LCL_SZ >= sizeof(TYPE) * WG_SZ
*/


__kernel void bitonic_fast(__global TYPE* g_mem, int j, int stage) {
    int to_cmp, i, direction;
    TYPE tmp;

    int id    = get_global_id(0);
    int lid   = get_local_id(0);
    // int wg_sz = get_local_size(0);
    // int glb_sz = get_global_size(0);
    // int k;
    // if (wg_sz >= stage)
        // k = 1;
    // else
        // k = stage / wg_sz;

    __local TYPE l_mem[LCL_SZ];
    // for (int i = 0; i < k; i++) {
        // l_mem[lid * wg_sz + i] = g_mem[id];
        l_mem[lid] = g_mem[id];
    // }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (i = j; i > 0; i /= 2) {
        to_cmp = i ^ lid;
        direction = (id & stage) ? 1 : 0;

        if (lid < to_cmp) {
            if (direction == (l_mem[to_cmp] > l_mem[lid])) {
                tmp = l_mem[to_cmp];
                l_mem[to_cmp] = l_mem[lid];
                l_mem[lid] = tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    g_mem[id] = l_mem[lid];
}


__kernel void bitonic_slow(__global TYPE* g_mem, int j, int stage) {
    int to_cmp, direction;
    TYPE tmp;
    int id = get_global_id(0);
    to_cmp = j ^ id;
    direction = (id & stage) ? 1 : 0;
    if (id < to_cmp) {
        if (direction == (g_mem[to_cmp] > g_mem[id])) {
            tmp = g_mem[to_cmp];
            g_mem[to_cmp] = g_mem[id];
            g_mem[id] = tmp;
        }
    }
}