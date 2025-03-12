#define TYPE int


__kernel void cas(__global TYPE* A, int j, int stage) {
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