import os
import random


UPPER_BOUND =  1000000
LOWER_BOUND = -1000000

generated = {}

def get_filename(group="Default"):
    if group not in generated:
        generated[group] = 0
    generated[group] += 1
    return group + "_" +  f"{generated[group]}".zfill(4) + "_test"


def generator(func):
    def __wrapper(N, group_name="Default", tryes=1, mx=UPPER_BOUND, mn=LOWER_BOUND):
        ans = []
        for _ in range(tryes):
            filename = get_filename(group_name)
            arr = func(N, mn=mn, mx=mx)
            with open(filename, "w") as f:
                f.write(f"{N} ")
                f.write(" ".join(map(str, arr)))

            if tryes != 1:
                ans.append(arr)
            else:
                ans = arr
        return ans

    return __wrapper


@generator
def generate_sorted_ascending(N, mx=UPPER_BOUND, mn=LOWER_BOUND):
    arr = [random.randint(mn, mx)]
    for i in range(N - 1):
        arr.append(arr[-1] + random.randint(1, 200))
    return arr


@generator
def generate_sorted_descending(N, mx=UPPER_BOUND, mn=LOWER_BOUND):
    arr = [random.randint(mn, mx)]
    for i in range(N - 1):
        arr.append(arr[-1] - random.randint(1, 200))
    return arr


@generator
def generate_random(N, mx=UPPER_BOUND, mn=LOWER_BOUND):
    arr = []
    for _ in range(N):
        arr.append(random.randint(mn, mx))
    return arr


# for file in os.listdir():
#     if file.endswith("_test"):
#         os.remove(file)


generate_sorted_ascending (2 ** 4, group_name="ascending_16elems" , tryes=5)
generate_sorted_descending(2 ** 4, group_name="descending_16elems", tryes=5)

generate_sorted_ascending (2 ** 7, group_name="ascending_128elems" , tryes=5)
generate_sorted_descending(2 ** 7, group_name="descending_128elems", tryes=5)

generate_sorted_ascending (2 ** 16, group_name="ascending_65536elems" , tryes=5)
generate_sorted_descending(2 ** 16, group_name="descending_65536elems", tryes=5)

generate_random(2 ** 3,  group_name="random_8elems",        tryes=1, mn=1, mx=50)
generate_random(2 ** 4,  group_name="random_16elems",       tryes=5, mn=1, mx=30)
generate_random(2 ** 7,  group_name="random_128elems",      tryes=5)
generate_random(2 ** 10, group_name="random_1024elems",     tryes=5)
generate_random(2 ** 16, group_name="random_65536elems",    tryes=5)
generate_random(2 ** 22, group_name="random_4193404elems",  tryes=3)
