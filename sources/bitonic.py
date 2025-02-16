def cas(arr, i, j, order):
    if len(arr) <= j:
        return
    if order == (arr[i] > arr[j]):
        arr[i], arr[j] = arr[j], arr[i]


def merge(arr, start, num, order):
    if num < 2:
        return
    median = num // 2 + num % 2
    for i in range(start, start + median):
        cas(arr, i, i + median, order)
    merge(arr, start, median, order)
    merge(arr, start + median, median, order)


def bitonic(arr, start, num, order):
    if num < 2:
        return
    median = num // 2 + num % 2
    bitonic(arr, start, median, 1)
    bitonic(arr, start + median, median, 0)
    merge(arr, start, num, order)


a = [5, 4, 3, 2, 1]
bitonic(a, 0, 5, 1)
print(a)