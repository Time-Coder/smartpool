import math


def is_prime(num:int):
    if num < 2:
        return False

    return all(num % i != 0 for i in range(2, int(math.sqrt(num)) + 1))


def count_prime(start:int, stop:int):
    count = 0
    for i in range(start, stop):
        if is_prime(i):
            count += 1
    return count
