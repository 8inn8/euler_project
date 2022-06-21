import math
import string

import numpy as np


def get_digit_number(number, n):
    return (number // 10**n) % 10


a = np.zeros(shape=(1000,), dtype=np.int64)

num = 2 ** 1000

for i in range(1000):
    digit = get_digit_number(num, i)
    a[i] = digit

print(np.sum(a))
