import math

import matplotlib.pyplot as plt
import numpy as np

path = "TestLab4GE.txt"

with open(path) as file:
    data = [float(line.strip().replace(',', '.')) for line in file]


def transversFilter(x):
    plt.figure(figsize=(19.2, 10.8))
    plt.subplots_adjust(hspace=0.7)

    delt = 0.2

    a = [1, -2, 1]
    iter_cnt = int(abs(a[1] * 2 - a[0]) / delt)
    # iter_cnt = 10
    for ic in range(iter_cnt):
        y = []
        for i in range(len(x)):
            temp_y = 0
            for j in range(len(a)):
                if i - j < 0:
                    break
                temp_y = x[i - j] * a[j]
            y.append(temp_y)
        plt.subplot(iter_cnt, 2, ic * 2 + 1)
        plt.plot(np.arange(len(x)), x)
        plt.subplot(iter_cnt, 2, ic * 2 + 2)
        plt.plot(np.arange(len(y)), y)
        plt.plot(x, y)
        a[1] += delt
    plt.show()


transversFilter(data)
