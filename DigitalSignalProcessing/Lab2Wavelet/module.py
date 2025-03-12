import matplotlib.pyplot as plt
import numpy as np
import math
import os

path = "N31.txt"

A0 = 1
f = 10
phi = 0
N = 1024
fd = 8000
dM = 0.04
M = 1


# with open(path) as file:
#     data = [float(line.strip().replace(',', '.')) for line in file]
#
# data = data[:N]

def save_signal(digitalSignal):
    with open("modul_signal.txt", 'w') as f:
        for value in digitalSignal:
            f.write(str(value) + "\n")

    return digitalSignal

delta = f/fd
A = [A0 * (1 + M) * np.sin(2 * np.pi * dM * i + phi) for i in range (0, N)]
s = [A[i] * np.sin(2 * np.pi * delta * i + phi) for i in range (0, N)]

save_signal(s)