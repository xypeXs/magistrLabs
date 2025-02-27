from matplotlib import pyplot as plt
import numpy as np


def W1(s, T):
    return 2/(s * s + s + 1)
    # return 1 / (2 * s + 1)


def W2(s, T):
    return 1 / (s + 1)
    # return 1 / (3 * s + 1)


def W3(s, T):
    return 1 / T / s
    # return T / (s + 1)


def W(s, t):
    return 0.5 * t * (s ** 4) + t * (s ** 3) + t * (s ** 2) + 0.5 * t * s + 1

def W_example(s, t):
    p = W1(s, t) * W2(s, t) * W3(s, t)
    return p / (1 + p)


def model(t, initS):
    s_arr = []
    s = initS
    for i in range(1, 100):
        s_out = W2(W1(s, t), t)
        s = -W3(s_out, t)
        s_arr.append(s_out)

    return np.arange(1, 100), s_arr

s_x = 0
t_x = 10
x_arr, s_arr = model(t_x, s_x)
plt.plot(x_arr, s_arr, label="t={}, s={}".format(t_x, s_x))
plt.legend()
plt.show()