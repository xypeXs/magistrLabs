import math

from matplotlib import pyplot as plt
import numpy as np


def W1(s, T):
    return 2 / (s * s + s + 1)
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


def U(w, T, t):
    return T * math.exp(t * w) * w ** 4 - 2 * T * math.exp(t * w) * w ** 2 + 2


def V(w, T, t):
    return T * math.exp(t * w) * w - 2 * T * math.exp(t * w) * w ** 3


def A(w, T, t):
    return math.sqrt(U(w, T, t) ** 2 + V(w, T, t) ** 2)


def phi(w, T, t):
    return math.atan(V(w, T, t) / U(w, T, t))


def lacx(w, T, t):
    return 20 * math.log(A(w, T, t)) / math.log(w)

def lfcx(w, T, t):
    return phi(w, T, t) / math.log(w)

# LAB1
# s_x = 0
# t_x = 10
# x_arr, s_arr = model(t_x, s_x)
# plt.plot(x_arr, s_arr, label="t={}, s={}".format(t_x, s_x))
# plt.legend()
# plt.show()

# LAB2
w_arr = np.linspace(0.01, 0.1, num = 100)
T = 3 / 4
t = 0.5
lacx_arr = [lacx(w, T, t) for w in w_arr]
lfcx_arr = [lfcx(w, T, t) for w in w_arr]

plt.subplots_adjust(hspace=0.3)

plt.subplot(2, 1, 1)
plt.title("ЛАЧХ")
plt.ylabel("Amplitude")
plt.plot(w_arr, [lacx(w, T, 0.5) for w in w_arr], color='b', label="t={}".format(0.5))
plt.plot(w_arr, [lacx(w, T, 15) for w in w_arr], color='g', label="t={}".format(15))
plt.legend()

plt.subplot(2, 1, 2)
plt.title("ЛФЧХ")
plt.ylabel("Phase")
plt.xlabel("Freq")
plt.plot(w_arr, [lfcx(w, T, 0.5) for w in w_arr], color='b', label="t={}".format(0.5))
plt.plot(w_arr, [lfcx(w, T, 15) for w in w_arr], color='g', label="t={}".format(15))
plt.legend()
plt.show()
