import numpy as np
import matplotlib.pyplot as plt
from _socket import herror


def getPopulation(init_x, init_r, init_b, init_T):
    pop = [init_x]
    for n in range(1, init_T):
        pop.append(pop[n - 1] * init_r * (init_b - pop[n - 1]))
    return pop


T0 = 40
b0 = 1
# x0 = b0 / 2

r_arr = [4, 3.5699, 3, 2, 1.25, 0.75, 0.3, 0.1]
# r_arr = np.linspace(0.9/b0, 1.1/b0, num=5)

x_arr = []
for r in r_arr:
    x0 = b0 - 1 / r + 0.1
    x_arr.append(getPopulation(x0, r, b0, T0))

T_arr = np.linspace(0, T0, num=T0)

plt.figure(figsize=(7.2, 6.4))
plt.subplots_adjust(hspace=0.5)
for i in range(len(r_arr)):
    plt.subplot(len(r_arr), 1, i + 1)
    plt.plot(T_arr, x_arr[i], label='r=' + str(r_arr[i]))
    plt.legend()

plt.show()

# b_arr = np.linspace(0.1, 1, num=20)
# x0_arr = np.linspace(0.1, 1, num=20)
r_arr = np.linspace(0, 5, num=1000)
b_arr = [0.8]
x0_arr = [0.4]

limx = []
r_x = []
for r in r_arr:
    lenBefore = len(limx)
    for b in b_arr:
        for x in x0_arr:
            x_pop = getPopulation(x, r, b, T0)
            distinct_pop = list(set(x_pop[int(0.9 * T0):]))
            for pop in distinct_pop:
                limx.append(pop)
    lenAfter = len(limx)
    for i in range(lenAfter - lenBefore):
        r_x.append(r)

plt.figure(figsize=(6.4, 4.8))
plt.plot(r_x, limx, ',', markersize=0.8)
plt.show()
