import numpy as np
import matplotlib.pyplot as plt


def gurvitz(a):
    d1 = a[1]
    d2 = a[1] * a[2] - a[0] * a[3]
    d3 = a[3] * d2 - a[1] * (a[1] * a[4] - a[0] * a[5])
    d4 = a[4] * d3 - a[5] * (a[2] * d2 - a[0] * (a[1] * a[4] - a[0] * a[5]))

    print('-------------------------Гурвиц-------------------------')
    print('d1 = ' + str(d1) + (' > ' if d1 > 0 else ' < ') + '0')
    print('d2 = ' + str(d2) + (' > ' if d2 > 0 else ' < ') + '0')
    print('d3 = ' + str(d3) + (' > ' if d3 > 0 else ' < ') + '0')
    print('d4 = ' + str(d4) + (' > ' if d4 > 0 else ' < ') + '0')
    print(('Не устойчива' if d1 < 0 or d2 < 0 or d3 < 0 or d4 < 0 else 'Устойчива') + ' по Гурвицу')


def raus(a):
    raus_table = [[], []]
    for i in range(0, len(a)):
        if i & 1:
            raus_table[1].append(a[i])
        else:
            raus_table[0].append(a[i])

    r_table = []
    for i in range(4):
        raus_table.append([])
        cur_ind = i + 2
        Rij = raus_table[cur_ind - 2][0] / raus_table[cur_ind - 1][0]
        r_table.append(Rij)
        for j in range(int(len(a) / 2) - 1):
            cij = raus_table[cur_ind - 2][j + 1] - Rij * raus_table[cur_ind - 1][j + 1]
            raus_table[cur_ind].append(cij)
        raus_table[cur_ind].append(0)

    print('-------------------------Раус-------------------------')
    print('{:<3} {:<5} {:<5} {:<5} {:<5}'.format('#', 'R', 'c1', 'c2', 'c3'))
    res_table = []
    isStable = True
    for i in range(len(raus_table)):
        r_str = ''
        if i > 1:
          r_str = ('{:.3f}'.format(r_table[i - 2]))
        print('{:<3} {:<5} {:<5.3f} {:<5.3f} {:<5.3f}'.format(i, r_str, *raus_table[i]))
        isStable &= (raus_table[i][0] > 0)
    print(('Не устойчива' if not isStable else 'Устойчива') + ' по Раусу')

def michailov(a):
    w_arr = np.linspace(0, 2, 12)
    uw_arr = []
    jvw_arr = []

    for w in w_arr:
        uw_arr.append(a[5] - a[3] * w ** 2 + a[1] * w ** 4)
        jvw_arr.append(w * (a[4] - a[2] * w ** 2 + a[0] * w ** 4))

    # D = b2 - 4ac
    D = a[2] ** 2 - 4 * a[0] * a[4]

    print('-------------------------Михаилов-------------------------')
    print(('{:<7}'.format('w')) + ' '.join(f"{w:<10.2f}" for w in w_arr))
    print(('{:<7}'.format('U(w)')) + ' '.join(f"{uw:<10.2f}" for uw in uw_arr))
    print(('{:<7}'.format('jV(w)')) + ' '.join(f"{jvw:<10.2f}" for jvw in jvw_arr))
    print('D = ' + str(D))
    print(('Не устойчива' if D < 0 else 'Устойчива') + ' по Михаилову')

    plt.plot(uw_arr, jvw_arr)
    plt.grid(True)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()

def evsukov(a):
    k = []
    isKSuccLessThenPred = True
    for i in range(1, len(a)):
        k.append(a[i] / a[i - 1])
        if i > 2 and k[i - 1] > k[i - 3]:
            isKSuccLessThenPred = False

    n = []
    for i in range(2, len(k)):
        n.append(k[i] / k[i - 2])

    print('-------------------------Евсюков-------------------------')
    print('K:')
    for i in range(len(k)):
        print('K{:<2} = {:<5.2f}'.format(i + 1, k[i]))
    print('{:4.2f}'.format(k[0]) + (' > ' if k[0] > k[2] else ' < ') + '{:4.2f}'.format(k[2]) + (' > ' if k[2] > k[4] else ' < ') + '{:4.2f}'.format(k[4]))
    print('{:4.2f}'.format(k[1]) + (' > ' if k[1] > k[3] else ' < ') + '{:4.2f}'.format(k[3]))

    print('N:')
    for i in range(len(n)):
        print('N{:<2} = {:<5.2f}'.format(i + 3, n[i]))
    n1 = n[0] + n[1]
    n2 = n[1] + n[2] - n[0] * n[1] * n[2]
    print('1 ' + ('>' if n1 < 1 else '<=') + '{:6.2f}'.format(n1))
    print('1 ' + ('>' if n2 < 1 else '<=') + '{:6.2f}'.format(n2))

    isStable = (1 > n1) and (1 > n2) and (isKSuccLessThenPred)
    print(('Устойчива' if isStable else 'Не устойчива') + ' по Евсюкову')


# a = [0.2, 0.8, 12, 2, 4, 10]
a = [0.01, 0.5, 0.8, 2, 1, 1]
# a = [1, 3, 4, 7, 2, 1]

gurvitz(a)
raus(a)
michailov(a)
evsukov(a)
