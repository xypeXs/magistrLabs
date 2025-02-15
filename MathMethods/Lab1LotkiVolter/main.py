import matplotlib.pyplot as plt


# y(i) = y(i-1) + (x(i) - x(i - 1)) * f(x(i - 1), y(i - 1))
def calc(yi1, dx, fxi1yi1):
    return yi1 + dx * fxi1yi1


def calcX(k1, k2, x, y):
    return k1 * x - k2 * x * y


def calcY(k3, k4, x, y):
    return -k3 * y + k4 * x * y


# dx/dt = k1x - k2xy
# 0 - t, 1 - count
def calcZayazAndLisa(x0, y0, dt, k1, k2, k3, k4, tborder):
    xarr = [[0], [x0]]
    yarr = [[0], [y0]]
    t = 0
    i = 0
    while t < tborder:
        t += dt
        i += 1
        # f(x(i - 1), y(i - 1))
        fi1x = calcX(k1, k2, xarr[1][i - 1], yarr[1][i - 1])
        fi1y = calcY(k3, k4, xarr[1][i - 1], yarr[1][i - 1])

        xi = calc(xarr[1][i - 1], dt, fi1x)
        yi = calc(yarr[1][i - 1], dt, fi1y)
        xarr[0].append(t)
        xarr[1].append(xi)
        yarr[0].append(t)
        yarr[1].append(yi)

    return xarr, yarr


def calcTask():
    xarr, yarr = calcZayazAndLisa(3, 2, 0.02, 0.4, 0.2, 0.3, 0.05, 100)

    fixure, axis = plt.subplots(1, 2)

    axis[0].plot(xarr[0], xarr[1], color='r', label='hare')
    axis[0].plot(yarr[0], yarr[1], color='b', label='fox')
    axis[0].legend()

    axis[1].plot(xarr[1], yarr[1], color='g', label='fox(hare)')
    axis[1].legend()

    plt.show()


calcTask()
