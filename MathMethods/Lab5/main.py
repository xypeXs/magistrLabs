def multiplyMatr(a, b):
    c = []
    for i in range(len(a)):
        c.append([])
        for j in range(len(b[0])):
            c[i].append(0)
            for k in range(len(a)):
                c[i][j] += a[i][k] * b[k][j]
    return c

def hasZeroElementsOnMainDiagonal(matr):
    for i in range(min(len(matr), len(matr[0]))):
        if matr[i][i] == 0:
            return True
    return False

def calcDistanceMatr(con_matr):
    cm = con_matr

    while hasZeroElementsOnMainDiagonal(cm):
        cm = multiplyMatr(cm, con_matr)

    return cm

def calcVertexBreak(con_matr):
    break_matr = [[], []]
    for i in range(len(con_matr)):
        break_matr[1].append(0)
        for j in range(len(con_matr[0])):
            if (len(break_matr[0]) - 1 < j):
                break_matr[0].append(0)
            break_matr[0][j] += con_matr[i][j]
            break_matr[1][i] += con_matr[i][j]
    return break_matr

def analyzeExcessiveness(matr):
    zeroRowAndColumnSum = 0
    elementSum = 0
    for i in range(len(matr)):
        for j in range(len(matr[0])):
            if i == j:
                continue
            if i == 0 or j == 0:
                zeroRowAndColumnSum += matr[i][j]
            elementSum += matr[i][j]

    k = len(matr)
    print("Избыточность", end=": ")
    print("Присутствует" if elementSum >= k else "Отсутствует")

    R = (elementSum - zeroRowAndColumnSum) / k - 1
    print("R = " + str(R))
    print("Менее надёжная" if R < 0 else "Более надёжная")

def calcConnectionCountMatr(weighed_matr):
    connectionCount_matr = []
    income_sum = 0
    outgoing_sum = 0
    total_sum = 0
    for i in range(len(weighed_matr)):
        connectionCount_matr.append([i, 0, 0, 0])
        for k in range(len(weighed_matr[0])):
            connectionCount_matr[i][1] += weighed_matr[k][i]
            connectionCount_matr[i][2] += weighed_matr[i][k]
        connectionCount_matr[i][3] = connectionCount_matr[i][1] + connectionCount_matr[i][2]
        income_sum += connectionCount_matr[i][1]
        outgoing_sum += connectionCount_matr[i][2]
        total_sum += connectionCount_matr[i][3]

    connectionCount_matr.append(['Итого ', income_sum, outgoing_sum, total_sum])
    return connectionCount_matr

def analyzeConnectionCount(connectionCount_matr):
    pavg = connectionCount_matr[len(connectionCount_matr) - 1][3] / (len(connectionCount_matr) - 1)
    square_deviation = 0
    for i in range(len(connectionCount_matr) - 1):
        square_deviation += (pavg - connectionCount_matr[i][3]) ** 2

    print("σ2 = " + str(square_deviation))

def calcMinimumDistance(connectivity_matr, distance_matr):
    res_matr = []
    for i in range(len(connectivity_matr)):
        res_matr.append([])
        for j in range(len(connectivity_matr)):
            if i == j:
                res_matr[i].append(0)
                continue
            if min(connectivity_matr[i][j], distance_matr[i][j]) == 0:
                res_matr[i].append(max(connectivity_matr[i][j], distance_matr[i][j]))
            else:
                res_matr[i].append(min(connectivity_matr[i][j], distance_matr[i][j]))
    return res_matr

def analyzeMinimumDistanceMatr(minDistance_matr):
    Q = 0
    res_matr = minDistance_matr
    k = len(minDistance_matr)
    for i in range(k):
        res_matr[i].append(0)
        for j in range(k):
            if i == j:
                continue
            res_matr[i][len(res_matr[i]) - 1] += minDistance_matr[i][j]
        Q += res_matr[i][len(res_matr[i]) - 1]

    k = len(minDistance_matr)
    Qmin = k * (k + 1) / 2
    Qrel = Q / Qmin - 1

    z = []
    for i in range(len(minDistance_matr)):
        z.append(Q / 2 / res_matr[i][len(res_matr[i]) - 1])
        res_matr[i].append(z[i])
    Zmax = max(z)
    centrInd = k * (2 * Zmax - (k + 1)) / (Zmax * (k - 1))

    printMatr(res_matr, "Матрица минимальной длины")
    print("Qотн = " + str(Qrel))
    print("Индекс центральности δ = " + str(centrInd))

def printMatr(matr, name):
    print(name)
    for i in range(len(matr)):
        for j in range(len(matr[0])):
            print("{:<7}".format(matr[i][j]), end="")
        print()

connectivity_matr = [
    [0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0]
]

weighed_matr = [
    [0, 7, 6, 7, 6, 7, 5],
    [0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [2, 0, 0, 0, 0, 2, 0],
    [2, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0]
]

# connectivity_matr = [
#     [0, 1, 1, 1, 1, 1],
#     [0, 0, 0, 1, 1, 0],
#     [1, 1, 0, 1, 0, 1],
#     [0, 1, 1, 0, 1, 1],
#     [1, 1, 0, 1, 0, 1],
#     [1, 0, 1, 0, 1, 0]
# ]

break_matr = calcVertexBreak(connectivity_matr)
printMatr(break_matr, 'Матрица наличия обрывов')

distance_matr = calcDistanceMatr(connectivity_matr)
printMatr(distance_matr, 'Матрица количества путей')

analyzeExcessiveness(connectivity_matr)

printMatr(weighed_matr, "Матрица взвешенных связей")

connectionCount_matr = calcConnectionCountMatr(weighed_matr)
printMatr(connectionCount_matr, "Матрица количества связей")
analyzeConnectionCount(connectionCount_matr)

minDistance_matr = calcMinimumDistance(connectivity_matr, distance_matr)
analyzeMinimumDistanceMatr(minDistance_matr)
