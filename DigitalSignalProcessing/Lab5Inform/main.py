import math

import matplotlib.pyplot as plt
import numpy as np

path = "inform/u/u_.txt"

with open(path) as file:
    data = [float(line.strip().replace(',', '.')) for line in file]

print(f'length of signal: {len(data)}')

data = data[: 2 ** math.floor(math.log2(len(data)))]

print(f'length of Ns sequence: {len(data)}')

fd = 8000

N = len(data)  # длина реализации
dt = 1 / int(fd)  # шаг дискретизации

def autoCorr(k_arr):
    avg = sum(data) / len(data)
    r = []
    for k in k_arr:
        r_u = 0
        r_d = 0
        for i in range(len(data)):
            if i < len(data) - k:
                r_u += (data[i] - avg) * (data[i + k] - avg)
            r_d += (data[i] - avg) ** 2
        r.append(r_u / r_d)

    return r

def wavelet():
    tempWaveletAprxLevels = [data]
    tempWaveletDetailLevel = [data]
    curLevelInd = 0
    while len(tempWaveletAprxLevels[curLevelInd]) >= 2:
        tempWaveletAprxLevels.append([])
        tempWaveletDetailLevel.append([])
        for j in range(math.floor(len(tempWaveletAprxLevels[curLevelInd]) / 2)):
            tempWaveletAprxLevels[curLevelInd + 1].append(
                (tempWaveletAprxLevels[curLevelInd][2 * j] + tempWaveletAprxLevels[curLevelInd][2 * j + 1]) / 2)
            tempWaveletDetailLevel[curLevelInd + 1].append(
                (tempWaveletAprxLevels[curLevelInd][2 * j] - tempWaveletAprxLevels[curLevelInd][2 * j + 1]) / 2)
        curLevelInd += 1

    return tempWaveletAprxLevels[1:], tempWaveletDetailLevel[1:]


def showWavelet(waveletApproximations, waveletDetails):
    plt.figure(figsize=(19.2, 10.8))
    plt.subplots_adjust(hspace=0.7)

    plt.subplot(len(waveletApproximations) + 1, 2, 1)
    plt.plot(np.arange(len(data)), data)
    plt.title(path)

    plt.subplot(len(waveletApproximations) + 1, 2, 2)
    plt.plot(np.arange(len(data)), data)
    plt.title(path)

    for i in range(len(waveletApproximations)):
        plt.subplot(len(waveletApproximations) + 1, 2, 2 * (i + 1) + 1)
        level_time = np.linspace(0, N * dt, len(waveletApproximations[i]) + 1)
        plt.step(level_time, [waveletApproximations[i][0]] + waveletApproximations[i], color='b',
                 label='appr L' + str(i))
        plt.subplot(len(waveletApproximations) + 1, 2, 2 * (i + 1) + 2)
        plt.step(level_time, [waveletDetails[i][0]] + waveletDetails[i], color='b', label='detail L' + str(i))

    for i in range(len(waveletApproximations)):
        plt.subplot(len(waveletApproximations) + 1, 2, 2 * (i + 1) + 1)
        plt.ylabel(f"{i}")
        plt.subplot(len(waveletApproximations) + 1, 2, 2 * (i + 1) + 2)
        plt.ylabel(f"{i}")

    plt.show()


def computeEnergyDensity(wlt):
    Ewlt = []
    for l in range(len(wlt)):
        Ewlt.append([])
        for t in range(len(wlt[l])):
            Ewlt[l].append(wlt[l][t] ** 2)
    return Ewlt


def computeGlobalEnergySpectrum(Ewlt):
    Ewl = []
    for l in range(len(Ewlt)):
        Ewl.append(0.0)
        for t in range(len(Ewlt[l])):
            Ewl[l] += Ewlt[l][t]
        #     normalize
        Ewl[l] /= len(data)

    return Ewl


def computeLocalPeremezhaemost(Ewlt):
    Awlt = []
    for l in range(len(Ewlt)):
        Awlt.append([])
        avg = sum(Ewlt[l]) / len(Ewlt[l])
        for t in range(len(Ewlt[l])):
            Awlt[l].append(Ewlt[l][t] / avg)
    return Awlt


def computeContrast(Ewlt):
    Cwlt = []
    for l in range(len(Ewlt)):
        Cwlt.append([])
        for t in range(len(Ewlt[l])):
            prevEwlt = [sum(v) for v in (Ewlt[j][t] for j in range(0, l + 1))]
            Cwlt[l].append(Ewlt[l][t] / prevEwlt)
    return Cwlt


def printMetric(name, mlt):
    print(name)
    print('\n'.join([''.join(['{:4.5f} '.format(item) for item in row])
                     for row in mlt]))

def printArr(name, arr):
    print(name)
    print('\n'.join([''.join(['{:4.5f} '.format(item) for item in arr])]))


def showMetric(name, mlt):
    plt.figure(figsize=(19.2, 10.8))
    plt.subplots_adjust(hspace=0.7)
    plt.title(name)
    for l in range(len(mlt)):
        plt.subplot(len(mlt), 1, l + 1)
        plt.plot(np.arange(len(mlt[l])), mlt[l])
    plt.show()


def printLocalEnergyCharacteristics(name, local_energy):
    print(name)
    for l in range(len(local_energy)):
        print(f'{l:4} {max(local_energy[l]):4.5f} {min(local_energy[l]):4.5f}')

waveletApproximations, waveletDetails = wavelet()

local_energy_approximations = computeEnergyDensity(waveletApproximations)
local_energy_details = computeEnergyDensity(waveletDetails)

printLocalEnergyCharacteristics('Local energy Approx', local_energy_approximations)
printLocalEnergyCharacteristics('Local energy Details', local_energy_details)

printArr('Global Energy Approx', computeGlobalEnergySpectrum(local_energy_approximations))
printArr('Global Energy Details', computeGlobalEnergySpectrum(local_energy_details))

printArr('Auto corr', autoCorr([1, 2, 5]))

# showMetric('Energy density', computeLocalPeremezhaemost(waveletDetails))

# printMetric('Local Peremezhaemost Approx', computeLocalPeremezhaemost(local_energy_approximations))
# printMetric('Local Peremezhaemost Details', computeLocalPeremezhaemost(local_energy_details))
#
# printMetric('Contrast Approx', computeContrast(local_energy_approximations))
# printMetric('Contrast Details', computeContrast(local_energy_details))