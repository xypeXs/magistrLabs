import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

path = "LabDictors/2TXT/DPavel_YesNo.txt"

bad_signal_median = 128

def normalize(data):
    for i in range(len(data)):
        if abs(data[i]) > 1:
            data[i] -= bad_signal_median


with open(path) as file:
    data = [float(line.strip().replace(',', '.')) for line in file]
    data = data + data[::-1]
    normalize(data)

print(f'length of signal: {len(data)}')

data = data[: 2 ** math.floor(math.log2(len(data)))]

print(f'length of Ns sequence: {len(data)}')

fd = 8000

N = len(data)  # длина реализации
dt = 1 / int(fd)  # шаг дискретизации

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

def getMaxLevelAndValueTwoDim(eArr):
    maxL = 0
    maxValue = -1
    for l in range(len(eArr)):
        for i in range(len(eArr[l])):
            if eArr[l][i] > maxValue:
                maxL = l
                maxValue = eArr[l][i]
    return f'{maxL}, {maxValue:.2f}'

def getMaxLevelAndValueOneDim(eArr):
    maxL = 0
    maxValue = -1
    for l in range(len(eArr)):
            if eArr[l] > maxValue:
                maxL = l
                maxValue = eArr[l]
    return f'{maxL}, {maxValue:.2f}'

def getMaxLevelAndValueACF(data):
    acf_arr = sm.tsa.acf(data, nlags=2)
    max_val = -100
    max_level = -1
    for i in range (1, len(acf_arr)):
        if acf_arr[i] > max_val:
            max_val = acf_arr[i]
            max_level = i

    return max_level, max_val


waveletApproximations, waveletDetails = wavelet()

local_energy_approximations = computeEnergyDensity(waveletApproximations)
local_energy_details = computeEnergyDensity(waveletDetails)
global_spectrum_energy_approximations = computeGlobalEnergySpectrum(local_energy_approximations)
global_spectrum_energy_details = computeGlobalEnergySpectrum(local_energy_details)

# print(f'Standard Deviation: {np.std(data):.5f}')
# print(f'Kurtosis: {stats.kurtosis(data):.5f}')
# print(f'Skewness: {stats.skew(data):.5f}')

# СКО, эксцесс, асимметрия
acf_max_level, acf_max_val = getMaxLevelAndValueACF(data)

print(getMaxLevelAndValueTwoDim(local_energy_approximations) + ', ' + getMaxLevelAndValueTwoDim(local_energy_details) + ', ' + getMaxLevelAndValueOneDim(global_spectrum_energy_approximations) + ', ' + getMaxLevelAndValueOneDim(global_spectrum_energy_details)
      + ', ' + f'{np.std(data):.5f}, {stats.kurtosis(data):.5f}, {stats.skew(data):.5f}, ', f'{acf_max_level:.5f}, {acf_max_val:.5f}')