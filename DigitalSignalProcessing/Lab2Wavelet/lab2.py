import matplotlib.pyplot as plt
import numpy as np
import math

with open('N31.txt') as file:
    data = [float(line.strip().replace(',', '.')) for line in file]

data = data[: 2 ** math.floor(math.log2(len(data)))]

def wavelet():
    tempWaveletAprxLevels = [data]
    tempWaveletDetailLevel = [data]
    curLevelInd = 0
    while len(tempWaveletAprxLevels[curLevelInd]) >= 2:
        tempWaveletAprxLevels.append([])
        tempWaveletDetailLevel.append([])
        for j in range (math.floor(len(tempWaveletAprxLevels[curLevelInd]) / 2)):
            tempWaveletAprxLevels[curLevelInd + 1].append((tempWaveletAprxLevels[curLevelInd][2 * j] + tempWaveletAprxLevels[curLevelInd][2 * j + 1]) / 2)
            tempWaveletDetailLevel[curLevelInd + 1].append((tempWaveletAprxLevels[curLevelInd][2 * j] - tempWaveletAprxLevels[curLevelInd][2 * j + 1]) / 2)
        curLevelInd += 1

    return tempWaveletAprxLevels[1:], tempWaveletDetailLevel[1:]

def showWavelet(waveletApproximations, waveletDetails):

    plt.figure(figsize=(19.2, 10.8))
    # plt.figure(figsize=(38.4, 21.6))
    # plt.figure(figsize=(76.8, 43.2))
    plt.subplots_adjust(hspace=0.6)

    plt.subplot(len(waveletApproximations) + 1, 2, 1)
    plt.plot(np.arange(len(data)), data)
    plt.subplot(len(waveletApproximations) + 1, 2, 2)
    plt.plot(np.arange(len(data)), data)

    for i in range(len(waveletApproximations)):
        plt.subplot(len(waveletApproximations) + 1, 2, 2 * (i + 1) + 1)
        plt.step(np.arange(len(waveletApproximations[i]) + 1), [waveletApproximations[i][0]] + waveletApproximations[i], color='b', label='appr L' + str(i))
        # plt.legend()
        plt.subplot(len(waveletApproximations) + 1, 2, 2 * (i + 1) + 2)
        plt.step(np.arange(len(waveletDetails[i]) + 1), [waveletDetails[i][0]] + waveletDetails[i], color='b', label='detail L' + str(i))
        # plt.legend()

    plt.show()

waveletApproximations, waveletDetails = wavelet()
showWavelet(waveletApproximations, waveletDetails)