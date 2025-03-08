import matplotlib.pyplot as plt
import numpy as np
import math
import os

print('full name of file:')
path = input()

filename = os.path.splitext(os.path.basename(path))[0]
approx_file = f"{filename}_wavelet_approx.txt"
detail_file = f"{filename}_wavelet_details.txt"

with open(path) as file:
    data = [float(line.strip().replace(',', '.')) for line in file]

print(f'length of signal: {len(data)}')

data = data[: 2 ** math.floor(math.log2(len(data)))]

print(f'length of Ns sequence: {len(data)}')

print('fd in Hz:')
fd = input()

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


def save_wavelet(waveletApproximations, waveletDetails):
    with open(approx_file, 'w') as f:
        for level in waveletApproximations:
            f.write("\n".join(map(str, level)) + "\n\n")

    with open(detail_file, 'w') as f:
        for level in waveletDetails:
            f.write("\n".join(map(str, level)) + "\n\n")

    return approx_file, detail_file


def load_wavelet():
    def load_file(filename):
        with open(filename, 'r') as f:
            content = f.read().strip().split("\n\n")
            return [list(map(float, level.split("\n"))) for level in content]

    return load_file(approx_file), load_file(detail_file)


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


waveletApproximations, waveletDetails = wavelet()
save_wavelet(waveletApproximations, waveletDetails)

loadedApproximations, loadedDetails = load_wavelet()

showWavelet(loadedApproximations, loadedDetails)
