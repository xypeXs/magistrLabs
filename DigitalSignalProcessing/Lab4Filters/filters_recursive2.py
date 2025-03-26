import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.patches import Circle

path = "modul_signal.txt"

with open(path) as file:
    data = [float(line.strip().replace(',', '.')) for line in file]

def calculate_resonance_frequency(pole):
    """Расчет резонансной частоты по положению полюса"""
    return np.abs(np.angle(pole)) / (2 * np.pi)


def calculate_quality_factor(pole):
    """Расчет добротности по положению полюса"""
    r = np.abs(pole)
    return np.sqrt(1 + r ** 2) / (2 * (1 - r))


def analyze_filter(b1, b2, input_signal, a_coeffs=[1, 0, 0]):
    # Коэффициенты фильтра
    b = a_coeffs
    a = [1, -b1, -b2]

    # Нули и полюса
    zeros = np.roots(b)
    poles = np.roots(a)

    # Частотная характеристика
    w, h = signal.freqz(b, a, worN=2000)
    h_mag = np.abs(h)

    # Импульсная характеристика
    impulse = signal.unit_impulse(50)
    h_imp = signal.lfilter(b, a, impulse)

    output_signal = signal.lfilter(b, a, input_signal)

    return zeros, poles, w, h_mag, h_imp, output_signal


# 1. Исследование при a1=a2=0, b2=-0.9, b1 от -2 до 2
b2 = -0.9
b1_values = np.linspace(-2, 2, 20)
res_freqs = []

plt.figure(figsize=(15, 10))
for i, b1 in enumerate(b1_values):
    zeros, poles, w, h_mag, h_imp, output_signal = analyze_filter(b1, b2, data)

    # Расчет резонансной частоты
    pole = poles[np.argmax(np.imag(poles))]  # Выбираем верхний полюс
    res_freq = calculate_resonance_frequency(pole)
    res_freqs.append(res_freq)

    # Визуализация для 5 характерных значений
    if i % 4 == 0:
        plt.subplot(2, 3, i // 4 + 1)
        plt.plot(w / np.pi, h_mag)
        plt.title(f'b1={b1:.2f}, ResF={res_freq:.2f}π')
        plt.xlabel('Нормированная частота')
        plt.grid(True)

plt.tight_layout()
plt.show()

# График зависимости резонансной частоты от b1
plt.figure()
plt.plot(b1_values, res_freqs, 'bo-')
plt.title('Зависимость резонансной частоты от b1')
plt.xlabel('b1')
plt.ylabel('Резонансная частота (×π)')
plt.grid(True)
plt.show()

# 2. Исследование при b1=0, b2 от 0.2 до 0.95
b1 = 0
b2_values = np.linspace(0.2, 0.95, 10)
q_factors = []
time_constants = []

plt.figure(figsize=(15, 10))
for i, b2 in enumerate(b2_values):
    zeros, poles, w, h_mag, h_imp, output_signal = analyze_filter(b1, b2, data)

    # Расчет характеристик
    pole = poles[np.argmax(np.imag(poles))]
    q = calculate_quality_factor(pole)
    tc = np.argmax(np.abs(h_imp) < 0.368)  # Постоянная времени
    q_factors.append(q)
    time_constants.append(tc)

    # Визуализация для 3 значений
    if i % 3 == 0:
        plt.subplot(2, 3, i // 3 + 1)
        plt.plot(w / np.pi, h_mag)
        plt.title(f'b2={b2:.2f}, Q={q:.2f}')
        plt.xlabel('Нормированная частота')
        plt.grid(True)

plt.tight_layout()
plt.show()

# Графики зависимостей
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(b2_values, q_factors, 'ro-')
plt.title('Зависимость добротности от b2')
plt.xlabel('b2')
plt.ylabel('Q')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(b2_values, time_constants, 'go-')
plt.title('Зависимость постоянной времени от b2')
plt.xlabel('b2')
plt.ylabel('τ (отсчеты)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Анализ трех вариантов коэффициентов
cases = [
    {'a': [1, 0, 0], 'b': [1, 0.218, -0.437]},  # Вариант 1
    {'a': [1, -2, 1], 'b': [1, 0, 0]},  # Вариант 2
    {'a': [1, -2, 1], 'b': [1, 0.218, -0.437]}  # Вариант 3
]

plt.figure(figsize=(15, 5))
for i, case in enumerate(cases):
    # Расчет характеристик
    zeros, poles, w, h_mag, h_imp, output_signal = analyze_filter(
        b1=case['b'][1],
        b2=case['b'][2],
        input_signal=data,
        a_coeffs=case['a']
    )

    # Построение АЧХ
    plt.subplot(1, 3, i + 1)
    plt.plot(w / np.pi, 20 * np.log10(h_mag))
    plt.title(f'Вариант {i + 1}')
    plt.xlabel('Нормированная частота')
    plt.ylabel('Амплитуда (дБ)')
    plt.grid(True)
    plt.ylim(-40, 10)

plt.tight_layout()
plt.show()