import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

path = "../TestLab4GE.txt"

try:
    with open(path) as file:
        input_signal = [float(line.strip().replace(',', '.')) for line in file]
except FileNotFoundError:
    print("Файл не найден. Используется тестовый сигнал.")
    input_signal = np.zeros(100)
    input_signal[0] = 1

# Параметры
fd = 8000  # Частота дискретизации [Гц]
dt = 1 / fd  # Интервал между отсчетами [с]

# Функция для анализа рекурсивного фильтра
def analyze_recursive_filter(b1, input_signal, fd=8000):
    dt = 1 / fd
    title = f'b1={b1}'

    # Коэффициенты фильтра
    b = [1]
    a = [1, -b1]

    # Нули и полюса
    zeros = np.roots(b)
    poles = np.roots(a)

    # АЧХ в Герцах
    w, h = signal.freqz(b, a, worN=2000, fs=fd)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))

    output_signal = signal.lfilter(b, a, input_signal)

    # Импульсная характеристика с временной осью
    impulse = np.zeros(100)
    impulse[0] = 1
    h_imp = signal.lfilter(b, a, impulse)
    t_imp = np.arange(len(h_imp)) * dt

    # Постоянная времени (время затухания до 1/e от максимума)
    h_max = np.max(np.abs(h_imp))
    tau_idx = np.where(np.abs(h_imp) <= h_max / np.e)[0]
    tau = t_imp[tau_idx[0]] if len(tau_idx) > 0 else t_imp[-1]

    # Временные оси для сигналов
    t_input = np.arange(len(input_signal)) * dt
    t_output = np.arange(len(output_signal)) * dt

    # Графики
    plt.figure(figsize=(15, 10))

    # Нули и полюса
    plt.subplot(2, 2, 1)
    plt.title(f'Нули и полюса ({title})')
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='r', label='Нули', s=150)
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='b', label='Полюса', s=150)
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Единичная окружность')
    for zero in zeros:
        plt.plot([0, np.real(zero)], [0, np.imag(zero)], 'g--', alpha=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend(loc='upper left')

    # АЧХ
    plt.subplot(2, 2, 2)
    plt.title(f'АЧХ ({title})')
    plt.plot(w, h_db)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда (дБ)')
    plt.grid(True)

    # Импульсная характеристика
    plt.subplot(2, 2, 3)
    plt.title(f'Импульсная характеристика ({title})')
    plt.stem(t_imp, h_imp)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    # Сигналы
    plt.subplot(2, 2, 4)
    plt.title(f'Сигнал ({title})')
    plt.plot(t_input, input_signal, '-b', label='Вход')
    plt.plot(t_output, output_signal, '-r', label='Выход')
    plt.xlabel('Время (с)')
    plt.ylabel('Сигнал')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return zeros, tau

# Параметры для исследования
b1_values = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]

# Списки для хранения характеристик
taus = []
W_values = []

# Анализ для каждого b1
for b1 in b1_values:
    zeros, tau = analyze_recursive_filter(b1, input_signal, fd=fd)
    taus.append(tau)
    # Вычисляем W как на графике
    W = -1.2 * (1 - b1**2)
    W_values.append(-W)

# График зависимости W от b1 (как на изображении)
plt.figure(figsize=(10, 6))
plt.plot(b1_values, W_values, 'bo-', linewidth=2, markersize=8, label='W')
plt.title('W')
plt.xlabel('b1')
plt.ylabel('W')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()