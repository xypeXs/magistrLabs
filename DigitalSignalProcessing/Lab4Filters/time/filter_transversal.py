import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

path = "../TestLab4GE.txt"

with open(path) as file:
    input_signal = [float(line.strip().replace(',', '.')) for line in file]

# NEW: Добавляем частоту дискретизации и Δt
fd = 8000  # Частота дискретизации [Гц]
dt = 1 / fd  # Интервал между отсчетами [с]


# Функция для анализа трансверсального фильтра (с изменениями)
def analyze_transversal_filter(a1_over_a0, a2_over_a0, title, input_signal, fd=8000):
    dt = 1 / fd  # NEW: Локальный Δt

    a0 = 1
    a1 = a1_over_a0 * a0
    a2 = a2_over_a0 * a0

    b = [a0, a1, a2]
    print(f'a0: {a0}, a1: {a1}, a2: {a2}')
    a = [1]

    zeros = np.roots(b)
    poles = np.roots(a)

    # NEW: Частота в Герцах вместо радиан
    w, h = signal.freqz(b, a, worN=2000, fs=fd)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))

    output_signal = signal.lfilter(b, a, input_signal)

    # Импульсная характеристика с временной осью
    impulse = np.zeros(10)
    impulse[0] = 1
    h_imp = signal.lfilter(b, a, impulse)
    t_imp = np.arange(len(h_imp)) * dt  # NEW: Время для импульсной характеристики

    # Временные оси для сигналов
    t_input = np.arange(len(input_signal)) * dt  # NEW
    t_output = np.arange(len(output_signal)) * dt  # NEW

    plt.figure(figsize=(15, 10))

    # Нули и полюса (без изменений)
    plt.subplot(2, 2, 1)
    plt.title(f'Нули и полюса ({title})')
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='r', label='Нули', s=150)
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='b', label='Полюса', s=150)
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--')
    for zero in zeros:
        plt.plot([0, np.real(zero)], [0, np.imag(zero)], 'g--', alpha=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend(loc='upper left')

    # АЧХ в Герцах
    plt.subplot(2, 2, 2)
    plt.title(f'АЧХ ({title})')
    plt.plot(w, h_db)  # NEW: w уже в Гц
    plt.xlabel('Частота (Гц)')  # NEW
    plt.ylabel('Амплитуда (дБ)')
    plt.grid(True)

    # Импульсная характеристика с временем
    plt.subplot(2, 2, 3)
    plt.title(f'Импульсная характеристика ({title})')
    plt.stem(t_imp, h_imp)  # NEW: время вместо отсчетов
    plt.xlabel('Время (с)')  # NEW
    plt.ylabel('Амплитуда')
    plt.grid(True)

    # Сигналы с временными осями
    plt.subplot(2, 2, 4)
    plt.title(f'Сигнал ({title})')
    plt.plot(t_input, input_signal, '-b', label='Вход')  # NEW: время
    plt.plot(t_output, output_signal, '-r', label='Выход')  # NEW: время
    plt.xlabel('Время (с)')  # NEW
    plt.ylabel('Сигнал')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return zeros


# Параметры (без изменений)
a2_over_a0 = 1
a1_values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

# # Анализ для каждого a1/a0 (без изменений)
# for a1 in a1_values:
#     zeros = analyze_transversal_filter(a1, a2_over_a0, f'a1/a0 = {a1}', input_signal, fd=fd)  # NEW: передаем fd
#
# # Траектория нулей (с временной подписью)
# plt.figure(figsize=(10, 8))
# theta = np.linspace(0, 2 * np.pi, 100)
# plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Единичная окружность')
#
# colors = plt.cm.viridis(np.linspace(0, 1, len(a1_values)))
#
# for a1, color in zip(a1_values, colors):
#     b = [1, a1, 1]
#     zeros = np.roots(b)
#     plt.scatter(np.real(zeros), np.imag(zeros), color=color, label=f'a1/a0 = {a1}')
#     for zero in zeros:
#         plt.plot([0, np.real(zero)], [0, np.imag(zero)], color=color, linestyle='--', alpha=0.3)

# NEW: График зависимости частоты нуля от a1/a0
zero_frequencies = []
for a1 in a1_values:
    b = [1, a1, 1]
    zeros = np.roots(b)
    positive_imag_zeros = zeros[zeros.imag > 0]
    if len(positive_imag_zeros) > 0:
        zero = positive_imag_zeros[0]
    else:
        zero = zeros[0]
    freq_zero = np.angle(zero) * fd / (2 * np.pi)
    if freq_zero < 0:
        freq_zero += fd
    zero_frequencies.append(freq_zero)

# Преобразуем частоты для отображения
nyquist_freq = fd / 2  # Частота Найквиста
display_frequencies = []
for f in zero_frequencies:
    if f <= nyquist_freq / 2:  # Первая половина (0–2000 Гц) остаётся без изменений
        display_frequencies.append(f)
    else:  # Вторая половина (2000–4000 Гц) отображается с инверсией
        display_frequencies.append(nyquist_freq - f)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(a1_values, display_frequencies, 'bo-', linewidth=2, markersize=8)
plt.title('Зависимость частоты нуля АЧХ от коэффициента a1')
plt.xlabel('a1')
plt.ylabel('Частота нуля (Гц)')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.title('Траектория нулей с лучами при изменении a1/a0')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.grid(True)
plt.axis('equal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

