import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

path = "modul_signal.txt"

with open(path) as file:
    input_signal = [float(line.strip().replace(',', '.')) for line in file]

# Функция для анализа трансверсального фильтра
def analyze_transversal_filter(a1_over_a0, a2_over_a0, title, input_signal):
    a0 = 1  # Нормировка
    a1 = a1_over_a0 * a0
    a2 = a2_over_a0 * a0

    # Коэффициенты фильтра (трансверсальный, поэтому знаменатель [1])
    b = [a0, a1, a2]
    a = [1]

    # Нули и полюса
    zeros = np.roots(b)
    poles = np.roots(a)

    # АЧХ
    w, h = signal.freqz(b, a, worN=2000, fs=2 * np.pi)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))  # Защита от log(0)

    output_signal = signal.lfilter(b, a, input_signal)

    # Импульсная характеристика
    impulse = np.zeros(10)
    impulse[0] = 1
    h_imp = signal.lfilter(b, a, impulse)

    # Графики
    plt.figure(figsize=(15, 10))

    # Нули и полюса
    plt.subplot(2, 2, 1)
    plt.title(f'Нули и полюса ({title})')
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='r', label='Нули')
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='b', label='Полюса')
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Единичная окружность')

    # Рисуем лучи от центра к нулям
    for zero in zeros:
        plt.plot([0, np.real(zero)], [0, np.imag(zero)], 'g--', alpha=0.5)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    # АЧХ
    plt.subplot(2, 2, 2)
    plt.title(f'АЧХ ({title})')
    plt.plot(w, h_db)
    plt.xlabel('Частота (рад/с)')
    plt.ylabel('Амплитуда (дБ)')
    plt.grid(True)

    # Импульсная характеристика
    plt.subplot(2, 2, 3)
    plt.title(f'Импульсная характеристика ({title})')
    plt.stem(np.arange(len(h_imp)), h_imp)
    plt.xlabel('Отсчеты')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.title(f'Сигнал ({title})')
    plt.plot(input_signal, '-b', label='Вход')
    plt.plot(output_signal, '-r', label='Выход')
    plt.xlabel('Отсчеты')
    plt.ylabel('Сигнал')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return zeros

# Параметры для исследования
a2_over_a0 = 1  # По условию a2/a0 = 1
a1_values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]  # Более детальный перебор значений

# Анализ для каждого a1/a0
for a1 in a1_values:
    zeros = analyze_transversal_filter(a1, a2_over_a0, f'a1/a0 = {a1}', input_signal)

# Построение траектории нулей с лучами
plt.figure(figsize=(10, 8))
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Единичная окружность')

# Цветовая карта для a1/a0
colors = plt.cm.viridis(np.linspace(0, 1, len(a1_values)))

for a1, color in zip(a1_values, colors):
    b = [1, a1, 1]
    zeros = np.roots(b)

    # Рисуем нули
    plt.scatter(np.real(zeros), np.imag(zeros), color=color, label=f'a1/a0 = {a1}')

    # Рисуем лучи к нулям
    for zero in zeros:
        plt.plot([0, np.real(zero)], [0, np.imag(zero)], color=color, linestyle='--', alpha=0.3)

plt.title('Траектория нулей с лучами при изменении a1/a0')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.grid(True)
plt.axis('equal')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()