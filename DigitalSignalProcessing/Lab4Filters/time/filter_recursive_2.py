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

# Функция для анализа рекурсивного фильтра 2-го порядка
def analyze_recursive_2_filter(a0, a1, a2, b1, b2, input_signal, title, fd=8000):
    print(f'a0: {a0}, a1: {a1}, a2: {a2}, b1: {b1}, b2: {b2}')

    dt = 1 / fd

    # Коэффициенты фильтра
    b = [a0, a1, a2]
    a = [1, -b1, -b2]

    # Нули и полюса
    zeros = np.roots(b)
    poles = np.roots(a)

    # АЧХ в Герцах
    w, h = signal.freqz(b, a, worN=2000, fs=fd)
    h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    h_linear = np.abs(h)  # Линейная шкала для АЧХ

    output_signal = signal.lfilter(b, a, input_signal)

    # Импульсная характеристика с временной осью
    impulse = np.zeros(100)
    impulse[0] = 1
    h_imp = signal.lfilter(b, a, impulse)
    t_imp = np.arange(len(h_imp)) * dt

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

    # АЧХ (в дБ)
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

    return zeros, w, h_linear  # Возвращаем линейную АЧХ

# Функции для вариантов из таблицы
def table_params1():
    zeros, w, h_linear = analyze_recursive_2_filter(
        a0=1, a1=0, a2=0,
        b1=0.218, b2=-0.437,
        input_signal=input_signal,
        title='Вариант 1',
        fd=fd
    )
    return w, h_linear

def table_params2():
    zeros, w, h_linear = analyze_recursive_2_filter(
        a0=1, a1=-2, a2=1,
        b1=0, b2=0,
        input_signal=input_signal,
        title='Вариант 2',
        fd=fd
    )
    return w, h_linear

def table_params3():
    zeros, w, h_linear = analyze_recursive_2_filter(
        a0=1, a1=-2, a2=1,
        b1=0.218, b2=-0.437,
        input_signal=input_signal,
        title='Вариант 3',
        fd=fd
    )
    return w, h_linear

if __name__ == '__main__':
    # Получаем АЧХ для всех вариантов
    w1, h_linear1 = table_params1()
    w2, h_linear2 = table_params2()
    w3, h_linear3 = table_params3()

    # Частоты для проверки
    freqs = [0, 20, 50]  # Гц
    indices = [np.argmin(np.abs(w1 - f)) for f in freqs]  # Индексы ближайших частот

    # Выводим значения АЧХ и проверяем произведение
    print("Проверка произведения АЧХ:")
    for i, f in enumerate(freqs):
        idx = indices[i]
        h1 = h_linear1[idx]
        h2 = h_linear2[idx]
        h3 = h_linear3[idx]
        product = h1 * h2
        print(f"При f={f} Гц:")
        print(f"  Вариант 1: {h1:.3f}")
        print(f"  Вариант 2: {h2:.3f}")
        print(f"  Произведение: {h1:.3f} * {h2:.3f} = {product:.3f}")
        print(f"  Вариант 3: {h3:.3f}")
        print(f"  Разница: {abs(h3 - product):.3f}\n")

    # График для сравнения АЧХ (в линейной шкале)
    plt.figure(figsize=(10, 6))
    plt.plot(w1, h_linear1, label='Вариант 1', color='blue')
    plt.plot(w2, h_linear2, label='Вариант 2', color='red')
    plt.plot(w3, h_linear3, label='Вариант 3', color='green')
    plt.plot(w1, h_linear1 * h_linear2, '--', label='Произведение (Вариант 1 * Вариант 2)', color='black')
    plt.title('Сравнение АЧХ (линейная шкала)')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()