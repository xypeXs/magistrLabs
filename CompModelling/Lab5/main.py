import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate, special
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Параметры варианта 1
# Закон распределения: f(z) = 1/(1+sin(z)), z ∈ [0, pi/2]
# Корреляционная функция: K(tau) = D * exp(-alpha*|tau|)
D = 2.0          # дисперсия для корреляционной функции (в исходных данных)
alpha = 5.0      # параметр корреляционной функции
h = 0.001        # шаг дискретизации
D_filt = 1.0     # дисперсия для формирующего фильтра (стандартизованный процесс)

# Параметры моделирования
N = 100000       # общее количество точек для моделирования
N_check = 10000  # количество точек для проверки корреляционной функции
burn_in = 1000   # количество точек для "прогрева" фильтра

# ---------------------------------------------------------------------
# 1. Генерация белого шума с равномерным распределением на [0, 1]
# ---------------------------------------------------------------------
print("1. Генерация белого шума с равномерным распределением...")
np.random.seed(42)  # для воспроизводимости
uniform_white = np.random.rand(N)

# ---------------------------------------------------------------------
# 2. Преобразование в стандартизованный нормальный белый шум (среднее 0, дисперсия 1)
#    Используем метод Бокса-Мюллера
# ---------------------------------------------------------------------
print("2. Преобразование в нормальный белый шум (метод Бокса-Мюллера)...")
# Преобразуем пары равномерных чисел в нормальные
if len(uniform_white) % 2 == 1:
    uniform_white = uniform_white[:-1]  # делаем чётное количество

U1 = uniform_white[:len(uniform_white)//2]
U2 = uniform_white[len(uniform_white)//2:]

# Метод Бокса-Мюллера
Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
Z1 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)

# Объединяем и перемешиваем
normal_white = np.concatenate([Z0, Z1])
np.random.shuffle(normal_white)

# Проверка параметров
print(f"   Мат. ожидание: {np.mean(normal_white):.6f} (теор. 0.0)")
print(f"   Дисперсия: {np.var(normal_white):.6f} (теор. 1.0)")

# ---------------------------------------------------------------------
# 3. Формирующий фильтр для получения коррелированного нормального процесса
#    Уравнение: x[k+1] = a * x[k] + b * xi[k]
#    где a = exp(-h/Tф), b = kф * (1 - exp(-h/Tф))
#    Tф = 1/alpha, kф = sqrt(2*D_filt/(alpha*S0)), S0 = h (дисперсия белого шума = 1)
# ---------------------------------------------------------------------
print("3. Применение формирующего фильтра...")
T_f = 1.0 / alpha
S0 = h  # интенсивность белого шума (дисперсия = 1, шаг = h)
k_f = np.sqrt(2 * D_filt / (alpha * S0))
a = np.exp(-h / T_f)
b = k_f * (1 - np.exp(-h / T_f))

print(f"   Параметры фильтра: T_f = {T_f:.4f}, k_f = {k_f:.4f}")
print(f"   Коэффициенты: a = {a:.6f}, b = {b:.6f}")

# Применяем фильтр
x = np.zeros(N)
x[0] = normal_white[0] * b  # начальное условие

for i in range(1, N):
    x[i] = a * x[i-1] + b * normal_white[i]

# Удаляем переходный процесс
x = x[burn_in:]

print(f"   После фильтра: мат. ожидание = {np.mean(x):.6f}, дисперсия = {np.var(x):.6f}")

# ---------------------------------------------------------------------
# 4. Проверка корреляционной функции после фильтра
# ---------------------------------------------------------------------
print("4. Проверка корреляционной функции...")
def theoretical_corr(tau):
    return D_filt * np.exp(-alpha * np.abs(tau))

def estimated_corr(signal, max_lag=100):
    n = len(signal)
    mean_signal = np.mean(signal)
    signal_centered = signal - mean_signal
    corr = np.zeros(max_lag)

    for lag in range(max_lag):
        if lag == 0:
            corr[lag] = np.mean(signal_centered**2)
        else:
            corr[lag] = np.mean(signal_centered[:-lag] * signal_centered[lag:])

    return corr

# Проверяем на первых N_check точках
max_lag = 100
corr_est = estimated_corr(x[:N_check], max_lag)
corr_theory = theoretical_corr(np.arange(max_lag) * h)

# Нормируем оценённую корреляционную функцию
corr_est_norm = corr_est / corr_est[0]

# ---------------------------------------------------------------------
# 5. Преобразование в равномерный процесс на [0, 1] с помощью ФРВ нормального распределения
# ---------------------------------------------------------------------
print("5. Преобразование в равномерный процесс...")
uniform_process = special.ndtr(x)  # ФРВ стандартного нормального распределения

print(f"   Равномерный процесс: мат. ожидание = {np.mean(uniform_process):.6f} (теор. 0.5)")
print(f"   Дисперсия = {np.var(uniform_process):.6f} (теор. {1/12:.6f})")

# ---------------------------------------------------------------------
# 6. Преобразование в заданный закон распределения методом обратных функций
#    f(z) = 1/(1+sin(z)), z ∈ [0, pi/2]
#    ФРВ: F(z) = ∫_0^z 1/(1+sin(t)) dt = 2 - 2/(1+tan(z/2))
#    Обратная функция: z = 2 * arctan(u/(2-u)), где u ∈ [0, 1]
# ---------------------------------------------------------------------
print("6. Преобразование в заданный закон распределения...")
def inverse_cdf(u):
    """Обратная функция распределения для f(z)=1/(1+sin(z)), z∈[0,π/2]"""
    return 2 * np.arctan(u / (2 - u))

final_process = inverse_cdf(uniform_process)

# ---------------------------------------------------------------------
# 7. Проверка конечного распределения
# ---------------------------------------------------------------------
print("7. Проверка конечного распределения...")

# Теоретические моменты
# Математическое ожидание: E[z] = ∫_0^{π/2} z * f(z) dz
def f_z(z):
    return 1 / (1 + np.sin(z))

def z_f_z(z):
    return z / (1 + np.sin(z))

# Вычисляем теоретические моменты численно
z_range = np.linspace(0, np.pi/2, 1000)
pdf_values = f_z(z_range)

# Математическое ожидание
E_z, _ = integrate.quad(z_f_z, 0, np.pi/2)

# E[z^2]
def z2_f_z(z):
    return z**2 / (1 + np.sin(z))

E_z2, _ = integrate.quad(z2_f_z, 0, np.pi/2)

# Дисперсия
D_z = E_z2 - E_z**2

print(f"   Теоретические значения:")
print(f"     Мат. ожидание: {E_z:.6f}")
print(f"     Дисперсия: {D_z:.6f}")

print(f"   Выборочные значения:")
print(f"     Мат. ожидание: {np.mean(final_process):.6f}")
print(f"     Дисперсия: {np.var(final_process):.6f}")

# ---------------------------------------------------------------------
# 8. Проверка согласия с помощью критерия Пирсона (хи-квадрат)
# ---------------------------------------------------------------------
print("\n8. Проверка критерием Пирсона (хи-квадрат)...")

# Разбиваем на интервалы
n_bins = 20  # количество интервалов
hist, bin_edges = np.histogram(final_process, bins=n_bins, range=(0, np.pi/2), density=True)

# Теоретические вероятности попадания в интервалы
def cdf_z(z):
    """Функция распределения для f(z)=1/(1+sin(z))"""
    return 2 - 2 / (1 + np.tan(z/2))

# Вероятности попадания в каждый интервал
p_theory = np.zeros(n_bins)
for i in range(n_bins):
    p_theory[i] = cdf_z(bin_edges[i+1]) - cdf_z(bin_edges[i])

# Частоты попадания
counts, _ = np.histogram(final_process, bins=n_bins, range=(0, np.pi/2))

# Критерий хи-квадрат
chi2_stat = np.sum((counts - len(final_process) * p_theory)**2 / (len(final_process) * p_theory))
df = n_bins - 1  # степени свободы

# p-value
p_value = 1 - stats.chi2.cdf(chi2_stat, df)

print(f"   Статистика хи-квадрат: {chi2_stat:.4f}")
print(f"   Число степеней свободы: {df}")
print(f"   p-value: {p_value:.6f}")

if p_value > 0.05:
    print("   Результат: нет оснований отвергать гипотезу о согласии (распределение соответствует заданному)")
else:
    print("   Результат: гипотеза о согласии отвергается (распределение не соответствует заданному)")

# ---------------------------------------------------------------------
# 9. Визуализация результатов
# ---------------------------------------------------------------------
print("\n9. Построение графиков...")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# 1. Исходный белый шум (нормальный)
axes[0, 0].hist(normal_white[:5000], bins=50, density=True, alpha=0.7, label='Выборка')
x_norm = np.linspace(-4, 4, 100)
axes[0, 0].plot(x_norm, stats.norm.pdf(x_norm), 'r-', linewidth=2, label='Теоретическая')
axes[0, 0].set_title('Нормальный белый шум')
axes[0, 0].set_xlabel('Значение')
axes[0, 0].set_ylabel('Плотность')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. После формирующего фильтра
axes[0, 1].hist(x[:5000], bins=50, density=True, alpha=0.7, label='Выборка')
axes[0, 1].plot(x_norm, stats.norm.pdf(x_norm), 'r-', linewidth=2, label='Теоретическая')
axes[0, 1].set_title('После формирующего фильтра')
axes[0, 1].set_xlabel('Значение')
axes[0, 1].set_ylabel('Плотность')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Корреляционная функция после фильтра
lags = np.arange(max_lag) * h
axes[0, 2].plot(lags, corr_est_norm, 'b-', linewidth=2, label='Оцененная')
axes[0, 2].plot(lags, np.exp(-alpha * lags), 'r--', linewidth=2, label='Теоретическая')
axes[0, 2].set_title('Корреляционная функция')
axes[0, 2].set_xlabel('τ')
axes[0, 2].set_ylabel('K(τ)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Равномерный процесс
axes[1, 0].hist(uniform_process[:5000], bins=50, density=True, alpha=0.7, label='Выборка')
axes[1, 0].axhline(1.0, 0, 1, color='r', linewidth=2, label='Теоретическая')
axes[1, 0].set_title('Равномерный процесс')
axes[1, 0].set_xlabel('Значение')
axes[1, 0].set_ylabel('Плотность')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Конечный процесс
axes[1, 1].hist(final_process[:5000], bins=50, density=True, alpha=0.7, label='Выборка')
axes[1, 1].plot(z_range, pdf_values, 'r-', linewidth=2, label='Теоретическая')
axes[1, 1].set_title('Конечный процесс')
axes[1, 1].set_xlabel('z')
axes[1, 1].set_ylabel('f(z)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Сравнение теоретической и выборочной ФРВ
sorted_process = np.sort(final_process)
empirical_cdf = np.arange(1, len(sorted_process)+1) / len(sorted_process)
theoretical_cdf_vals = cdf_z(sorted_process)

axes[1, 2].plot(sorted_process, empirical_cdf, 'b-', linewidth=1, label='Выборочная ФРВ')
axes[1, 2].plot(sorted_process, theoretical_cdf_vals, 'r--', linewidth=1, label='Теоретическая ФРВ')
axes[1, 2].set_title('Функции распределения')
axes[1, 2].set_xlabel('z')
axes[1, 2].set_ylabel('F(z)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# 7. Пример реализации процесса
axes[2, 0].plot(np.arange(500) * h, final_process[:500], 'b-', linewidth=1)
axes[2, 0].set_title('Пример реализации процесса (500 точек)')
axes[2, 0].set_xlabel('Время')
axes[2, 0].set_ylabel('Значение')
axes[2, 0].grid(True, alpha=0.3)

# 8. Зависимость мат. ожидания от объема выборки
sample_sizes = [50, 100, 1000, 10000, 100000]
means = []
variances = []

for size in sample_sizes:
    if size <= len(final_process):
        sample = final_process[:size]
        means.append(np.mean(sample))
        variances.append(np.var(sample))

axes[2, 1].plot(sample_sizes[:len(means)], means, 'bo-', linewidth=2, markersize=6)
axes[2, 1].axhline(E_z, color='r', linestyle='--', linewidth=2, label='Теоретическое')
axes[2, 1].set_title('Зависимость мат. ожидания от объема выборки')
axes[2, 1].set_xlabel('Объем выборки')
axes[2, 1].set_ylabel('Мат. ожидание')
axes[2, 1].set_xscale('log')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# 9. Зависимость дисперсии от объема выборки
axes[2, 2].plot(sample_sizes[:len(variances)], variances, 'go-', linewidth=2, markersize=6)
axes[2, 2].axhline(D_z, color='r', linestyle='--', linewidth=2, label='Теоретическое')
axes[2, 2].set_title('Зависимость дисперсии от объема выборки')
axes[2, 2].set_xlabel('Объем выборки')
axes[2, 2].set_ylabel('Дисперсия')
axes[2, 2].set_xscale('log')
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab5_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nРезультаты сохранены в файл 'lab5_results.png'")
print("\nЛабораторная работа №5 выполнена успешно!")