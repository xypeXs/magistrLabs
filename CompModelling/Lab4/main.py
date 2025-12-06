import numpy as np
import matplotlib.pyplot as plt

# ==================== ЗАДАНИЕ ПАРАМЕТРОВ (вариант 1) ====================
D = 2.0
alpha = 5.0
h = 0.001
T_end = 10.0  # общее время моделирования
# =========================================================================

# Рассчёт параметров формирующего фильтра
T_f = 1.0 / alpha
S0 = h / 12.0
k_f = np.sqrt(2.0 * D / (alpha * S0))

# Количество шагов
N = int(T_end / h)

# Генерация белого шума с равномерным распределением [0, 1]
xi = np.random.rand(N)
# Центрирование белого шума
g = xi - 0.5

# Дискретная модель формирующего фильтра (метод Эйлера)
X = np.zeros(N)
for i in range(1, N):
    X[i] = X[i-1] + (k_f * g[i-1] - X[i-1]) * h / T_f

# Оценка математического ожидания
m_est = np.mean(X)

# Вычисление оценки корреляционной функции до лага 3/alpha
lag_max = int(3.0 / alpha / h)  # максимальный лаг в отсчётах
lags = np.arange(0, lag_max + 1)
K_est = np.zeros(lag_max + 1)
for j in range(lag_max + 1):
    if j == 0:
        K_est[j] = np.mean((X - m_est)**2)
    else:
        K_est[j] = np.mean((X[:-j] - m_est) * (X[j:] - m_est))

# Теоретическая корреляционная функция
tau = lags * h  # перевод отсчётов во время
K_theory = D * np.exp(-alpha * np.abs(tau))

# Построение графиков
plt.figure(figsize=(12, 6))

# График 1: реализация случайного процесса
plt.subplot(2, 2, 1)
plt.plot(np.arange(0, T_end, h)[:lag_max], X[:lag_max])
plt.xlabel('Время, с')
plt.ylabel('X(t)')
plt.title('Реализация случайного процесса (интервал [0; 3/a])')
plt.grid(True)

# График 2: оценённая и теоретическая корреляционные функции
plt.subplot(2, 2, 2)
plt.plot(tau, K_est, label='Оценка по модели')
plt.plot(tau, K_theory, label='Теоретическая K(t)=D*exp(-α|t|)', linestyle='--')
plt.xlabel('Время τ, с')
plt.ylabel('K(τ)')
plt.title('Корреляционная функция')
plt.legend()
plt.grid(True)

# График 3: разность между оценкой и теорией
plt.subplot(2, 2, 3)
plt.plot(tau, K_est - K_theory)
plt.xlabel('Время τ, с')
plt.ylabel('ΔK(τ)')
plt.title('Разность между оценкой и теорией')
plt.grid(True)

# График 4: гистограмма значений процесса
plt.subplot(2, 2, 4)
plt.hist(X, bins=50, density=True, edgecolor='black')
plt.xlabel('Значение X')
plt.ylabel('Плотность')
plt.title('Гистограмма значений процесса')
plt.grid(True)

plt.tight_layout()
plt.show()

# Вывод параметров
print(f"Параметры модели:")
print(f"  D = {D}, α = {alpha}, h = {h}")
print(f"Параметры фильтра:")
print(f"  T_f = {T_f:.4f}, S0 = {S0:.6f}, k_f = {k_f:.4f}")
print(f"Оценка математического ожидания: {m_est:.6f}")
print(f"Оценка дисперсии: {K_est[0]:.6f} (теоретическая D = {D})")