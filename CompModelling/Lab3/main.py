import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, stats
import random
import math


# Вариант 1: f(z) = 1/(1+sin(z)) на [0, pi/2], критерий Пирсона (1)

class RandomGenerator:
    def __init__(self):
        self.a = 0  # левая граница
        self.b = np.pi / 2  # правая граница

    def pdf(self, z):
        """Плотность распределения"""
        return 1 / (1 + np.sin(z))

    def cdf(self, z):
        """Функция распределения (аналитическое выражение)"""
        return 2 - 2 / (1 + np.tan(z / 2))

    def inverse_cdf(self, u):
        """Обратная функция распределения (аналитическое выражение)"""
        return 2 * np.arctan(u / (2 - u))

    def generate(self, n):
        """Генерация n случайных чисел с заданным распределением"""
        # Генерируем равномерные числа
        uniform_numbers = np.random.random(n)
        # Применяем метод обратных функций
        return np.array([self.inverse_cdf(u) for u in uniform_numbers])

    def theoretical_mean(self):
        """Теоретическое математическое ожидание"""
        # E[X] = ∫z*f(z)dz
        result, _ = integrate.quad(lambda z: z * self.pdf(z), self.a, self.b)
        return result

    def theoretical_variance(self):
        """Теоретическая дисперсия"""
        # E[X^2] = ∫z^2*f(z)dz
        ex2, _ = integrate.quad(lambda z: z ** 2 * self.pdf(z), self.a, self.b)
        ex = self.theoretical_mean()
        return ex2 - ex ** 2


def estimate_mean_variance(sample):
    """Оценка математического ожидания и дисперсии по выборке"""
    n = len(sample)
    mean = np.mean(sample)
    variance = np.var(sample, ddof=1)  # несмещенная оценка
    return mean, variance


def pearson_test(sample, pdf, a, b, m=10):
    """Критерий согласия Пирсона"""
    n = len(sample)

    # Создаем гистограмму
    hist, bin_edges = np.histogram(sample, bins=m, range=(a, b))

    # Теоретические вероятности для каждого интервала
    p_theoretical = np.zeros(m)
    for i in range(m):
        # Вероятность попадания в i-й интервал
        p, _ = integrate.quad(pdf, bin_edges[i], bin_edges[i + 1])
        p_theoretical[i] = p

    # Статистика хи-квадрат
    chi2_stat = 0
    for i in range(m):
        if p_theoretical[i] > 0:
            expected = n * p_theoretical[i]
            chi2_stat += (hist[i] - expected) ** 2 / expected

    # Число степеней свободы
    df = m - 1

    # p-значение
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    return chi2_stat, df, p_value, hist, bin_edges, p_theoretical


def main():
    # Инициализация генератора
    generator = RandomGenerator()

    # Объемы выборок
    sample_sizes = [50, 100, 1000, 100000]

    # Теоретические значения
    theoretical_mean = generator.theoretical_mean()
    theoretical_var = generator.theoretical_variance()

    print("=" * 60)
    print("Лабораторная работа №3: Построение генератора случайных чисел")
    print(f"Плотность распределения: f(z) = 1/(1+sin(z)) на [0, π/2]")
    print("Критерий согласия: Пирсона (χ²)")
    print("=" * 60)
    print(f"\nТеоретические значения:")
    print(f"Математическое ожидание: {theoretical_mean:.6f}")
    print(f"Дисперсия: {theoretical_var:.6f}")
    print("=" * 60)

    # Для каждой выборки
    for n in sample_sizes:
        print(f"\n{'=' * 40}")
        print(f"Объем выборки: n = {n}")
        print(f"{'=' * 40}")

        # Генерация выборки
        sample = generator.generate(n)

        # Оценки по выборке
        sample_mean, sample_var = estimate_mean_variance(sample)

        print(f"\nОценки по выборке:")
        print(f"Математическое ожидание: {sample_mean:.6f}")
        print(f"Выборочная дисперсия: {sample_var:.6f}")
        print(f"Отклонение от теоретического:")
        print(f"  По мат ожиданию: {abs(sample_mean - theoretical_mean):.6f}")
        print(f"  По дисперсии: {abs(sample_var - theoretical_var):.6f}")

        # Критерий Пирсона
        m = min(20, max(5, int(np.sqrt(n))))  # Число интервалов
        chi2_stat, df, p_value, hist, bin_edges, p_theoretical = pearson_test(
            sample, generator.pdf, generator.a, generator.b, m
        )

        print(f"\nКритерий Пирсона (χ²):")
        print(f"Число интервалов: m = {m}")
        print(f"Статистика χ²: {chi2_stat:.4f}")
        print(f"Число степеней свободы: df = {df}")
        print(f"p-значение: {p_value:.4f}")

        # Решение по критерию
        alpha = 0.05  # Уровень значимости
        critical_value = stats.chi2.ppf(1 - alpha, df)

        print(f"\nКритическое значение (α={alpha}): {critical_value:.4f}")
        if chi2_stat < critical_value:
            print("✓ Гипотеза о согласии с заданным распределением НЕ отвергается")
        else:
            print("✗ Гипотеза о согласии с заданным распределением ОТВЕРГАЕТСЯ")

        # Построение графиков
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Гистограмма и теоретическая плотность
        bin_width = bin_edges[1] - bin_edges[0]
        x_hist = bin_edges[:-1] + bin_width / 2
        ax1.bar(x_hist, hist / n, width=bin_width * 0.8,
                alpha=0.7, label='Выборочная плотность')

        # Теоретическая плотность
        x_theor = np.linspace(generator.a, generator.b, 200)
        y_theor = generator.pdf(x_theor)
        ax1.plot(x_theor, y_theor, 'r-', linewidth=2, label='Теоретическая плотность')

        ax1.set_xlabel('z')
        ax1.set_ylabel('Плотность вероятности')
        ax1.set_title(f'Гистограмма (n={n}, m={m})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Сравнение теоретических и выборочных вероятностей
        x_pos = np.arange(m)
        width = 0.35

        ax2.bar(x_pos - width / 2, p_theoretical, width,
                label='Теоретическая', alpha=0.7)
        ax2.bar(x_pos + width / 2, hist / n, width,
                label='Выборочная', alpha=0.7)

        ax2.set_xlabel('Номер интервала')
        ax2.set_ylabel('Вероятность')
        ax2.set_title('Сравнение вероятностей по интервалам')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'lr3_sample_{n}.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Вывод информации по интервалам
        print(f"\nРаспределение по интервалам:")
        print("Интервал\tТеор. вер.\tВыб. вер.\tРазность")
        for i in range(m):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            p_th = p_theoretical[i]
            p_emp = hist[i] / n
            diff = abs(p_emp - p_th)
            print(f"[{lower:.3f},{upper:.3f})\t{p_th:.4f}\t\t{p_emp:.4f}\t\t{diff:.4f}")

    # Дополнительный анализ: генерация большой выборки для точных оценок
    print(f"\n{'=' * 60}")
    print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
    print(f"{'=' * 60}")

    n_large = 1000000
    large_sample = generator.generate(n_large)
    mean_large, var_large = estimate_mean_variance(large_sample)

    print(f"\nБольшая выборка (n={n_large}):")
    print(f"Выборочное среднее: {mean_large:.6f}")
    print(f"Теоретическое среднее: {theoretical_mean:.6f}")
    print(f"Относительная ошибка: {abs(mean_large - theoretical_mean) / theoretical_mean * 100:.4f}%")
    print(f"\nВыборочная дисперсия: {var_large:.6f}")
    print(f"Теоретическая дисперсия: {theoretical_var:.6f}")
    print(f"Относительная ошибка: {abs(var_large - theoretical_var) / theoretical_var * 100:.4f}%")


if __name__ == "__main__":
    # Установка seed для воспроизводимости
    np.random.seed(42)
    random.seed(42)

    main()
