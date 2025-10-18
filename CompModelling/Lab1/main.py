import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

class NonlinearDynamicSystem:
    def __init__(self, model_type, variant=1):
        self.model_type = model_type
        self.variant = variant
        self.set_parameters()

    def set_parameters(self):
        """Установка параметров по варианту 1"""
        if self.model_type == 1:
            # Модель 1 - система уравнений 5-го порядка
            self.p = 1e5
            self.a = 0.6
            self.m = 2000
            self.u = 10
            self.c_r = 0.05
            self.c_f = 0.01
            self.m1 = 0.1
            self.m2 = 0.01
            self.T = 10
            self.x0 = [1800, 0.8, 0, 0, 0.8]
            self.error_var = 3  # Погрешность по x4 (индекс 3)

        elif self.model_type == 2:
            # Модель 2 - система уравнений 5-го порядка
            self.k = 1
            self.l = 10
            self.m = 2
            self.n = 8
            self.k_t = 100
            self.b = 22000
            self.i1 = 10
            self.i2 = 1
            self.s = 100
            self.V = 800
            self.T = 11
            self.delta_max = 0.5
            self.x0 = [1, 1, 0, 0, 0]
            self.error_var = 4  # Погрешность по x5

        elif self.model_type == 3:
            # Модель 3 - система уравнений 5-го порядка
            self.k = 1
            self.l = 12
            self.m = 1
            self.n = 8
            self.k_t = 100
            self.b = 30000
            self.i1 = 10
            self.i2 = 1
            self.s = 100
            self.V = 800
            self.T = 11
            self.alpha_max = 0.5
            self.x0 = [1, 1, 0, 0, 0]
            self.error_var = 4  # Погрешность по x5

        elif self.model_type == 4:
            # Модель 4 - система уравнений 3-го порядка
            self.c = 9000
            self.u = 10
            self.T = 11
            self.h_in = 8480
            self.x0 = [0, 0, 600]
            self.error_var = 0  # Погрешность по x1

        self.g = 9.81

    def model1_rhs(self, x, t):
        """Правая часть для модели 1"""
        x1, x2, x3, x4, x5 = x

        # Защита от деления на ноль
        if self.m - self.u * t <= 0:
            return [0, 0, 0, 0, 0]

        dx1dt = -self.g * np.sin(x2) + (self.p - self.a * self.c_r * x1**2) / (self.m - self.u * t)
        dx2dt = (-self.g + (self.p * np.sin(x5 - x2) + self.a * self.c_f * x1**2) / (self.m - self.u * t)) / max(x1, 1e-6)
        dx3dt = (self.m1 * self.a * (x2 - x5) * x1**2 - self.m2 * self.a * x1**2 * x3) / (self.m - self.u * t)
        dx4dt = x1 * np.sin(x2)
        dx5dt = x3

        return [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt]

    def model2_rhs(self, x, t):
        """Правая часть для модели 2"""
        x1, x2, x3, x4, x5 = x

        theta = (10000 - x5) / max(self.b - self.V * t, 1e-6)

        # Ограничение delta
        delta = -self.k_t * x4 - self.i1 * x2 - self.i2 * x3 + self.s * (theta - x2)
        if abs(delta) <= self.delta_max:
            x4_val = delta
        else:
            x4_val = self.delta_max * np.sign(delta)

        dx1dt = self.k * x2 - self.k * x1
        dx2dt = x3
        dx3dt = self.l * x1 - self.l * x2 - self.m * x3 + self.n * x4_val
        dx4dt = delta
        dx5dt = self.V * np.sin(x1)

        return [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt]

    def model3_rhs(self, x, t):
        """Правая часть для модели 3"""
        x1, x2, x3, x4, x5 = x

        alpha = x2 - x1
        if abs(alpha) <= self.alpha_max:
            alpha_star = alpha
        else:
            alpha_star = self.alpha_max * np.sign(alpha)

        theta = (10000 - x5) / max(self.b - self.V * t, 1e-6)

        dx1dt = self.k * alpha_star
        dx2dt = x3
        dx3dt = self.l * x1 - self.l * x2 - self.m * x3 + self.n * x4
        dx4dt = -self.k_t * x4 - self.i1 * x2 - self.i2 * x3 + self.s * (theta - x1)
        dx5dt = self.V * np.sin(x1)

        return [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt]

    def model4_rhs(self, x, t):
        """Правая часть для модели 4"""
        x1, x2, x3 = x

        # Вычисление параметра r
        if self.variant % 2 == 1:  # нечетные варианты
            r = 0.1 * np.exp(-x1 / self.h_in)
        else:  # четные варианты
            # Линейная интерполяция по таблице (упрощенная версия)
            r = max(0.0016, 0.015 - 0.0000013 * x1)

        dx1dt = x2
        dx2dt = (self.c * self.u) / max(x3, 1e-6) - self.g - (r * x2**2) / max(x3, 1e-6)
        dx3dt = -self.u

        return [dx1dt, dx2dt, dx3dt]

    def solve_system(self, h, method='euler'):
        """Решение системы дифференциальных уравнений"""
        t = np.arange(0, self.T + h, h)

        if method == 'euler':
            n = len(t)
            x = np.zeros((n, len(self.x0)))
            x[0] = self.x0

            for i in range(1, n):
                if self.model_type == 1:
                    dxdt = self.model1_rhs(x[i-1], t[i-1])
                elif self.model_type == 2:
                    dxdt = self.model2_rhs(x[i-1], t[i-1])
                elif self.model_type == 3:
                    dxdt = self.model3_rhs(x[i-1], t[i-1])
                elif self.model_type == 4:
                    dxdt = self.model4_rhs(x[i-1], t[i-1])

                x[i] = x[i-1] + np.array(dxdt) * h

            return t, x

    def calculate_error(self, h):
        """Вычисление относительной погрешности"""
        # Решение с шагом h
        t1, x1 = self.solve_system(h)
        y_h = x1[-1, self.error_var]

        # Решение с шагом h/2
        t2, x2 = self.solve_system(h/2)
        y_h2 = x2[-1, self.error_var]

        # Относительная погрешность
        if abs(y_h2) > 1e-10:
            relative_error = abs(y_h - y_h2) / abs(y_h2) * 100
        else:
            relative_error = abs(y_h - y_h2) * 100

        return relative_error, len(t1)

    def auto_select_step(self, max_error=1.0, initial_h=1.0):
        """Автоматический выбор шага интегрирования"""
        h = initial_h
        errors = []
        steps = []
        computations = []

        print("Автоматический подбор шага интегрирования:")
        print(f"Целевая погрешность: {max_error}%")

        for i in range(20):  # ограничение итераций
            error, n_steps = self.calculate_error(h)
            errors.append(error)
            steps.append(h)
            computations.append(n_steps)

            print(f"Шаг h={h:.6f}, погрешность δ={error:.4f}%, шагов вычислений: {n_steps}")

            if error <= max_error:
                print(f"Достигнута требуемая точность при h={h:.6f}")
                break
            else:
                h = h / 2
        else:
            print("Достигнуто максимальное количество итераций")

        return h, steps, errors, computations

    def analyze_step_dependence(self):
        """Анализ зависимости точности и трудоемкости от шага"""
        h_values = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
        errors = []
        computations = []

        print("\nАнализ зависимости точности от шага:")
        for h in h_values:
            error, n_steps = self.calculate_error(h)
            errors.append(error)
            computations.append(n_steps)
            print(f"h={h:.3f}, δ={error:.4f}%, шагов: {n_steps}")

        return h_values, errors, computations

    def plot_results(self, t, x, optimal_h=None):
        """Построение графиков результатов"""
        n_vars = x.shape[1]

        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3*n_vars))
        if n_vars == 1:
            axes = [axes]

        for i in range(n_vars):
            axes[i].plot(t, x[:, i], 'b-', linewidth=2)
            axes[i].set_ylabel(f'$x_{i+1}(t)$')
            axes[i].set_xlabel('Время, с')
            axes[i].grid(True)
            axes[i].set_title(f'Переменная состояния $x_{i+1}$')

        plt.tight_layout()

        if optimal_h is not None:
            plt.suptitle(f'Результаты моделирования (шаг h={optimal_h:.6f})', y=1.02)
        plt.show()

    def plot_analysis(self, h_values, errors, computations, optimal_h=None):
        """Построение графиков анализа"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # График зависимости погрешности от шага
        ax1.loglog(h_values, errors, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Шаг интегрирования h, с')
        ax1.set_ylabel('Относительная погрешность δ, %')
        ax1.set_title('Зависимость точности от шага интегрирования')
        ax1.grid(True, which="both", ls="-")

        if optimal_h is not None:
            ax1.axvline(x=optimal_h, color='r', linestyle='--',
                        label=f'Оптимальный h={optimal_h:.6f}')
            ax1.legend()

        # График зависимости трудоемкости от шага
        ax2.loglog(h_values, computations, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Шаг интегрирования h, с')
        ax2.set_ylabel('Количество шагов вычислений')
        ax2.set_title('Зависимость трудоемкости от шага интегрирования')
        ax2.grid(True, which="both", ls="-")

        plt.tight_layout()
        plt.show()

def main():
    """Основная функция выполнения лабораторной работы"""
    print("Лабораторная работа №1: Моделирование нелинейных динамических систем")
    print("=" * 70)

    # Анализ для всех моделей по варианту 1
    for model_type in [1, 2, 3, 4]:
        print(f"\n{'='*50}")
        print(f"МОДЕЛЬ {model_type}")
        print(f"{'='*50}")

        # Создание системы
        system = NonlinearDynamicSystem(model_type, variant=1)

        # 1. Решение с произвольным шагом
        print("\n1. Решение системы с шагом h=0.01:")
        t, x = system.solve_system(0.01)
        final_value = x[-1, system.error_var]
        print(f"Значение x_{system.error_var+1}(T) = {final_value:.6f}")

        # 2. Анализ зависимости точности от шага
        h_values, errors, computations = system.analyze_step_dependence()

        # 3. Автоматический выбор шага
        optimal_h, opt_steps, opt_errors, opt_computations = system.auto_select_step()

        # Решение с оптимальным шагом
        t_opt, x_opt = system.solve_system(optimal_h)
        final_value_opt = x_opt[-1, system.error_var]
        error_opt, _ = system.calculate_error(optimal_h)

        print(f"\nИтоговые результаты для оптимального шага h={optimal_h:.6f}:")
        print(f"x_{system.error_var+1}(T) = {final_value_opt:.6f}")
        print(f"Относительная погрешность δ = {error_opt:.4f}%")

        # Построение графиков
        system.plot_results(t_opt, x_opt, optimal_h)
        system.plot_analysis(h_values, errors, computations, optimal_h)

        print(f"\nЗавершена обработка модели {model_type}")

if __name__ == "__main__":
    main()