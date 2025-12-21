import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors


class CrowdCellularAutomaton:
    """
    Клеточный автомат для моделирования движения толпы
    """

    def __init__(self, grid, depth=3):
        self.grid = grid

        self.width = grid.shape[1]
        self.height = grid.shape[0]
        self.depth = depth

        # Приоритетное направление (по умолчанию - вправо, к выходу)
        self.priority_direction = "right"

        # Статистика
        self.people_count = 0
        self.evacuated_count = 0
        self.step_count = 0
        self.statistics = []

    # def __init__(self, width=40, height=30, depth=3, exit_width=3):
    #     """
    #     Инициализация автомата
    #
    #     Параметры:
    #     - width: ширина поля (в клетках)
    #     - height: высота поля (в клетках)
    #     - depth: глубина анализа (N в формулах)
    #     - exit_width: ширина выхода (в клетках)
    #     """
    #     self.width = width
    #     self.height = height
    #     self.depth = depth
    #
    #     # Состояния клеток: 0 - пусто, 1 - человек, 2 - стена/препятствие
    #     self.grid = np.zeros((height, width), dtype=int)
    #
    #     # Создаем стены по периметру
    #     self.grid[0, :] = 2  # верхняя стена
    #     self.grid[-1, :] = 2  # нижняя стена
    #     self.grid[:, 0] = 2  # левая стена
    #     self.grid[:, -1] = 2  # правая стена
    #
    #     # Создаем выход (убираем часть стены)
    #     exit_start = (height - exit_width) // 2
    #     for i in range(exit_start, exit_start + exit_width):
    #         self.grid[i, -1] = 0  # убираем правую стену для выхода
    #
    #     # Приоритетное направление (по умолчанию - вправо, к выходу)
    #     self.priority_direction = "right"
    #
    #     # Статистика
    #     self.people_count = 0
    #     self.evacuated_count = 0
    #     self.step_count = 0
    #     self.statistics = []

    def add_people_random(self, density=0.3):
        """Добавляем людей случайным образом с заданной плотностью"""
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                if self.grid[i, j] == 0 and random.random() < density:
                    self.grid[i, j] = 1
                    self.people_count += 1

    def add_people_pattern(self, pattern="block", x=10, y=10, size=5):
        """Добавляем людей по определенному шаблону"""
        if pattern == "block":
            for i in range(y, min(y + size, self.height - 1)):
                for j in range(x, min(x + size, self.width - 1)):
                    if self.grid[i, j] == 0:
                        self.grid[i, j] = 1
                        self.people_count += 1

    def add_obstacle(self, x1, y1, x2, y2):
        """Добавляем препятствие (прямоугольник)"""
        for i in range(max(1, y1), min(y2, self.height - 1)):
            for j in range(max(1, x1), min(x2, self.width - 1)):
                if self.grid[i, j] == 0:
                    self.grid[i, j] = 2

    def calculate_probabilities(self, i, j):
        """
        Вычисление вероятностей движения по формулам (5)
        
        Возвращает: (Pf, Pr, Pl) - вероятности движения вперед, вправо, влево
        """
        if self.grid[i, j] != 1:  # Не человек
            return (0, 0, 0)

        Pf, Pr, Pl = 0, 0, 0

        # В зависимости от приоритетного направления определяем направления
        if self.priority_direction == "right":  # Выход справа
            # Вперед = вправо
            for k in range(1, self.depth + 1):
                if j + k < self.width:
                    if self.grid[i, j + k] in [1, 2]:
                        Pf += 1
                    # Если встретили стену, считаем все последующие занятыми
                    elif self.grid[i, j + k] == 2:
                        Pf += (self.depth - k + 1)
                        break
            Pf = 1 - Pf / self.depth

            # Вправо = вниз (относительно направления движения)
            for k in range(1, self.depth + 1):
                if i + k < self.height:
                    if self.grid[i + k, j] in [1, 2]:
                        Pr += 1
                    elif self.grid[i + k, j] == 2:
                        Pr += (self.depth - k + 1)
                        break
            Pr = 1 - Pr / self.depth

            # Влево = вверх
            for k in range(1, self.depth + 1):
                if i - k >= 0:
                    if self.grid[i - k, j] in [1, 2]:
                        Pl += 1
                    elif self.grid[i - k, j] == 2:
                        Pl += (self.depth - k + 1)
                        break
            Pl = 1 - Pl / self.depth

        return max(0, Pf), max(0, Pr), max(0, Pl)

    def step(self):
        """Выполняем один шаг эволюции автомата"""
        # Создаем копию сетки для следующего шага
        new_grid = self.grid.copy()

        # Список людей с их позициями и вероятностями
        people = []
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 1:
                    Pf, Pr, Pl = self.calculate_probabilities(i, j)
                    people.append((i, j, Pf, Pr, Pl))

        # Перемещаем людей в случайном порядке
        random.shuffle(people)

        for i, j, Pf, Pr, Pl in people:
            if new_grid[i, j] != 1:  # Если человека уже переместили
                continue

            # Проверяем, не достиг ли человек выхода
            if self.priority_direction == "right" and j == self.width - 1:
                new_grid[i, j] = 0
                self.people_count -= 1
                self.evacuated_count += 1
                continue

            # Определяем возможные направления движения
            directions = []

            # Вперед (к выходу)
            if self.priority_direction == "right" and j + 1 < self.width:
                if new_grid[i, j + 1] == 0:
                    directions.append(("forward", Pf, i, j + 1))

            # Вправо (вниз)
            if i + 1 < self.height and new_grid[i + 1, j] == 0:
                directions.append(("right", Pr, i + 1, j))

            # Влево (вверх)
            if i - 1 >= 0 and new_grid[i - 1, j] == 0:
                directions.append(("left", Pl, i - 1, j))

            # Назад (влево от выхода)
            if self.priority_direction == "right" and j - 1 >= 0:
                if new_grid[i, j - 1] == 0:
                    # Вероятность движения назад ниже
                    directions.append(("backward", 0.1, i, j - 1))

            if directions:
                # Выбираем направление с максимальной вероятностью
                directions.sort(key=lambda x: x[1], reverse=True)
                _, _, new_i, new_j = directions[0]

                # Перемещаем человека
                new_grid[i, j] = 0
                new_grid[new_i, new_j] = 1

        self.grid = new_grid
        self.step_count += 1

        # Сохраняем статистику
        self.statistics.append({
            'step': self.step_count,
            'people_remaining': self.people_count,
            'evacuated': self.evacuated_count
        })

        return self.people_count > 0

    def simulate(self, max_steps=200):
        """Запускаем симуляцию"""
        steps = 0
        while steps < max_steps and self.people_count > 0:
            self.step()
            steps += 1

        return self.statistics

    def visualize(self, interval=100):
        """Визуализация работы автомата"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Цветовая карта: пусто - белый, человек - синий, стена - серый
        cmap = colors.ListedColormap(['white', 'blue', 'gray'])
        bounds = [0, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        im = ax1.imshow(self.grid, cmap=cmap, norm=norm, interpolation='nearest')
        ax1.set_title(f'Текущее состояние (шаг: {self.step_count})')
        ax1.grid(True, which='both', color='lightgray', linewidth=0.5)
        ax1.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, self.height, 1), minor=True)

        # График статистики
        ax2.set_title('Статистика эвакуации')
        ax2.set_xlabel('Шаг')
        ax2.set_ylabel('Количество людей')

        # Данные для графика
        steps = [s['step'] for s in self.statistics]
        remaining = [s['people_remaining'] for s in self.statistics]
        evacuated = [s['evacuated'] for s in self.statistics]

        line1, = ax2.plot(steps, remaining, 'r-', label='Осталось в зоне')
        line2, = ax2.plot(steps, evacuated, 'g-', label='Эвакуировано')
        ax2.legend()
        ax2.grid(True)

        def update(frame):
            if self.people_count > 0:
                self.step()
                im.set_array(self.grid)
                ax1.set_title(f'Текущее состояние (шаг: {self.step_count})')

                # Обновляем график
                steps = [s['step'] for s in self.statistics]
                remaining = [s['people_remaining'] for s in self.statistics]
                evacuated = [s['evacuated'] for s in self.statistics]

                line1.set_data(steps, remaining)
                line2.set_data(steps, evacuated)
                ax2.relim()
                ax2.autoscale_view()

            return im, line1, line2

        ani = animation.FuncAnimation(fig, update, frames=200,
                                      interval=interval, blit=True, repeat=False)

        plt.tight_layout()
        plt.show()

        return ani


# Демонстрационная программа
if __name__ == "__main__":
    # Создаем автомат с параметрами из задания 9.4

    arrays = [
        np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]),
        np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]),
        np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]),
        np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]),
        np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]),
        np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]),
        np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]),
        np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]),
        np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ])
    ]

    for arr in arrays:

        automaton = CrowdCellularAutomaton(arr, depth=3)

        # Добавляем людей с плотностью ~30%
        automaton.add_people_random(density=0.3)

        print(f"Начальное количество людей: {automaton.people_count}")
        print(f"Размер поля: {automaton.width}x{automaton.height}")
        print(f"Глубина анализа: {automaton.depth}")
        print(f"Ширина выхода: 3 клетки")

        # Запускаем симуляцию
        stats = automaton.simulate(max_steps=100)

        # Визуализируем результаты
        print(f"\nРезультаты симуляции:")
        print(f"Всего шагов: {automaton.step_count}")
        print(f"Эвакуировано людей: {automaton.evacuated_count}")

        # Строим график зависимости числа вышедших людей от времени
        plt.figure(figsize=(10, 6))
        steps = [s['step'] for s in stats]
        evacuated = [s['evacuated'] for s in stats]

        plt.plot(steps, evacuated, 'b-', linewidth=2)
        plt.xlabel('Время (шаги)', fontsize=12)
        plt.ylabel('Число вышедших людей', fontsize=12)
        plt.title('Зависимость числа вышедших людей от времени', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
