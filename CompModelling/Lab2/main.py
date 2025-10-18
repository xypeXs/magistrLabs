import numpy as np

class MooreAutomaton:
    def __init__(self):
        # Таблица переходов (4 состояния × 4 входа)
        # states: 0, 1, 2, 3
        # inputs: 0, 1, 2, 3
        self.transition_table = np.array([
            [2, 1, 0, 3],  # Из состояния 0
            [3, 2, 1, 0],  # Из состояния 1
            [0, 3, 2, 1],  # Из состояния 2
            [1, 0, 3, 2]   # Из состояния 3
        ])

        # Таблица выходов (для каждого состояния)
        # outputs: 0, 1
        self.output_table = np.array([0, 1, 0, 1])

        # Начальное состояние
        self.current_state = 0

    def step(self, input_signal):
        """Выполняет один такт работы автомата"""
        if input_signal < 0 or input_signal > 3:
            raise ValueError("Входной сигнал должен быть в диапазоне 0-3")

        # Вычисляем новое состояние
        self.current_state = self.transition_table[self.current_state, input_signal]

        # Возвращаем выходной сигнал
        return self.output_table[self.current_state]

    def get_current_state(self):
        return self.current_state

    def get_current_output(self):
        return self.output_table[self.current_state]

def main():
    automaton = MooreAutomaton()

    print("Детерминированный конечный автомат Мура")
    print("Количество входов: 4 (0-3)")
    print("Количество состояний: 4 (0-3)")
    print("Количество выходов: 2 (0-1)")
    print("Начальное состояние: 0")
    print("Выход в начальном состоянии:", automaton.get_current_output())
    print("\nДля работы автомата вводите номер входного сигнала (0-3)")
    print("Для выхода введите 'q'")

    step_count = 0
    state_history = [automaton.get_current_state()]
    output_history = [automaton.get_current_output()]

    while True:
        try:
            user_input = input(f"\nТакт {step_count + 1}. Введите входной сигнал (0-3): ").strip()

            if user_input.lower() == 'q':
                break

            input_signal = int(user_input)

            if input_signal < 0 or input_signal > 3:
                print("Ошибка: входной сигнал должен быть в диапазоне 0-3")
                continue

            # Выполняем такт
            output = automaton.step(input_signal)
            step_count += 1

            # Сохраняем историю
            state_history.append(automaton.get_current_state())
            output_history.append(output)

            # Выводим результаты такта
            print(f"Такт {step_count}:")
            print(f"  Входной сигнал: x{input_signal}")
            print(f"  Новое состояние: z{automaton.get_current_state()}")
            print(f"  Выходной сигнал: y{output}")

        except ValueError:
            print("Ошибка: введите число от 0 до 3 или 'q' для выхода")
        except KeyboardInterrupt:
            break

    # Выводим полную историю работы
    print("\n" + "="*50)
    print("Полная история работы автомата:")
    print("Такт | Состояние | Выход")
    print("-" * 25)
    for i in range(len(state_history)):
        print(f"{i:4} | z{state_history[i]:1}       | y{output_history[i]}")

if __name__ == "__main__":
    main()