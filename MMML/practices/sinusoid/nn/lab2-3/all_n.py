import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Функция для генерации фрагмента синусоиды на интервале [-pi, pi]
def generate_sin_data(num_samples):
    x = np.linspace(0, np.pi, num_samples)
    y = np.sin(x)
    return x, y


# Создаем модель с помощью Sequential API
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='sigmoid', input_shape=[1]))
model.add(tf.keras.layers.Dense(1))

# Компилируем модель с помощью функции потерь среднеквадратической ошибки и оптимизатора Adam
model.compile(optimizer='adam', loss='mean_squared_error')

# Генерируем данные для обучения
num_samples_train = 1000
x_train, y_train = generate_sin_data(num_samples_train)

# Обучаем модель на обучающих данных
num_epochs = 1000
model.fit(x_train, y_train, epochs=num_epochs)

# Создаем модель, которая выводит активации для всех слоев
layer_outputs = [layer.output for layer in model.layers]
activations_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

# Генерируем данные для проверки
num_samples_test = 100
x_test, y_test = generate_sin_data(num_samples_test)

# Предсказываем значения для проверки
y_pred = model.predict(x_test)

# Визуализируем датасет и аппроксимацию функции
# plt.plot(x_train, y_train, 'ro', label='Обучающие данные')
plt.plot(x_test, y_test, 'b-', label='Оригинал функции')
plt.plot(x_test, y_pred, 'g-', label='Результат аппроксимации')
# plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация синусоиды. Интервал [-pi;pi]')

# Вычисляем активации всех нейронов
activations = activations_model.predict(x_test)

# Визуализируем функции активации всех нейронов
num_layers = len(activations)
for layer_i in range(num_layers):
    num_neurons = activations[layer_i].shape[1]
    for neuron_i in range(num_neurons):
        plt.plot(x_test, activations[layer_i][:, neuron_i], '--', label=f'Layer {layer_i + 1}, Neuron {neuron_i + 1}')
# plt.legend(loc='lower right')
plt.show()
