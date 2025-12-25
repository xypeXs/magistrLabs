import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def generate_sin_data():
    x = np.linspace(-np.pi, np.pi, 1000)
    y = np.sin(x)
    return x, y


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='tanh', input_shape=[1]))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

x_train, y_train = generate_sin_data()

model.fit(x_train, y_train, epochs=200)

layer_outputs = [layer.output for layer in model.layers]
activations_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

x_test = np.linspace(-np.pi, np.pi, 1000)
y_test = np.sin(x_test)

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

# Визуализируем функции активации всех нейронов
activations = activations_model.predict(x_test)
num_layers = len(activations)
for layer_i in range(1, num_layers):
    num_neurons = activations[layer_i].shape[1]
    for neuron_i in range(num_neurons):
        plt.plot(x_test, activations[layer_i][:, neuron_i], '--', label=f'Layer {layer_i}, Neuron {neuron_i}')
# plt.legend(loc='lower right')
plt.show()
