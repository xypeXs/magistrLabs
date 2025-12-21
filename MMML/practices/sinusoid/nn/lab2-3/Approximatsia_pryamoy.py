# http://synset.com/ai/ru/nn/NeuralNet_01_Intro.html

import tensorflow as tf
import numpy as np
from tensorflow import keras

# Создаем модель с помощью Sequential API
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Компилируем модель с помощью функции потерь среднеквадратической ошибки и оптимизатором стохастического градиентного спуска
model.compile(optimizer='sgd', loss='mean_squared_error')

# Создаем обучающий набор данных
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Создаем класс для получения весов в конце каждой эпохи
class GetWeights(tf.keras.callbacks.Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch

        # loop over each layer and get weights and biases
        # Получаем веса и смещения для каждого слоя
        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]
            print('Layer %s has weights of shape %s and biases of shape %s' % (
                layer_i, np.shape(w), np.shape(b)))
            print('солой %s коф. %s and смещение %s' % (
                layer_i, w, b))

            # Сохраняем веса и смещения в словарь
            if epoch == 0:
                # Создаем массив для хранения весов и смещений
                self.weight_dict['w_' + str(layer_i + 1)] = w
                self.weight_dict['b_' + str(layer_i + 1)] = b
            else:
                # Добавляем новые веса к ранее созданному массиву весов
                self.weight_dict['w_' + str(layer_i + 1)] = np.dstack(
                    (self.weight_dict['w_' + str(layer_i + 1)], w))
                # Добавляем новые смещения к ранее созданному массиву смещений
                self.weight_dict['b_' + str(layer_i + 1)] = np.dstack(
                    (self.weight_dict['b_' + str(layer_i + 1)], b))


gw = GetWeights()

# Обучаем модель на обучающих данных
model.fit(xs, ys, epochs=5, callbacks=[gw])

from keras.utils.vis_utils import plot_model
# Сохраняем визуализацию модели в файл model_plot.png
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Выводим полученные веса для слоя
print(model.layers[0].get_weights())
for key in gw.weight_dict:
    # Выводим размерность массивов весов и смещений после каждой эпохи
    print(str(key) + ' shape: %s' % str(np.shape(gw.weight_dict[key])))

# Предсказываем значение при x = 0.1
print(model.predict([0.1]))
