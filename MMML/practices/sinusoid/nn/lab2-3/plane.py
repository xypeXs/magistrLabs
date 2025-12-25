import random

import tensorflow as tf
from tensorflow import keras


# W1X+W2y+b=z
# Создаем класс для получения весов в конце каждой эпохи
class GetWeights(tf.keras.callbacks.Callback):
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        # Перебираем каждый слой и получаем веса и смещения
        for layer_i in range(len(self.model.layers)):
            weights = self.model.layers[layer_i].get_weights()
            weights_shapes = [w.shape for w in weights]
            weights_vals = [w for w in weights]
            print(f'Layer {layer_i} has weights of shape {weights_shapes}')

            if epoch == 0:

                self.weight_dict['weights_' + str(layer_i + 1)] = [weights_vals]
            else:

                self.weight_dict['weights_' + str(layer_i + 1)].append(weights_vals)


gw = GetWeights()

model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
# model.add(keras.layers.Dense(50, activation='sigmoid'))  # Скрытый слой с 5 нейронами
model.add(keras.layers.Dense(1))  # Выходной слой с 1 нейроном

model.compile(optimizer='sgd', loss='mean_squared_error')

num_points = 21

arr_xy = []
arr_z = []

for i in range(num_points):
    x = random.uniform(0.0, 1.0)
    y = random.uniform(0.0, 1.0)
    arr_xy.append([x, y])
    arr_z.append(3 * x + 4 * y + 2)

model.fit(arr_xy, arr_z, epochs=1000, callbacks=[gw])

from keras.utils import plot_model

plot_model(model, to_file='model_p.png', show_shapes=True, show_layer_names=True)

for key in gw.weight_dict:
    print(f'Weighs for layer {key}:')
    for i, weights in enumerate(gw.weight_dict[key]):
        print(f'Epoch {i + 1}: {weights}')
    print()

# Предсказываем значение для нового примера
print(model.predict([[0.2, 0.8]]))
