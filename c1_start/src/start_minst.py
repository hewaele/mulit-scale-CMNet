import tensorflow as tf
import numpy as np
from tensorflow import keras as keras
import time
import os
import matplotlib.pyplot as plt


def load_mnist(path):
    #download
    if path is None:
        return keras.datasets.mnist.load_data()
    #load direct
    data = np.load(path)

    return data['x_train'], data['y_train'], data['x_test'], data['y_test']

def main():
    x_train, y_train, x_test, y_test = load_mnist('../data/mnist.npz')
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)
    print(y_test)
    # plt.imshow(x_train[2])
    # plt.show()

    #将输入变成一维
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                                     keras.layers.Dense(128, activation='relu'),
                                     keras.layers.Dropout(0.2),
                                     keras.layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    #保存模型权重
    log_path = '../log/{}/'.format(time.strftime('%Y%m%d-%H%M%S'))
    cb_checkpoint = keras.callbacks.ModelCheckpoint(log_path, save_weights_only=True, verbose=2)
    model.summary()
    model.fit(x_train, y_train, steps_per_epoch=100, epochs=3,
              callbacks=[cb_checkpoint],
              validation_data=[x_test, y_test],
              validation_steps=1000)
    model.fit(x_train, y_train, steps_per_epoch=100, epochs=2,
              callbacks=[])
    #保存整个模型
    model.save(os.path.join(log_path, 'my_model.h5'))
    # result = model.evaluate(x_test, y_test, steps=1000)
    # print(result)

if __name__ == "__main__":
    main()
