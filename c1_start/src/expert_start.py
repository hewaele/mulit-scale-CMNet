import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import sys

def load_data(path):
    data = np.load(path)
    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

    return x_train, y_train, x_test, y_test

def data_preprocess(data):
    x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = np.reshape(x_train, [-1, 28, 28, 1])
    x_test = np.reshape(x_test, [-1, 28, 28, 1])
    #ont-hot
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    return x_train, y_train, x_test, y_test

#创建自己的模型
class MyModel(keras.Model):
    def _init__(self):
        super(MyModel, self).__init()

    def call(self, x):
        x = keras.layers.Conv2D(32, [3, 3], strides=[1, 1], padding='same',activation='relu')(x)
        x = keras.layers.Flatten(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        out = keras.layers.Dense(10, activation='softmax')(x)

        return out



def main():
    path = '../data/mnist.npz'
    data = load_data(path)
    x_train, y_train, x_test, y_test = data_preprocess(data)

    model = MyModel()
    # model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    # model.fit(x_train, y_train, steps_per_epoch=100, epochs=2)
    loss = keras.losses.sparse_categorical_crossentropy()
    opt = keras.optimizers.Adam()

    train_loss = tf.metrics.mean(name='train_loss')
    train_accuracy = tf.metrics.mean_per_class_accuracy(name='train_accuracy')

    test_loss = tf.metrics.mean(name='test_loss')
    test_accuracy = tf.metrics.mean_per_class_accuracy(name='test_accuracy')

    #TODO 后续学习更新此代码



if __name__ == "__main__":
    main()

