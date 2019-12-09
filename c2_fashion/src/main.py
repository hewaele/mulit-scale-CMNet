import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow.keras as keras
import gzip

def load_data(path):
    # keras.datasets.fashion_mnist.load_data()
    file_list = ['train-images-idx3-ubyte.gz',
                 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz']

    #load flie
    with gzip.open(os.path.join(path, file_list[0]), 'rb') as fb:
        train_x = np.frombuffer(fb.read(), dtype=np.uint8, offset=16).reshape([-1, 28, 28])
    with gzip.open(os.path.join(path, file_list[1]), 'rb') as fb:
        train_y = np.frombuffer(fb.read(), dtype=np.uint8, offset=8)
    with gzip.open(os.path.join(path, file_list[2]), 'rb') as fb:
        test_x = np.frombuffer(fb.read(), dtype=np.uint8, offset=16).reshape([-1, 28, 28])
    with gzip.open(os.path.join(path, file_list[3]), 'rb') as fb:
        test_y = np.frombuffer(fb.read(), dtype=np.uint8, offset=8)

    return train_x, train_y, test_x, test_y

def data_preprocess(data):
    pass


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, x):

        return x

def main():
    path = '../data'
    train_x, train_y, test_x, test_y = load_data(path)

    #构建模型
    #TODO 学习完高级自定义回来设计模型


if __name__ == "__main__":
    main()