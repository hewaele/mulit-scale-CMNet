import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(0)
sess = tf.Session()
x = [random.randint(0, 5) for i in range(2*16*16*5)]
x = np.array(x).reshape([2, 16, 16, 5])
x = tf.constant(x, dtype=tf.float32)
bsize, nb_rows, nb_cols, nb_feats = keras.backend.int_shape(x)
nb_maps = nb_rows * nb_cols
#欧式距离
x_3d = keras.backend.reshape(x, tf.stack([-1, nb_maps, nb_feats]))
#用于存储欧氏距离
# s = []

# for i in range(nb_maps):
#     # t = x_3d[:, i:i+1, :] - x_3d
#     t = tf.slice(x_3d, [0, i, 0], [-1, 1, nb_feats])
#     print(t)
#     t = t-x_3d
#     # print(t)
#     t = tf.multiply(t, t)
#     t = tf.keras.backend.sum(t, axis=2)
#     t = tf.sqrt(t)
#     s.append(t)

c_x_3d = tf.tile(x_3d, [1, nb_maps, 1])
t2 = tf.reshape(x_3d, shape=[bsize, nb_rows, nb_cols, 1, nb_feats])
t2 = tf.reshape(tf.tile(t2, [1, 1, 1, nb_maps, 1]), shape=[2, nb_maps*nb_maps, nb_feats])
t3 = tf.subtract(c_x_3d, t2)
print(c_x_3d)
print(t2)
t = tf.multiply(t3, t3)
t = tf.keras.backend.sum(t, axis=2)
r = tf.reshape(t, shape=[bsize, nb_maps, nb_maps])
r = tf.sqrt(r)
r = 10 - r
x_sort, _ = tf.nn.top_k(r, nb_maps, sorted=True)
print(t)
# print(sess.run(c_x_3d))
# print(sess.run(t2))
# print(sess.run(t3))
# print(sess.run(t))
print(sess.run(x_sort))
#将结果连接
# x_corr_3d = tf.concat(s, axis=1)
# x_corr = tf.reshape(x_corr_3d, [-1, nb_rows, nb_cols, nb_maps])
# x_sort, _ = tf.nn.top_k(x_corr, nb_maps, sorted=True)
# ranks = tf.range(256 - 128 - 1, 256-1)
# sess = tf.Session()
# print(sess.run(x))
# print(sess.run(x_sort))