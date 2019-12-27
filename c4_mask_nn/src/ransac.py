#测试随机采样一致性算法
import tensorflow as tf
import os
import numpy as np
from model_core import creat_backbone, std_norm_along_chs
from PIL import Image
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.enable_eager_execution()
# pre_weight_path = '../pre_model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# #创建模型获取提取特征
# input, output = creat_backbone([256, 256, 3], pre_weight_path)
# model = tf.keras.Model(input, [output[2], output[1]])
# model.summary()
#
# #载入一张图片
# img = Image.open('./074_F.png').convert('RGB').resize([256, 256])
# img = np.array(img).reshape([-1, 256, 256, 3])/255.0
# result1, result2 = model.predict(img)
# plt.figure(1)
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     print(result1)
#     plt.imshow(result1[0, :, :, i])
# plt.show()
#
# #计算相似度
# from model_core import SelfCorrelationPercPooling
#
# result1 = tf.keras.layers.Activation(std_norm_along_chs, name='_sn4')(tf.constant(result1))
# corr_result = SelfCorrelationPercPooling(name='test', nb_pools=64)(result1)
# # print(corr_result.shape)
# #
# plt.figure(1)
# index = 1
# for i in range(1, 64, 7):
#     plt.subplot(2, 5, index)
#     index += 1
#     print(np.round(corr_result[0, :, :, i]))
#     plt.imshow(np.round(corr_result[0, :, :, i])*255)
# plt.show()
#

import random
a = [random.randint(0, 9) for i in range(9)]
x = tf.constant(a)
x = tf.reshape(x, [-1, 3, 3, 1])
print(x)
y = 1-x
print(y)