"""
function: 制作tfdata数据集解决oom问题
time:2019年12月06日10:40:02
author： hewa

"""

import tensorflow as tf
import numpy as np
import os
import time
import sys
import re
import matplotlib.pyplot as plt


#过滤图片选择用于训练的图片及mask
def filter_image(image_path):
    m_rep = re.compile('\d{3}_B')
    t_rep = re.compile('\d{3}_F')
    image_list = sorted(os.listdir(image_path))
    image_list = [v for v in image_list if m_rep.match(v) or t_rep.match(v)]
    mask_image = []
    train_image = []

    for image in image_list:
        if m_rep.match(image):
            mask_image.append(os.path.join(image_path, image))
        else:
            train_image.append(os.path.join(image_path, image))
    mask_image = mask_image*25
    mask_image = sorted(mask_image)

    # print(len(mask_image))
    # print(len(train_image))
    # for i, j in zip(mask_image, train_image):
    #     print('{}  {}'.format(i, j))
    return train_image, mask_image


def load_and_prepro_image(image, c, new_size):
    img = tf.read_file(image)
    if tf.image.is_jpeg:
        img = tf.image.decode_jpeg(img, channels=c)
    else:
        img = tf.image.decode_png(img, channels=c)

    #TODO 添加图片处理，比如裁剪 翻转 旋转，噪声 以此增加样本
    #不可以调换resize 和 convert的位置，否则，程序有bug
    # tf.image.resize_image_with_crop_or_pad()
    img = tf.image.resize_images(img, [new_size, new_size])
    # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img /= 255.0
    #判断是否进行标签转换
    if c == 1:
        img = tf.round(img)

    return img

#创建tf dataset
def creat_tfdata(images_path, c, new_size):
    tf_data = tf.data.Dataset.from_tensor_slices(images_path)
    #映射动态生成数据集
    tf_data = tf_data.map(lambda x: load_and_prepro_image(x, c, new_size))

    return tf_data

if __name__ == "__main__":
    #开启eager模式
    tf.enable_eager_execution()
    image_path = '../data/CoMoFoD_small'
    train_x, train_y = filter_image(image_path)
    tf_data_x = creat_tfdata(train_x, 3, 512)
    tf_data_y = creat_tfdata(train_y, 1, 512)
    plt.figure(1)
    '''
    tf_data_x = tf_data_x.make_one_shot_iterator()
    tf_data_y = tf_data_y.make_one_shot_iterator()
    x_element = tf_data_x.get_next()
    y_element = tf_data_y.get_next()
    # with tf.Session() as sess:
    #     for i in range(5000):
    #         print(x_element)
    #         # temp = sess.run(y_element)
    #         # if y_element.shape[-1] == 4:
    #         #     print(i)
    '''
    #
    # img = tf.read_file(train_x[0])
    # img = tf.image.decode_png(img, 1)
    # img = tf.image.resize_images(img, [256, 256])
    # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # print(img)
    # img /= 255.0
    # print(tf.round(img))
    # img = tf.reshape(img, [-1, 256, 256, 3])
    for ei in tf_data_y.take(4):
        print(np.sum(ei))
        plt.imshow(np.array(ei).reshape([512, 512]))
        plt.show()

    for ei in train_x[:4]:
        img = tf.read_file(ei)
        img = tf.image.decode_jpeg(img, channels=3)

        # 不可以调换resize 和 convert的位置，否则，程序有bug
        img = tf.image.resize_images(img, [512, 512])
        # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img /= 255.0
        # 判断是否进行标签转换
        if 0:
            img = tf.round(img)

        print(np.sum(img))


