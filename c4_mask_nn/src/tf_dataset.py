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
import random
from PIL import Image

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

def creat_augmentation_data(images_path, mask_path, new_size):
    for image, target in zip(images_path, mask_path):
        img = tf.read_file(image)
        mask = tf.read_file(target)

        if tf.image.is_jpeg(img) is not None:
            img = tf.image.decode_jpeg(img, channels=3)
        else:
            img = tf.image.decode_png(img, channels=3)

        if tf.image.is_jpeg(mask) is not None:
            mask = tf.image.decode_jpeg(mask, channels=1)
        else:
            mask = tf.image.decode_png(mask, channels=1)

        #TODO 添加图片处理，高斯噪声，亮度，上下旋转， 左右旋转， 旋转角度， 裁剪， 以此增加样本
        #上下旋转
        if random.random() > 0.5:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        #左右旋转
        if random.random() > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)

        #90度旋转
        random_k = random.randint(0, 3)
        img = tf.image.rot90(img, k=random_k)
        mask = tf.image.rot90(mask, k=random_k)
        # 随机设置图片的亮度
        if random.random() > 0.5:
            img = tf.image.random_brightness(img, max_delta=0.05)
            # 随机设置图片的对比度
            img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
            # 随机设置图片的色度
            img = tf.image.random_hue(img, max_delta=0.2)
            # 随机设置图片的饱和度
            img = tf.image.random_saturation(img, lower=0.5, upper=1.5)

        img = tf.image.resize_images(img, [new_size, new_size])
        mask = tf.image.resize_images(mask, [new_size, new_size])

        #
        noise = tf.random_normal(shape=tf.shape(img), mean=0.0, stddev=0.05,
                                 dtype=tf.float32)
        img = tf.add(img, noise)

        #判断是否进行标签转换
        mask = tf.round(mask)

        yield [img, mask]

def load_and_prepro_image(image_path, c, new_size):
    img = tf.read_file(image_path)
    if tf.image.is_jpeg(img) is not None:
        img = tf.image.decode_jpeg(img, channels=c)
    else:
        img = tf.image.decode_png(img, channels=c)

    img = tf.image.resize_images(img, [new_size, new_size])
    img /= 255.0

    #判断是否进行标签转换
    if c == 1:
        img = tf.round(img)

    return img

#创建tf dataset
def creat_tfdata(images_path, mask_path, new_size):
    #生成x数据并处理
    tfdata_x = tf.data.Dataset.from_tensor_slices(images_path)
    tfdata_y = tf.data.Dataset.from_tensor_slices(mask_path)
    #映射动态生成数据集
    tfdata_x = tfdata_x.map(lambda x: load_and_prepro_image(x, 3, new_size))
    tfdata_y = tfdata_y.map(lambda x: load_and_prepro_image(x, 1, new_size))
    return tfdata_x, tfdata_y

if __name__ == "__main__":
    #开启eager模式
    from casia_data_process import get_casiadataset
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 屏蔽警告
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.enable_eager_execution()

    target_path = '../data/casia-dataset/target'
    mask_path = '../data/casia-dataset/mask'
    x_list, y_list = get_casiadataset(target_path, mask_path)
    seed = time.ctime()
    random.seed(seed)
    count = 0
    for i in range(25):
        print(i)
        tfdata_xy = creat_augmentation_data(x_list[:], y_list[:], 256)
        # tfdata_xy = tf.data.Dataset.zip((tfdata_x, tfdata_y))
        # tfdata_xy = tfdata_xy.shuffle(buffer_size=len(x_list[:]))
        # tfdata_xy = tfdata_xy.repeat(10 + 1)
        # tfdata_xy = tfdata_xy.batch(2)
        # print('...............')
        # random.seed(seed)
        # tf_data_y = creat_tfdata(y_list[:20], 1, 512)
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

        test_x = []
        test_y = []
        for xy in tfdata_xy:
            # print(np.sum(ei))
            # test_x.append(xy[0])
            # test_y.append(xy[1])
            # plt.subplot(121)
            # plt.imshow(np.array(xy[0]).reshape([256, 256, 3]))
            # plt.subplot(122)
            # plt.imshow(np.array(xy[1]).reshape([256, 256]))
            # plt.show()
            # print(np.uint8(xy[1]))
            img = Image.fromarray(np.uint8(xy[0]))
            img.save('../data/augmentation_data/image/'+str(count)+'.png')
            mask = Image.fromarray(np.uint8(np.reshape(xy[1], [256, 256])), 'L')
            mask.save('../data/augmentation_data/mask/' + str(count) + '.png')
            count += 1
            # if count == 10:
            #     break


