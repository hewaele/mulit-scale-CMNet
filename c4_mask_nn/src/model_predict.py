import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
import os
import sys
from model_core import creat_my_model
from tf_dataset import filter_image


def pre2img(result, channel=1):
    """
    将预测结果转换为图片显示
    :param result:numpy shape[-1, 256, 256, 1]
    :return:
    """
    result = np.reshape(result, [256, 256])
    result = result * 255
    result = result.astype(np.uint8)
    # print(type(result))
    img = Image.fromarray(result)
    return img


def img2pre(img, channel = 3):
    """
    将图片转换为数组用于预测
    :param img:
    :return:
    """
    img = Image.open(img)
    if channel == 3:
        img = img.convert('RGB').resize([256, 256])

    else:
        channel = 1
        img = img.convert('L').resize([256, 256])
    img = np.array(img)/255.0
    return img.reshape([1, 256, 256, channel])

def plt_show(pre, mask, source):
    plt.figure(1)
    if pre is not None:
        plt.subplot(131)
        plt.imshow(pre)
    if mask is not None:
        plt.subplot(132)
        plt.imshow(mask)
    if source is not None:
        plt.subplot(133)
        plt.imshow(source)
    plt.show()

def eval_B(pre, mask, image=None):
    """
    :param pre: Image type  [256, 256, 1]
    :param mask: Image type [256, 256, 4]
    :param image: Image type [256, 256, 3]
    :return:
    """
    #将预测结果二值化
    pre = np.array(pre)/255
    pre = pre.round()
    #mask 转化
    mask = np.array(mask.convert('L'))/255
    mask = mask.round()
    result = np.equal(pre, mask).astype(np.uint8)

    ac = result.sum()/(256**2)

    #计算TP 真正
    TP = np.sum(np.logical_and(pre, mask).astype(np.int8))
    FP = np.sum(pre)-TP


    TN = np.sum(np.logical_and(np.logical_not(pre), np.logical_not(mask)).astype(np.uint8))
    FN = np.sum(np.logical_not(pre))-TN

    # print(type(TP), FP, TN, FN)
    precision = TP/(TP+FP+0.0000001)
    recall = TP/(TP+FN+0.000001)
    F1 = (2*precision*recall)/(precision+recall+0.00001)
    return ac, precision, recall, F1, TP, FP, TN, FN

def start_eval(star, end, step=1, show=False):
    model_path = '../log/20191212-100324/my_model.h5'
    model = creat_my_model()
    model.load_weights(model_path)

    #载入测试数据
    img_path = '../data/CoMoFoD_small/'
    x_list, y_list = filter_image(img_path)

    ac = 0
    precision = 0
    recall = 0
    F1 = 0
    count = 0
    TP, FP, TN, FN = 0, 0, 0, 0
    for x, y in zip(x_list[star:end:step], y_list[star:end:step]):
        # print(x)
        img_data = img2pre(x)
        pre = model.predict(img_data/255)
        print(pre)
        show_pre = pre2img(pre)
        show_y = Image.open(y).resize([256, 256])
        show_x = Image.open(x).resize([256, 256])

        # Image.Image.show(show_pre)
        # Image.Image.show(show_y)
        # Image.Image.show(show_x)
        if show:
            plt_show(show_pre, show_y, show_x)
        a, b, c, d, tp, fp, tn, fn = eval_B(show_pre, show_y)
        print(a, b, c, d, tp, fp, tn, fn)
        TP += tp
        FP += fp
        TN += tn
        FN += fn

        ac += a
        precision += b
        recall += c
        F1 += d
        count += 1

    print('protocal A: ac:{} precision:{} recall:{} F1:{}'.
          format((TP+TN)/(TP+TN+FP+FN), TP/(TP+FP), TP/(TP+FN), 2*TP/(2*TP+FP+FN)))

    print('protocal B: ac:{} precision:{} recall:{} F1:{}'.
          format(ac/count, precision/count, recall/count, F1/count))

def pre_test_img(img_path):
    model_path = '../log/source_v1/my_model.h5'
    model = creat_my_model()
    model.load_weights(model_path)

    img_data = img2pre(img_path)
    pre = model.predict(img_data / 255)
    show_pre = pre2img(pre)
    show_x = Image.open(img_path).resize([256, 256])

    Image.Image.show(show_pre)
    Image.Image.show(show_x)

    plt_show(show_pre, None, show_x)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    start_eval(0, 200, step=25, show=True)
    # pre_test_img('../test_img/161_F.png')


    # #载入数据
    # image_path = '../data/CoMoFoD_small/'
    # from tf_dataset import creat_tfdata
    # x_list, y_list = filter_image(image_path)
    #
    # #载入模型
    # model = creat_my_model()
    # weight_path = '../log/20191211-121150/my_model.h5'
    # model.load_weights(weight_path)
    #
    # x_data = creat_tfdata(x_list[4000:], 3, 256)
    # x_data = x_data.make_one_shot_iterator()
    #
    # print(x_data)
    # result = model.predict(x_data.get_next(), steps=1)
    # print(result.shape)
    # img = pre2img(result[0], 1)
    # plt.imshow(img)
    # plt.show()
    #
    # # 单张图片预测
    # for img_name in x_list[:4]:
    #     img = tf.read_file(img_name)
    #     img = tf.image.decode_jpeg(img, 3)
    #     img = tf.image.resize_images(img, [256, 256])
    #
    #     # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    #     img /= 255.0
    #
    #     #执行预测
    #     pre_result = model.predict(tf.reshape(img, [-1, 256, 256, 3]), steps=1)
    #     print(pre_result.shape)
    #     img = pre2img(pre_result, 1)
    #     print(np.sum(pre_result.round()))
    #     plt.imshow(img)
    #     plt.show()

if __name__ == "__main__":
    main()

