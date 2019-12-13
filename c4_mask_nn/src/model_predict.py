import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
import os
import sys
from model_core import creat_my_model
from tf_dataset import filter_image, load_and_prepro_image
from casia_data_process import get_casiadataset

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


def show_result(pre_result, mask, source):
    plt.figure()

    if pre_result is not None:
        plt.subplot(131)
        pre_result = pre2img(pre_result, 1)
        plt.imshow(pre_result)
    if mask is not None:
        plt.subplot(132)
        mask = Image.open(mask).convert('L').resize([256, 256])
        plt.imshow(mask)
    if source is not None:
        plt.subplot(133)
        source = Image.open(source).convert('RGB').resize([256, 256])
        plt.imshow(source)
    plt.show()


def eval_protcal(pre_result, mask):
    #现将预测结果01化
    pre_result = np.round(pre_result).reshape([256, 256])
    mask = Image.open(mask).convert('L').resize([256, 256])
    mask = np.array(mask).reshape([256, 256])/255.0
    mask = np.round(mask)

    #两种评价方式，一种计算整体，一种计算单张
    TP = np.sum(np.logical_and(pre_result, mask).astype(np.uint8))
    FP = np.sum(pre_result) - TP

    TN = np.sum(np.logical_and(np.logical_not(pre_result), np.logical_not(mask)).astype(np.uint8))
    FN = 256*256 - TP - FP - TN

    #计算precision, recall, F1
    ac = (TP+TN)/(TP+FP+TN+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*TP/(2*TP+FP+FN)

    flag = TP/(np.sum(pre_result) + np.sum(mask) - TP)

    return TP, FP, TN, FN, ac, precision, recall, F1, flag



def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # start_eval(0, 200, step=25, show=True)

    # #载入数据
    image_path = '../data/CoMoFoD_small/'
    x_list, y_list = filter_image(image_path)
    #测试casia数据
    # target_path = '../data/casia-dataset/target'
    # mask_path = '../data/casia-dataset/mask'
    # x_list, y_list = get_casiadataset(target_path, mask_path)
    #
    #载入模型
    model = creat_my_model()
    weight_path = '../log/20191212-183929_v2/my_model.h5'
    model.load_weights(weight_path)
    correct = 0
    count = 0
    start = 0
    end = 5000
    step = 25
    TP, FP, TN, FN, accuracy, precision, recall, F1 = 0, 0, 0, 0, 0, 0, 0, 0

    TP_c, FP_c, TN_c, FN_c, accuracy_c, precision_c, recall_c, F1_c = 0, 0, 0, 0, 0, 0, 0, 0
    # 单张图片预测
    for source, mask in zip(x_list[start:end:step], y_list[start:end:step]):
        img = Image.open(source).convert('RGB').resize([256, 256])
        #执行预测
        pre_result = model.predict(np.array(img).reshape([1, 256, 256, 3])/255.0)
        if True:
            show_result(pre_result, mask, source)

        #开始进行评价
        tp, fp, tn, fn, ac, pre, rc, f1, flag = eval_protcal(pre_result, mask)

        TP += tp
        FP += fp
        TN += tn
        FN += fn
        accuracy += ac
        precision += pre
        recall += rc
        F1 += f1
        count += 1
        if flag >= 0.5 :
            print(count)
            TP_c += tp
            FP_c += fp
            TN_c += tn
            FN_c += fn
            accuracy_c += ac
            precision_c += pre
            recall_c += rc
            F1_c += f1
            correct += 1

    #输出评价结果
    print('protocal A: ac:{} precision:{} recall:{} F1:{}'.
          format((TP + TN) / (TP + TN + FP + FN), TP / (TP + FP), TP / (TP + FN), 2 * TP / (2 * TP + FP + FN)))

    print('protocal B: ac:{} precision:{} recall:{} F1:{}'.
          format(accuracy / count, precision / count, recall / count, F1 / count))

    print('\ncorrect:{}'.format(correct))
    print('protocal correct A: ac:{} precision:{} recall:{} F1:{}'.
          format((TP_c + TN_c) / (TP_c + TN_c + FP_c + FN_c), TP_c / (TP_c + FP_c),
                 TP_c / (TP_c + FN_c), 2 * TP_c / (2 * TP_c + FP_c + FN_c)))

    print('protocal correct B: ac:{} precision:{} recall:{} F1:{}'.
          format(accuracy_c / correct, precision_c / correct, recall_c / correct, F1_c / correct))

if __name__ == "__main__":
    main()

