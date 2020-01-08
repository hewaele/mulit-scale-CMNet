import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
import os
import sys
from model_core import creat_my_model, creat_my_model_newaug
from tf_dataset import filter_image, load_and_prepro_image
from casia_data_process import get_casiadataset
from source_buster_model import create_BusterNet_model
from sklearn.metrics import precision_recall_fscore_support
# tf.enable_eager_execution()
image_size = 256
def pre2img(result, channel=1):
    """
    将预测结果转换为图片显示
    :param result:numpy shape[-1, 256, 256, 1]
    :return:
    """
    result = np.round(result)
    result = np.reshape(result, [image_size, image_size])
    result = result * 255
    result = result.astype(np.uint8)
    # print(type(result))
    img = Image.fromarray(result)
    return img

def show_result(pre_result, mask, source):
    plt.figure()

    if pre_result is not None:
        plt.subplot(131)
        pre_result = pre2img(pre_result, 1)
        plt.imshow(pre_result)
    if mask is not None:
        plt.subplot(132)
        mask = pre2img(mask, 1)
        plt.imshow(mask)
    if source is not None:
        plt.subplot(133)
        plt.imshow(source)
    plt.show()

def show_tensor(*arg, position=0):
    l = len(arg)
    plt.figure()
    for index, ti in enumerate(arg):
        plt.subplot('1'+str(l)+str(index+1))
        plt.imshow(ti)
    plt.show()


def eval_protcal(pre_result, mask):
    #现将预测结果01化
    pre_result = np.round(pre_result)
    mask = np.round(mask)

    #两种评价方式，一种计算整体，一种计算单张
    TP = np.sum(np.logical_and(pre_result, mask).astype(np.uint8))
    FP = np.sum(pre_result) - TP

    TN = np.sum(np.logical_and(np.logical_not(pre_result), np.logical_not(mask)).astype(np.uint8))
    FN = image_size**2 - TP - FP - TN

    #计算precision, recall, F1
    ac = (TP+TN)/(image_size**2)
    precision = TP/(TP+FP+0.0000001)
    recall = TP/(TP+FN+0.0000001)
    F1 = 2*TP/(2*TP+FP+FN+0.000001)

    return TP, FP, TN, FN, ac, precision, recall, F1


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # start_eval(0, 200, step=25, show=True)

    # #载入数据
    image_path = '../data/CoMoFoD_small/'
    x_list, y_list = filter_image(image_path)
    #测试casia数据
    # target_path = '../data/casia-dataset/target'
    # mask_path = '../data/casia-dataset/mask'
    # x_list, y_list = get_casiadataset(target_path, mask_path)

    # 测试casia增强数据
    # target_path = '../data/augmentation_data/image'
    # mask_path = '../data/augmentation_data/mask'
    # x_list, y_list = get_casiadataset(target_path, mask_path)

    #载入模型
    weight_path = '../pre_model/pretrained_busterNet.hd5'
    model = create_BusterNet_model(weight_path)
    model.summary()
    correct = 0
    count = 0
    start = 0
    end = 5000
    step = 25
    threshold = 0.
    TP, FP, TN, FN, accuracy, precision, recall, F1 = 0, 0, 0, 0, 0, 0, 0, 0

    TP_c, FP_c, TN_c, FN_c, accuracy_c, precision_c, recall_c, F1_c = 0, 0, 0, 0, 0, 0, 0, 0
    # 单张图片预测
    statr_time = time.time()
    for source, mask in zip(['./2m2d.png'], y_list[start:start+1:step]):
        img = Image.open(source).convert('RGB').resize([image_size, image_size])
        #执行预测
        pre_result = model.predict(np.array(img).reshape([1, image_size, image_size, 3]))
        pre_result = pre_result[0, :, :, 2].ravel() <= 0.5
        mask = Image.open(mask).convert('L').resize([image_size, image_size])
        mask = np.array(mask).ravel() != 0
        show_result(pre_result, mask, img)
        # show_tensor(x2, x3, x4, position=2)
        # print(x2.shape)
        # t1 = np.sum(x3, axis=3)
        # for ti in t1[0]:
        #     print('{}'.format(ti))

        #开始进行评价
        tp, fp, tn, fn, ac, pre, rc, f1 = eval_protcal(pre_result, mask)

        TP += tp
        FP += fp
        TN += tn
        FN += fn
        accuracy += ac
        precision += pre
        recall += rc
        F1 += f1
        count += 1
        if f1 >= 0.5:
            print(count)
            # show_result(pre_result, mask, source)
            # show_tensor(x2, x3, x4, position=1)
            TP_c += tp
            FP_c += fp
            TN_c += tn
            FN_c += fn
            accuracy_c += ac
            precision_c += pre
            recall_c += rc
            F1_c += f1
            correct += 1
        else:
            print(source)
            # show_result(pre_result, mask, source)
            # show_tensor(x2, x3, x4, position=1)


    end_time = time.time()
    print("consume time:{} process time per img:{}".format(end_time - statr_time, (end_time-statr_time)/count))
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

