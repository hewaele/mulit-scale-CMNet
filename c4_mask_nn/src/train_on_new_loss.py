import tensorflow as tf
from tensorflow import keras
import time
import os
from model_core import creat_my_model
from tensorflow.keras.callbacks import TensorBoard
from tf_dataset import creat_tfdata, filter_image, load_and_prepro_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from casia_data_process import get_casiadataset
import random
from sklearn.utils import shuffle
from utiles import my_generator
import tensorflow.keras.backend as K

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #屏蔽警告
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # tf.enable_eager_execution()

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    image_size = 256
    batchs = 2
    epochs = 200

    #load data
    log = '../log/' + time.strftime('%Y%m%d-%H%M%S')+'_v4_vgg_new_loss/'
    backbone = 'vgg'
    if backbone == 'vgg':
        pre_weight_path = '../pre_model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    else:
        pre_weight_path = '../pre_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    retrain = True
    ckpt_path = '../log/20200107-094903_v4_vgg/weight_0001.ckpt'

    #comofod数据集
    image_path = '../data/CoMoFoD_small'
    x_list, y_list = filter_image(image_path)
    #创建测试训练集
    test_xy = my_generator(x_list[::25], y_list[::25], batchs, 256, rescale=False)

    #训练casia数据集，测试comofod数据集
    target_path = '../data/casia-dataset/target'
    mask_path = '../data/casia-dataset/mask'
    x_list, y_list = get_casiadataset(target_path, mask_path)

    #训练casia增强数据集，测试comofod数据集
    # target_path = '../data/augmentation_data/image'
    # mask_path = '../data/augmentation_data/mask'
    # x_list, y_list = get_casiadataset(target_path, mask_path)
    x_list = x_list[:]
    y_list = y_list[:]
    nums = len(x_list)

    x_list, y_list = shuffle(x_list, y_list)
    train_xy = my_generator(x_list, y_list, batchs, 256, rescale=False)
    print('data load done')

    #定义tensorboard回调可视化
    TBCallback = TensorBoard(log_dir=log)
    cpCallback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(log, 'weight_{epoch:04d}.ckpt'), period=1)
    my_model = creat_my_model([image_size, image_size, 3], backbone=backbone, pre_weight_path=None, mode='train')
    my_model.summary()


    my_model.load_weights(ckpt_path)

    my_model.compile(optimizer=keras.optimizers.Adam(0.0001),
                     loss=binary_focal_loss(2, 0.25),
                     metrics=['accuracy'])

    print('start train......')
    my_model.fit_generator(train_xy,
                 steps_per_epoch=nums//batchs,
                 epochs=epochs,
                 validation_data=test_xy,
                 validation_steps=100,
                 # workers=4,
                 # use_multiprocessing=True,
                 callbacks=[TBCallback, cpCallback],
                 )
    #
    my_model.save(os.path.join(log, 'my_model.h5'))