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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#屏蔽警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# tf.enable_eager_execution()

image_size = 256
batchs = 2
epochs = 50

#load data
log = '../log/' + time.strftime('%Y%m%d-%H%M%S')+'_v4/'
backbone = 'vgg'
if backbone == 'vgg':
    pre_weight_path = '../pre_model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
else:
    pre_weight_path = '../pre_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

retrain = True
ckpt_path = '../log/20191219-134527_v4/weight_0165.ckpt'

def my_generator(x_list, y_list, batchs, new_size=256):
    x = np.zeros([batchs, new_size, new_size, 3])
    y = np.zeros([batchs, new_size, new_size, 1])
    b = 0
    for image_path, mask_path, in zip(x_list, y_list):
        img = Image.open(image_path).convert('RGB').resize([new_size, new_size])
        mask = Image.open(image_path).convert('L').resize([new_size, new_size])

        img = np.array(img).reshape([new_size, new_size, 3])
        mask = np.array(mask).reshape([new_size, new_size, 1])
        img = np.multiply(img, 1/255.0)
        mask = np.multiply(mask, 1/255.0)
        # 判断是否进行标签转换
        mask = np.round(mask)
        x[b] = img
        y[b] = mask
        b += 1
        if b >= batchs:
            b = 0
            yield x, y

#comofod数据集
image_path = '../data/CoMoFoD_small'
x_list, y_list = filter_image(image_path)
#创建测试训练集
test_xy = my_generator(x_list, y_list, batchs, 256)

#训练casia数据集，测试comofod数据集
# target_path = '../data/casia-dataset/target'
# mask_path = '../data/casia-dataset/mask'
# x_list, y_list = get_casiadataset(target_path, mask_path)

#训练casia增强数据集，测试comofod数据集
target_path = '../data/augmentation_data/image'
mask_path = '../data/augmentation_data/mask'
x_list, y_list = get_casiadataset(target_path, mask_path)
x_list = x_list[:]
y_list = y_list[:]
train_xy = my_generator(x_list, y_list, batchs, 256)
print('data load done')

#定义tensorboard回调可视化
TBCallback = TensorBoard(log_dir=log)
cpCallback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(log, 'weight_{epoch:04d}.ckpt'), period=1)
my_model = creat_my_model([image_size, image_size, 3], backbone=backbone, pre_weight_path=None, mode='train')
my_model.summary()


my_model.load_weights(ckpt_path)

my_model.compile(optimizer=keras.optimizers.Adam(0.001),
                 loss=keras.losses.binary_crossentropy,
                 metrics=['accuracy'])

print('start train......')
my_model.fit_generator(train_xy,
             steps_per_epoch=len(x_list)//batchs,
             epochs=epochs,
             validation_data=test_xy,
             validation_steps=100,
             # workers=4,
             # use_multiprocessing=True,
             callbacks=[TBCallback, cpCallback],
             )
#
my_model.save(os.path.join(log, 'my_model.h5'))