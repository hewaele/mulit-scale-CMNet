import tensorflow as tf
from tensorflow import keras
import time
import os
from model_core import creat_my_model
from tensorflow.keras.callbacks import TensorBoard
from tf_dataset import creat_tfdata, filter_image
import numpy as np
import matplotlib.pyplot as plt
from casia_data_process import get_casiadataset
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#屏蔽警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# tf.enable_eager_execution()

image_size = 256
batchs = 2
epochs = 20

#load data
log = '../log/' + time.strftime('%Y%m%d-%H%M%S')+'_v4/'
backbone = 'vgg'
if backbone == 'vgg':
    pre_weight_path = '../pre_model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
else:
    pre_weight_path = '../pre_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

retrain = True
ckpt_path = '../log/20191219-134527_v4/weight_0165.ckpt'

#comofod数据集
image_path = '../data/CoMoFoD_small'
x_list, y_list = filter_image(image_path)
#创建测试训练集
test_x, test_y = creat_tfdata(x_list, y_list, image_size)
test_xy = tf.data.Dataset.zip((test_x, test_y)).shuffle(len(x_list)).repeat().batch(2)

#训练casia数据集，测试comofod数据集
# target_path = '../data/casia-dataset/target'
# mask_path = '../data/casia-dataset/mask'
# x_list, y_list = get_casiadataset(target_path, mask_path)

#训练casia增强数据集，测试comofod数据集
target_path = '../data/augmentation_data/image'
mask_path = '../data/augmentation_data/mask'
x_list, y_list = get_casiadataset(target_path, mask_path)
x_list = x_list[:15000]
y_list = y_list[:15000]
print(x_list)
print(y_list)
tfdata_x, tfdata_y = creat_tfdata(x_list, y_list, image_size)
tfdata_xy = tf.data.Dataset.zip((tfdata_x, tfdata_y))
tfdata_xy = tfdata_xy.shuffle(buffer_size=len(x_list))
tfdata_xy = tfdata_xy.repeat(epochs+1)
tfdata_xy = tfdata_xy.batch(batchs)

#定义tensorboard回调可视化
TBCallback = TensorBoard(log_dir=log)
cpCallback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(log, 'weight_{epoch:04d}.ckpt'), period=2)
my_model = creat_my_model([image_size, image_size, 3], backbone=backbone, pre_weight_path=None, mode='train')
my_model.summary()

if retrain:
    my_model.load_weights(ckpt_path)

my_model.compile(optimizer=keras.optimizers.Adam(0.0001),
                 loss=keras.losses.binary_crossentropy,
                 metrics=['accuracy'])

print('start train......')
my_model.fit(tfdata_xy,
             steps_per_epoch=len(x_list)//batchs,
             epochs=epochs,
             validation_data=test_xy,
             validation_steps=100,
             callbacks=[TBCallback, cpCallback],
             )


#执行后半部分数据集训练
target_path = '../data/augmentation_data/image'
mask_path = '../data/augmentation_data/mask'
x_list, y_list = get_casiadataset(target_path, mask_path)
x_list = x_list[15000:30000]
y_list = y_list[15000:30000]
print(x_list)
print(y_list)
tfdata_x, tfdata_y = creat_tfdata(x_list, y_list, image_size)
tfdata_xy = tf.data.Dataset.zip((tfdata_x, tfdata_y))
tfdata_xy = tfdata_xy.shuffle(buffer_size=len(x_list))
tfdata_xy = tfdata_xy.repeat(epochs+1)
tfdata_xy = tfdata_xy.batch(batchs)

cpCallback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(log, 'weight_s2_{epoch:04d}.ckpt'), period=2)
print('second train......')
my_model.fit(tfdata_xy,
             steps_per_epoch=len(x_list)//batchs,
             epochs=epochs,
             validation_data=test_xy,
             validation_steps=100,
             callbacks=[TBCallback, cpCallback],
             )

my_model.save(os.path.join(log, 'my_model.h5'))