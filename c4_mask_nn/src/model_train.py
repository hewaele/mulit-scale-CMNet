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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#屏蔽警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# tf.enable_eager_execution()


image_size = 256
#load data
image_path = '../data/CoMoFoD_small'
log = '../log/' + time.strftime('%Y%m%d-%H%M%S')+'_v3'

#定义tensorboard回调可视化
TBCallback = TensorBoard(log_dir=log)
#单独训练comofod数据集
# x_list, y_list = filter_image(image_path)
#训练casia数据集，测试comofod数据集
target_path = '../data/casia-dataset/target'
mask_path = '../data/casia-dataset/mask'
x_list, y_list = get_casiadataset(target_path, mask_path)
tfdata_x = creat_tfdata(x_list[:], 3, image_size)
tfdata_y = creat_tfdata(y_list[:], 1, image_size)
batchs = 2
epochs = 200
tfdata_xy = tf.data.Dataset.zip((tfdata_x, tfdata_y))
tfdata_xy = tfdata_xy.shuffle(buffer_size=len(x_list))
tfdata_xy = tfdata_xy.repeat(epochs+1)
tfdata_xy = tfdata_xy.batch(batchs)

def show_xy(x, y):
    plt.figure()
    plt.subplot(121)
    x = np.array(x)
    plt.imshow(x.reshape((256, 256, 3)))
    y = np.array(y)
    plt.subplot(122)
    plt.imshow(y.reshape([256, 256]))
    plt.show()

# for i, j in tfdata_xy.take(1):
#     # show_xy(i, j)
#     print(i.shape)
#     print(np.sum(np.round(j)))

my_model = creat_my_model([image_size, image_size, 3], 'my')
print(my_model.input)
print(my_model.output)
my_model.summary()
my_model.compile(optimizer=keras.optimizers.Adam(0.001),
                 loss=keras.losses.binary_crossentropy,
                 metrics=['accuracy'])
my_model.fit(tfdata_xy,
             steps_per_epoch=len(x_list)//batchs,
             epochs=epochs,
             callbacks=[TBCallback])

my_model.save(os.path.join(log, 'my_model.h5'))