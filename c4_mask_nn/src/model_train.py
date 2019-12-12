import tensorflow as tf
from tensorflow import keras
import time
import os
from model_core import creat_my_model
from tensorflow.keras.callbacks import TensorBoard
from tf_dataset import creat_tfdata, filter_image
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#屏蔽警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# tf.enable_eager_execution()


image_size = 256
#load data
image_path = '../data/CoMoFoD_small'
log = '../log/' + time.strftime('%Y%m%d-%H%M%S')

#定义tensorboard回调可视化
TBCallback = TensorBoard(log_dir=log)
x_list, y_list = filter_image(image_path)
tfdata_x = creat_tfdata(x_list[:4000], 3, image_size)
tfdata_y = creat_tfdata(y_list[:4000], 1, image_size)

tfdata_xy = tf.data.Dataset.zip((tfdata_x, tfdata_y))
tfdata_xy = tfdata_xy.shuffle(buffer_size=4000)
tfdata_xy = tfdata_xy.repeat(50)
tfdata_xy = tfdata_xy.batch(2)

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
             steps_per_epoch=2000,
             epochs=50,
             callbacks=[TBCallback])

my_model.save(os.path.join(log, 'my_model.h5'))