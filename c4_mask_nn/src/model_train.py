import tensorflow as tf
from tensorflow import keras
import time
import os
from model_core import creat_my_model
from tensorflow.keras.callbacks import TensorBoard
from tf_dataset import creat_tfdata, filter_image

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#屏蔽警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

image_size = 256
#load data
image_path = '../data/CoMoFoD_small'
log = '../log/' + time.strftime('%Y%m%d-%H%M%S')
my_model = creat_my_model([image_size, image_size, 3], 'my')
print(my_model.input)
print(my_model.output)
my_model.summary()

#定义tensorboard回调可视化
TBCallback = TensorBoard(log_dir=log)
x_list, y_list = filter_image(image_path)
tfdata_x = creat_tfdata(x_list, 3, image_size)
tfdata_y = creat_tfdata(y_list, 1, image_size)

tfdata_xy = tf.data.Dataset.zip((tfdata_x, tfdata_y))
tfdata_xy = tfdata_xy.shuffle(buffer_size=5000)
tfdata_xy = tfdata_xy.repeat()
tfdata_xy = tfdata_xy.batch(32)
tfdata_xy = tfdata_xy.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
my_model.compile(optimizer=keras.optimizers.Adam(0.001),
                 loss=keras.losses.binary_crossentropy,
                 metrics=['accuracy'])
my_model.fit(tfdata_xy,
             steps_per_epoch=200,
             epochs=10,
             # validation_split=0.2,
             shuffle=True,
             callbacks=[TBCallback])
my_model.save(os.path.join(log, 'my_model.h5'))
import tensorflow