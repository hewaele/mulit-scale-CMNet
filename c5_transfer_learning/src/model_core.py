import tensorflow as tf
import numpy
import os
from PIL import Image
import numpy as np
import os

def creat_test_model():
    pre_model = tf.keras.applications.VGG16(input_shape=[256, 256, 3], include_top=False, weights=None)
    pre_model.summary()
    w0 = pre_model.get_weights()[0]
    pre_model.load_weights('../pre_model_weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    my_in = pre_model.input
    w1 = pre_model.get_weights()[0]
    print(w0[0][0])

    print(w1[0][0])
    print(pre_model.get_layer('block4_conv1'))
    print(pre_model.layers[-8])
    x = tf.keras.layers.Flatten(name='flatten')(pre_model.output)
    # print(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    my_model = tf.keras.Model(my_in, x)
    return my_model

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = creat_test_model()
    # model.summary()
    img = Image.open('../test_image/cat.png')
    img = img.convert('RGB').resize([256, 256])
    img = np.array(img).reshape([-1, 256, 256, 3])
    pre_result = model.predict(img)
    print(model.get_weights()[0][0][0])
    print(np.argmax(pre_result))

if __name__ == "__main__":
    main()