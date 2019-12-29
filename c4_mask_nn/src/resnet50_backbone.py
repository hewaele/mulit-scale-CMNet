import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.engine.network import get_source_inputs
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
  """The identity block is the block that has no conv layer at shortcut.

  Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

  Returns:
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  x = Conv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
          x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  x = layers.add([x, input_tensor])
  x = Activation('relu')(x)
  return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,
                                                                          2)):
  """A block that has a conv layer at shortcut.

  Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the first conv layer in the block.

  Returns:
      Output tensor for the block.

  Note that from stage 3,
  the first conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(
      filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(
          input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  x = Conv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
          x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  shortcut = Conv2D(
      filters3, (1, 1), strides=strides, name=conv_name_base + '1')(
          input_tensor)
  shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

  x = layers.add([x, shortcut])
  x = Activation('relu')(x)
  return x


@tf_export('keras.applications.ResNet50',
           'keras.applications.resnet50.ResNet50')
def ResNet50(input_shape=None, weights='imagenet'):
  """Instantiates the ResNet50 architecture.

  Optionally loads weights pre-trained
  on ImageNet. Note that when using TensorFlow,
  for best performance you should set
  `image_data_format='channels_last'` in your Keras config
  at ~/.keras/keras.json.

  The model and the weights are compatible with both
  TensorFlow and Theano. The data format
  convention used by the model is the one
  specified in your Keras config file.

  Arguments:
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)` (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 inputs channels,
          and width and height should be no smaller than 197.
          E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.

  Returns:
      A Keras model instance.

  Raises:
      ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
  """
  img_input = Input(shape=input_shape)
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1

#block 1 256x256
  x1 = Conv2D(
      64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
  x1 = BatchNormalization(axis=bn_axis, name='bn_conv1')(x1)
  x1 = Activation('relu')(x1)
  x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x1)
#block 2 128x128
  x2 = conv_block(x1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
  x2 = identity_block(x2, 3, [64, 64, 256], stage=2, block='b')
  x2 = identity_block(x2, 3, [64, 64, 256], stage=2, block='c')

#block 3 64x64
  x3 = conv_block(x2, 3, [128, 128, 512], stage=3, block='a')
  x3 = identity_block(x3, 3, [128, 128, 512], stage=3, block='b')
  x3 = identity_block(x3, 3, [128, 128, 512], stage=3, block='c')
  x3 = identity_block(x3, 3, [128, 128, 512], stage=3, block='d')

#block 4 32x32
  x4 = conv_block(x3, 3, [256, 256, 1024], stage=4, block='a')
  x4 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='b')
  x4 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='c')
  x4 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='d')
  x4 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='e')
  x4 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='f')
#block 5 16x16
  x5 = conv_block(x4, 3, [512, 512, 2048], stage=5, block='a')
  x5 = identity_block(x5, 3, [512, 512, 2048], stage=5, block='b')
  x5 = identity_block(x5, 3, [512, 512, 2048], stage=5, block='c')

  out = AveragePooling2D((7, 7), name='avg_pool')(x5)

  out = GlobalMaxPooling2D()(out)

  inputs = img_input
  # Create model.
  model = Model(inputs, [x2, x3, x4, out], name='resnet50')

  # load weights
  if weights is not None:
    model.load_weights(weights)

  return model.input, model.output

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    weight_path = '../pre_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    input, output = ResNet50([256, 256, 3], weights=weight_path)
    x2, x3, x4, _ = output[0], output[1], output[2], output[3]
    model = Model(input, _)
    model.summary()
