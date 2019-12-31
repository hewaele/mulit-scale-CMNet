import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
import time
from tensorflow.keras.callbacks import TensorBoard
from resnet50_backbone import ResNet50

def std_norm_along_chs(x):
    '''Data normalization along the channle axis
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        xn = tensor4d, same shape as x, normalized version of x
    '''
    avg = keras.backend.mean(x, axis=-1, keepdims=True)
    std = keras.backend.maximum(1e-4, keras.backend.std(x, axis=-1, keepdims=True))
    return (x - avg) / std

class SelfCorrelationPercPooling( keras.layers.Layer ) :
    '''Custom Self-Correlation Percentile Pooling Layer
    Arugment:
        nb_pools = int, number of percentile poolings
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        x_pool = tensor4d, (n_samples, n_rows, n_cols, nb_pools)
    '''
    def __init__(self, nb_pools=256, **kwargs ) :
        self.nb_pools = nb_pools
        super( SelfCorrelationPercPooling, self ).__init__( **kwargs )
    def build( self, input_shape ) :
        self.built = True
    def call( self, x, mask=None ):
        # parse input feature shape
        #获取输入shape
        bsize, nb_rows, nb_cols, nb_feats = keras.backend.int_shape(x)
        nb_maps = nb_rows * nb_cols
        #矩阵扩充
        # x_3d = keras.backend.reshape(x, tf.stack([-1, nb_maps, nb_feats]))
        # x_3d = tf.tile(x_3d, [1, nb_maps, 1])
        # temp_x_3d = tf.reshape(x, shape=[-1, nb_rows, nb_cols, 1, nb_feats])
        # temp_x_3d = tf.reshape(tf.tile(temp_x_3d, [1, 1, 1, nb_maps, 1]), shape=[-1, nb_maps*nb_maps, nb_feats])
        #
        # #计算欧式距离
        # x_corr_3d = tf.subtract(x_3d, temp_x_3d)
        # x_corr_3d = tf.multiply(x_corr_3d, x_corr_3d)
        # x_corr_3d = tf.keras.backend.sum(x_corr_3d, axis=2)
        # x_corr_3d = tf.sqrt(x_corr_3d)
        # x_corr = tf.reshape(x_corr_3d, [-1, nb_rows, nb_cols, nb_maps])
        # x_corr = 10 - x_corr
        # x_sort, _ = tf.nn.top_k(x_corr, nb_maps, sorted=True)
        #
        # #选择一定个数的返回
        # # ranks = tf.range(nb_maps - self.nb_pools-1, nb_maps)
        # ranks = tf.range(1, self.nb_pools+1)
        # x_f1st_sort = keras.backend.permute_dimensions(x_sort, (3, 0, 1, 2))
        # # 这里将最大值抛弃，选择剩下的所有结果 （抛弃自身的匹配）
        # x_f1st_pool = tf.gather(x_f1st_sort, ranks)
        #
        # x_pool = keras.backend.permute_dimensions(x_f1st_pool, (1, 2, 3, 0))


        # self correlation
        #计算自相关系数
        x_3d = keras.backend.reshape(x, tf.stack([-1, nb_maps, nb_feats]))
        x_corr_3d = tf.matmul(x_3d, x_3d, transpose_a = False, transpose_b = True )/nb_feats
        x_corr = keras.backend.reshape( x_corr_3d, tf.stack( [ -1, nb_rows, nb_cols, nb_maps ] ) )
        # argsort response maps along the translaton dimension
        if (self.nb_pools is not None ):
            ranks = keras.backend.cast( keras.backend.round(tf.lin_space(1., nb_maps - 1, self.nb_pools)), 'int32')
        else:
            ranks = tf.range(1, nb_maps, dtype = 'int32' )
        #排序相关系数 并选择，这里选择所有的结果
        x_sort, _ = tf.nn.top_k(x_corr, k = nb_maps, sorted = True )
        # pool out x features at interested ranks
        # NOTE: tf v1.1 only support indexing at the 1st dimension
        x_f1st_sort = keras.backend.permute_dimensions( x_sort, ( 3, 0, 1, 2 ) )
        #这里将最大值抛弃，选择剩下的所有结果 （抛弃自身的匹配）
        x_f1st_pool = tf.gather(x_f1st_sort, ranks)

        x_pool = keras.backend.permute_dimensions( x_f1st_pool, ( 1, 2, 3, 0 ) )


        return x_pool
    def compute_output_shape( self, input_shape ) :
        bsize, nb_rows, nb_cols, nb_feats = input_shape
        nb_pools = self.nb_pools if (self.nb_pools is not None) else (nb_rows * nb_cols - 1)
        return tuple([bsize, nb_rows, nb_cols, nb_pools])

def BnInception(x, nb_inc=16, inc_filt_list=[(1,1), (3,3), (5,5)], name='uinc') :
    '''Basic Google inception module with batch normalization
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
        nb_inc = int, number of filters in individual Conv2D
        inc_filt_list = list of kernel sizes, individual Conv2D kernel size
        name = str, name of module
    Output:
        xn = tensor4d, (n_samples, n_rows, n_cols, n_new_feats)
    '''
    uc_list = []
    for idx, ftuple in enumerate(inc_filt_list ) :
        uc = keras.layers.Conv2D( nb_inc, ftuple, activation='linear', padding='same', name=name+'_c%d' % idx)(x)
        uc_list.append(uc)
    if ( len( uc_list ) > 1 ):
        uc_merge = keras.layers.Concatenate( axis=-1, name=name+'_merge')(uc_list)
    else :
        uc_merge = uc_list[0]
    uc_norm = keras.layers.BatchNormalization(name=name+'_bn')(uc_merge)
    xn = keras.layers.Activation('relu', name=name+'_re')(uc_norm)
    return xn

class BilinearUpSampling2D( keras.layers.Layer ) :
    '''Custom 2x bilinear upsampling layer
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        x2 = tensor4d, (n_samples, 2*n_rows, 2*n_cols, n_feats)
    '''
    def call( self, x, mask=None ) :
        bsize, nb_rows, nb_cols, nb_filts = keras.backend.int_shape(x)
        new_size = tf.constant( [ nb_rows * 2, nb_cols * 2 ], dtype = tf.int32 )
        return tf.image.resize_images( x, new_size, align_corners=True )
    def compute_output_shape( self, input_shape ) :
        bsize, nb_rows, nb_cols, nb_filts = input_shape
        return tuple( [ bsize, nb_rows * 2, nb_cols * 2, nb_filts ] )

def mrcnn_mask_loss_graph(target_masks, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.

    mask_shape = tf.shape(target_masks)
    target_masks = keras.backend.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = keras.backend.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = keras.backend.switch(tf.size(target_masks) > 0,
                                keras.backend.binary_crossentropy(target=target_masks, output=pred_masks),
                                tf.constant(0.0))
    loss = keras.backend.mean(loss)
    return loss

def creat_backbone(input_shape=None, weights=None):

    train = True
    img_input = tf.keras.layers.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=train)(
        img_input)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=train)(
        x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x2 = tf.keras.layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=train)(
        x)
    x2 = tf.keras.layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=train)(
        x2)
    x2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x2)

    # Block 3
    x3 = tf.keras.layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=train)(
        x2)
    x3 = tf.keras.layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=train)(
        x3)
    x3 = tf.keras.layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=train)(
        x3)
    x3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x3)

    # Block 4
    x4 = tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=train)(
        x3)
    x4 = tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=train)(
        x4)
    x4 = tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=train)(
        x4)
    x4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x4)

    # Block 5
    x5 = tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=train)(
        x4)
    x5 = tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=train)(
        x5)
    x5 = tf.keras.layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=train)(
        x5)
    x5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x5)

    y = tf.keras.layers.GlobalMaxPooling2D()(x5)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    model = tf.keras.Model(img_input, [x2, x3, x4, y], name='vgg16')

    if weights is not None:
        model.load_weights(weights)

    return model.input, model.output

def creat_my_model(img_shape, backbone, pre_weight_path, name='my', mode='train'):

    #定义特征提取网络
    '''Create the similarity branch for copy-move forgery detection
        '''
    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------
    if backbone == 'vgg':
        img_input, xx = creat_backbone(img_shape, pre_weight_path)
    else:
        img_input, xx = ResNet50(img_shape, pre_weight_path)
    x2 = xx[0]
    x3 = xx[1]
    x4 = xx[2]
    # ---------------------------------------------------------
    bname = name

    # Local Std-Norm Normalization (within each sample)
    xx4 = keras.layers.Activation(std_norm_along_chs, name=bname + '_sn4')(x4)
    xx2 = keras.layers.Activation(std_norm_along_chs, name=bname + '_sn2')(x2)
    xx3 = keras.layers.Activation(std_norm_along_chs, name=bname + '_sn3')(x3)

    # ---------------------------------------------------------
    # Self Correlation Pooling
    # ---------------------------------------------------------
    bname = name + '_corr'
    ## Self Correlation

    #TODO 蚕食修改nb——pools 参数 缩减参数 但前选择了一半
    xcorr4 = SelfCorrelationPercPooling(name=bname + '_corr', nb_pools=256)(xx4)

    #将x2 x3计算自相关
    #todo  更改参数为64进行测试
    xcorr3 = SelfCorrelationPercPooling(name=bname + '_corr3', nb_pools=8)(xx3)
    xcorr2 = SelfCorrelationPercPooling(name=bname + '_corr2', nb_pools=6)(xx2)
    ## Global Batch Normalization (across samples)
    xcorr4_1 = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name=bname+"_cn4")(xcorr4)
    xn4 = keras.layers.BatchNormalization(name=bname + '_bn4')(xcorr4_1)
    xcorr3_1 = keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', name=bname + "_cn3")(xcorr3)
    xn3 = keras.layers.BatchNormalization(name=bname + '_bn3')(xcorr3_1)
    xcorr2_1 = keras.layers.Conv2D(6, (3, 3), padding='same', activation='relu', name=bname + "_cn2")(xcorr2)
    xn2 = keras.layers.BatchNormalization(name=bname + '_bn2')(xcorr2_1)
    # ---------------------------------------------------------
    # Deconvolution Network
    # ---------------------------------------------------------
    patch_list = [(1, 1), (3, 3), (5, 5)]
    # MultiPatch Featex
    bname = name + '_dconv'
    f16 = BnInception(xn4, 8, patch_list, name=bname + '_mpf')
    # Deconv x2
    f32 = BilinearUpSampling2D(name=bname + '_bx2')(f16)
    f32 = keras.layers.Concatenate(axis=-1, name=name + '_dx2_m')([f32, xn3])
    dx32 = BnInception(f32, 6, patch_list, name=bname + '_dx2')
    # Deconv x4
    f64a = BilinearUpSampling2D(name=bname + '_bx4a')(f32)
    f64b = BilinearUpSampling2D(name=bname + '_bx4b')(dx32)
    f64 = keras.layers.Concatenate(axis=-1, name=name + '_dx4_m')([f64a, f64b, xn2])
    dx64 = BnInception(f64, 4, patch_list, name=bname + '_dx4')
    # Deconv x8
    f128a = BilinearUpSampling2D(name=bname + '_bx8a')(f64a)
    f128b = BilinearUpSampling2D(name=bname + '_bx8b')(dx64)
    f128 = keras.layers.Concatenate(axis=-1, name=name + '_dx8_m')([f128a, f128b])
    dx128 = BnInception(f128, 2, patch_list, name=bname + '_dx8')
    # Deconv x16
    f256a = BilinearUpSampling2D(name=bname + '_bx16a')(f128a)
    f256b = BilinearUpSampling2D(name=bname + '_bx16b')(dx128)
    f256 = keras.layers.Concatenate(axis=-1, name=name + '_dx16_m')([f256a, f256b])
    dx256 = BnInception(f256, 2, patch_list, name=bname + '_dx16')
    # Summerize
    fm256 = keras.layers.Concatenate(axis=-1, name=name + '_mfeat')([f256a, dx256])
    masks = BnInception(fm256, 2, [(5, 5), (7, 7), (11, 11)], name=bname + '_dxF')
    # ---------------------------------------------------------
    # Output for Auxiliary Task
    # ---------------------------------------------------------
    pred_mask = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', name=name + '_pred_mask', padding='same')(masks)
    # ---------------------------------------------------------
    # End to End
    # ---------------------------------------------------------
    if mode == 'train':
        model = keras.Model(inputs=img_input, outputs=pred_mask, name=name)
    else:
        model = keras.Model(inputs=img_input, outputs=[pred_mask, masks], name=name)
    return model
