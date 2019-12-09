import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
import time
from tensorflow.keras.callbacks import TensorBoard


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
    def __init__( self, nb_pools=256, **kwargs ) :
        self.nb_pools = nb_pools
        super( SelfCorrelationPercPooling, self ).__init__( **kwargs )
    def build( self, input_shape ) :
        self.built = True
    def call( self, x, mask=None ) :
        # parse input feature shape
        #获取输入shape
        bsize, nb_rows, nb_cols, nb_feats = keras.backend.int_shape( x )
        nb_maps = nb_rows * nb_cols
        # self correlation
        #计算自相关系数
        x_3d = keras.backend.reshape( x, tf.stack( [ -1, nb_maps, nb_feats ] ) )
        x_corr_3d = tf.matmul( x_3d, x_3d, transpose_a = False, transpose_b = True ) / nb_feats
        x_corr = keras.backend.reshape( x_corr_3d, tf.stack( [ -1, nb_rows, nb_cols, nb_maps ] ) )
        # argsort response maps along the translaton dimension
        if ( self.nb_pools is not None ) :
            ranks = keras.backend.cast( keras.backend.round( tf.lin_space( 1., nb_maps - 1, self.nb_pools ) ), 'int32' )
        else:
            ranks = tf.range(1, nb_maps, dtype = 'int32' )
        #排序相关系数 并选择，这里选择所有的结果
        x_sort, _ = tf.nn.top_k(x_corr, k = nb_maps, sorted = True )
        # pool out x features at interested ranks
        # NOTE: tf v1.1 only support indexing at the 1st dimension
        x_f1st_sort = keras.backend.permute_dimensions( x_sort, ( 3, 0, 1, 2 ) )
        #这里将最大值抛弃，选择剩下的所有结果 （抛弃自身的匹配）
        x_f1st_pool = tf.gather(x_f1st_sort, ranks )


        x_pool = keras.backend.permute_dimensions( x_f1st_pool, ( 1, 2, 3, 0 ) )
        return x_pool
    def compute_output_shape( self, input_shape ) :
        bsize, nb_rows, nb_cols, nb_feats = input_shape
        nb_pools = self.nb_pools if ( self.nb_pools is not None ) else ( nb_rows * nb_cols - 1 )
        return tuple( [ bsize, nb_rows, nb_cols, nb_pools])

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
    if ( len( uc_list ) > 1 ) :
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

def creat_my_model(img_shape=[256, 256, 3], name='my', train=True):

    #定义特征提取网络
    '''Create the similarity branch for copy-move forgery detection
        '''
    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------
    img_input = keras.Input(shape=img_shape, name=name + '_in')
    # ---------------------------------------------------------
    # VGG16 Conv Featex
    # ---------------------------------------------------------
    bname = name + '_cnn'
    ## Block 1
    x1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name=bname + '_b1c1')(img_input)
    x1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name=bname + '_b1c2')(x1)
    x1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name=bname + '_b1p')(x1)
    # Block 2
    x2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name=bname + '_b2c1')(x1)
    x2 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name=bname + '_b2c2')(x2)
    x2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name=bname + '_b2p')(x2)
    # Block 3
    x3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name=bname + '_b3c1')(x2)
    x3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name=bname + '_b3c2')(x3)
    x3 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name=bname + '_b3c3')(x3)
    x3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name=bname + '_b3p')(x3)
    # Block 4
    x4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name=bname + '_b4c1')(x3)
    x4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name=bname + '_b4c2')(x4)
    x4 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name=bname + '_b4c3')(x4)
    x4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name=bname + '_b4p')(x4)

    #将x2 x3 经过卷积成为16X16  x2 64X64 to 16X16   x3 32X32 to 16X16

    # Local Std-Norm Normalization (within each sample)
    xx = keras.layers.Activation(std_norm_along_chs, name=bname + '_sn')(x4)
    # ---------------------------------------------------------
    # Self Correlation Pooling
    # ---------------------------------------------------------
    bname = name + '_corr'
    ## Self Correlation

    xcorr = SelfCorrelationPercPooling(name=bname + '_corr')(xx)

    ## Global Batch Normalization (across samples)
    xn = keras.layers.BatchNormalization(name=bname + '_bn')(xcorr)
    # ---------------------------------------------------------
    # Deconvolution Network
    # ---------------------------------------------------------
    patch_list = [(1, 1), (3, 3), (5, 5)]
    # MultiPatch Featex
    bname = name + '_dconv'
    f16 = BnInception(xn, 8, patch_list, name=bname + '_mpf')
    # Deconv x2
    f32 = BilinearUpSampling2D(name=bname + '_bx2')(f16)
    dx32 = BnInception(f32, 6, patch_list, name=bname + '_dx2')
    # Deconv x4
    f64a = BilinearUpSampling2D(name=bname + '_bx4a')(f32)
    f64b = BilinearUpSampling2D(name=bname + '_bx4b')(dx32)
    f64 = keras.layers.Concatenate(axis=-1, name=name + '_dx4_m')([f64a, f64b])
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
    model = keras.Model(inputs=img_input, outputs=pred_mask, name=name)
    return model

if __name__ == "__main__":
    from data_preprocess import filter_image, image_preprocess

    #load data
    image_path = '../data/CoMoFoD_small'
    log = '../log/' + time.strftime('%Y%m%d-%H%M%S')
    my_model = creat_my_model([256, 256, 3], 'my')
    print(my_model.input)
    print(my_model.output)
    my_model.summary()

    #定义tensorboard回调可视化
    TBCallback = TensorBoard(log_dir=log)
    x_list, y_list = filter_image(image_path)
    x, y = image_preprocess(image_path, x_list, y_list)

    my_model.compile(optimizer=keras.optimizers.Adam(0.001),
                     loss=keras.losses.binary_crossentropy,
                     metrics=['accuracy'])
    my_model.fit(x, y,
                 batch_size=2,
                 epochs=1,
                 # validation_split=0.2,
                 shuffle=True,
                 callbacks=[TBCallback])
    my_model.save(os.path.join(log, 'my_model.h5'))