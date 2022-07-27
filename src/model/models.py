# Import internal libraries
import src.config as config

# Import external libraries
import pdb
import traceback
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

############################################################
#                 FLIPOUT MODELS - FocusNet                #
############################################################

class ConvBlockFNet(tf.keras.layers.Layer):
    """
    Folows "A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation in 3D Head and Neck CT Images"
    Changelog
     - residual blocks: x--> ReLu(Conv-GN)-ReLu(Conv-GN + x)
     - use GroupNorm 
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation=tf.nn.relu
                    , group_factor=2
                    , trainable=False
                    , residual=True
                    , init_filters=False
                    , pool=None
                    , name=''):
        super(ConvBlockFNet, self).__init__(name='{}_ConvBlockFNet'.format(name))

        # Step 0 - Init
        self.init_filters = init_filters
        self.residual = residual
        self.pool = pool
        if type(filters) == int:
            filters = [filters]
        
        # Step 1 - Reset the number of starting filters
        if self.init_filters:
            self.init_layer = tf.keras.Sequential(name='{}_InitConv'.format(self.name))
            self.init_layer.add(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=strides, padding=padding
                        , dilation_rate=1
                        , activation=None
                        , name='Conv1x1')
                    )
            # self.init_layer.add(tf.keras.layers.BatchNormalization(trainable=trainable))
            self.init_layer.add(tfa.layers.GroupNormalization(groups=filters[0]//group_factor, trainable=trainable))
            self.init_layer.add(tf.keras.layers.Activation(activation))
        
        # Step 2 - Conv Block
        self.conv_layer = tf.keras.Sequential(name='{}__ConvSeq'.format(self.name))
        for filter_id, filter_count in enumerate(filters):
            
            self.conv_layer.add(
                tf.keras.layers.Conv3D(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                        , dilation_rate=dilation_rate
                        , activation=None
                        , name='{}__ConvSeq__Conv_{}'.format(self.name, filter_id))
            )
            # self.conv_layer.add(tf.keras.layers.BatchNormalization(trainable=trainable))
            self.conv_layer.add(tfa.layers.GroupNormalization(groups=filter_count//group_factor, trainable=trainable))
            if self.residual:
                if filter_id == 0:
                    self.conv_layer.add(tf.keras.layers.Activation(activation))
            else:
                self.conv_layer.add(tf.keras.layers.Activation(activation))
        
        if self.residual:
            self.activation_layer = tf.keras.layers.Activation(activation)
        
        # Step 3 - Pooling
        if self.pool is not None:
            # self.pool_layer = tf.keras.layers.MaxPooling3D(pool_size=self.pool, strides=self.pool, name='{}_ConvBlockONet__Pool'.format(name))
            self.pool_layer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=self.pool, strides=self.pool, padding=padding
                        , dilation_rate=1
                        , activation=None
                        , name='{}__ConvPooling'.format(self.name)
                    )
                  
    @tf.function
    def call(self, x):
        
        # Step 1 - Init
        if self.init_filters:
            x = self.init_layer(x)

        if self.residual:
            x_ = self.conv_layer(x) # Conv-BN-ReLU -- Conv-BN
            x  = self.activation_layer(x_ + x) # RelU
        else:
            x = self.conv_layer(x)
        
        # Step 2 - Pooling
        if self.pool:
            x_pool = self.pool_layer(x)
            return x, x_pool
        else:
            return x

class ConvBlock3DSERes(tf.keras.layers.Layer):
    """
    For channel-wise attention
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation='relu'
                    , trainable=False
                    , pool=False
                    , squeeze_ratio=None
                    , init=False
                    , name=''):

        super(ConvBlock3DSERes, self).__init__(name='{}_ConvBlock3DSERes'.format(name))

        # Step 0 - Init
        self.init = init
        self.trainable = trainable

        # Step 1 - Init (to get equivalent feature map count)
        if self.init:
            self.convblock_filterequalizer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                                    , activation='relu'
                                                    )

        # Step 2- Conv Block
        self.convblock_res = ConvBlockFNet(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
                            , dilation_rate=dilation_rate
                            , activation=activation
                            , trainable=trainable
                            , pool=False
                            , name=name
                            )

        # Step 3 - Squeeze Block
        """
        Ref: https://github.com/imkhan2/se-resnet/blob/master/se_resnet.py
        """
        self.squeeze_ratio = squeeze_ratio
        if self.squeeze_ratio is not None:
            self.seblock = tf.keras.Sequential()
            self.seblock.add(tf.keras.layers.GlobalAveragePooling3D())
            self.seblock.add(tf.keras.layers.Reshape(target_shape=(1,1,1,filters[0])))
            self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0]//squeeze_ratio, kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                    , activation='relu'))
            self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                    , activation='sigmoid'))

        self.pool = pool
        if self.pool:
            self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), name='{}_Pool'.format(name))

    @tf.function
    def call(self, x):
        
        if self.init:
            x = self.convblock_filterequalizer(x)

        x_res = self.convblock_res(x)

        if self.squeeze_ratio is not None:
            x_se = self.seblock(x_res) # squeeze and then get excitation factor
            x_res = tf.math.multiply(x_res, x_se) # excited block

        y = x + x_res

        if self.pool:
            return y, self.pool_layer(y)
        else:
            return y

class UpConvBlock3D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(2,2,2), strides=(2, 2, 2), padding='same', trainable=False, name=''):
        super(UpConvBlock3D, self).__init__(name='{}_UpConv3D'.format(name))
        
        self.trainable = trainable
        self.upconv_layer = tf.keras.Sequential()
        self.upconv_layer.add(tf.keras.layers.Conv3DTranspose(filters, kernel_size, strides, padding=padding
                        , activation='relu'
                        , kernel_regularizer=None
                        , name='UpConv_{}'.format(self.name))
                    )
        # self.upconv_layer.add(tf.keras.layers.BatchNormalization(trainable=trainable))
    
    @tf.function
    def call(self, x):
        return self.upconv_layer(x)

class ConvBlock3DFlipOut(tf.keras.layers.Layer):
    """
    Ref
    - https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution3DFlipout
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation='relu'
                    , trainable=False
                    , pool=False
                    , name=''):
        super(ConvBlock3DFlipOut, self).__init__(name='{}ConvBlock3DFlipOut'.format(name))

        self.pool = pool
        self.filters = filters

        if type(filters) == int:
            filters = [filters]
        
        self.conv_layer = tf.keras.Sequential()
        for filter_id, filter_count in enumerate(filters):
            self.conv_layer.add(
                tfp.layers.Convolution3DFlipout(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                        , dilation_rate=dilation_rate
                        , activation=activation
                        # , kernel_prior_fn=?
                        , name='Conv3DFlip_{}'.format(filter_id))
            )
            self.conv_layer.add(tfa.layers.GroupNormalization(groups=filter_count//2, trainable=trainable))
            
        if self.pool:
            self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), name='Pool')
            
    def call(self, x):
        
        x = self.conv_layer(x)
        
        if self.pool:
            return x, self.pool_layer(x)
        else:
            return x

class ConvBlock3DSEResFlipOut(tf.keras.layers.Layer):
    """
    For channel-wise attention
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation='relu'
                    , trainable=False
                    , pool=False
                    , squeeze_ratio=None
                    , init=False
                    , name=''):

        super(ConvBlock3DSEResFlipOut, self).__init__(name='{}ConvBlock3DSEResFlipOut'.format(name))

        self.init = init
        if self.init:
            self.convblock_filterequalizer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                                    , activation='relu'
                                                    , kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None)

        self.convblock_res = ConvBlock3DFlipOut(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
                            , dilation_rate=dilation_rate
                            , activation=activation
                            , trainable=trainable
                            , pool=False
                            , name=name
                            )

        """
        Ref: https://github.com/imkhan2/se-resnet/blob/master/se_resnet.py
        """
        self.squeeze_ratio = squeeze_ratio
        if self.squeeze_ratio is not None:
            self.seblock = tf.keras.Sequential()
            self.seblock.add(tf.keras.layers.GlobalAveragePooling3D())
            self.seblock.add(tf.keras.layers.Reshape(target_shape=(1,1,1,filters[0])))
            self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0]//squeeze_ratio, kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                    , activation='relu'
                                    , kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None))
            self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                    , activation='sigmoid'
                                    , kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None))

        self.pool = pool
        if self.pool:
            self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), name='{}_Pool'.format(name))

    def call(self, x):
        
        if self.init:
            x = self.convblock_filterequalizer(x)

        x_res = self.convblock_res(x)

        if self.squeeze_ratio is not None:
            x_se = self.seblock(x_res) # squeeze and then get excitation factor
            x_res = tf.math.multiply(x_res, x_se) # excited block

        y = x + x_res

        if self.pool:
            return y, self.pool_layer(y)
        else:
            return y

class ModelFocusNetFlipOut(tf.keras.Model):

    def __init__(self, class_count, trainable=False, verbose=False):
        super(ModelFocusNetFlipOut, self).__init__(name='ModelFocusNetFlipOut')

        # Step 0 - Init
        self.verbose = verbose
        
        if 1:
            filters  = [[16,16], [32,32]]
            dilation_xy = [1, 2, 3, 6, 12, 18]
            dilation_z  = [1, 1, 1, 1, 1 , 1]
        
        # Se-Res Blocks
        self.convblock1 = ConvBlock3DSERes(filters=filters[0], kernel_size=(3,3,1), dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, pool=True , squeeze_ratio=2, name='Block1') # Dim/2 (e.g. 96/2=48, 240/2=120)(rp=(3,5,10),(1,1,2))
        

        # Dense ASPP
        self.convblock2 = ConvBlock3DSERes(filters=filters[0]                     , dilation_rate=(dilation_xy[1], dilation_xy[1], dilation_z[1]), trainable=trainable, pool=False, squeeze_ratio=2, name='Block2') # Dim/2 (e.g. 96/2=48, 240/2=120)(rp=(14,18) ,(4,6))
        self.convblock3 = ConvBlock3DFlipOut(filters=filters[1], dilation_rate=(dilation_xy[2], dilation_xy[2], dilation_z[2]), trainable=trainable, pool=False, name='Block3_ASPP') # Dim/2 (e.g. 96/2=48, 240/2=120) (rp=(24,30),(16,18))
        self.convblock4 = ConvBlock3DFlipOut(filters=filters[1], dilation_rate=(dilation_xy[3], dilation_xy[3], dilation_z[3]), trainable=trainable, pool=False, name='Block4_ASPP') # Dim/2 (e.g. 96/2=48, 240/2=120) (rp=(42,54),(20,22))
        self.convblock5 = ConvBlock3DFlipOut(filters=filters[1], dilation_rate=(dilation_xy[4], dilation_xy[4], dilation_z[4]), trainable=trainable, pool=False, name='Block5_ASPP') # Dim/2 (e.g. 96/2=48, 240/2=120) (rp=(78,102),(24,26))
        self.convblock6 = ConvBlock3DFlipOut(filters=filters[1], dilation_rate=(dilation_xy[5], dilation_xy[5], dilation_z[5]), trainable=trainable, pool=False, name='Block6_ASPP') # Dim/2 (e.g. 96/2=48, 240/2=120) (rp=(138,176),(28,30))
        self.convblock7 = ConvBlock3DSEResFlipOut(filters=filters[1], dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, pool=False, squeeze_ratio=2, init=True, name='Block7') # Dim/2 (e.g. 96/2=48) (rp=(176,180),(32,34))

        # Upstream
        self.convblock8 = ConvBlock3DSERes(filters=filters[1], dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, pool=False, squeeze_ratio=2, init=True, name='Block8') # Dim/2 (e.g. 96/2=48)

        self.upconvblock9 = UpConvBlock3D(filters=filters[0][0], trainable=trainable, name='Block9_1') # Dim/1 (e.g. 96/1 = 96)
        self.convblock9 = ConvBlock3DSERes(filters=filters[0], dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, pool=False, squeeze_ratio=2, init=True, name='Block9') # Dim/1 (e.g. 96/1 = 96)
        
        self.convblock10 = tf.keras.layers.Conv3D(filters=class_count, strides=(1,1,1), kernel_size=(3,3,3), padding='same'
                                , dilation_rate=(1,1,1)
                                , activation='softmax'
                                , name='Block10')

    # @tf.function (cant call model.losses if this is enabled)
    def call(self, x):
        
        # SE-Res Blocks
        conv1, pool1 = self.convblock1(x)
        conv2        = self.convblock2(pool1)
        
        # Dense ASPP
        conv3 = self.convblock3(conv2)
        conv3_op = tf.concat([conv2, conv3], axis=-1)
        
        conv4 = self.convblock4(conv3_op)
        conv4_op = tf.concat([conv3_op, conv4], axis=-1)
        
        conv5 = self.convblock5(conv4_op)
        conv5_op = tf.concat([conv4_op, conv5], axis=-1)
        
        conv6 = self.convblock6(conv5_op)
        conv6_op = tf.concat([conv5_op, conv6], axis=-1)
        
        conv7 = self.convblock7(conv6_op)
        
        # Upstream
        conv8 = self.convblock8(tf.concat([pool1, conv7], axis=-1))
        
        up9 = self.upconvblock9(conv8)
        conv9 = self.convblock9(tf.concat([conv1, up9], axis=-1))
        
        # Final
        conv10 = self.convblock10(conv9)

        if self.verbose:
            print (' ---------- Model: ', self.name)
            print (' - x: ', x.shape)
            print (' - conv1: ', conv1.shape)
            print (' - conv2: ', conv2.shape)
            print (' - conv3_op: ', conv3_op.shape)
            print (' - conv4_op: ', conv4_op.shape)
            print (' - conv5_op: ', conv5_op.shape)
            print (' - conv6_op: ', conv6_op.shape)
            print (' - conv7: ', conv7.shape)
            print (' - conv8: ', conv8.shape)
            print (' - conv9: ', conv9.shape)
            print (' - conv10: ', conv10.shape)


        return conv10

############################################################
#                 FLIPOUT MODELS-OrganNet2.5D              #
############################################################

class ConvBlockONet(tf.keras.layers.Layer):
    """
    Folows "A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation in 3D Head and Neck CT Images"
    Changelog
     - residual blocks: x--> ReLu(Conv-GN)-ReLu(Conv-GN + x)
     - use GroupNorm 
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation=tf.nn.relu
                    , group_factor=2
                    , trainable=False
                    , residual=True
                    , init_filters=False
                    , pool=None
                    , name=''):
        super(ConvBlockONet, self).__init__(name='{}_ConvBlockONet'.format(name))

        # Step 0 - Init
        self.init_filters = init_filters
        self.residual = residual
        self.pool = pool
        if type(filters) == int:
            filters = [filters]
        
        # Step 1 - Reset the number of starting filters
        if self.init_filters:
            self.init_layer = tf.keras.Sequential(name='{}_InitConv'.format(self.name))
            self.init_layer.add(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=strides, padding=padding
                        , dilation_rate=1
                        , activation=None
                        , name='Conv1x1')
                    )
            # self.init_layer.add(tf.keras.layers.BatchNormalization(trainable=trainable))
            self.init_layer.add(tfa.layers.GroupNormalization(groups=filters[0]//group_factor, trainable=trainable))
            self.init_layer.add(tf.keras.layers.Activation(activation))
        
        # Step 2 - Conv Block
        self.conv_layer = tf.keras.Sequential(name='{}__ConvSeq'.format(self.name))
        for filter_id, filter_count in enumerate(filters):
            
            self.conv_layer.add(
                tf.keras.layers.Conv3D(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                        , dilation_rate=dilation_rate
                        , activation=None
                        , name='{}__ConvSeq__Conv_{}'.format(self.name, filter_id))
            )
            # self.conv_layer.add(tf.keras.layers.BatchNormalization(trainable=trainable))
            self.conv_layer.add(tfa.layers.GroupNormalization(groups=filter_count//group_factor, trainable=trainable))
            if self.residual:
                if filter_id == 0:
                    self.conv_layer.add(tf.keras.layers.Activation(activation))
            else:
                self.conv_layer.add(tf.keras.layers.Activation(activation))
        
        if self.residual:
            self.activation_layer = tf.keras.layers.Activation(activation)
        
        # Step 3 - Pooling
        if self.pool is not None:
            # self.pool_layer = tf.keras.layers.MaxPooling3D(pool_size=self.pool, strides=self.pool, name='{}_ConvBlockONet__Pool'.format(name))
            self.pool_layer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=self.pool, strides=self.pool, padding=padding
                        , dilation_rate=1
                        , activation=None
                        , name='{}__ConvPooling'.format(self.name)
                    )
                  
    @tf.function
    def call(self, x):
        
        # Step 1 - Init
        if self.init_filters:
            x = self.init_layer(x)
        
        if self.residual:
            x_ = self.conv_layer(x) # Conv-BN-ReLU -- Conv-BN
            x  = self.activation_layer(x_ + x) # RelU
        else:
            x = self.conv_layer(x)
        
        # Step 2 - Pooling
        if self.pool:
            x_pool = self.pool_layer(x)
            return x, x_pool
        else:
            return x

class ConvBlockOnetSERes(tf.keras.layers.Layer):
    """
    Folows "A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation in 3D Head and Neck CT Images"
     - for channel-wise attention
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation=tf.nn.relu
                    , group_factor=2
                    , trainable=False
                    , residual=True
                    , init_filters=False
                    , pool=None
                    , squeeze_ratio=2
                    , name=''):

        super(ConvBlockOnetSERes, self).__init__(name='{}_ConvBlockOnetSERes'.format(name))

        # Step 1 - ConvBlock    
        self.convblock_res = ConvBlockONet(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
                                , dilation_rate=dilation_rate
                                , activation=activation
                                , group_factor=group_factor
                                , trainable=trainable
                                , residual=residual
                                , init_filters=init_filters
                                , pool=None
                                , name='{}_'.format(self.name)
                            )

        # Step 2 - Squeeze and Excitation 
        ## Ref: https://github.com/imkhan2/se-resnet/blob/master/se_resnet.py
        self.seblock = tf.keras.Sequential(name='{}__SE'.format(self.name))
        self.seblock.add(tf.keras.layers.GlobalAveragePooling3D())
        self.seblock.add(tf.keras.layers.Reshape(target_shape=(1,1,1,filters[0])))
        self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0]//squeeze_ratio, kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                , activation=tf.nn.relu))
        self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                , activation=tf.nn.sigmoid))

        # Step 3 - Pooling
        self.pool = pool
        if self.pool is not None:
            # self.pool_layer = tf.keras.layers.MaxPooling3D(pool_size=self.pool, strides=self.pool, name='{}_ConvBlockOnetSERes__Pool'.format(name))
            self.pool_layer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=self.pool, strides=self.pool, padding=padding
                        , dilation_rate=1
                        , activation=None
                        , name='{}__ConvPool'.format(self.name)
                    )

    @tf.function
    def call(self, x):
        
        # Step 1 - Conv Block
        x_res = self.convblock_res(x)
        
        # Step 2.1 - Squeeze and Excitation 
        x_se  = self.seblock(x_res) # squeeze and then get excitation factor
        
        # Step 2.2
        y = x_res + tf.math.multiply(x_res, x_se) # excited block
        
        # Step 3 - Pooling
        if self.pool is not None:
            return y, self.pool_layer(y)
        else:
            return y

class ConvBlockOnetFlipOutOld(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation=tf.nn.relu
                    , group_factor=2
                    , trainable=False
                    , pool=None
                    , name=''):
        super(ConvBlockOnetFlipOutOld, self).__init__(name='{}_ConvBlockOnetFlipOutOld'.format(name))

        # Step 0 - Init
        self.pool    = pool
        self.filters = filters
        if type(filters) == int:
            filters = [filters]
        
        # Step 1 - Conv Block
        self.conv_layer = tf.keras.Sequential(name='{}__ConvSeq'.format(self.name))
        for filter_id, filter_count in enumerate(filters):
            self.conv_layer.add(
                tfp.layers.Convolution3DFlipout(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                        , dilation_rate=dilation_rate
                        , activation=None
                        , name='{}___ConvFlip_{}'.format(self.name, filter_id))
            )
            self.conv_layer.add(tfa.layers.GroupNormalization(groups=filter_count//group_factor, trainable=trainable))
            self.conv_layer.add(tf.keras.layers.Activation(activation))
        
        # Step 2 - Pooling
        if self.pool is not None:
            self.pool_layer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=self.pool, strides=self.pool, padding=padding
                        , dilation_rate=1
                        , activation=None
                        , name='{}__ConvPool'.format(self.name)
                    )
    
    def call(self, x):
        
        # Step 1 - Conv Block
        x = self.conv_layer(x)
        
        # Step 2 - Pooling
        if self.pool:
            return x, self.pool_layer(x)
        else:
            return x

class ConvBlockOnetFlipOut(tf.keras.layers.Layer):
    """
    Folows "A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation in 3D Head and Neck CT Images"
    Changelog
     - residual blocks: x--> ReLu(Conv-GN)-ReLu(Conv-GN + x)
     - use GroupNorm 
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation=tf.nn.relu
                    , group_factor=2
                    , trainable=False
                    , residual=True
                    , init_filters=False
                    , pool=None
                    , name=''):
        super(ConvBlockOnetFlipOut, self).__init__(name='{}_ConvBlockOnetFlipOut'.format(name))

        # Step 0 - Init
        self.init_filters = init_filters
        self.residual = residual
        self.pool = pool
        if type(filters) == int:
            filters = [filters]
        
        # Step 1 - Reset the number of starting filters
        if self.init_filters:
            self.init_layer = tf.keras.Sequential(name='{}_InitConv'.format(self.name))
            self.init_layer.add(tfp.layers.Convolution3DFlipout(filters=filters[0], kernel_size=(1,1,1), strides=strides, padding=padding
                        , dilation_rate=1
                        , activation=None
                        , name='Conv1x1')
                    )
            self.init_layer.add(tfa.layers.GroupNormalization(groups=filters[0]//group_factor, trainable=trainable))
            self.init_layer.add(tf.keras.layers.Activation(activation))
        
        # Step 2 - Conv Block
        self.conv_layer = tf.keras.Sequential(name='{}__ConvSeq'.format(self.name))
        for filter_id, filter_count in enumerate(filters):
            
            self.conv_layer.add(
                tfp.layers.Convolution3DFlipout(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                        , dilation_rate=dilation_rate
                        , activation=None
                        , name='{}__ConvSeq__Conv_{}'.format(self.name, filter_id))
            )
            self.conv_layer.add(tfa.layers.GroupNormalization(groups=filter_count//group_factor, trainable=trainable))
            if self.residual:
                if filter_id == 0:
                    self.conv_layer.add(tf.keras.layers.Activation(activation))
            else:
                self.conv_layer.add(tf.keras.layers.Activation(activation))
        
        if self.residual:
            self.activation_layer = tf.keras.layers.Activation(activation)
        
        # Step 3 - Pooling
        if self.pool is not None:
            self.pool_layer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=self.pool, strides=self.pool, padding=padding
                        , dilation_rate=1
                        , activation=None
                        , name='{}__ConvPooling'.format(self.name)
                    )
                  
    @tf.function
    def call(self, x):
        
        # Step 1 - Init
        if self.init_filters:
            x = self.init_layer(x)
        
        if self.residual:
            x_ = self.conv_layer(x) # Conv-BN-ReLU -- Conv-BN
            x  = self.activation_layer(x_ + x) # RelU
        else:
            x = self.conv_layer(x)
        
        # Step 2 - Pooling
        if self.pool:
            x_pool = self.pool_layer(x)
            return x, x_pool
        else:
            return x

class UpConvBlockOnet(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(2,2,2), strides=(2, 2, 2), padding='same', trainable=False, name=''):
        super(UpConvBlockOnet, self).__init__(name='{}_UpConvBlockOnet'.format(name))
        
        self.upconv_layer = tf.keras.Sequential()
        self.upconv_layer.add(tf.keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=kernel_size, padding=padding
                        , activation=None
                        , name='{}_UpConvBlockOnet__ConvTranspose'.format(name))
                    )
    
    @tf.function
    def call(self, x):
        return self.upconv_layer(x)

class ModelONetFlipOutDenseASPP(tf.keras.Model):
    """
    Folows "A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation in 3D Head and Neck CT Images"
    Main Pointers
     - Pooling/Upsampling : separate pooling/upsampling for xy and z axes
     - 2D Conv vs 3D SERes: added start and end layer as 2D Conv followed by 3D SERes
    """

    def __init__(self, class_count, deepsup=False, trainable=False, verbose=False):
        super(ModelONetFlipOutDenseASPP, self).__init__(name='ModelONetFlipOutDenseASPP')

        # Step 0 - Init
        self.verbose = verbose
        self.deepsup = deepsup 

        if 1:
            filters  = [[16,16], [32,32]]
            dilation_xy = [2, 3, 5, 7, 9]
            dilation_z  = [1, 1, 1, 2, 2]
        elif 0:
            filters  = [[32,32], [48,48]]
            dilation_xy = [2, 3, 5, 7, 9]
            dilation_z  = [1, 1, 1, 2, 2]
        
        # Se-Res Blocks
        print ('\n ======================= {} ======================='.format(self.name))
        print (' - [ModelONetFlipOuttt()] filters: ', filters)
        print (' - [ModelONetFlipOuttt()] dilation_xy: ', dilation_xy)
        print (' - [ModelONetFlipOuttt()] dilation_z: ', dilation_z)

        # Step 1 - Basic 2D Conv
        self.convblock1 = ConvBlockONet(filters=filters[0], kernel_size=(3,3,1), trainable=trainable, pool=(2,2,1), name='Block1') # e.g. [140,140,40] -> [70,70,40], rp=(3->5->10),(1,1,1)

        # Step 2 - Se-Res Blocks
        self.convblock2 = ConvBlockOnetSERes(filters=filters[0]                   , trainable=trainable, pool=(2,2,2), squeeze_ratio=2, name='Block2') # e.g. [70,70,40] -> [35,35,20], rp=(12->14->28) ,(3->5->10)
        self.convblock3 = ConvBlockOnetSERes(filters=filters[1], init_filters=True, trainable=trainable, pool=None   , squeeze_ratio=2, name='Block3') # e.g. [35,35,20] -> [35,35,20], rp=(30->32)     ,(12->14)

        # Step 3 - Dense ASPP
        self.convblock4 = ConvBlockOnetFlipOut(filters=filters[1], init_filters=True, dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, name='Block4_ASPP') # e.g. [35,35,20] -> [35,35,20], rp=(36->40),(16->18)
        self.convblock5 = ConvBlockOnetFlipOut(filters=filters[1], init_filters=True, dilation_rate=(dilation_xy[1], dilation_xy[1], dilation_z[1]), trainable=trainable, name='Block5_ASPP') # e.g. [35,35,20] -> [35,35,20], rp=(46->52),(20->22)
        self.convblock6 = ConvBlockOnetFlipOut(filters=filters[1], init_filters=True, dilation_rate=(dilation_xy[2], dilation_xy[2], dilation_z[2]), trainable=trainable, name='Block6_ASPP') # e.g. [35,35,20] -> [35,35,20], rp=(62->72),(24->26)
        self.convblock7 = ConvBlockOnetFlipOut(filters=filters[1], init_filters=True, dilation_rate=(dilation_xy[3], dilation_xy[3], dilation_z[3]), trainable=trainable, name='Block7_ASPP') # e.g. [35,35,20] -> [35,35,20], rp=(86->100),(30->34)
        self.convblock8 = ConvBlockOnetFlipOut(filters=filters[1], init_filters=True, dilation_rate=(dilation_xy[4], dilation_xy[4], dilation_z[4]), trainable=trainable, name='Block8_ASPP') # e.g. [35,35,20] -> [35,35,20], rp=(118->136),(38->42)

        # Step 4 - Se-Res Blocks (for upstream)
        self.convblock9    = ConvBlockOnetSERes(filters=filters[1], init_filters=True, trainable=trainable, pool=None, squeeze_ratio=2, name='Block9')  # e.g. [35,35,20] -> [35,35,20], rp=(138->140) ,(44->46)
        self.upconvblock9  = UpConvBlockOnet(filters=filters[0][0], kernel_size=(2,2,2), trainable=trainable, name='Block9_1')       # e.g. [35,35,20] -> [70,70,40]
        self.convblock10   = ConvBlockOnetSERes(filters=filters[0], init_filters=True, trainable=trainable, pool=None, squeeze_ratio=2, name='Block10') # e.g. [70,70,40] -> [70,70,40]
        self.upconvblock10 = UpConvBlockOnet(filters=filters[0][0], kernel_size=(2,2,1), trainable=trainable, name='Block10_1')      # e.g. [70,70,40] -> [140,140,40]

        # Step 5 - Finalize
        self.convblock11  = ConvBlockONet(filters=filters[0], init_filters=True, kernel_size=(3,3,1), trainable=trainable, name='Block11') # e.g. [140,140,40] -> [140,140,40]
        self.convblock12 = tf.keras.layers.Convolution3D(filters=class_count, strides=(1,1,1), kernel_size=(1,1,1), padding='same'
                                , dilation_rate=(1,1,1)
                                , activation=tf.nn.softmax
                                , name='Block12'
                                )

    # @tf.function (cant call model.losses if this is enabled)
    def call(self, x):
        
        # Step 1 - Basic 2D Conv
        conv1, pool1 = self.convblock1(x) # e.g. [140,140,40], [70,70,40]
        
        # Step 2 - Se-Res Blocks
        conv2, pool2 = self.convblock2(pool1) # e.g. [70,70,40], [35,35,20]
        conv3        = self.convblock3(pool2) # e.g. [35,35,20]

        # Step 3 - Dense ASPP
        conv4 = self.convblock4(conv3) # e.g. [35,35,20]
        
        conv5    = self.convblock5(conv4)
        conv5_op = tf.concat([conv4, conv5], axis=-1)
        conv6    = self.convblock6(conv5_op)
        conv6_op = tf.concat([conv5_op, conv6], axis=-1)
        conv7    = self.convblock7(conv6_op)
        conv7_op = tf.concat([conv6_op, conv7], axis=-1)
        
        conv8    = self.convblock8(conv7_op) # e.g. [35,35,20]
        
        # Step 4 - Se-Res Blocks (for upstream)
        conv9 = self.convblock9(tf.concat([conv3, conv8], axis=-1)) # e.g. tf.concat([35,35,20], [35,35,20])
        up9   = self.upconvblock9(conv9) # e.g. [70,70,40]
        
        conv10 = self.convblock10(tf.concat([conv2, up9], axis=-1))  # e.g. tf.concat([70,70,40], [70,70,40])
        up10   = self.upconvblock10(conv10) # e.g. [140,140,40]
        
        # Step 5 - Finalize
        conv11 = self.convblock11(tf.concat([conv1, up10], axis=-1)) # e.g. tf.concat([140,140,40], [140,140,40])
        conv12 = self.convblock12(conv11)

        if self.verbose:
            print (' ---------- Model: ', self.name)
            print (' - x: ', x.shape)
            print (' - conv1   : ', conv1.shape)
            print (' - conv2   : ', conv2.shape)
            print (' - conv3   : ', conv3.shape)
            print (' - conv4   : ', conv4.shape)
            print (' - conv5_op: ', conv5_op.shape)
            print (' - conv6_op: ', conv6_op.shape)
            print (' - conv7_op: ', conv7_op.shape)
            print (' - conv8   : ', conv8.shape)
            print (' - conv9   : ', conv9.shape)
            print (' - conv10  : ', conv10.shape)
            print (' - conv11  : ', conv11.shape)
            print (' - conv12  : ', conv12.shape)


        if self.deepsup:
            return _, conv12
        else:
            return conv12

############################################################
#                            MAIN                          #
############################################################

if __name__ == "__main__":
    
    try:
        X         = tf.random.normal((2,140,140,40,1))

        if 1:
            print ('\n ------------------- ModelONetFlipOutDenseASPP ------------------- ')
            model     = ModelONetFlipOutDenseASPP(class_count=10, trainable=True, verbose=True)
            y_predict = model(X, training=True)
            model.summary()

        elif 0:
            print ('\n ------------------- ModelFocusNetFlipOut ------------------- ')
            model     = ModelFocusNetFlipOut(class_count=10, trainable=True)
            y_predict = model(X, training=True)
            model.summary()
    
    except:
        traceback.print_exc()
        pdb.set_trace()
    