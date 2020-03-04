import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras import utils

from tensorflow.python import keras 

from keras_applications import get_submodules_from_kwargs


#MODIFIED FROM https://github.com/qubvel/segmentation_models/
# @misc{Yakubovskiy:2019,
#   Author = {Pavel Yakubovskiy},
#   Title = {Segmentation Models},
#   Year = {2019},
#   Publisher = {GitHub},
#   Journal = {GitHub repository},
#   Howpublished = {\url{https://github.com/qubvel/segmentation_models}}
# }

def get_submodules():
    return {
        'backend': K,
        'models': models,
        'layers': layers,
        'utils': utils,
    }

def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3 if K.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper

def Conv2dBn(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs
):
    """Extension of Conv2D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)
    backend, layers, models, utils = get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor):

        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper



def create_vgg16_unet(img_shape):
    vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=img_shape)

    encoder_features = ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2') #skip_connection_layers
    decoder_filters = (256, 128, 64, 32, 16)
    skip_connections = ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2','block5_pool', 'block4_pool', 'block3_pool', 'block2_pool', 'block1_pool',)

    inputs = vgg16.input
    x = vgg16.output

    skips = ([vgg16.get_layer(name=i).output if isinstance(i, str)
              else vgg16.get_layer(index=i).output for i in encoder_features])

    if isinstance(vgg16.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, True, name='center_block1')(x)
        x = Conv3x3BnReLU(512, True, name='center_block2')(x)


    for i in range(5):
        if i < len(skips):
            skip = skips[i]
        else:
            skip = None
        x = DecoderUpsamplingX2Block(decoder_filters[i], stage=i, use_batchnorm=True)(x, skip)


    x = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)

    x = layers.Activation('sigmoid', name='sigmoid')(x)

    model = models.Model(inputs, x)

    return model



create_vgg16_unet(img_shape=(256,256,3))
