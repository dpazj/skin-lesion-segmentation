import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K 



# Modified from https://github.com/DeepTrial/Retina-VesselNet/blob/master/perception/models/dense_unet.py


def DenseBlock(inputs, outdim):

		inputshape = K.int_shape(inputs)
		bn = layers.normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(inputs)
		act = layers.Activation('relu')(bn)
		conv1 = layers.Conv2D(outdim, (3, 3), activation=None, padding='same')(act)

		if inputshape[3] != outdim:
			shortcut = layers.Conv2D(outdim, (1, 1), padding='same')(inputs)
		else:
			shortcut = inputs
		result1 = layers.add([conv1, shortcut])

		bn = layers.normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(result1)
		act = layers.Activation('relu')(bn)
		conv2 = layers.Conv2D(outdim, (3, 3), activation=None, padding='same')(act)
		result = layers.add([result1, conv2, shortcut])
		result = layers.Activation('relu')(result)
		return result

def create_dense_unet(img_size):
    inputs = layers.Input(shape=img_size)
    # conv1 = layers.Conv2D(32, (1, 1), activation=None, padding='same')(inputs)
    # conv1 = layers.normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
    #                                             beta_initializer='zero', gamma_initializer='one')(conv1)
    # conv1 = layers.Activation('relu')(conv1)



    conv2 = DenseBlock(inputs, 64)  # 24
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = DenseBlock(pool2, 128)  # 12
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = DenseBlock(pool3, 256)  # 12
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = DenseBlock(pool4, 512)  # 12
    pool5 = layers.MaxPooling2D(pool_size=(2, 2))(conv5)

    center = DenseBlock(pool5, 1024)  # 12

    up1 = layers.Conv2DTranspose(512, (3, 3), strides=2, activation='relu', padding='same')(center)
    up1 = layers.concatenate([up1, conv5], axis=-1) #5
    conv5 = DenseBlock(up1, 512)

    up2 = layers.Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same')(conv5)
    up2 = layers.concatenate([up2, conv4], axis=-1) #4
    conv6 = DenseBlock(up2, 256)

    up3 = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(conv6)
    up3 = layers.concatenate([up3, conv3], axis=-1)
    conv7 = DenseBlock(up3, 128)

    up4 = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(conv7)
    up4 = layers.concatenate([up4, conv2], axis=-1)
    conv8 = DenseBlock(up4, 64)

    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv8)
    
    

    model = models.Model(inputs, conv10)
    return model

