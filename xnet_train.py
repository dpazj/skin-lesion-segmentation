from __future__ import absolute_import, division, print_function, unicode_literals

import os
import math
import pathlib
import numpy as np
import tensorflow as tf
import keras

from skimage import io
from skimage import transform
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from keras import optimizers

#import segmentation_models as sm

import matplotlib.pyplot as plt
from tqdm import tqdm

from UNetPlusPlus.segmentation_models import Xnet, Nestnet

from sklearn.model_selection import train_test_split
from config import * 
from metrics import * 

MODEL_ARCHITECTURE = Xnet(backbone_name='vgg19', encoder_weights='imagenet', decoder_block_type='transpose', input_shape=SHAPE, classes=1)


def load_image_by_pathname(image_path, mask=False):
    
    if mask:
        image = keras.preprocessing.image.load_img(image_path, target_size=(SHAPE[0], SHAPE[0]), color_mode="grayscale")
        image = np.array(image) 
        image = np.expand_dims(image,axis=-1)
    else:
        image = keras.preprocessing.image.load_img(image_path, target_size=(SHAPE[0], SHAPE[0]))
        image = np.array(image) 

    image = image.astype(np.uint8)

    return image
    

def load_images(paths, mask=False):
    images = []
    for path in tqdm(paths):
        image = load_image_by_pathname(path, mask)
        images.append(image)

    return images
    



def load(image_paths, mask_paths):

    images = []
    if os.path.exists("npsave/images.npy"):
        images = np.load('npsave/images.npy')
    else:
        images = load_images(image_paths)
        images = np.stack(images).astype(np.uint8)
        np.save('npsave/images.npy', images)

    masks = []
    if os.path.exists("npsave/masks.npy"):
        masks = np.load('npsave/masks.npy')
    else:
        masks = load_images(mask_paths, True)
        masks = np.stack(masks).astype(np.uint8)
        np.save('npsave/masks.npy', masks)

    images = images * SCALE
    masks = masks * SCALE

    return images, masks

    
def plot_images(a, b):
    
    for x,y in zip(a,b):
        plt.figure(figsize=(10, 10))
        img = x[0]

        print(x[0][125])
        print(y[0][125])


        plt.subplot(1, 2, 1)
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.imshow(y[0,:, :, 0])
        plt.show()   
    
      
    
#images
image_dir = pathlib.Path(INPUT_PATH)
image_paths = list(image_dir.glob('*.jpg'))
image_paths = [str(path) for path in image_paths]


mask_dir = pathlib.Path(GROUNDTRUTH_PATH)
mask_paths = list(mask_dir.glob('*.png'))
mask_paths = [str(path) for path in mask_paths]


images, masks = load(image_paths, mask_paths)

#plot_images(images,masks)


x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
num_train = len(x_train)
num_val = len(x_val)


img_gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=40, width_shift_range=0.1, height_shift_range=0.1,  horizontal_flip=True, vertical_flip=True)
mask_gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=40, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=True)

seed = 42

img_gen.fit(x_train, augment=True, seed=seed)
mask_gen.fit(y_train, augment=True, seed=seed)

x = img_gen.flow(x=x_train, batch_size=BATCH_SIZE, seed=seed)
y = mask_gen.flow(x=y_train, batch_size=BATCH_SIZE, seed=seed)

train = zip(x,y)

adam = optimizers.Adam(lr=INITIAL_LR)

#UNET RESENT BACKBONE

model = MODEL_ARCHITECTURE

model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=[jaccard_loss, jaccard_index, dice_coeff, pixelwise_specificity, pixelwise_sensitivity, pixelwise_accuracy])

save_path = './experiment_models/model1.hdf5'
checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_jaccard_index', save_best_only=True, verbose=1)


history = model.fit_generator(
    generator=train, 
    steps_per_epoch=int(np.ceil(num_train / float(BATCH_SIZE))),
    epochs=EPOCHS, 
    validation_data=(x_val, y_val),
    callbacks=[checkpoint]
    )

jaccard = history.history['jaccard_index']
val_jaccard = history.history['val_jaccard_index']

dice = history.history['dice_coeff']
val_dice = history.history['val_dice_coeff']


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs_range, jaccard, label='Training Jaccard Loss')
plt.plot(epochs_range, val_jaccard, label='Validation Jaccard Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Jaccard Index')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')


plt.subplot(2, 2, 3)
plt.plot(epochs_range, dice, label='Training Dice Coeff')
plt.plot(epochs_range, val_dice, label='Validation Dice Coeff')
plt.legend(loc='upper right')
plt.title('Training and Validation Dice Loss')

plt.savefig('train-img/training_loss.png')
