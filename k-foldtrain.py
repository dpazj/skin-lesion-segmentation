from __future__ import absolute_import, division, print_function, unicode_literals

import os
import math
import pathlib
import numpy as np
import tensorflow as tf
import gc
import keras

from skimage import io
from skimage import transform
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from keras import optimizers

#import segmentation_models as sm

import matplotlib.pyplot as plt
from tqdm import tqdm

from UNetPlusPlus import segmentation_models
from UNetPlusPlus.segmentation_models import Xnet, Nestnet

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from config import * 
from metrics import * 
from unet import * 

MODEL_ARCHITECTURE = Xnet(backbone_name='resnext101', encoder_weights='imagenet', decoder_block_type='transpose', input_shape=SHAPE, classes=1)

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
        img = x

        plt.subplot(1, 2, 1)
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.imshow(y[:, :, 0])
        plt.savefig('foo.png') 
        input("Press Enter to continue...")
    



    
#images
image_dir = pathlib.Path(INPUT_PATH) 
image_paths = list(image_dir.glob('*.jpg'))
image_paths = [str(path) for path in image_paths]
image_paths = sorted(image_paths)

mask_dir = pathlib.Path(GROUNDTRUTH_PATH)
mask_paths = list(mask_dir.glob('*.png'))
mask_paths = [str(path) for path in mask_paths]
mask_paths = sorted(mask_paths)

images, masks = load(image_paths, mask_paths)

#plot_images(images,masks)

kfold = KFold(n_splits=5, random_state=42, shuffle=True)

idx = 0

for train_idx, test_idx in kfold.split(images):
        
    x_train = images[train_idx]
    y_train = masks[train_idx]
    
    x_test = images[test_idx]
    y_test = masks[test_idx]
    
    
    num_train = len(x_train)

    img_gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=40, width_shift_range=0.1, height_shift_range=0.1,  horizontal_flip=True, vertical_flip=True)
    mask_gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=40, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=True)

    seed = 42

    img_gen.fit(x_train, augment=True, seed=seed)
    mask_gen.fit(y_train, augment=True, seed=seed)

    x = img_gen.flow(x=x_train, batch_size=BATCH_SIZE, seed=seed)
    y = mask_gen.flow(x=y_train, batch_size=BATCH_SIZE, seed=seed)
    
    
    
    train = zip(x,y)

    adam = tf.keras.optimizers.Adam(lr=INITIAL_LR)
    


    model = MODEL_ARCHITECTURE
    
    model.compile(optimizer=adam, loss=bce_jaccard_loss, metrics=[jaccard_loss, jaccard_index, dice_coeff, pixelwise_specificity, pixelwise_sensitivity, pixelwise_accuracy])

    save_path = './models/kfmodel_unetbcej06loss_'+str(idx)+'.hdf5'
    checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_jaccard_index', save_best_only=True, verbose=1)


    model.fit_generator(
        generator=train, 
        steps_per_epoch=int(np.ceil(num_train / float(BATCH_SIZE))),
        epochs=EPOCHS, 
        validation_data=(x_test, y_test),
        callbacks=[checkpoint]
        )
    
    del model
    gc.collect()
    idx += 1
