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

import segmentation_models as sm

import matplotlib.pyplot as plt
from tqdm import tqdm


from sklearn.model_selection import train_test_split
from data import * 
from config import * 
from metrics import * 
from pretrained_backbone_unet import * 


def partition_data(x, y, k=5, i=0, test_split=0., seed=42):
    
    n = x.shape[0]

    n_set = int(n * (1. - test_split)) // k
    # divide the data into (k + 1) sets, -1 is test set, [0, k) are for train and validation
    indices = np.array([i for i in range(k) for _ in range(n_set)] +
                       [-1] * (n - n_set * k),
                       dtype=np.int8)

    np.random.seed(seed)
    np.random.shuffle(indices)

    valid_indices = (indices == i)
    test_indices = (indices == -1)
    train_indices = ~(valid_indices | test_indices)

    x_valid = x[valid_indices]
    y_valid = y[valid_indices]

    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def load_image_by_pathname(image_path, mask=False):
    
    #image = io.imread(image_path)
    
    

    if mask:
        #image = image.split()[0]
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(SHAPE[0], SHAPE[0]), color_mode="grayscale")
        image = np.array(image) 
        image = np.expand_dims(image,axis=-1)
    else:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(SHAPE[0], SHAPE[0]))
        image = np.array(image) 


        
    # image = transform.resize(image, (SHAPE[0], SHAPE[0]),
    #                             order=1, mode='constant',
    #                             cval=0, clip=True,
    #                             preserve_range=True,
    #                             anti_aliasing=True)
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
image_dir = pathlib.Path("../Data/ISIC2018/Training/ISIC2018_Task1-2_Training_Input")
image_paths = list(image_dir.glob('*.jpg'))
image_paths = [str(path) for path in image_paths]


mask_dir = pathlib.Path("../Data/ISIC2018/Training/ISIC2018_Task1_Training_GroundTruth")
mask_paths = list(mask_dir.glob('*.png'))
mask_paths = [str(path) for path in mask_paths]




images, masks = load(image_paths, mask_paths)

# images = images[:10]
# masks = masks[:10]
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

adam = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
model = sm.Unet('resnet34', encoder_weights='imagenet', input_shape=SHAPE, classes=1)
model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=[jaccard_loss, jaccard_index, dice_coeff, pixelwise_specificity, pixelwise_sensitivity, pixelwise_accuracy])

save_path = './models/model1.hdf5'
checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_jaccard_index', save_best_only=True, verbose=1)


model.fit_generator(
    generator=train, 
    steps_per_epoch=int(np.ceil(num_train / float(BATCH_SIZE))),
    epochs=EPOCHS, 
    validation_data=(x_val, y_val),
    callbacks=[checkpoint]
    )
