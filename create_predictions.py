from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
from unet import * 
from config import * 
from metrics import * 
from skimage.transform import resize 
from tqdm import tqdm

MODEL_PATH = "./models/"
ENSEMBLE_MODELS = ["NestNET-RESNET50-EX10.hdf5", "VGG16UNET-EX2.hdf5" , "RES34-UNET-EX3.hdf5" , "UNETPP-RESNET50-EX8.hdf5", "UNETPP-VGG16-EX9.hdf5"]
OUTPUT_DIR = "./test_out/"

def inv_sigmoid(x):
    eps = np.finfo(np.float32).eps
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1. - x))

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def load_image_by_pathname(image_path):
    image_size = None
   
    image = tf.keras.preprocessing.image.load_img(image_path)
    width, height = image.size
    image_size = (width, height)
    image = image.resize((SHAPE[0], SHAPE[0]))
    image = np.array(image) 

    image = image.astype(np.uint8)

    return image, image_size
    

def load_images(paths):
    images = []
    image_sizes = []
    for path in tqdm(paths):
        image, size = load_image_by_pathname(path)
        images.append(image)

        if size is not None:
            image_sizes.append(image_sizes)

    return images, image_sizes
    

def load(image_paths):

    images = []
    sizes = []
    if os.path.exists("npsave/images_test.npy") and os.path.exists("npsave/images_test_sizes.npy"):
        images = np.load('npsave/images_test.npy')
        sizes = np.load('npsave/images_test_sizes.npy')
    else:
        images, sizes = load_images(image_paths)
        images = np.stack(images).astype(np.uint8)
        sizes = np.stack(sizes)

        np.save('npsave/images_test.npy', images)
        np.save('npsave/images_test_sizes.npy', images)


    images = images * SCALE


    return images, sizes

image_dir = pathlib.Path("../Data/ISIC2018/ISIC2018/Test/ISIC2018_Task1-2_Test_Input")
image_paths = list(image_dir.glob('*.jpg'))
image_paths = [str(path) for path in image_paths]


image_number = len(image_paths)
images, sizes = load(image_paths)

predictions = np.zeros(shape=(image_number, SHAPE[0], SHAPE[0]))
for model_name in ENSEMBLE_MODELS:
    model = models.load_model(MODEL_PATH + model_name, custom_objects={
        'bce_jaccard_loss':bce_jaccard_loss, 
        'bce_dice_loss':bce_dice_loss, 
        'jaccard_loss':jaccard_loss,
        'jaccard_index': jaccard_index, 
        'dice_loss' : dice_loss, 
        'dice_coeff' : dice_coeff,
        'pixelwise_specificity' : pixelwise_specificity,
        'pixelwise_sensitivity' : pixelwise_sensitivity,
        'pixelwise_accuracy' : pixelwise_accuracy
    })

    predictions += inv_sigmoid(model.predict(images))
    

predictions = predictions / len(ENSEMBLE_MODELS) 
predictions = sigmoid(predictions)

#predictions = post_process(predictions, threshold=0.6)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)




for i_image, path in enumerate(image_paths):
    base = os.path.basename(image_dir+path)
    image_name = os.path.splitext(base)[0]

    current_pred = predictions[i_image]
    current_pred = current_pred * 255

    resized_pred = resize(current_pred, output_shape=sizes[i_image],preserve_range=True,mode='reflect', anti_aliasing=True)

    resized_pred[resized_pred > 128] = 255
    resized_pred[resized_pred <= 128] = 0

    img = Image.fromarray(resized_pred.astype(np.uint8))
    img.save(OUTPUT_DIR + '/' + image_name + '_segmentation.png')
