from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow_examples.models.pix2pix import pix2pix

import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data import * 
from model import * 
from config import * 
from metrics import * 

MODEL_PATH = "./models/model1.hdf5"


def resize(x,y):
    image = tf.image.resize(x, [SHAPE[0], SHAPE[0]])
    mask = tf.image.resize(y, [SHAPE[0], SHAPE[0]])
    image = tf.dtypes.cast(image, tf.float32) * SCALE
    mask = tf.dtypes.cast(mask, tf.float32) * SCALE 
    return image, mask

image_dir = pathlib.Path("../Data/ISIC2018/Training/ISIC2018_Task1-2_Training_Input")
image_paths = list(image_dir.glob('*.jpg'))
image_paths = [str(path) for path in image_paths]

mask_dir = pathlib.Path("../Data/ISIC2018/Training/ISIC2018_Task1_Training_GroundTruth")
mask_paths = list(mask_dir.glob('*.png'))
mask_paths = [str(path) for path in mask_paths]


image_paths = image_paths[:5]
mask_paths = mask_paths[:5]

images = []
masks = []
for i in range(len(image_paths)):
    image, mask = process_paths(image_paths[i], mask_paths[i])

    image = tf.image.resize(image, [SHAPE[0],SHAPE[0]])
    mask = tf.image.resize(mask, [SHAPE[0],SHAPE[0]])
    image = tf.dtypes.cast(image, tf.float32) * SCALE
    mask = tf.dtypes.cast(mask, tf.float32) * SCALE 

    images.append(image)
    masks.append(mask)


model = models.load_model(MODEL_PATH, custom_objects={'bce_jaccard_loss':bce_jaccard_loss, 'jaccard_index': jaccard_index, 'dice_loss' : dice_loss, 'dice_coeff' : dice_coeff})


plt.figure(figsize=(10, 20))
for i in range(5):
  
  img = tf.expand_dims(images[i], axis=0)  
  predicted_label = model.predict(img)[0]

  plt.subplot(5, 3, 3 * i + 1)
  plt.imshow(images[i])
  plt.title("Input image")
  
  plt.subplot(5, 3, 3 * i + 2)
  plt.imshow(masks[i][:, :, 0])
  plt.title("Actual Mask")

  plt.subplot(5, 3, 3 * i + 3)
  plt.imshow(predicted_label[:, :, 0])
  plt.title("Predicted Mask")

plt.suptitle("Examples of Input Image, Label, and Prediction")
plt.show()