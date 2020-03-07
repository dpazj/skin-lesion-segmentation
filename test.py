from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from unet import * 
from config import * 
from metrics import * 
#from pretrained_backbone_unet import * 

MODEL_PATH = "./models/UNETPP-RES50.hdf5"


def post_process(predictions):
  THRESHOLD = 0.5
  for i in range(50):
    predictions[i] = np.round(predictions[i])

  return predictions

def jaccard(y_true, y_pred):
    intersect = np.sum(y_true * y_pred) 
    union = np.sum(y_true) + np.sum(y_pred) 
    return (float(intersect))/(union - intersect)

def dice(y_true, y_pred):
  intersect = np.sum(y_true * y_pred) 
  union = np.sum(y_true) + np.sum(y_pred) 
  return (2. * intersect) / union

def pix_sensitivity(y_true, y_pred):
  true = np.sum(y_true * y_pred)
  total = np.sum(y_true)
  return true/total

def pix_specificity(y_true, y_pred):
  true = np.sum((1 - y_true) * (1-y_pred))
  total = np.sum(1 - y_true)
  return true/total

def pix_accuracy(y_true, y_pred):
  true = np.sum(y_true * y_pred)
  total = np.sum(y_pred)
  return true/total


def process_paths(image_path, mask_path):
  image_str = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(image_str, channels = 3)

  mask_str = tf.io.read_file(mask_path)
  mask_img = tf.image.decode_png(mask_str)
  mask_img = mask_img[:,:,0]
  mask_img = tf.expand_dims(mask_img, axis=-1)
  return img, mask_img

def augment(image, mask, resize=None, scale=1, validate=False):

    if resize is not None:
        image = tf.image.resize(image, resize)
        mask = tf.image.resize(mask, resize)

    image = tf.dtypes.cast(image, tf.float32) * scale
    mask = tf.dtypes.cast(mask, tf.float32) * scale 

    
    return image,mask

   

image_dir = pathlib.Path("../Data/ISIC2018/VAL/input")
image_paths = list(image_dir.glob('*.jpg'))
image_paths = [str(path) for path in image_paths]

mask_dir = pathlib.Path("../Data/ISIC2018/VAL/mask")
mask_paths = list(mask_dir.glob('*.png'))
mask_paths = [str(path) for path in mask_paths]



images = []
masks = []
for i in range(len(image_paths)):
    image, mask = process_paths(image_paths[i], mask_paths[i])
    image, mask = augment(image, mask, resize=[SHAPE[0], SHAPE[0]], scale=SCALE, validate=True)
    images.append(image)
    masks.append(mask)



model = models.load_model(MODEL_PATH, custom_objects={
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

images = np.squeeze(tf.expand_dims(images, axis=0))
masks = tf.expand_dims(masks, axis=0)
masks = np.reshape(masks, masks.shape[1:])

predictions = model.predict(images)
predictions = post_process(predictions)
print(predictions.shape)



idx = 0
mean_jaccard = 0.
mean_dice = 0.
sensitivity = 0.
specificity = 0.
accuracy = 0.

thresholded_jaccard = 0.

for x in predictions:
  score = jaccard(masks[idx], x)
  
  sensitivity += pix_sensitivity(masks[idx], x)
  specificity += pix_specificity(masks[idx], x)
  accuracy += pix_accuracy(masks[idx], x)
  

  mean_dice += dice(masks[idx], x)
  mean_jaccard += score
  if score >= 0.65:
    thresholded_jaccard += score
  idx += 1

print(idx)

mean_jaccard = mean_jaccard/idx
mean_dice = mean_dice/idx
specificity = specificity/idx
sensitivity = sensitivity/idx
accuracy = accuracy/idx

thresholded_jaccard = thresholded_jaccard/idx

print('Mean Dice = %.4f, Mean jaccard = %.4f, Thresholded Jaccard = %.4f ' % (mean_dice,mean_jaccard, thresholded_jaccard))
print('Pixelwise Specificity = %.4f, Pixelwise Sensitivity = %.4f, Accuracy = %.4f'  % (specificity, sensitivity, accuracy))



for j in range(10):
  plt.figure(figsize=(10, 20))

  index = j * 5

  for i in range(5):
    img = tf.expand_dims(images[index + i], axis=0)  
    predicted_label = predictions[index + i]

    plt.subplot(5, 3, 3 * i + 1)
    plt.imshow(images[index + i])
    plt.title("Input image")
    
    plt.subplot(5, 3, 3 * i + 2)
    plt.imshow(masks[index + i][:, :, 0])
    plt.title("Actual Mask")

    plt.subplot(5, 3, 3 * i + 3)
    plt.imshow(predicted_label[:, :, 0])
    plt.title("Predicted Mask")

  plt.suptitle("Examples of Input Image, Label, and Prediction")
  plt.show()