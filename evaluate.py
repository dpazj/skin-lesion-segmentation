from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from tensorflow.python.keras import models
from PIL import Image
from config import * 
from metrics import * 
from skimage.transform import resize 
from tqdm import tqdm


from metrics import * 
from post_process import * 

import time

now = int(time.time()) 


THRESH_HOLD_CUTOFF = 0.5
MODEL_PATH = "./models/" 
ENSEMBLE_MODELS = ["5","6","7","8","9"]
OUTPUT_DIR = "./test_out/"

CRF_POST_PROCESS = False
CRF_CLEAN = False

GAUSSIAN_FILTER_POST_PROCESS = False
GAUSSIAN_SIGMA = 1.0

EVAL_INPUT_PATH = "../Data/ISIC2018/EVAL/input"
EVAL_MASK_PATH = "../Data/ISIC2018/EVAL/mask"


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
  return float(true)/total

def pix_specificity(y_true, y_pred):
  true = np.sum((1 - y_true) * (1-y_pred))
  total = np.sum(1 - y_true)
  return float(true)/total

def pix_accuracy(y_true, y_pred):
  return np.mean(np.equal(y_true, y_pred))


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
    image_size = (height, width)
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
            image_sizes.append(size)

    return images, image_sizes
    

def load(image_paths):

    images = []
    sizes = []

    nppathim = 'npsave/images_eval.npy'
    nppathsizes = 'npsave/images_eval_sizes.npy'

    #10 tes images
    # nppathim = 'npsave/images_testT.npy'
    # nppathsizes = 'npsave/images_test_sizesT.npy'

    if os.path.exists(nppathim) and os.path.exists(nppathsizes):
        images = np.load(nppathim)
        sizes = np.load(nppathsizes)
    else:
        images, sizes = load_images(image_paths)
        images = np.stack(images).astype(np.uint8)

        np.save(nppathim, images)
        np.save(nppathsizes, sizes)


    images = images * SCALE


    return images, sizes

image_dir = pathlib.Path(INPUT_PATH)
image_paths = list(image_dir.glob('*.jpg'))
image_paths = [str(path) for path in image_paths]

images, sizes = load(image_paths)

# image_paths = image_paths[:20] #FOR TESTING
# images = images[:20]

image_number = len(image_paths)
print("Images loaded")

predictions = np.zeros(shape=(image_number, SHAPE[0], SHAPE[0],1))

print("Loading Models")

for model_name in ENSEMBLE_MODELS:

    path = MODEL_PATH + model_name + '.hdf5'
    print(path)
    model = models.load_model(path, custom_objects={
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

    predictions += inv_sigmoid(model.predict(images)) #Logits 
    

predictions = predictions / len(ENSEMBLE_MODELS) 
predictions = sigmoid(predictions)

predictions = np.squeeze(predictions)

print("Post Processing")
if CRF_POST_PROCESS: 
    predictions = post_process_crf(predictions, images, CRF_CLEAN)

if GAUSSIAN_FILTER_POST_PROCESS:
    predictions = post_process_mask(predictions, GAUSSIAN_SIGMA)

   
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

np.set_printoptions(threshold=sys.maxsize)


mean_jaccard = 0.
mean_dice = 0.
mean_sensitivity = 0.
mean_specificity = 0.
mean_accuracy = 0.

idx = len(image_paths)

print("Loading GT Masks:")


results = {
    'dice' : [],
    'jaccard' : [],
    'sensitivity' : [],
    'specificity' : [],
    'accuracy' : []
}


gt_masks = []
for i, path in enumerate(tqdm(image_paths)):
    base = os.path.basename(EVAL_INPUT_PATH+path)
    image_name = os.path.splitext(base)[0]
    gt_mask = tf.keras.preprocessing.image.load_img(EVAL_MASK_PATH + '/' + image_name + '_segmentation.png', color_mode="grayscale")
    gt_mask = np.array(gt_mask)
    gt_mask = gt_mask.astype(np.float32) * 1/255
    gt_masks.append(gt_mask)


print("Evaluating...")

for i_image, path in enumerate(tqdm(image_paths, miniters=1)):
    
    base = os.path.basename(INPUT_PATH+path)
    image_name = os.path.splitext(base)[0]

    pred = predictions[i_image] * 255
    
    pred  = resize(pred , output_shape=sizes[i_image],preserve_range=True,mode='reflect', anti_aliasing=True)

    threshold = THRESH_HOLD_CUTOFF * 255

    pred[pred > threshold] = 255
    pred[pred <= threshold] = 0

    # img = Image.fromarray(pred.astype(np.uint8))
    # img.save("TEST_CRF" + '/' + str(i_image) + '_crf.png')    

    pred = pred.astype(np.float32) * 1/255

    j = jaccard(gt_masks[i_image],pred)
    d = dice(gt_masks[i_image],pred)
    sen = pix_sensitivity(gt_masks[i_image],pred)
    spec = pix_specificity(gt_masks[i_image],pred)
    acc = pix_accuracy(gt_masks[i_image],pred)

    mean_jaccard += j
    mean_dice += d
    mean_sensitivity += sen
    mean_specificity += spec
    mean_accuracy += acc

    results['dice'].append(d)
    results['jaccard'].append(j)
    results['sensitivity'].append(sen)
    results['specificity'].append(spec)
    results['accuracy'].append(acc)
   


mean_jaccard = mean_jaccard/idx
mean_dice = mean_dice/idx
mean_specificity = mean_specificity/idx
mean_sensitivity = mean_sensitivity/idx
mean_accuracy = mean_accuracy/idx

print('Mean Dice = %.4f, Mean jaccard = %.4f, Pixelwise Specificity = %.4f, Pixelwise Sensitivity = %.4f, Accuracy = %.4f' % (mean_dice,mean_jaccard, mean_specificity, mean_sensitivity, mean_accuracy))
print('%.4f & %.4f &, %.4f & %.4f & %.4f' % (np.std(results['jaccard']),np.std(results['dice']), np.std(results['sensitivity']), np.std(results['specificity']), np.std(results['accuracy'])))

with open('pickle/tmp' + str(now) + '.pickle','wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)



  
