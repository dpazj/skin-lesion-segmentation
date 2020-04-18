from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.python.keras import models
from PIL import Image
from config import * 
from metrics import * 
from skimage.transform import resize 
from tqdm import tqdm

from metrics import * 
from post_process import * 

MODEL_PATH = "./models/" 
ENSEMBLE_MODELS = ["kfmodel0","kfmodel1","kfmodel2","kfmodel3","kfmodel4","kfmodel_0","kfmodel_1","kfmodel_2","kfmodel_3","kfmodel_4","kfmodel1_0","kfmodel1_1","kfmodel1_2","kfmodel1_3","kfmodel1_4"] #["VGG16UNET-EX2","RES34-UNET-EX3","UNETPP-RESNET50-EX8", "UNETPP-VGG16-EX9", "NestNET-RESNET50-EX10", "NestNET-VGG16-EX11", "NESTNET-EX12"] #
OUTPUT_DIR = "./predictions_out/"

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

    #1000 images
    nppathim = 'npsave/images_test.npy'
    nppathsizes = 'npsave/images_test_sizes.npy'

    #10 tes images
    # nppathim = 'npsave/images_testT.npy'
    # nppathsizes = 'npsave/images_test_sizesT.npy'

    if os.path.exists(nppathim) and os.path.exists("npsave/images_test_sizes.npy"):
        images = np.load(nppathim)
        sizes = np.load(nppathsizes)
    else:
        images, sizes = load_images(image_paths)
        images = np.stack(images).astype(np.uint8)

        np.save(nppathim, images)
        np.save(nppathsizes, sizes)


    images = images * SCALE


    return images, sizes

image_dir = pathlib.Path("../Data/ISIC2018/Test/ISIC2018_Task1-2_Test_Input")
image_paths = list(image_dir.glob('*.jpg'))
image_paths = [str(path) for path in image_paths]





images, sizes = load(image_paths)

# image_paths = image_paths[:20] #FOR TESTING
# images = images[:20]

image_number = len(images)
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

predictions = post_process_crf(predictions, images)
#predictions = post_process_mask(predictions, 1.0)


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


print("Creating images")

for i_image, path in enumerate(tqdm(image_paths)):
    
    base = os.path.basename("../Data/ISIC2018/Test/ISIC2018_Task1-2_Test_Input"+path)
    image_name = os.path.splitext(base)[0]

    current_pred = predictions[i_image]
    current_pred = current_pred * 255

    resized_pred = resize(current_pred, output_shape=sizes[i_image],preserve_range=True,mode='reflect', anti_aliasing=True)



    threshold = 0.5 * 255
    resized_pred[resized_pred > threshold] = 255
    resized_pred[resized_pred <= threshold] = 0


    img = Image.fromarray(resized_pred.astype(np.uint8))

    img.save(OUTPUT_DIR + '/' + image_name + '_segmentation.png')
