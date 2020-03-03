import tensorflow as tf
import functools
import numpy as np
from config import *  

import tensorflow_addons as tfa

import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

def process_paths(image_path, mask_path):
    image_str = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(image_str, channels = 3)

    mask_str = tf.io.read_file(mask_path)
    mask_img = tf.image.decode_png(mask_str)
    mask_img = mask_img[:,:,0]
    mask_img = tf.expand_dims(mask_img, axis=-1)
    return img, mask_img



def flip_img(image, mask, horizontal_flip = True):
    do_flip = tf.random.uniform([]) > 0.5

    if horizontal_flip:
        image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
        mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
    else:
        image = tf.cond(do_flip, lambda: tf.image.flip_up_down(image), lambda: image)
        mask = tf.cond(do_flip, lambda: tf.image.flip_up_down(mask), lambda: mask)
    return image, mask

def shift_img(image, mask, width_shift_range, height_shift_range):
  
    if width_shift_range:
        width_shift_range = tf.random.uniform([], 
                                            -width_shift_range * SHAPE[1],
                                            width_shift_range * SHAPE[1])
    if height_shift_range:
        height_shift_range = tf.random.uniform([],
                                            -height_shift_range * SHAPE[0],
                                            height_shift_range * SHAPE[0])
      # Translate both 
    image = tfa.image.translate(image, [width_shift_range, height_shift_range])
    mask = tfa.image.translate(mask, [width_shift_range, height_shift_range])
    return image, mask


def hue(image):
    return tf.image.random_hue(image,HUE_DELTA)


def rot_image(image, mask):
    angle = tf.random.uniform(shape=[], minval=0, maxval=40 * (np.pi/180) , seed=42)
    image = tfa.image.rotate(image, angle)
    mask = tfa.image.rotate(mask, angle)
    return image, mask

def brightness(image):
    return tf.image.random_brightness(image,BRIGHTNESS_DELTA)

def augment(image, mask, resize=None, scale=1, validate=False):

    if resize is not None:
        image = tf.image.resize(image, resize)
        mask = tf.image.resize(mask, resize)

    image = tf.dtypes.cast(image, tf.float32) * scale
    mask = tf.dtypes.cast(mask, tf.float32) * scale 

    if validate:
        return image,mask

    image = hue(image)
    image = brightness(image)

    image, mask = flip_img(image, mask)
    image, mask = flip_img(image, mask, False)
    image, mask = shift_img(image,mask,WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE)
    image, mask = rot_image(image,mask)


   
    return image, mask


def create_dataset(image_paths, mask_paths, preprocess_fn=functools.partial(augment), batch_size=3, shuffle=True):
    length = len(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    dataset = dataset.map(process_paths, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=AUTOTUNE)

    # if shuffle:
    #     dataset = dataset.shuffle(length)

    dataset = dataset.repeat().batch(batch_size)


    return dataset