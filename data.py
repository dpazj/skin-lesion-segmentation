import tensorflow as tf
import functools
from config import *  

AUTOTUNE = tf.data.experimental.AUTOTUNE

def process_paths(image_path, mask_path):
    image_str = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(image_str, channels = 3)

    mask_str = tf.io.read_file(mask_path)
    mask_img = tf.image.decode_png(mask_str)
    mask_img = mask_img[:,:,0]
    mask_img = tf.expand_dims(mask_img, axis=-1)
    return img, mask_img


def rotate_img(image, mask):
    # image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # mask = tf.image.rot90(mask, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image, mask

def flip_img(image, mask, horizontal_flip = True):
    # do_flip = tf.random.uniform([]) > 0.5

    # if horizontal_flip:
    #     image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
    #     mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
    # else:
    #     image = tf.cond(do_flip, lambda: tf.image.flip_up_down(image), lambda: image)
    #     mask = tf.cond(do_flip, lambda: tf.image.flip_up_down(mask), lambda: mask)
    return image, mask

def augment(image, mask, resize=None, scale=1):

    if resize is not None:
        image = tf.image.resize(image, resize)
        mask = tf.image.resize(mask, resize)

    image = tf.dtypes.cast(image, tf.float32) * scale
    mask = tf.dtypes.cast(mask, tf.float32) * scale 

    #TODO add more augmentations
   
    return image, mask


def create_dataset(image_paths, mask_paths, preprocess_fn=functools.partial(augment), batch_size=3, shuffle=True):
    length = len(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(process_paths, num_parallel_calls=AUTOTUNE)

    dataset = dataset.map(preprocess_fn, num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(length)
    dataset = dataset.repeat().batch(batch_size)
    return dataset