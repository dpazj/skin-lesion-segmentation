import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K 


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def jaccard_index(y_true, y_pred):
    y_true_f = K.round(y_true)
    y_pred_f = K.round(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=[1, 2, 3])
    union = K.sum(K.abs(y_true_f) + K.abs(y_pred_f), axis=[1, 2, 3])
    iou = intersection / K.clip(union - intersection, K.epsilon(), None)
    return iou




