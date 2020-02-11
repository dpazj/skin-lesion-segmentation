import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as K 


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def bce_jaccard_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + jaccard_loss(y_true, y_pred)
    return loss

def jaccard_loss(y_true, y_pred):
    loss = 1 - jaccard_index(y_true,y_pred)
    return loss

def jaccard_index(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    jac = (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
    return jac
