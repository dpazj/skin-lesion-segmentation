import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import backend as backend


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
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
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    jac = (intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) - intersection + smooth)
    return jac


#https://en.wikipedia.org/wiki/Sensitivity_and_specificity

def pixelwise_specificity(y_true, y_pred):
    true = backend.sum(backend.abs((1. - y_true) * (1. - y_pred)), axis=[1, 2, 3])
    total = backend.sum(backend.abs(1. - y_true), axis=[1, 2, 3])
    return true / backend.clip(total, backend.epsilon(), None)

def pixelwise_sensitivity(y_true, y_pred):
    y_true = backend.round(y_true)
    true = backend.sum(backend.abs(y_true * y_pred), axis=[1, 2, 3])
    total = backend.sum(backend.abs(y_true), axis=[1, 2, 3])
    return true / backend.clip(total, backend.epsilon(), None)


def pixelwise_accuracy(y_true, y_pred):
    true = backend.sum(backend.abs(y_true * y_pred), axis=[1, 2, 3])
    total= backend.sum(backend.abs(y_pred), axis=[1, 2, 3])
    return true / backend.clip(total, backend.epsilon(), None)
