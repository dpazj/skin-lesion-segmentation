import pydensecrf
import numpy as np
from config import *
from PIL import Image

import matplotlib.pyplot as plt

from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from skimage.morphology import label


def post_process_mask(predictions, gaussian_sigma=2.0):
    #for x in range(predictions.shape[0]):
    for x in range(len(predictions)):
        predictions[x] = gaussian_filter(input=predictions[x], sigma=gaussian_sigma)
    return predictions


def clean_crf(prediction):

    labels, nlabels = label(prediction, return_num=True, background=-1)
    max_bg_idx = -1
    max_bg = 0

    max_l_idx = -1
    max_l = 0

    
    for i in range(0, nlabels+1):

        size = np.sum(labels == i)

        #if background
        if np.sum(prediction[labels == i]) == 0:
            if size > max_bg:
                max_bg = size
                max_bg_idx = i              
        else:
           if size > max_l:
                max_l = size
                max_l_idx = i     

    if max_bg_idx == -1 or max_l_idx == -1:
        return prediction
        
    for i in range(0, nlabels+1):
        if np.sum(prediction[labels == i]) == 0:
            if i != max_bg_idx:
                prediction[labels == i] = 1
        else:
            if i != max_l_idx:
                prediction[labels == i] = 0

    return prediction


def post_process_crf(predictions,base_images, clean=False):

    #TODO CREATE FROM LABELS

    new_predictions = []




    images = base_images * 255

    for prediction, base in tqdm(zip(predictions, images)):
        

        base = base.astype(np.uint8)
        

        d = dcrf.DenseCRF2D(SHAPE[0], SHAPE[1], 2)

    
        probs = np.stack([1-prediction, prediction])
        unary = unary_from_softmax(probs)
        
        d.setUnaryEnergy(unary)
        
        d.addPairwiseGaussian(sxy=10, compat=3)
        d.addPairwiseBilateral(sxy=30, srgb=15, rgbim=base, compat=10)

        Q = d.inference(5)

        res = np.argmax(Q, axis=0).reshape((SHAPE[0], SHAPE[1]))

        if clean:
            res =clean_crf(res)

        new_predictions.append(res)



    return new_predictions
        
