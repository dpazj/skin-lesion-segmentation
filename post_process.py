import pydensecrf
import numpy as np
from config import *
from PIL import Image

import matplotlib.pyplot as plt

from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm


def post_process_mask(predictions, gaussian_sigma=2.0):
    #for x in range(predictions.shape[0]):
    for x in range(len(predictions)):
        predictions[x] = gaussian_filter(input=predictions[x], sigma=gaussian_sigma)
    return predictions



def post_process_crf(predictions,base_images):

    #TODO CREATE FROM LABELS

    new_predictions = []



    images = base_images * 255

    for prediction, base in tqdm(zip(predictions, images)):
        

        base = base.astype(np.uint8)
        

        d = dcrf.DenseCRF2D(SHAPE[0], SHAPE[1], 2)

    
        probs = np.stack([1-prediction, prediction])
        unary = unary_from_softmax(probs)
        
       
        d.setUnaryEnergy(unary)
        
        d.addPairwiseGaussian(sxy=1, compat=1)
        d.addPairwiseBilateral(sxy=5, srgb=1, rgbim=base, compat=1)

        Q = d.inference(10)

        res = np.argmax(Q, axis=0).reshape((SHAPE[0], SHAPE[1]))
        new_predictions.append(res)


    return new_predictions
        
