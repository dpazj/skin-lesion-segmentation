import numpy as np
import os
import shutil

in_GT_folder = "../Data/ISIC2018/Training/ISIC2018_Task1_Training_GroundTruth/"
in_input_folder = "../Data/ISIC2018/Training/ISIC2018_Task1-2_Training_Input/"

out_GT_folder = '../Data/ISIC2018/EVAL/mask/'
out_input_folder = '../Data/ISIC2018/EVAL/input/'



files = [f for f in os.listdir(in_input_folder)]
# select 0.1 of the files randomly 
random_files = np.random.choice(files, 500, replace = False) # get 500 samples

for x in random_files:
    y = x.split('.')[0] + '_segmentation.png'
    print(x)
    print(y)
    shutil.move(in_GT_folder + y, out_GT_folder)
    shutil.move(in_input_folder + x, out_input_folder)

