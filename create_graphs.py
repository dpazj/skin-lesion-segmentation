import pickle
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

pickle_names = ["Loss1", "Loss2", "Loss3", "Loss4", "Loss5", "Loss6"]#"5","6","7","8","10","u1","u2","u3","u4", "u5"]

label_names = ["BCE", "Jaccard Loss", "Dice Loss", "Weighted alpha = 0.5", "Weighted alpha = 0.4", "Weighted alpha = 0.6"]
#label_names = ["U-Net", "U-Net++ - VGG16", "U-Net++ - VGG19", "U-Net++ - ResNet50", "U-Net++ - ResNet101", "U-Net++ - ResNet152", "U-Net - VGG16", "U-Net - ResNet34", "U-Net - ResNet50", "U-Net - ResNet101", "U-Net - ResNet152"]

idx = 0


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')
plt.figure(figsize=(8,8), dpi=100)

X_MIN = 0.0

tst = []



for x in pickle_names:


    with open('pickle/' + x + '.pickle', 'rb') as handle:
        results = pickle.load(handle)


    arry = results['accuracy'] 

    plt.hist(arry, density=True, bins=50,facecolor='white', edgecolor='black')
    density = stats.gaussian_kde(arry)
    x = np.arange(X_MIN, 1.0 + 0.001, 0.001)
    plt.plot(x, density(x), 'black')
    plt.title(label_names[idx], fontsize = 22)
    plt.xlabel('Accuracy', fontsize=22)
    plt.ylabel('Density', fontsize=22)
    plt.xticks(np.arange(X_MIN, 1.0 + 0.05, 0.1))
    plt.yticks(np.arange(0, 32, 1))
    plt.xlim(X_MIN, 1.05)
    plt.legend()
    plt.savefig('./Figures/' + label_names[idx] + "_acc.png")
    plt.clf()


    print( str('{} & {:.4f} $\pm$ {:.4f} & {:.4f} $\pm$ {:.4f} & {:.4f} $\pm$ {:.4f} & {:.4f} $\pm$ {:.4f} & {:.4f} $\pm$ {:.4f} \\\\').format( label_names[idx], 
                                np.mean(results['dice']), np.std(results['dice']),  
                                np.mean(results['jaccard']), np.std(results['jaccard']), 
                                np.mean(results['sensitivity']), np.std(results['sensitivity']), 
                                np.mean(results['specificity']), np.std(results['specificity']), 
                                np.mean(results['accuracy']), np.std(results['accuracy']), 
                                ))
    
    idx+=1
    



