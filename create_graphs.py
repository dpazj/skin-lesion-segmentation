import pickle
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

pickle_names = ["Loss1", "Loss2","Loss3","Loss4","Loss5","Loss6"]

label_names = ["Binary Cross entropy", "Jaccard Loss", "Dice Loss", "Weighted: alpha = 1 beta = 1", "Weighted: alpha = 0.4 beta = 0.6", "Weighted: alpha = 0.6 beta = 0.4"]

idx = 0


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.figure(figsize=(8,8), dpi=100)

X_MIN = 0.0

tst = []
for x in pickle_names:

    with open('pickle/' + x + '.pickle', 'rb') as handle:
        results = pickle.load(handle)

    # density = stats.gaussian_kde(results['jaccard'])
    # x = np.arange(X_MIN, 1.0 + 0.001, 0.001)

    # plt.plot(x, density(x), label = label_names[idx])
    sns.distplot(results['jaccard'], hist = False, kde = True, kde_kws = {'linewidth': 1.0, "clip":(0,1.0)}, label = label_names[idx])
    
    idx+=1
    print(np.std(results['jaccard']))
print('\n')


plt.xlabel('Jaccard Index')
plt.ylabel('Density')
plt.xticks(np.arange(X_MIN, 1.0 + 0.05, 0.05))
plt.xlim(X_MIN, 1.05)
plt.legend()
plt.show()


plt.figure(figsize=(8,8), dpi=100)
idx = 0
for x in pickle_names:

    with open('pickle/' + x + '.pickle', 'rb') as handle:
        results = pickle.load(handle)

    sns.distplot(results['dice'], hist = False, kde = True, kde_kws = {'linewidth': 1.0, "clip":(0,1.0)}, label = label_names[idx])
    idx+=1

    print(np.std(results['dice']))
print('\n')
plt.xlabel('Dice Coefficient')
plt.ylabel('Density')
plt.xticks(np.arange(X_MIN, 1.0 + 0.05, 0.05))
plt.xlim(X_MIN, 1.05)
plt.legend()
plt.show()  



plt.figure(figsize=(8,8), dpi=100)
idx = 0
for x in pickle_names:

    with open('pickle/' + x + '.pickle', 'rb') as handle:
        results = pickle.load(handle)

    sns.distplot(results['accuracy'], hist = False, kde = True, kde_kws = {'linewidth': 1.0, "clip":(0,1.0)}, label = label_names[idx])
    idx+=1

    print(np.std(results['accuracy']))
print('\n')
plt.xlabel('Accuracy')
plt.ylabel('Density')
plt.xticks(np.arange(X_MIN, 1.0 + 0.05, 0.05))
plt.xlim(X_MIN, 1.05)
plt.legend()
plt.show()  



