#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 09:35:15 2020

@author: Si
"""

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import os
from functions import spectrograph



### USER DEFINED

# set the number k of nearest neighours to find
k = 1000


#############

if os.path.isdir('./models') == False:
    print(f'Creating ./models')
    os.mkdir('./models')    


print('Importing data...')


joblib_save_name = './models/%s_KNNClassification_model.k=%s.joblib' %(spectrograph, k)
pca_folder = './PCA_ready/'
X = np.loadtxt(pca_folder + 'PCAReady.%s.txt' %spectrograph )
y = np.loadtxt(pca_folder + 'classlist.%s.txt' %spectrograph, dtype='str')


X_train = X
y_train = y

# print('Performing pca')
# pca=PCA(n_components=0.95)

# pca.fit(X_train)


# # now find the principal components
# X_reduced =pca.fit_transform(X_train)

# # This reconstructs the original data from the pca transform
# X_return = pca.inverse_transform(X_reduced)

# score = np.mean(cross_val_score(pca, X_reduced))
# print(score)

# plt.plot(np.arange(1, pca.n_components_ + 1),
#           pca.explained_variance_ratio_, '+', linewidth=2)
# plt.yscale('log')
# plt.show()




print('Fitting model')
clf = make_pipeline( StandardScaler(), PCA(n_components = 0.95),
                    KNeighborsClassifier(n_neighbors = k, weights = 'distance') )
print(clf)
clf.fit(X_train,y_train)



# clf_nn = make_pipeline( StandardScaler(), PCA(n_components=0.95),
#                     NearestNeighbors(n_neighbors=k) )
# print(clf_nn)
# clf_nn.fit(X_train,y_train)

# pca_pipe = make_pipeline( StandardScaler(), PCA(n_components=0.95) )
# pca_pipe.fit(X_train)

# X_pca = pca_pipe.fit_transform(X_train)

# model = KNeighborsClassifier(n_neighbors=k, weights = 'distance')
# model.fit(X_pca, y_train)



from joblib import dump
dump(clf, joblib_save_name) 


if os.path.isdir('./for_classification') == False:
    print(f'Creating ./for_classification')
    os.mkdir('./for_classification')    

#dump(clf_nn, 'NN_Classification_model.k=%s.joblib' %(k)) 
# dump(model, 'KNNmodel.k=%s.joblib' %(k)) 
# dump(pca_pipe, 'PCAfit.k=%s.joblib' %(k)) 


