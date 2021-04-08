#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 09:35:15 2020

@author: SPrentice
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
#spectrograph  = '2021-03-04'

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


print('Fitting model')
clf = make_pipeline( StandardScaler(), PCA(n_components = 0.95),
                    KNeighborsClassifier(n_neighbors = k, weights = 'distance') )
print(clf)
clf.fit(X_train,y_train)



from joblib import dump
dump(clf, joblib_save_name) 


if os.path.isdir('./for_classification') == False:
    print(f'Creating ./for_classification')
    os.mkdir('./for_classification')    



