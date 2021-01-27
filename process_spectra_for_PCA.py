#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:17:38 2021

@author: Si
"""

from functions import ProcessSpectra, sn_types, target_wavelength_range, spectrograph, z_range
import numpy as np
import os
from glob import glob

### USER DEFINED




################################



# Set folders
pca_folder = './PCA_ready/'
files = glob('./spectra/*.txt' )
pca_ready_save = pca_folder + 'PCAReady.%s.txt' %spectrograph
classification_list = pca_folder + 'classlist.%s.txt' %spectrograph


if os.path.isdir(pca_folder) == False:
    print(f'Creating {pca_folder}')
    os.mkdir(pca_folder)
    
    
    
    
main=[]
lab =[]
count =0
for f in files:
    count  +=1

    base = os.path.basename(f)
    print('%s/%s Processing %s ' %(count,len(files),base))
    
    x0, y0 = np.loadtxt(f,unpack=True,usecols=(0,1))
    
    if np.all(x0[1:] >= x0[:-1], axis=0) != False:
            
        # this shouldn't process the spectra, instead we'll pass this to
        # the wavelength range checker next and then process the spectrum
        x,y = x0, y0 #ProcessSpectra(x=x0,y=y0,wave_bins=8)


        
        for z in z_range:
            x_z = x0 * (1 + z)
            
            if (x_z[0] < target_wavelength_range[0] - 50) and (x_z[-1] > target_wavelength_range[1] + 50):
                #print('%s/%s Processing %s at z = %.3f' %(count,len(files),base,z))
                
                # cut extraneous stuff
                lowcut = 0#np.argmin(abs( x_z - (target_wavelength_range[0] - 50)) )
                hicut = np.argmin(abs( x_z - (target_wavelength_range[1] + 50)) )
                
                x_z = x_z[lowcut:hicut]
                y_z = y0[lowcut:hicut]
                #print(min(x_z), max(x_z))
                x_z, y = ProcessSpectra(x = x_z, y = y_z ,wave_bins=8)
                

                if str(min(y))[0] in '0123456789':
                    
                    main.append(y)
                    lab.append('%.3f_%s'%(z,base))
                else:
                    print(base, 'NaN detected',str(min(y))[0], z)
                    
                # if reguired uncomment to save the individual spectra    
                #m=list(zip(x_z,y))
                #np.savetxt('/Users/Si/Documents/searn/SupernovaClassifier/processedspectra/%.3f_%s'%(z,base),m,fmt="%s")

np.savetxt(pca_ready_save,main,fmt="%s")  
np.savetxt(classification_list,lab,"%s")     