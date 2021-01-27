#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:41:36 2021

@author: Si
"""

from functions import *
from glob import glob
import os
import matplotlib.pyplot as plt

### User defined

location = './json_files/'
savelocation = '/Users/Si/Desktop/SN_Classifier/new/'
spec_saveloc = './spectra/'
tsv_save = './tsv_files'


####
if os.path.isdir(location) == False:
    print(f'Creating {location}')
    os.mkdir(location)

if os.path.isdir(spec_saveloc) == False:
    print(f'Creating {spec_saveloc}')
    os.mkdir(spec_saveloc)
    
if os.path.isdir(tsv_save) == False:
    print(f'Creating {tsv_save}')
    os.mkdir(tsv_save)    



files = glob(location+'*.json')
band_to_use = 'R'
colors = {'r':'red', 'R':'red','I':'yellow','i':'yellow'}
label = {'r':'r', 'R':'Rb','I':'Ib','i':'i'}
master_list =[]
for f in files:
    print(f'Processing{f}')
    name,data = openJSON(f)
    
    # get the SNe class
    # name_type = ''    
    # base =os.path.basename(f)
    # for j in range(len(base)):
    #     if base[j]!='.':
    #         name_type = name_type+base[j]
    #     else:
    #         break
        
    # set the redshift
    z = float(JSONz(location=f,name=name))
    
    sn_type = sn_types.get(name, 'UNK')

    #print(name)

    mjd, m = phot(data = data, bands=['R','r','rp'])
    
    if len(mjd) < 5:
        mjd, m = phot(data = data, bands=['V'])

    
    
    
    spec = spectra(data)
    
    if len(mjd)>0:
        tmax = mjd[ np.argmin(m)   ]

        for specmjd in spec:
            
            #interp_mag= estmag(lctime=mjd, lcmag = m, specmjd= float(specmjd),k=1,s=0)
            #print(specmjd,interp_mag)
            #interp_mag=17
            
            spec_time = (float(specmjd)-tmax)/(1.+z)
            
            wavelength,flux = np.array(spec[specmjd][0], dtype='float'),np.array(spec[specmjd][1], dtype='float')
            
            
            # if the wavelength axis is not inascending order            
            if wavelength[0]>wavelength[-1]:
                idx_sort = np.argsort(wavelength)
                flux = flux[idx_sort]
                wavelength.sort()
            
            # account for NaNs
            if str(min(flux))[0] not in '-0123456789':
                #print('Correcting for NaN',str(min(flux))[0])
                w1,f1=[],[]
                for j in range(len(flux)):
                    if str(flux[j])[0] in '-0123456789':
                        w1.append(wavelength[j])
                        f1.append(flux[j])
                wavelength,flux=np.array(w1),np.array(f1)            
            
            
            
            
            
            if (min(wavelength/(1+z)) < (target_wavelength_range[0] - 50 ) ) and (max(wavelength/(1+z)) > (target_wavelength_range[1] + 50) ):
                            
                # Once the redshift is accounted for and the spectrum in the
                # rest frame, we don't need anything outside of z of 0.1
                lowcut = np.argmin(abs( (wavelength / (1++z)) - (target_wavelength_range[0] / (1. + (max(z_range) +0.01 )) ) ) )
                hicut = np.argmin(abs( (wavelength / (1+z)) - (target_wavelength_range[1] + 50)) )
                            
                wavelength = wavelength[lowcut:hicut]
                flux = flux[lowcut:hicut]
                            
                flux = flux/ max(flux)
                            
                #flux = spike_reduction(y = flux)
                            
                if (float(specmjd) - tmax) / (1. + z) < 350:
                    savename=('%s_%s_%.2f_.txt'%(sn_type, name, spec_time))
                    #print(savename)
                                
                    spec_to_save = list(zip(wavelength,flux))
                                
                    np.savetxt(spec_saveloc + savename, spec_to_save, fmt="%s")
                                
                    plt.plot(wavelength, flux)
                    plt.savefig(spec_saveloc + savename +'.pdf')
                    plt.close()
                                # Note that R is outside the limits of some so we use 'V
                                # specname = name+'.'+label[band_to_use]+'.'+str(specmjd)+'.txt'
                                # print(specname)
                                # calib(x = wavelength, y=flux, 
                                #   specfile=specname, 
                                #   saveloc=spec_saveloc, 
                                #   band='r', mag=19, 
                                #   savename=('%s.%s.%.2f.txt'%(name_type,name,(float(specmjd)-tmax)/(1.+z))),
                                #   z=z
                                #   )
                    master_list.append([name, z, specmjd, spec_time, savename])
#                else:
#                    print('Rejecting %s min %.1f max %.1f'%(specmjd,min(wavelength/(1+z)),max(wavelength/(1+z))))

            
#    else:
#        print('LC length <1')
        
        
np.savetxt(tsv_save + '/spectra.'+label[band_to_use]+'.tsv', master_list, fmt="%s", delimiter = '\t', header = 'SN, z, obs-mjd, [t-t(rmax)]/z, filename')