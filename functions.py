#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:35:02 2021

@author: Si
"""
import numpy as np
import json
from scipy.interpolate import UnivariateSpline as spline
import os
import matplotlib.pyplot as plt



### User defined

# set the wavelength bounds of the target spectrograph
target_wavelength_range = [3700, 9000]

# This changes the name of the saved files in process_spectra_for_PCA.py
spectrograph = 'Gr13'

# the wavelength range and interval over which to produce the spectra
z_range = np.arange(0, 0.15, 0.002)

# load and set dictionary of object classifications
sn_types = {}
if os.path.isfile('./SNList.csv'):
    rows = np.loadtxt('./SNList.csv', delimiter = ',', dtype = 'str')
    sn_types = {}
    for sn, typ in rows:
        sn_types[sn] = typ
 	
    


###

def clip_lines(wav,flux, window_width= 340):
    '''
    Clips anrrow emission from 1D spectra.
    
    >>> CRReject(x,y, window=37)
    numpy.array(y)
    '''
    scale = np.median(flux)
    x,y = wav, flux/scale
    
    # set the window
    window = int(np.ceil(window_width/(x[1]-x[0])))
    
    for j in range(len(x)-window):
        region = y[j:j+window+1]
        
        magic_index = int(j+np.ceil(window/2))
        
        if y[magic_index] > (np.median(region) + 2.5*np.std(region)):
            
            y[magic_index] = np.median(region)
    
    # deal with the beginning
    region = y[:window+1]   
    for idx in range(len(region)):
        if y[idx] > (np.median(region) + 2.5*np.std(region)):
            y[idx] = np.median(region)
            
    # deal with the beginning
    region = y[j-window:]   
    for idx in range(j-window, len(region)):
        if y[idx] > (np.median(region) + 2.5*np.std(region)):
            y[idx] = np.median(region)        
            
    return y*scale 


def spike_reduction(y = [], sigma = 3 , window = 10 ):
    for i in range(len(y) - window):
        batch = y[i : i + window + 1]
        
        median, std = np.median(batch), np.std(batch)
        
        batch_bool = np.array(batch) > median + (std * sigma)
        
        idx = np.argmax(batch_bool)
        
        y[i + idx ] = median
        
        #if (y[i] > median + (std * sigma)) or (y[i] < median - (std * sigma)):
        #    #print(f'clipping {y[i]} to {median}')
        #    y[i] = median
            
    return y


def JSONz(location='',name=''):
    '''Extracts the redshift from the JSON file
    '''
    redshift=0
    with open(location, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())  

        for f in data:
            if data[f]['name'] == name:
                if 'redshift' in data[f]:  
                    z  = data[f]['redshift']
                    
                    
                    for l in z:
                        if 'kind' in l:
                            if 'host' == l['kind']:
                                #print (name,'host', l['value'])
                                redshift=l['value']

                    if redshift==0:
                            #print(name,    l['value'])
                            redshift=l['value']
                            
    return redshift
        
                
def openJSON(location):
    '''Opens a JSON file and accounts for ascii errors
    '''
    
    with open(location, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())  
       # print(len(data))
        
        for f in data:

            return data[f]['name'], data[f]
    
    
def spectra(data):
    '''Extracts the spectra from a json file.
    returns dictionary of spectra where the objects are named according
    to date and the lists are the spectrum
    
    Example:
    
    allspec ={53467:[ [4000,5000,6000],[5,6,5] ]}
    '''
    
    allspec={}
    if 'spectra' in data:
        s = data['spectra']
        
        for spectrum in s:
            headers=[]
            for header in spectrum:
                headers.append(header)
                if header=='time':
                    date = float(spectrum['time'])
                    while date in allspec:
                        date = date+0.01
                        
                    wave,flux = [],[]
                    for head in spectrum['data']:
                        wave.append(head[0])
                        flux.append(head[1])
                    allspec[date] =  [wave,flux]  
            #print(headers)
                  
    return allspec

    
        
                
def phot(data={}, bands = ['R']):
    '''Extracts the photometry of one band from a JSON file from the OSC
    Final lists are sorted and output given as two lists -- mjd and magnitude
    '''
    
    if 'photometry' in data:
        p = data['photometry']
        time,mag=[],[]
        for header in p:
            #print(header)
            if 'band' in header:
                if header['band'] in bands:
                    if 'upperlimit' not in header:
                        t = header['time']
                        m= header['magnitude']
                        
                        time_to_add = float(t)
                        
                        while time_to_add in time:
                            #print('Duplicate time found, adding small variation to mjd', time_to_add, '->', time_to_add+1e-3)
                            time_to_add = time_to_add+1e-4
                        
                        time.append(time_to_add)    

                        mag.append(float(m))
        #return sorted(zip(time,mag))
        if len(time)> 1:
            mjd,m= map(list,zip(*sorted(zip(time, mag))))

            return mjd, m
        else:

            return [],[]

def estmag(lctime =[], lcmag=[], specmjd=0, k=2, s=5):
    '''Fits a spline to the light curve to give an
    estimated magnitude at the time of the spectrum'''
    
    if len(lctime)<1:
        print('LC length <1')
        return None
    
    if specmjd > max(lctime)+25:
        print ('Spectrum date > last obs by', abs(specmjd-max(lctime)) )
        return None
    elif specmjd < min(lctime)-2:
        print ('Spectrum date < last obs by', abs(specmjd-min(lctime)) ) 
        return None
        
    else:
        fit = spline(lctime,lcmag, k=1,s=s)
        
        return fit(specmjd)


def ProcessSpectra(x=[],y=[],wave_bins=8, 
                   target_wavelength_range = target_wavelength_range, smoothed = True, order = 3):
    y=y / max(y)

    dx = x[1]-x[0]    

    
    # ensure we get one more than the total needed
    bins = int( np.ceil(wave_bins / dx) ) + 1
   
    wave = np.arange(target_wavelength_range[0], 
                     target_wavelength_range[1] + wave_bins, wave_bins)
    
    
    if smoothed: 
        # smooth with SG
        window = (6000./( 300000*  (x[1]-x[0]) / 6200 ))
        window_sg = np.ceil((window + 1)/2)*2 - 1
        if window_sg < 5:
            window_sg=5.
            #print (SN,'window_sg adjusted')
        y_smoothed = sg(y,window_sg,3,deriv=0)  
    
    else:
        y_smoothed = y

    
    y= [ np.median(y_smoothed[np.argmin(abs(x-w))-max(1,int(bins/2)) : np.argmin(abs(x-w))+max(1,int(bins/2))]) 
    for w in wave ]

    
    fit = spline(wave,y,k = order)
    y= y - fit(wave)

    ### Set the min to 0 and max to 1

    m=1/(max(y)-min(y))
    con=-m*min(y)    
    y = np.array([m*d+con for d in y])

    return wave,y


def sg(y, window_size, order, deriv=0):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')



####### For Classify_Supernovae
def BinSpec(x=[],y=[],wave_bin=5):
    y=y/max(y)
    fit = spline(x,y,k=2)
    #plt.plot(x,y)
    #plt.plot(x,fit(x))
    y= y-fit(x)
    
    
    delx = x[1]-x[0]
    bin = int(np.ceil(wave_bin/delx))+1
    
    
    n=0
    xlist,ylist=[],[]
    newx,newy=[],[]
    for k in range(len(x)-bin):
            xlist.append(x[k])
            ylist.append(y[k])
            
            if n <bin:                
                n+=1
                
            else:
                n=0               
                newx.append(np.mean(xlist))
                newy.append(np.median(ylist))
                
                xlist,ylist=[],[]
                
    x= np.array(newx)
    y = np.array(newy)
    
    locut = np.argmin(abs(x-3500))
    hicut= np.argmin(abs(x-8000))
    
    x= x[locut:hicut]
    y = y[locut:hicut]
    m=1/(max(y)-min(y))
    con=-m*min(y)    
    y = np.array([m*d+con for d in y])    
    #print('Wavelength binned from',delx,'to', x[1]-x[0], 'Angstroms')
    return x,y


#def split(string_list = [], z = False):
#    
#    types, sne, redshifts , epochs, full = [], [],[],[],[]
#    for string in string_list:
#        s = string.split('_')
#        
#        
#        redshift, typ, sn, epoch, _ = s
#        
#        fname = f'{typ}_{sn}_{epoch}_{_}'
#        
#        if z:
#            
#            if ( z - 0.003) < redshift < (z + 0.003):
#                
#                #typ, sn, epoch1, epoch2, _ = s[1].split('.')
#                types.append(typ)
#                sne.append(sn)
#                redshifts.append(float(redshift))
#                epochs.append( float(epoch))
#                full.append(fname)
#        
#        else:
#            #typ, sn, epoch1, epoch2, *_ = s[1].split('.')
#            types.append(typ)
#            sne.append(sn)
#            redshifts.append(float(redshift))
#            epochs.append( float(epoch))
#            full.append(fname)
#        
#    return list(zip(types,sne,redshifts,epochs, full))    



def split(string_list = [], z = False):
    
    types, sne, redshifts , epochs, full = [], [],[],[],[]
    for string in string_list:
        s = string.split('_')
             
        redshift, typ, sn, epoch, _ = s
        
        fname = f'{typ}_{sn}_{epoch}_{_}'
        
        types.append(typ)
        sne.append(sn)
        redshifts.append(float(redshift))
        epochs.append( float(epoch))
        full.append(fname)
    
    results = list(zip(types,sne,redshifts,epochs, full))
    
    if z:
        redshift_sorted_idxs = np.argsort(abs(np.array(redshifts) - z))
        results = [results[idx] for idx in redshift_sorted_idxs]
        
    return    results


def pad_spectrum(x, y, target_wavelength_range = target_wavelength_range):
    
    # get the waveength intervals
    dx = x[1] - x[0]
    
    if min(x) > min(target_wavelength_range):        
        
        pad_x = np.arange(min(target_wavelength_range) - dx * 5, min(x), dx  )
        pad_y = [ np.mean(y[0:30]) for n in range(len(pad_x))]
        
        x = np.concatenate((pad_x, x))
        y = np.concatenate((pad_y, y))
    
    
    if max(x) < max(target_wavelength_range):   
        pad_x = np.arange(max(x)+ dx, max(target_wavelength_range) + dx * 5, dx  )
        pad_y = [np.mean(y[-30:-1]) for n in range(len(pad_x))]
        
        x = np.concatenate((x, pad_x) )
        y = np.concatenate((y, pad_y))
   
    return x, y 

        
        
def get_classification(spectrum, clf, pad = False, E = False, clip=False):
    
    x,y = np.loadtxt(spectrum,unpack=True,usecols=(0,1))
    
    x = x / 1.0
    
    if clip:
        y = clip_lines(x, y)
        print('Clipping input spectrum')
    
    # if the spectrum need to be dereddened
    if E:
        y = dered(x, y, 3.1, E)
        
    # if the spectrum is not quite in the wavelength range    
    if pad:
        x, y = pad_spectrum(x ,y, target_wavelength_range = target_wavelength_range)
    
    
    #noise = EstimateNoise(x=x,y=y)
    #print('Noise = %.4f' %noise)
    
    window = (6000./( 300000*  (x[1]-x[0]) / 6200 ))
    window_sg = np.ceil((window + 1)/2)*2 - 1
    if window_sg < 5:
            window_sg=5.
            #print (SN,'window_sg adjusted')
    y_smoothed = sg(y,window_sg,3,deriv=0)    
    
    
    xp,yp = ProcessSpectra(x=x,y=y_smoothed)
    y_shape = yp.reshape(1,-1)
    
    # get the list of predicted objects
    predicted = clf.predict(y_shape)
    #print(f'pred from pipeline = {predicted}')
    
#     # Try to decompose
#     for_pca = clf.named_steps.standardscaler.transform(y_shape)
#     #print(for_pca)
#     for_knn = clf.named_steps.pca.transform(for_pca)
#     #print(for_knn)
#     final = clf.named_steps.kneighborsclassifier.kneighbors(X = for_knn.reshape(1,-1))
#     #print(final)
#     pred = clf.named_steps.kneighborsclassifier.predict(X = for_knn.reshape(1,-1))
#     print(f'pred from stepwise = {pred}')
    
    # get the scores of these objects
    scores = clf.predict_proba(y_shape)
    
    #graph =clf.get_params()
    
    # get a list of indices to match against the actual SN template spectra
    idxs=[]
    final_scores = []
    for j in range(len(scores[0])):
        if scores[0,j]>0:
            idxs.append(j)
            final_scores.append(scores[0,j])
            
    return predicted, idxs, final_scores

def get_spec_matches(idxs = [], z = False, classlist = []):
    
    all_matches = []
    for idx in idxs:
        all_matches.append(classlist[idx])
        
    matches_list = split(all_matches, z)
    
    return matches_list


def dered(wav,flux,Rv,Ebv):
    lam=wav*0.0001
    Av=Ebv*Rv
    x=1/lam
    y=x-1.82
    a=1+(0.17699*y)-(0.50477*y**2)-(0.02427*y**3)+(0.72085*y**4)+(0.01979*y**5)-(0.77530*y**6)+(0.32999*y**7)
    b=(1.41338*y)+(2.28305*y**2)+(1.07233*y**3)-(5.38434*y**4)-(0.62251*y**5)+(5.30260*y**6)-(2.09002*y**7)
    AlAv=a+b/Rv
    Al=AlAv*Av
    #print Al
    F=10**(Al/2.5)
    delF= flux*F
    return delF

def fit_extinction(w = [], f = [], lam_ref = [], f_ref = [], runs = 50):
    
    E = 0.4
    offset = 0
    
    results = [ [1e9, E]   ]
    final_f = []
    final_w = []
    
    for j in range(runs):
    
        E = results[-1][1]
    
        E = np.random.normal(E, E*0.1)
        
        
        new_f = dered(w, f, 3.1, E)
        #new_f = dereddened_f / max(dereddened_f) + offset
        
        scale = f_ref[np.argmin(abs(6000 - lam_ref))] / new_f[np.argmin(abs(6000 - w))]
        new_f = new_f * scale
    
        lo,hi = np.argmin(abs(w - lam_ref[0])), np.argmin(abs(w - lam_ref[-1]))
    
        ws = w[lo:hi]
        new_f = new_f[lo:hi]
    
        convolved_to_spec = []
        for j in range(len(lam_ref)):
            idx = np.argmin(abs(ws - lam_ref[j]   ))
            convolved_to_spec.append(new_f[idx])
        
        mae = sum( abs( f_ref - np.array(convolved_to_spec) )**2 ) / len(f_ref)
    
        if mae < results[-1][0]:
            results.append( [mae, E]        )
            final_f = new_f
            final_w = ws
    
                       
    return  (final_w, final_f), results


### Plotting and stats
    
def get_z_stats(sne, zs, types, sorted_scores, get_sn = False, get_type = False, plot = True ):
    '''
    Returns the z stats for either a particular supernova as defined by get_sn,
    or a particular type as defined by get_type.
    Output is one tuple and two arrays; 
    
    [0] = (median, mean, std)
    [1] = redshift array
    [2] = weights array for plt.hist weights argument
    '''
    
    
    # set the empty lists
    stats = []
    w = []
    
    # set parameters for a supernova or a type
    if get_sn:
        sn_matches = [get_sn]
        list_to_use = sne
        
    if get_type:
        sn_matches = set(sne)
        list_to_use = types
     
    # iterate through the lists to get the redshifts that match the conditions
    for this_sn in sn_matches:        
        for idx ,obj in enumerate(list_to_use):
            
            if (obj == this_sn) or (obj == get_type):
                stats.append(zs[idx])
                w. append(sorted_scores[idx])
                
    # normalise the scores to the score in the list. This is for 'weights' in plt.hist            
    w =  ( w / max(sorted_scores) ) ** 4 
    
    # cast as numpy array
    stats = np.array(stats)
    w = np.array(w)
    
    if plot:
        plt.hist(stats, weights = w, color = 'k', facecolor = 'tab:green', zorder = 0, rwidth= 0.9, density = True)
        plt.axvline(x = np.average(stats, weights = w), linestyle = 'dashed' , color = 'k')
        plt.axvline(x = np.average(stats, weights = w) + np.std(stats), linestyle = 'dotted', color = 'k' )
        plt.axvline(x = np.average(stats, weights = w) - np.std(stats), linestyle = 'dotted', color = 'k' )
        plt.xlabel('Redshift')
        plt.ylabel('weighted counts')
    
    return (np.median(stats), np.average(stats, weights = w), np.std(stats) ), stats, w

def plot(spectrum, types, sne, zs, epochs, all_specs, sorted_scores, ref_location = './spectra/', 
         E=False, i=15, scaling=6000, clip=False):
    print(f'Top {i} matches to {os.path.basename(spectrum)}')
    
    x, y = np.loadtxt(spectrum, usecols = (0,1) ,unpack= True )
    
    if clip:
        y = clip_lines(x, y)

    if E:
        y = dered(x, y, 3.1, E)

    #ref_lam = 6000

    #scale = y[np.argmin(abs(x - ref_lam))]
    #y = y / scale
    
    if scaling == 'normed':
        m=1/(max(y)-min(y))
        con=-m*min(y)    
        y = np.array([m*d+con for d in y])
    else:  
        scale = y[np.argmin(abs(x - scaling))]
        y = y / scale    

    plt.figure(figsize = (15,19))
    used =[]
    n = 0
    for i, spec in enumerate(all_specs):
        #if i <15:
        if (sne[i] not in used) and (n < 15):
            used.append(sne[i])
        
            plt.subplot(5,3, n + 1)
            plt.plot(x, y)
    
            x_, y_ = np.loadtxt(ref_location + spec, unpack = True)
    
            x_ = x_ * (1 + zs[i])
    
            #scale = y_[np.argmin(abs(x_ - ref_lam))]
            #y_ = y_ / scale
            
            if scaling == 'normed':
                m = 1 / (max(y_) - min(y_))
                con = -m * min(y_)    
                y_ = np.array([m*d+con for d in y_])
            else:  
                scale = y_[np.argmin(abs(x_ - scaling))]
                y_ = y_ / scale
        
            fit_score = sorted_scores[i] / max(sorted_scores)
            plt.plot(x_, y_, label = f'{types[i]} {sne[i]} (t = {epochs[i]})\n@ $z=$ {zs[i]}, score = {fit_score:.2f}', alpha = 0.8)

            plt.legend(frameon = False)
            plt.minorticks_on()
            # this messes up if i changes
            if n in  [0, 3, 6, 9, 12]:
                plt.ylabel('Scaled flux')
            if n in [14, 13 ,12]:
                plt.xlabel('Observed wavelength [$\mathrm{\AA}$]')
        
            n+=1