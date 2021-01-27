#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:21:43 2021

@author: Si
"""

from functions import sn_types
import os
from requests import get  # to make GET request


def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)

json_folder = './json_files/'     
        
if os.path.isdir(json_folder) == False:
    print(f'Creating {json_folder}')
    os.mkdir(json_folder) 
    
        
for sn in sn_types:
    fname = f'{json_folder}/{sn}.json'
    
    if not os.path.isfile(fname):
        print(f'Downloading {sn}.json')
        url = f'https://sne.space/sne/{sn}.json'
        download(url, fname)
        
    else:
        print(f'{fname} already exists, skipping.')