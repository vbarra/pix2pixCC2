#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:52:51 2022

@author: madannehl

The script aims at providing an overview over the existing aia and hmi channels 
by plotting them together in a plot.

The script traverses directories recursively and takes the first n images of
each channel to plot them together.
Make sure that the chosen root directory contains directly the folders of the
different channels e.g. AIA_0304 or HMI_Bz. This is important, because the 
script parses the names of the folders that are found directly inside of the 
root directory and treats them as the channel names.
The script only considers the first n_show images of each channel that are found.

This file was created at the beginning of the internship before knowing about
sunpy. For that reason matplotlib is used instead.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


##############################################################################
### Please specify:

### where to find the solar data, must contain subfolders of the different channels e.g. AIA_0304 or HMI_Bz
base_data_path = os.path.expanduser('~/git/solari2i/data/solar-data/2014')

### how many images to show per channel
n_show = 7

##############################################################################


DATA = {} ## will contain the file names, for each channel seperately

for img_typ in os.listdir(base_data_path):
    DATA[img_typ] = [] 
    img_dir = base_data_path + "/" + img_typ
    count = 1
    for root,dirs,files in os.walk(img_dir):    
        if count <= n_show:
            for file in files:
                if count <= n_show:
                    DATA[img_typ].append(np.load(root+"/"+file)['x'])
                    count += 1
                else:
                    break
        else:
            break
            


### plotting starts here
fig, axs = plt.subplots(n_show, len(DATA))

idxCol = 0
idxRow = 0
for img_typ in sorted(DATA.keys()):
    for img in np.array(DATA[img_typ]):
        
        img_norm = img
        #img_norm = img - np.min(img)
        #img_norm = img_norm / np.max(np.abs(img_norm))*255
        
        # Log transform by liam:
        #img_norm = 255/np.log(255+1) * np.log(1 + img_norm);
        
        ax = axs[idxRow, idxCol]
        if idxRow==0:
            ax.set_title(img_typ)
        maxVal = np.max(np.abs(img_norm))
        
        if img_typ.startswith("AIA"):
            # Log transform by liam:
            img_norm = 255/np.log(255+1) * np.log(1 + img_norm);
            
            maxVal = np.max(np.abs(img_norm))
            ax.imshow(img_norm, cmap="twilight", vmin=0,vmax=maxVal)     
            #ax.imshow(img_norm, cmap="twilight", vmin=-maxVal,vmax=maxVal)

        else:
            ax.imshow(img_norm, cmap="seismic", vmin=-200, vmax=200) #norm=mlp.colors.LogNorm(vmin=-maxVal,vmax=maxVal))
            #ax.imshow(img_norm, cmap="seismic", norm=mlp.colors.LogNorm(vmin=-maxVal,vmax=maxVal))
               
        ax.set_axis_off()
        idxRow += 1
        
    idxCol += 1
    idxRow = 0
    
plt.tight_layout()
fig.subplots_adjust(hspace=0.001)
fig.subplots_adjust(wspace=0.001)

plt.show()
