#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:02:42 2022

@author: madannehl

For each wavelength channel of the AIA images the script creates histograms as 
well as time-series diagrams of the raw values, the medians, the 95th percentile 
values and the maximum values.
For each channel (Bx, By, Bz) of the HMI images currently only the raw histograms 
are created.

By changing the settings you are able to change which channels are analyzed, 
how much images are supposed to be taken into account, the images of which 
timestamps are used, whether the plots are supposed to be shown on screen or not
and whether to save the created plots to disk.

!! In order to run this file, make sure the path to the file containing the 
relevant timestamps is correctly set and that the corresponding sdo data exists
in the specified data root directory. !!

"""

import numpy as np
import os
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from datetime import datetime

#############################################################################
### Please specify:  
n = 8 # number of images to download per channel, -1=all
input_channel_combination = [171,193,304] # please also edit relevant timestamps appropriately
output_channels = []#['bz']
relevant_timestmaps_path = 'relevant_timestamps/relevant_timestamps_short_train.txt'
data_root_dir = '../data/solar-data2' # where the sdo data can be found 
fig_save_dir = 'histograms' # where to save the generated figures
save_figs=False
show_plots=True

### The settings below were used to run the script on the hpc2 server: 
#n = -1 # number of images to download per channel, -1=all
#input_channel_combination = [171,193,304] # please also edit relevant timestamps appropriately 
#output_channels = ['bx','by','bz']
#relevant_timestmaps_path = 'relevant_timestamps/relevant_timestamps_171_193_304_2010-2017_c4h_10m_15sp.txt'
#data_root_dir = '/storage/databanks/machinelearning/NASA-Solar-Dynamics-Observatory-Atmospheric-Imaging-Assembly/'
#fig_save_dir = 'histograms' 
#save_figs=True
#show_plots=False

#############################################################################

plt.ioff() # disable interactive mode, so that we have to call plt.show() explicitly

## preprocess path of all aia and hmi files ##
aia_paths = []
hmi_paths = []
timestamps = []
with open(relevant_timestmaps_path) as f:
    for line in f.readlines():
        date_time = line.strip().split("_") # yyyy_mm_dd_hhmm
        year_str = date_time[0]
        month_str = date_time[1]
        day_str =  date_time[2]
        time_str = date_time[3]                          
        
        ### Prepare input paths ###
        AIA_paths = {}
        for channel in input_channel_combination:
            c_str = str(channel).zfill(4)
            AIA_paths[c_str] = os.path.join(os.path.expanduser(data_root_dir), year_str,"AIA_"+c_str,month_str,day_str,"AIA"+year_str+month_str+day_str+"_"+time_str+"_"+c_str+".npz")
        aia_paths.append(AIA_paths)
        
        ### Prepare target paths ###
        HMI_paths = {}                  
        HMI_paths['bx'] = os.path.join(os.path.expanduser(data_root_dir), year_str,"HMI_Bx",month_str,day_str,"HMI"+year_str+month_str+day_str+"_"+time_str+"_bx.npz")
        HMI_paths['by'] = os.path.join(os.path.expanduser(data_root_dir), year_str,"HMI_By",month_str,day_str,"HMI"+year_str+month_str+day_str+"_"+time_str+"_by.npz")
        HMI_paths['bz'] = os.path.join(os.path.expanduser(data_root_dir), year_str,"HMI_Bz",month_str,day_str,"HMI"+year_str+month_str+day_str+"_"+time_str+"_bz.npz")
        hmi_paths.append(HMI_paths)
        
        ### Additional information ###
        timestamps.append(year_str+month_str+day_str+'_'+time_str)
                
def plot_histogram_raw_values(data, n, c_str, color):
    
    mean = np.mean(data) 
    median = np.median(data)
    
    fig = plt.figure(figsize=(13,9))
    ax = fig.add_subplot(111)

    plt.hist(data,bins="doane", log=True, density=False, color=color)
    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    ax.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.grid(True, which='both')
    plt.title(c_str+("" if "b" in c_str else " A")+', logarithmic histogram, ' + str(n) + ' data points (images), mean='+str(mean)+', median='+str(median), loc='center')

    plt.tight_layout()
    if save_figs:
        plt.savefig(os.path.join(fig_save_dir,'Histogram raw values' + c_str))
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_histogram_statistics(medians, n, c_str, color, name='medians'):
    
    mean = np.mean(medians) # mean of the medians
    median = np.median(medians) # median of the medians
    
    fig = plt.figure(figsize=(13,9))
    ax = fig.add_subplot(111)

    plt.hist(medians, bins=n//3, log=False, density=False, color=color)

    ax.grid(True, which='both')
    plt.title(c_str+("" if "b" in c_str else " A")+', histogram of the '+name+', ' + str(n) + ' data points (images), mean='+str(mean)+', median='+str(median), loc='center')

    plt.tight_layout()
    if save_figs:
        plt.savefig(os.path.join(fig_save_dir,'Histogram ' + name + ' ' + c_str))
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_timeseries_statistics(timestamps, medians, n, c_str, color, name='medians'):
    
    mean = np.mean(medians) # mean of the medians
    median = np.median(medians) # median of the medians
    
    fig = plt.figure(figsize=(13,9))
    ax = fig.add_subplot(111)

    plt.plot(timestamps, medians, linestyle='None',marker='o',color=color,alpha=0.5)

    ax.grid(True, which='both')
    plt.title(c_str+("" if "b" in c_str else " A")+', time series of the '+name+', ' + str(n) + ' data points (images), mean='+str(mean)+', median='+str(median), loc='center')

    plt.tight_layout()
    if save_figs:
        plt.savefig(os.path.join(fig_save_dir,'Histogram ' + name + ' ' + c_str))
    if show_plots:
        plt.show()
    else:
        plt.close()
        
## create fig_save directory if not exist
if save_figs:
    os.makedirs(fig_save_dir, exist_ok=True)

## load the npz files ##
if n==-1:
    n = len(timestamps) ## use all images
nc = len(input_channel_combination)

import matplotlib.ticker

for channel in input_channel_combination:
    c_str = str(channel).zfill(4)
    
    AIA_imgs = []
    AIA_medians = []
    AIA_perc_95s = []
    AIA_maxs = []
    datetimes = []
    for i in tqdm(range(n)):
        aia_img_np = np.load(aia_paths[i][c_str])['x'].flatten()
        AIA_imgs.append(aia_img_np)
        AIA_medians.append(np.median(aia_img_np))
        AIA_perc_95s.append(np.percentile(aia_img_np,95))
        AIA_maxs.append(np.max(aia_img_np))
        datetimes.append(datetime.strptime(timestamps[i],'%Y%m%d_%H%M'))
        
    AIA_imgs = np.stack(AIA_imgs).flatten() # len = n*512*512
            
    plot_histogram_raw_values(AIA_imgs, n, c_str, 'red')
    plot_histogram_statistics(AIA_medians, n, c_str, 'red', name='medians')
    plot_histogram_statistics(AIA_perc_95s, n, c_str, 'red', name='95th percentiles')
    plot_histogram_statistics(AIA_maxs, n, c_str, 'red', name='maximums')
    
    plot_timeseries_statistics(datetimes,AIA_medians, n, c_str, 'red', name='medians')
    plot_timeseries_statistics(datetimes,AIA_perc_95s, n, c_str, 'red', name='95th percentiles')
    plot_timeseries_statistics(datetimes,AIA_maxs, n, c_str, 'red', name='maximums')    
    
for c_str in output_channels:
    
    HMI_imgs = []
    for i in tqdm(range(n)):
        HMI_imgs.append(np.load(hmi_paths[i][c_str])['x'].flatten())
    HMI_imgs = np.stack(HMI_imgs).flatten() # len = n*512*512
       
    plot_histogram_raw_values(HMI_imgs, n, c_str, 'grey')