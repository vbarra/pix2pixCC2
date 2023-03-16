#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:53:45 2022

@author: mdannehl

This file provides a simple routine to load and plot aia and hmi data on a
specified timestamp using local npz files and sunpy.

It works as following:
You choose a timestamp and specify the channels you are interested in.
The script constructs the corresponding paths and tries to load the npz data.
You need to make sure that the data for all channels exists for the given 
timestamp, otherwise the script will fail.
 

"""

import numpy as np
import os

from pix2pixCC.aia_hmi_sunpy_plotter import AIA_HMI_Plotter


#############################################################################
### Please specify for which timestamp you want to plot each channel:
timestamp_to_plot = '2014_01_01_0000' # YYYY_MM_DD_hhmm

### Please make sure that all channels have data on the given timestamp_to_plot
input_channel_combination = [171,304]  # e.g. [171,193,304]
output_channels = ['bz'] # e.g. ['bx','by','bz']

data_root_dir = '../data/solar-data2'

show_plots=False

save_fig = True
fig_save_dir = '.' 


### input channel preprocessing
logscale_input = False
saturation_lower_limit_input = 1
saturation_upper_limit_input = 200
saturation_clip_input = False

### output channel preprocessing
logscale_target = False
saturation_lower_limit_target = -1
saturation_upper_limit_target = 1
saturation_clip_target = False
#############################################################################



### Parse timestamp
line = timestamp_to_plot
date_time = line.strip().split("_") # yyyy_mm_dd_hhmm
year_str = date_time[0]
month_str = date_time[1]
day_str =  date_time[2]
time_str = date_time[3]                          

### Prepare input channel paths ###
aia_paths = []
hmi_paths = []

AIA_paths = {}
for channel in input_channel_combination:
    c_str = str(channel).zfill(4) # make sure we obtain a zero padded 4 character string
    AIA_paths[c_str] = os.path.join(os.path.expanduser(data_root_dir), year_str,"AIA_"+c_str,month_str,day_str,"AIA"+year_str+month_str+day_str+"_"+time_str+"_"+c_str+".npz")
aia_paths.append(AIA_paths)

### Prepare output channel paths ###
HMI_paths = {}                  
HMI_paths['bx'] = os.path.join(os.path.expanduser(data_root_dir), year_str,"HMI_Bx",month_str,day_str,"HMI"+year_str+month_str+day_str+"_"+time_str+"_bx.npz")
HMI_paths['by'] = os.path.join(os.path.expanduser(data_root_dir), year_str,"HMI_By",month_str,day_str,"HMI"+year_str+month_str+day_str+"_"+time_str+"_by.npz")
HMI_paths['bz'] = os.path.join(os.path.expanduser(data_root_dir), year_str,"HMI_Bz",month_str,day_str,"HMI"+year_str+month_str+day_str+"_"+time_str+"_bz.npz")
hmi_paths.append(HMI_paths)

AIA_composed = None
### Load input channel npz ###
if len(input_channel_combination)>0:
    AIA_imgs = []
    for channel in input_channel_combination:
        c_str = str(channel).zfill(4)
        AIA_imgs.append(np.load(aia_paths[0][c_str])['x'])
        
    AIA_composed = np.stack(AIA_imgs, axis=0) 
            
    if logscale_input == True:
        AIA_composed[np.isnan(AIA_composed)] = 0.1
        AIA_composed[AIA_composed == 0] = 0.1
        AIA_composed = np.log10(AIA_composed)
    else:
        AIA_composed[np.isnan(AIA_composed)] = 0
        
    ## clip and normalize input if wished ##
    UpIA = float(saturation_upper_limit_input)
    LoIA = float(saturation_lower_limit_input)
    if saturation_clip_input == True:
        AIA_composed = (np.clip(AIA_composed, LoIA, UpIA)-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
    else:
        AIA_composed = (AIA_composed-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)

HMI_composed=None
### Load target channel npz ###
if len(output_channels)>0:
    HMI_composed = []
    for out_channel in output_channels:
        HMI_composed.append(np.load(hmi_paths[0][out_channel])['x'])
    HMI_composed = np.stack(
        HMI_composed
    , axis=0)
    
    if logscale_target == True:
        HMI_composed[np.isnan(HMI_composed)] = 0.1
        HMI_composed[HMI_composed == 0] = 0.1
        HMI_composed = np.log10(HMI_composed)
    else:
        HMI_composed[np.isnan(HMI_composed)] = 0
        
    ## clip and normalize target if wished ##
    UpIB = float(saturation_upper_limit_target)
    LoIB = float(saturation_lower_limit_target)
    if saturation_clip_target == True:
        HMI_composed = (np.clip(HMI_composed, LoIB, UpIB)-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
    else:
        HMI_composed = (HMI_composed-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)



### Initialize plotter for aia and hmi data
plotter = AIA_HMI_Plotter(512, show_plots)


### Plot input channels ###
for i, channel in enumerate(input_channel_combination):
    
    c_str = str(channel).zfill(4) # make sure we obtain a zero padded 4 character string
    
    _, fig = plotter.plot_aia_fig(AIA_composed[i,:,:], channel, title=c_str+r'$\,\AA$'+ ' on '+ timestamp_to_plot)
    
    if save_fig:
        plotter.save_figure(fig, fig_save_dir, c_str+'_on_'+ timestamp_to_plot)


### Plot output channels ###             
for i in range(0,len(output_channels)):
    
    _, fig = plotter.plot_hmi_fig(HMI_composed[i,:,:], title=output_channels[i]+' on '+ timestamp_to_plot)
    
    if save_fig:
        plotter.save_figure(fig, fig_save_dir, output_channels[i]+'_on_'+ timestamp_to_plot)
