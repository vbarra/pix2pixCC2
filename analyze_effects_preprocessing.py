#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:41:39 2022

@author: mdannehl

This script opens a plot to visualize the effect of different preprocessing 
settings. Those settings cover applying a logarithm, a square root as well as 
value clipping. 
You can choose between all available aia channels (wavelengths), edit the 
clipping limits and profit from two different plotters (sunpy and matplotlib).

This script loads all timestamps from a specified file, constructs the according
paths of the images for all channels (no check if they exist or not) and then
opens the plot by plotting the first image (Index 0). 
With the slider "Image index" in the top of the image you can select which 
timestamp to use and hence which image to plot.
Use the other widgets to experiment with the settings.

!! To make this script work you need to specify the file that contains the 
relevant timestamps as well as a directory where to find the sdo data !!

"""

import numpy as np
import os

import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.coordinates import frames
from sunpy.visualization.colormaps.color_tables import aia_color_table

import warnings
from sunpy.util import SunpyDeprecationWarning, SunpyUserWarning
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=SunpyDeprecationWarning)
warnings.filterwarnings("ignore", category=SunpyUserWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
from matplotlib.widgets import Slider,CheckButtons,RadioButtons,TextBox

#############################################################################
# Please specify which timestamps to use and where to find the sdo data
relevant_timestamp_path = 'relevant_timestamps/relevant_timestamps_all_2010-2017_c8h_12m_1sp.txt'
data_root_dir = '../data/solar-data' #data_root_dir = '/storage/databanks/machinelearning/NASA-Solar-Dynamics-Observatory-Atmospheric-Imaging-Assembly/'

data_size=512
typ = "AIA"
#############################################################################

AIA_CHANNELS = [
    '0094',
    '0131',
    '0171',
    '0193',
    '0211',
    '0304',
    '0335',
    '1600',
    '1700',
    '4500'
]

### prepare paths ###
aia_paths = []
hmi_paths = []
timestamps = []

line = []
with open(relevant_timestamp_path) as f:
    idx = 0 
    for line in f.readlines():
        date_time = line.strip().split("_") # yyyy_mm_dd_hhmm
        year_str = date_time[0]
        month_str = date_time[1]
        day_str =  date_time[2]
        time_str = date_time[3]                          
        
        ### Prepare input paths ###
        AIA_paths = {}
        for channel in AIA_CHANNELS:
            c_str = str(channel).zfill(4)
            AIA_paths[c_str] = os.path.join(os.path.expanduser(data_root_dir), year_str,"AIA_"+c_str,month_str,day_str,"AIA"+year_str+month_str+day_str+"_"+time_str+"_"+c_str+".npz")
        aia_paths.append(AIA_paths)
        
        ### Additional information ###
        timestamps.append(year_str+month_str+day_str+'_'+time_str)
            
        idx += 1
    dataset_length = idx

idx = 0

def load_npz(path, typ="AIA"):
    global logscale, saturation_clip, saturation_lower_limit, saturation_upper_limit
    
    try:
        ### Load npz ###
        if typ=="AIA":
            aia_np = np.load(path)['x']
                
            if logscale == True:
                aia_np[np.isnan(aia_np)] = 0.1
                aia_np[aia_np == 0] = 0.1
                aia_np = np.log10(aia_np)
            else:
                aia_np[np.isnan(aia_np)] = 0
                
            if square_root == True:
                aia_np[aia_np<1] = 0
                aia_np = np.sqrt(aia_np)
                
            ## clip and normalize input if wished ##
            UpIA = float(saturation_upper_limit)
            LoIA = float(saturation_lower_limit)
            if saturation_clip == True:
                aia_np = (np.clip(aia_np, LoIA, UpIA)-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
            else:
                aia_np = (aia_np-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
                
            return aia_np
        else:
            return None
    except FileNotFoundError as e:
        print(e.strerror)
        return None

obstime = '2017-09-01 12:00' # placeholder, does only affect the plot titles
observer = 'earth' 
arcsec_per_pixel = 4.8
rsun = 976  # in arcsec

coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=obstime,
                  observer=observer, frame=frames.Helioprojective)

header = sunpy.map.make_fitswcs_header((data_size,data_size), coord,
                                   reference_pixel=None,
                                   scale=[arcsec_per_pixel, arcsec_per_pixel]*u.arcsec/u.pixel)
header['rsun_obs'] = rsun
channel = str(channel).zfill(4) 

kwargs = {
    'title':''
}

fontsize_subtitle = 10
            
fig = plt.figure(figsize=(13,9))


ax_idx = plt.axes([0.215, 0.93, 0.65, 0.06])
ax_preprocessing = plt.axes([0.05,0.748,0.14,0.1])
ax_sat_low_limit = plt.axes([0.056,0.085,0.06,0.6])
ax_sat_upp_limit = plt.axes([0.12,0.085,0.06,0.6])
ax_plotter = plt.axes([0.05,0.873,0.08,0.1])
ax_channel = plt.axes([0.94,0.674,0.045,0.3])
ax_max_val_sat_low = plt.axes([0.06,0.695, 0.05,0.03])
ax_max_val_sat_upp = plt.axes([0.124,0.695,0.05,0.03])
ax = None #ax = fig.add_subplot(111)


########################## Initial values of widgets #########################
imgIdx = 0
logscale = False
square_root = False
saturation_clip = False
saturation_lower_limit = 1
saturation_upper_limit = 200
plotter=0 # 0='sunpy', 1='matplotlib' 
channel = "0304" # please also edit relevant timestamps appropriately 
##############################################################################

maxValSliderLow = 500 # to be updated at runtime
maxValSliderUpp = 3e3 # to be updated at runtime

sliderIdx = Slider(
    ax=ax_idx,
    label='Image index',
    valmin=0,
    valmax=dataset_length-1,
    valstep=1,
    valinit=imgIdx,
    color='green',
    initcolor='none'
)
sliderSatLow = Slider(
    ax=ax_sat_low_limit,
    label='sat.\nlow.\nlim.',
    orientation='vertical',
    valmin=0,
    valmax=1,#2e3,
    valstep=1e-9,
    valinit=saturation_lower_limit/maxValSliderLow,
    color='blue',
    initcolor='none'
)
sliderSatUpp = Slider(
    ax=ax_sat_upp_limit,
    label='sat.\nupp.\nlim.',
    orientation='vertical',
    valmin=0,#2
    valmax=1,#40e3,
    valstep=1e-9,
    valinit=saturation_upper_limit/maxValSliderUpp,
    color='red',
    initcolor='none'
)

sliderIdx.label.set_position((0.56,0.48))
sliderSatLow.label.set_position((0.5,0.5))
sliderSatUpp.label.set_position((0.5,0.5))

def calc_pre_processing(val, maxVal):
        
    val = maxVal * val
    
    if logscale:
        if val < 1: val = 1
        val = np.log10(val)
    
    if square_root:
        val = np.sqrt(val)
        
    if not logscale and not square_root:
        val = int(np.round(val,0)) # get rid of .0 suffix
    else:
        val = np.round(val,5)
        
    return val
    

# define slider update functions that are called when the slider values change
def updateIdx(i):
    global imgIdx
    imgIdx = i
    refresh_plot()

def updateSliderSatLow(val):
    global logscale,square_root, saturation_lower_limit, saturation_upper_limit
    
    val = calc_pre_processing(val, maxValSliderLow)
    
    saturation_lower_limit = val
    if saturation_lower_limit == saturation_upper_limit:
        saturation_lower_limit -= 1
    sliderSatLow.valtext.set_text(saturation_lower_limit)
    
    refresh_plot()

def updateSliderSatUpp(val):
    global logscale, square_root, saturation_lower_limit, saturation_upper_limit
    
    val = calc_pre_processing(val, maxValSliderUpp)
    
    saturation_upper_limit = val
    if saturation_lower_limit == saturation_upper_limit:
        saturation_upper_limit += 1
        
    sliderSatUpp.valtext.set_text(saturation_upper_limit)
    refresh_plot()

# set slider update functions
sliderIdx.on_changed(updateIdx)
sliderSatLow.on_changed(updateSliderSatLow)
sliderSatUpp.on_changed(updateSliderSatUpp)

tbMaxValSatLow = TextBox(ax_max_val_sat_low, "max: ",initial=str(calc_pre_processing(1,maxValSliderLow)), textalignment='center')
tbMaxValSatUpp = TextBox(ax_max_val_sat_upp, "",initial=str(calc_pre_processing(1,maxValSliderUpp)), textalignment='center')

def updateMaxValSatLow(new_text):
    global logscale,square_root, maxValSliderLow
    try:
        
        val = float(new_text)

        if square_root:
            val = np.math.pow(val,2)
            
        if logscale:
            if val < 0: val = 0
            val = np.math.pow(10,val)
            
        val = int(np.round(val,0)) # get rid of .0 suffix

        maxValSliderLow = val

        updateSliderSatLow(sliderSatLow.val)
        
    except:
        print('Warning invalid text for maxValSatLow')

def updateMaxValSatUpp(new_text):
    global logscale,square_root, maxValSliderUpp
    try:
        
        val = float(new_text)

        if square_root:
            val = np.math.pow(val,2)
            
        if logscale:
            if val < 0: val = 0
            val = np.math.pow(10,val)
            
        val = int(np.round(val,0)) # get rid of .0 suffix

        maxValSliderUpp = val

        updateSliderSatUpp(sliderSatUpp.val)
        
    except:
        print('Warning invalid text for maxValSatUpp')

tbMaxValSatLow.on_submit(updateMaxValSatLow)
tbMaxValSatUpp.on_submit(updateMaxValSatUpp)


cbPreprocessing = CheckButtons(
    ax=ax_preprocessing,
    labels=[
        "logscale",
        "square root",
        "saturation clipping"
    ],
    actives=[logscale, square_root, saturation_clip] # initial values
)

def updatePreprocessing(label):
    global logscale, square_root, saturation_clip
    if label == "logscale":
        logscale = not logscale
    if label == "square root":
        square_root = not square_root
    if label == "saturation clipping":
        saturation_clip = not saturation_clip
    
    updateSliderSatLow(sliderSatLow.val)
    updateSliderSatUpp(sliderSatUpp.val)
    
    tbMaxValSatLow.set_val(str(calc_pre_processing(1,maxValSliderLow)))
    tbMaxValSatUpp.set_val(str(calc_pre_processing(1,maxValSliderUpp)))
    
    refresh_plot()
    

cbPreprocessing.on_clicked(updatePreprocessing)



rbPlotter = RadioButtons(
    ax=ax_plotter,
    labels = ['sunpy','matplotlib'],
    active=plotter
)

def updatePlotter(plotval):
    global plotter
    if plotval=='sunpy':
        plotter=0
    if plotval=='matplotlib':
        plotter=1
    refresh_plot()
    
rbPlotter.on_clicked(updatePlotter)

rbChannel = RadioButtons(
    ax=ax_channel,
    labels = AIA_CHANNELS,
    active=AIA_CHANNELS.index(channel)
)   
for c in rbChannel.circles: 
#    c.set_radius(0.04)
    c.set_height(0.03) # make circles round
    c.set_width(0.15) # make circles round
for l in rbChannel.labels:
    l.set_x(l.get_position()[0]+0.06) # enhance label x position
    l.set_y(l.get_position()[1]-0.001) # enhance label y position

def updateChannel(val):
    global channel
    channel = str(val).zfill(4) # just make sure we have str with length of 4
    refresh_plot()
    
rbChannel.on_clicked(updateChannel)

p = None
### opening the interactive plot ###
def refresh_plot():    
    global p, imgIdx, typ, channel
    if p is not None:
        for to_del in fig.axes[-2:]:
            fig.delaxes(to_del)
        #p.clear()
    
    ax = plt.axes([0.1,0.1,0.9,0.8])
    
    channel = str(channel).zfill(4)
    if typ=="AIA":
        
        aia_clipping_interval = (0.25, 99.75)*u.percent # for all AIA wavelengths
        
        
        aia_img = load_npz(aia_paths[imgIdx][channel])
        
        if aia_img is not None:
            ax.set_title(str(channel)+r'$\,\AA$'+", t="+timestamps[imgIdx] , fontsize=fontsize_subtitle)
            
            aia_map = sunpy.map.Map(aia_img, header)
            aia_cmap = aia_color_table(int(channel)*u.angstrom)
    
            if plotter == 0:
                p=aia_map.plot(axes=ax, clip_interval=aia_clipping_interval, cmap=aia_cmap,**kwargs)
            else:
                p = plt.imshow(aia_img, origin='lower', cmap='rainbow')
            #p.set_clim([aia_img.min(), aia_img.max()]) # refresh colorbar
            plt.gca().grid(False)
            
            plt.colorbar()
    
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_xticks([])              
            
        else:
            # just print a black image if the image does not exist on disk.
            p = plt.imshow(np.zeros((data_size,data_size)), cmap='gray', vmax=100)
            plt.colorbar()
            print('Image does not exist.')
    fig.canvas.draw() # refresh drawing on screen            
    
    
refresh_plot()
updateSliderSatLow(sliderSatLow.val)
updateSliderSatUpp(sliderSatUpp.val)
plt.show()
