#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:22:32 2022

@author: mdannehl

This file contains multiple methods to plot aia and hmi data using sunpy. A
dditionally, the file provides two small methods: one forsaving figures to disk 
and one to calculate the solar radius in pixels depending on the current 
settings of the plotter.

"""


import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.coordinates import frames
from sunpy.visualization.colormaps.color_tables import aia_color_table
import copy
import os

import warnings
from sunpy.util import SunpyDeprecationWarning, SunpyUserWarning
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=SunpyDeprecationWarning)
warnings.filterwarnings("ignore", category=SunpyUserWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


class AIA_HMI_Plotter():
    
    def __init__(self, data_size, show_plots):
        
        self.data_size = data_size # 512
        self.show_plots = show_plots
        
        ### prepare aia / hmi plotting ###
        
        plt.ioff() # disable interactive mode, so that we have to call plt.show() explicitly
        
        obstime = '2017-09-01 12:00' # placeholder, does only affect the plot titles
        observer = 'earth' 
        self.arcsec_per_pixel = 4.8
        self.rsun = 976  # in arcsec
        
        self.coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=obstime,
                              observer=observer, frame=frames.Helioprojective)
        
        self.header = sunpy.map.make_fitswcs_header((self.data_size,self.data_size), self.coord,
                                               reference_pixel=None,
                                               scale=[self.arcsec_per_pixel, self.arcsec_per_pixel]*u.arcsec/u.pixel)
        self.header['rsun_obs'] = self.rsun
    
    #--------------------------------------------------------------------------
      
    ### used by several scripts in solari2i/code
    def plot_aia_fig(self, np_aia, channel, title='', show_plots=None):
        return self.plot_aia(np_aia, channel, title=title, show_plots=show_plots,
                      clip_interval=None, ax=None, cmap=None, plot_colorbar=True)
    
    
    ### used by several scripts in solari2i/code/
    def plot_hmi_fig(self, np_hmi, title='', show_plots=None):
        return self.plot_hmi(np_hmi, title=title, show_plots=show_plots,
                      clip_interval=None, ax=None, cmap=None, plot_colorbar=True)
    
    
    ### used by pix2pixCC_Utils.py
    def plot_aia_hmi_all_in_one(self, aia_channel_combination, aia_np, hmi_real_np, hmi_fake_np, suptitle):
        
        fig = plt.figure(figsize=(13,9))
        fig.suptitle(suptitle)
        nCols = max(3,aia_np.shape[0]) # cols for: input images of different wavelengths
        nRows = 3 # rows for: input aia, real hmi, fake hmi
        
        axes = fig.subplots(nRows, nCols)
        
        hmi_names = ["Bx", "By", "Bz"]
        
        
        ### Plot inputs ###
        for i, channel in enumerate(aia_channel_combination):
            self.plot_aia(aia_np[i,:,:], channel, ax=axes[0,i], plot_colorbar=True, title=str(channel)+r'$\,\AA$', show_plots=False)
                    
        
        ### Plot target outputs ###         
        for i in range(0,3):
            self.plot_hmi(hmi_real_np[i,:,:], ax=axes[1,i], plot_colorbar=True, title="Target " + hmi_names[i], show_plots=False)
                   
            
        ### Plot generated outputs ###      
        for i in range(0,3):
            self.plot_hmi(hmi_fake_np[i,:,:], ax=axes[2,i], plot_colorbar=True, title="Generated " + hmi_names[i], show_plots=False)
        
        
        ## remove unimportant labels and ticks ##
        for axrow in axes:
            for ax in axrow:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_yticks([])
                ax.set_xticks([])
        
        ## add row titles ##
        axes[0,0].set_ylabel("Input: AIA images")
        axes[1,0].set_ylabel("Target: HMI images")
        axes[2,0].set_ylabel("Generated: HMI images")
        
        ## delete unused plots ##
        for row in range(1,3):
            for col in range(3,nCols):
                fig.delaxes(axes[row,col])
        if aia_np.shape[0] <= 2:
            fig.delaxes(axes[0,2])
        if aia_np.shape[0] == 1:
            fig.delaxes(axes[0,1])
            
        
        #fig.subplots_adjust(hspace=0.001)
        #fig.subplots_adjust(wspace=0.2)
        plt.tight_layout()

        if self.show_plots:
            plt.show()
        else:
            plt.close()
            
        return fig
    




    ##########################################################################

    
    def plot_aia(self, np_aia, channel, clip_interval=None, ax=None, cmap=None, plot_colorbar=True, title='', show_plots=None, kwargs=None):
                
        fontsize_subtitle = 10
        
        fig = None
    
        if ax is None:
            fig = plt.figure(figsize=(13,9))
            ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=fontsize_subtitle)
        
        aia_map = sunpy.map.Map(np_aia, self.header)
        
        if cmap is None:
            cmap = aia_color_table(channel*u.angstrom)

        if clip_interval is None:
            clip_interval = (0.25, 99.75)*u.percent # for all AIA wavelengths
            
        if kwargs is None:
            kwargs = {}
            
        if show_plots is None:
            show_plots = self.show_plots
                
        ax.grid(False)
        
        aia_map.plot(axes=ax, annotate=False, clip_interval=clip_interval, cmap=cmap,**kwargs)
        plt.gca().grid(False)
    
        if show_plots:
            plt.show()
        else:
            if fig is not None:
                plt.close()
        
        if plot_colorbar:
            ax.figure.colorbar(ax.figure._gci())
            
        return ax, fig
        
    
    def plot_hmi(self, np_hmi, clip_interval=None, ax=None, cmap=None, plot_colorbar=True, title='', show_plots=None, kwargs=None):
        
        fontsize_subtitle = 10
        
        fig = None
    
        if ax is None:
            fig = plt.figure(figsize=(13,9))
            ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=fontsize_subtitle)
    
        hmi_map = sunpy.map.Map(np_hmi, self.header)
        centers = sunpy.map.all_coordinates_from_map(hmi_map)
        interior = sunpy.map.coordinate_is_on_solar_disk(centers)
        hmi_map.data[~interior] = float('nan')
        
        if cmap is None:
            cmap = copy.copy(hmi_map.cmap)
            cmap.set_bad('k')

        if clip_interval is None:
            clip_interval = (0.05, 99.95)*u.percent
            
        if kwargs is None:
            kwargs = {}
            
        if show_plots is None:
            show_plots = self.show_plots
                
        ax.grid(False)
        
        hmi_map.plot(axes=ax, annotate=False, clip_interval=clip_interval, cmap=cmap,**kwargs)
        plt.gca().grid(False)
            
        if show_plots:
            plt.show()
        else:
            if fig is not None:
                plt.close()
        
        if plot_colorbar:
            ax.figure.colorbar(ax.figure._gci())
            
        return ax, fig
    
    ##########################################################################
    
    
    def save_figure(self, figure, directory, filename):
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, filename))
    
    
    ##########################################################################
        
    def get_rsun_in_pixels(self):
        return self.rsun / self.arcsec_per_pixel
