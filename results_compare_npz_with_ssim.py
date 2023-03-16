#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:11:07 2022

@author: mdannehl

This file is used to compare the results from different trained models. The 
script opens two figures that are both interactive and that are also connected 
to each other. Each figure consists of four plots: the original hmi image, 
the generated hmi image, the structural similarity index map and a combination-
plot that combines the ssim map with the original hmi image.

You can zoom in into one plot and the zoom will apply automatically to all other
plots of both figures. This is done by sharing the axes objects with each other.

A certain model is defined by it’s configuration and can be referenced by the date
on which the training had started. During the training checkpoints of the models
are exported. Those checkpoints of the models are therefore referenced by the
epoch number that is usually between 5 and 200.

Use the widgets to select a model, it’s checkpoint as well as the timestamp of 
the image that you would like to compare.

Suggested use:
Open the two figures side by side on your monitor. Then select a different model
in the left and right figure. After that select the same image timestamp. Like
that you can easily compare the results of the two models. The zoom, that is
automatically applied to all plots of both figures, might help you with that.

"""

import os
import numpy as np
from skimage.metrics import structural_similarity as skssim
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pix2pixCC.aia_hmi_sunpy_plotter import AIA_HMI_Plotter

import astropy.units as u


class SSIM_Figure():

    def __init__(self, Epoch=None, runTimestamp=None, useTorchMetrics=True):
        
        ######################################################################
        ### Which relevant timestamps file to use:
        self.relevant_timestamp_path = 'relevant_timestamps_171_193_304_2010-2017_c4h_10m_15sp.txt'
        
        ### Where to find the sdo data
        self.data_root_dir = '../data/solar-data'
        
        ### where to find the trained pix2pixCC models and the files that were exported during testing
        self.result_dir = os.path.expanduser('~/git/solari2i/code/pix2pixCC/results/solar/')

        ### Configuration of SSIM (attention: skimage and tochmetrics have different options)            
        self.useTorchMetrics=useTorchMetrics
        if self.useTorchMetrics:
            self.optionsSSIM = {
                'return_full_image':True, 
                'kernel_size': 11, 
                'gaussian_kernel':False, 
                'reduction':None,
                'sigma': 1.5,
                'k1': 0.01,
                'k2': 0.03,
                }
        else:
            self.optionsSSIM = {
                'full':True, 
                'win_size': 11, 
                'gaussian_weights': False,
                'sigma': 1.5,
                'k1': 0.01,
                'k2': 0.03,
                }
        ######################################################################

        self.selectedRunIdx = 0
        self.selectedEpochIdx = 0
        self.selectedImageIdx = 0
        self.availableRuns = None
        self.availableEpochs = None
        self.availableImages = None
        
        self.refresh_available_runs()
        
        # try to select the choosen run
        if runTimestamp is not None:
            if runTimestamp in self.availableRuns:
                self.selectedRunIdx = self.availableRuns.index(runTimestamp)
            else:    
                self.selectedRunIdx = 0
                print('Warning: Selected run',runTimestamp,'does not exist.')     
            runTimestamp = self.availableRuns[self.selectedRunIdx]
            self.refresh_available_epochs()
            
        # try to select the choosen epoch 
        if Epoch is not None:
            if Epoch in self.availableEpochs:
                self.selectedEpochIdx = self.availableEpochs.index(Epoch)
            else:    
                self.selectedEpochIdx = 0
                print('Warning: Selected Epoch',Epoch,'does not exist for the selected run.')        
            Epoch = self.availableEpochs[self.selectedEpochIdx]
            self.refresh_available_images()


        ## plotting ##
    
        self.plotter = AIA_HMI_Plotter(512,True)
        
        self.fig = None 
        self.ax0 = None # upper left 
        self.ax1 = None # upper right
        self.ax2 = None # lower left
        self.ax3 = None # lower right
        
        self.second_ssim_fig_obj = None # space to connect a second figure
        
        self.refresh_plot()
    
    def refresh_available_runs(self):
        available_runs = []
               
        for x in glob(self.result_dir+"20*"): 
            if '_orig' not in x and not x.startswith('_'):
                available_runs.append(os.path.basename(x))
    
        available_runs.sort()
        
        if self.selectedRunIdx >= len(available_runs):
            self.selectedRunIdx = 0
        
        self.availableRuns = available_runs
        self.refresh_available_epochs()
    
    def refresh_available_epochs(self):
        if self.availableEpochs is None:
            epochOld = None
        else:
            epochOld = self.availableEpochs[self.selectedEpochIdx]
        availableEpochs = []
        selectedRunTimestamp = self.availableRuns[self.selectedRunIdx]
        runDir = os.path.join(self.result_dir, selectedRunTimestamp, 'Image','Test','')
        for x in glob(runDir+"E*"): 
            basename = os.path.basename(x)
            if '_' not in basename:
                availableEpochs.append(int(basename.replace('E','')))
        availableEpochs.sort()
        if epochOld in availableEpochs:
            self.selectedEpochIdx = availableEpochs.index(epochOld)
        elif self.selectedEpochIdx >= len(availableEpochs):
            self.selectedEpochIdx = 0
            
        self.availableEpochs = availableEpochs
        self.refresh_available_images()
        
    def refresh_available_images(self):
        if self.availableImages is None:
            imgTimestampOld = None
        else:
            imgTimestampOld = self.availableImages[self.selectedImageIdx]
        
        availableImages = []
        selectedRunTimestamp = self.availableRuns[self.selectedRunIdx]
        selectedEpoch = self.availableEpochs[self.selectedEpochIdx]
        
        epochDir = os.path.join(self.result_dir, selectedRunTimestamp, 'Image','Test', 'E'+str(selectedEpoch),'')
        for x in glob(epochDir+"HMI*_fake.npz"): 
            basename = os.path.basename(x)
            availableImages.append(basename.replace('HMI','').replace('_fake.npz',''))
        availableImages.sort()
        
        if imgTimestampOld in availableImages:
            self.selectedImageIdx = availableImages.index(imgTimestampOld)
        elif self.selectedImageIdx >= len(availableImages):
            self.selectedImageIdx = 0 
        
        self.availableImages = availableImages

        

    
    def refresh_plot(self):
        
        selectedEpoch = self.availableEpochs[self.selectedEpochIdx]
        selectedImgTimestamp = self.availableImages[self.selectedImageIdx]
        selectedRunTimestamp = self.availableRuns[self.selectedRunIdx]
            
        epochDir = os.path.join(self.result_dir, selectedRunTimestamp, 'Image','Test', 'E'+str(selectedEpoch))
        filename1 = os.path.join(epochDir,'HMI'+selectedImgTimestamp+'_target.npz')
        filename2 = os.path.join(epochDir,'HMI'+selectedImgTimestamp+'_fake.npz')

        npz1 = np.load(filename1)['z']
        npz2 = np.load(filename2)['z']
        #npz2[npz2<9e22] = 0
        
#        UpIB = 5000
#        LoIB = -5000
#        npz1 = (npz1-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
#        npz2 = (npz2-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)

        npz1[np.isnan(npz1)]=0
        npz2[np.isnan(npz2)]=0
        
#        
        if self.useTorchMetrics:
            sval,smap = ssim(torch.tensor(npz1).unsqueeze(0).unsqueeze(0),
                             torch.tensor(npz2).unsqueeze(0).unsqueeze(0), 
                             **self.optionsSSIM);sval=sval.numpy();smap=smap.numpy()
        else:
            sval,smap = skssim(npz1,npz2,**self.optionsSSIM)
            

        ### Plot ###             
        
        if self.fig is None: 
            self.fig = plt.figure(figsize=(13,9))
            
            fontsize_text = 10
            
            xpos = 0.79; 
            ypos = 0.966
            self.axEpochNext = plt.axes([xpos+0.15,ypos,0.02,0.02])
            self.axEpochPrev = plt.axes([xpos+0.086,ypos,0.02,0.02])     
            self.axImgNext = plt.axes([xpos+0.186,ypos-0.03,0.02,0.02])
            self.axImgPrev = plt.axes([xpos+0.05,ypos-0.03,0.02,0.02])
            self.axRunNext = plt.axes([xpos+0.186,ypos-0.06,0.02,0.02])
            self.axRunPrev = plt.axes([xpos+0.05,ypos-0.06,0.02,0.02])
            self.axResetZoom = plt.axes([xpos+0.083,ypos-0.09,0.09,0.02])
                               
            
            ## add widgets and texts ##
            self.fig.text(xpos,ypos+0.004,'Epoch:', fontsize=fontsize_text)
            self.tEpoch = self.fig.text(xpos+0.127,ypos+0.004,selectedEpoch,horizontalalignment='center',fontsize=fontsize_text)
            self.bEpochNext = Button(self.axEpochNext,'>')
            self.bEpochPrev = Button(self.axEpochPrev,'<')
            
            self.fig.text(xpos,ypos-0.03+0.004,'Image:',fontsize=fontsize_text)
            self.bImgNext = Button(self.axImgNext,'>')
            self.bImgPrev = Button(self.axImgPrev,'<')
            self.tImgTimestamp = self.fig.text(xpos+0.129,ypos-0.03+0.004,selectedImgTimestamp,horizontalalignment='center',
                                               fontsize=fontsize_text,fontfamily='monospace')
            
            self.fig.text(xpos,ypos-0.06+0.004,'Run:',fontsize=fontsize_text)
            self.bRunNext = Button(self.axRunNext,'>')
            self.bRunPrev = Button(self.axRunPrev,'<')
            self.tRunTimestamp = self.fig.text(xpos+0.129,ypos-0.06+0.004,selectedRunTimestamp,horizontalalignment='center',
                                               fontsize=fontsize_text,fontfamily='monospace')
            
            self.bResetZoom = Button(self.axResetZoom, 'Reset Zoom')
            
            self.bEpochNext.on_clicked(self.bEpochNext_clicked)
            self.bEpochPrev.on_clicked(self.bEpochPrev_clicked)
            self.bImgNext.on_clicked(self.bImgNext_clicked)
            self.bImgPrev.on_clicked(self.bImgPrev_clicked)
            self.bRunNext.on_clicked(self.bRunNext_clicked)
            self.bRunPrev.on_clicked(self.bRunPrev_clicked)
            self.bResetZoom.on_clicked(self.bResetZoom_clicked)
            self.fig.my_widgets = self # prevent garbage collection by keeping a reference inside the figure object        
                    
        else:
            # delete all axes but those of the widgets (the first six)
            for ax in self.fig.axes[7:]:
                self.fig.delaxes(ax)
                
        ## update title ##
        title="SSIM="+str(sval)+ ", for the following settings:"
        self.fig.suptitle(title)
        
        ## update widget texts ##
        self.tEpoch.set_text(str(selectedEpoch))
        self.tImgTimestamp.set_text(selectedImgTimestamp)
        self.tRunTimestamp.set_text(selectedRunTimestamp)
        
        ## create subplots with hmi and ssim map ##
        
        self.ax0 = self.fig.add_subplot(221);
        self.plotter.plot_hmi(npz1,ax=self.ax0,title="target")
        
        self.ax1 = self.fig.add_subplot(222,sharex=self.ax0,sharey=self.ax0);
        alpha=1.0
        self.plotter.plot_hmi(smap,ax=self.ax1,title="structural similarity map", 
                    clip_interval=(0,100)*u.percent, cmap="Reds_r", kwargs={'alpha':alpha})
        
        
        self.ax2 = self.fig.add_subplot(223,sharex=self.ax0,sharey=self.ax0) 
        self.plotter.plot_hmi(npz2,ax=self.ax2,title="generated")
        
        self.ax3 = self.fig.add_subplot(224,sharex=self.ax0,sharey=self.ax0)
        alpha=0.4
        self.plotter.plot_hmi(npz2,ax=self.ax3, plot_colorbar=False)
        self.plotter.plot_hmi(smap,ax=self.ax3,title="generated + structural similarity (alpha=" + str(alpha)+")  ",
                    clip_interval=(0,100)*u.percent, cmap="Reds_r", kwargs={'alpha':alpha})
        
        if self.second_ssim_fig_obj is not None:
            # connect x and y axes to the other figure
            self.ax0.get_shared_x_axes().join(self.ax0, self.second_ssim_fig_obj.ax0)
            self.ax0.get_shared_y_axes().join(self.ax0, self.second_ssim_fig_obj.ax0)
            
            # keep previous zoom
            self.ax0.set_xlim(self.second_ssim_fig_obj.ax0.get_xlim())
            self.ax0.set_ylim(self.second_ssim_fig_obj.ax0.get_ylim())
        else:
            # remember the default zoom
            self.xlim_default = self.ax0.get_xlim()
            self.ylim_default = self.ax0.get_ylim()
            
        self.fig.canvas.draw()

    def bEpochNext_clicked(self,e):
        self.selectedEpochIdx += 1
        if len(self.availableEpochs) == self.selectedEpochIdx:
            self.selectedEpochIdx = 0
        self.refresh_available_images()
        self.refresh_plot()
        
    def bEpochPrev_clicked(self,e):
        self.selectedEpochIdx -= 1
        if self.selectedEpochIdx < 0:
            self.selectedEpochIdx = len(self.availableEpochs) - 1
        self.refresh_available_images()
        self.refresh_plot()    

    def bImgNext_clicked(self,e):
        self.selectedImageIdx += 1
        if len(self.availableImages) == self.selectedImageIdx:
            self.selectedImageIdx = 0
        self.refresh_plot()
        
    def bImgPrev_clicked(self,e):
        self.selectedImageIdx -= 1
        if self.selectedImageIdx < 0:
            self.selectedImageIdx = len(self.availableImages) - 1
        self.refresh_plot()

    def bRunNext_clicked(self,e):
        self.selectedRunIdx += 1
        if len(self.availableRuns) == self.selectedRunIdx:
            self.selectedRunIdx = 0
        self.refresh_available_epochs()
        self.refresh_plot()
        
    def bRunPrev_clicked(self,e):        
        self.selectedRunIdx -= 1
        if self.selectedRunIdx < 0:
            self.selectedRunIdx = len(self.availableRuns) - 1
        self.refresh_available_epochs()
        self.refresh_plot()
        
    def bResetZoom_clicked(self,e):
        self.ax0.set_xlim(self.xlim_default)
        self.ax0.set_ylim(self.ylim_default)
        self.fig.canvas.draw()

    
    def join_fig_axes(ssim_fig1, ssim_fig2):
        ssim_fig1.second_ssim_fig_obj = ssim_fig2
        ssim_fig2.second_ssim_fig_obj = ssim_fig1
        ssim_fig1.refresh_plot()
        ssim_fig2.refresh_plot()
        
    
if __name__=='__main__':
    s1=SSIM_Figure(Epoch=200, runTimestamp="20220701_1058",useTorchMetrics=False)
    s2=SSIM_Figure(Epoch=200, runTimestamp="20220723_1102",useTorchMetrics=False)
    SSIM_Figure.join_fig_axes(s1,s2)
    plt.show()