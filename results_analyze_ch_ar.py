#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:25:58 2022

@author: mdannehl

This script analyses the results by building histograms of the hmi target and
hmi fake image while only taking the regions of coronal holes and of active 
regions (separately) into account.
Also only the inner circle of 2/3 of the solar radius are taken into account as
proposed by Veronique.

You can specify which trained model and which images you want to analyze.
Please make sure that all the results of the specified model and images exist
in the specified directory.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from skimage.transform import resize
from scipy.stats import skew,kurtosis
from pix2pixCC.aia_hmi_sunpy_plotter import AIA_HMI_Plotter

##############################################################################
### Please specify the images and models you wish to analyze:

### Select images:
timestamp_imgs = [
#    '20180115_2000',
#    '20180219_2000',
#    '20180313_0800',
#    '20180408_2012',
#    '20180514_2000',
#    '20180515_0000',
#    '20180530_1200',
#    '20180609_1200',
#    '20180623_1200',
#    '20180721_2000',
    '20180722_0000',
#    '20180811_2000',
#    '20180812_0000',
#    '20180908_2000',
#    '20180909_0000',
#    '20181006_2000',
#    '20181007_0000',
]

### Select a model with by timestamp and training epoch:
runs = {   
#    '20220804_1726': '185',
#    '20220723_1102': '150',
    '20220704_1816': '200',
#    '20220704_1804': '190',
#    '20220701_1058': '200',
#    '20220629_1001': '150',
#    '20220628_1512': '75',
#    '20220621_1215': '85',
#    '20220621_1048': '85',
#    '20220621_1041': '155',
#    '20220618_0030': '80',
#    '20220611_0516': '110',
#    '20220601_1059': '145',
#    '20220531_1050': '100',
#    '20220526_1758': '100',
#    '20220516_1135': '100',
#    '20220511_1634': '200',
#    '20220511_1628': '150',
#    '20220509_1307': '50',
}

size_hmi = 512 # size of the hmi images

### Limit for absolute intensity of hmi images to classify pixels as part of active region
limit_ar_hmi = 250

### Directory that contains the coronal hole maps as .fits files
dir_ch_maps = '/home/mdannehl/Downloads/forMarkus'

### Where to store the resulting histograms and time series
dir_export_histograms = '/home/mdannehl/Downloads/forMarkus/result_histograms'

### Where to find the models and their npz files that were exported during testing 
dir_model_results = '/home/mdannehl/git/solari2i/code/pix2pixCC/results/solar'

### Whether to show and or save the plots
show_plots = True
save_result_histograms = False

### Define a mask for the inner 2/3 of the radius as proposed by Veronique
plotter = AIA_HMI_Plotter(size_hmi, True)
r = plotter.get_rsun_in_pixels()*2//3 # only analyze two thirds of solar radius as proposed by Veronique
x = np.arange(size_hmi)
y = np.arange(size_hmi)
xx, yy = np.meshgrid(x, y)
zz = np.sqrt((xx-size_hmi//2)**2 + (yy-size_hmi//2)**2)
mask_inner = zz <= r

##############################################################################



for timestamp_run in runs.keys():
    best_epoch = runs[timestamp_run]
    print('\n--------------------------------\nModel: ' +timestamp_run+ ', E=' + best_epoch)
    for timestamp_img in timestamp_imgs:
    
        try:
            ## Load and scale coronal holes map
            print('\nLoading data for timestamp '+timestamp_img)
            path_fits = os.path.join(dir_ch_maps, timestamp_img+'00.CHMap.fits')
            f=fits.open(path_fits)
            #print(f.info())

            preserve_range=True
            ch=f[1].data
            ch = ch / np.max(ch); limit=0.5; ch[ch>=limit] =1; ch[ch<limit] = 0
            ch_512 = resize(ch,(size_hmi,size_hmi),anti_aliasing=True,preserve_range=preserve_range)
            
            ch_512 = ch_512 / np.max(ch_512); limit=0.5; ch_512[ch_512>=limit] =1; ch_512[ch_512<limit] = 0

            ## Load hmi data (generated and target)
            hmi_target = np.load(os.path.join(dir_model_results, timestamp_run, 'Image', 'Test', 'E'+best_epoch, 'HMI'+timestamp_img+'_target.npz'))['z']
            hmi_fake = np.load(os.path.join(dir_model_results, timestamp_run, 'Image', 'Test', 'E'+best_epoch, 'HMI'+timestamp_img+'_fake.npz'))['z']
            
            
            ## disregard everything outside 2/3 of the solar radius as proposed by Veronique
            hmi_target_only_23 = np.copy(hmi_target)
            hmi_target_only_23[~mask_inner] = np.nan
            hmi_fake_only_23 = np.copy(hmi_fake)
            hmi_fake_only_23[~mask_inner] = np.nan
             
            ## disregard everything outside the coronal holes
            hmi_target_only_23_only_ch = np.copy(hmi_target_only_23)
            hmi_target_only_23_only_ch[ch_512 == 0] = np.nan
            hmi_fake_only_23_only_ch = np.copy(hmi_fake_only_23)
            hmi_fake_only_23_only_ch[ch_512 == 0] = np.nan
            
            ## disregard everything outside active regions
            hmi_only_23_only_ar_index = np.abs(hmi_target_only_23) >= limit_ar_hmi
            hmi_target_only_23_only_ar = np.copy(hmi_target_only_23)
            hmi_target_only_23_only_ar[~hmi_only_23_only_ar_index] = np.nan
            hmi_fake_only_23_only_ar = np.copy(hmi_fake_only_23)
            hmi_fake_only_23_only_ar[~hmi_only_23_only_ar_index] = np.nan
            
            ## calculate skewness and kurtosis, only coronal holes, only 2/3 rsun
            skewness_ch_target = skew(hmi_target_only_23_only_ch, axis=None,nan_policy='omit')
            skewness_ch_fake = skew(hmi_fake_only_23_only_ch, axis=None,nan_policy='omit')
            kurtosis_ch_target = kurtosis(hmi_target_only_23_only_ch, axis=None,nan_policy='omit')
            kurtosis_ch_fake = kurtosis(hmi_fake_only_23_only_ch, axis=None,nan_policy='omit')
            print('Skewness CH_HMI_Target', skewness_ch_target)
            print('Skewness CH_HMI_Fake', skewness_ch_fake)
            print('Kurtosis CH_HMI_Target',kurtosis_ch_target)
            print('Kurtosis CH_HMI_Fake',kurtosis_ch_fake)
            
            ## calculate skewness and kurtosis, only active regions, only 2/3 rsun            
            skewness_ar_target = skew(hmi_target_only_23_only_ar, axis=None,nan_policy='omit')
            skewness_ar_fake = skew(hmi_fake_only_23_only_ar, axis=None,nan_policy='omit')
            kurtosis_ar_target = kurtosis(hmi_target_only_23_only_ar, axis=None,nan_policy='omit')
            kurtosis_ar_fake = kurtosis(hmi_fake_only_23_only_ar, axis=None,nan_policy='omit')
            print('Skewness AR_HMI_Target', skewness_ar_target)
            print('Skewness AR_HMI_Fake', skewness_ar_fake)
            print('Kurtosis AR_HMI_Target',kurtosis_ar_target)
            print('Kurtosis AR_HMI_Fake',kurtosis_ar_fake)
            
            ## safety checks to only plot when appropriate 
            hasInnerCoronalHoles23 = np.any(~np.isnan(hmi_target_only_23_only_ch))
            hasInnerActiveRegions23 = np.any(~np.isnan(hmi_target_only_23_only_ar))
            
            ## plotting if enabled
            if show_plots:
                ## plot histogram of the loaded map of coronal holes
                plt.figure()
                plt.title('Histogram of original CH map')
                plt.hist(ch.flatten(),bins=70, log=True)
    
                ## plot of loaded map of coronal holes
                plt.figure()
                plt.imshow(ch,origin='lower')
                plt.title(str(ch.shape) + ", original CH")
                plt.colorbar()
                
                ## plot rescaled map of coronal holes
                plt.figure()
                plt.imshow(ch_512,origin='lower')
                plt.title(str(ch_512.shape) + ", CH, anti_aliasing=True")
                plt.colorbar()
                
                ## plot loaded hmi data
                ax = plotter.plot_hmi_fig(hmi_target, title='target, '+timestamp_img)
                ax = plotter.plot_hmi_fig(hmi_fake, title='fake, '+timestamp_img)
    
                ## plot hmi data reduced to 2/3 of solar radius
                ax = plotter.plot_hmi_fig(hmi_target_only_23, title='target, '+timestamp_img+', 2/3 inner radius')
                ax = plotter.plot_hmi_fig(hmi_fake_only_23, title='fake, '+timestamp_img+', 2/3 inner radius')
    
                ## plot hmi data reduced to 2/3 of solar radius only ch
                if hasInnerCoronalHoles23:
                    ax = plotter.plot_hmi_fig(hmi_target_only_23_only_ch, title='target, '+timestamp_img+', 2/3 inner radius, only CH')
                    ax = plotter.plot_hmi_fig(hmi_fake_only_23_only_ch, title='fake, '+timestamp_img+', 2/3 inner radius, only CH')
                else:
                    print("No coronal holes inside 2/3 inner radius.");
                
                # plot hmi data reduced to 2/3 of solar radius only ar
                if hasInnerActiveRegions23:
                    ax = plotter.plot_hmi_fig(hmi_target_only_23_only_ar, title='target, '+timestamp_img+', 2/3 inner radius, only AR')
                    ax = plotter.plot_hmi_fig(hmi_fake_only_23_only_ar, title='fake, '+timestamp_img+', 2/3 inner radius, only AR')
                else:
                    print("No active regions inside 2/3 inner radius.");
            
            
            ## plot histogram with statistical information
            if hasInnerCoronalHoles23:
                fig = plt.figure()
                plt.title('Model='+timestamp_run+', E='+best_epoch+', img='+timestamp_img+', CH, 2/3 rsun')
                plt.hist(hmi_fake_only_23_only_ch.flatten(), log=True, label='fake',alpha=0.5,bins=300,color='blue')
                plt.hist(hmi_target_only_23_only_ch.flatten(), log=True, label='target',alpha=0.5,bins=300,color='darkgreen')
                plt.legend()
                plt.annotate('skewness_ch_target='+str(np.round(skewness_ch_target,2)),(0.15,0.85),xycoords='figure fraction', fontfamily='monospace')
                plt.annotate('skewness_ch_fake  ='+str(np.round(skewness_ch_fake,2)), (0.15,0.80),xycoords='figure fraction', fontfamily='monospace')
                plt.annotate('kurtosis_ch_target='+str(np.round(kurtosis_ch_target,2)), (0.15,0.75),xycoords='figure fraction', fontfamily='monospace')
                plt.annotate('kurtosis_ch_fake  ='+str(np.round(kurtosis_ch_fake,2)), (0.15,0.70),xycoords='figure fraction', fontfamily='monospace')
        
                if save_result_histograms:
                    plotter.save_figure(fig, os.path.join(dir_export_histograms, timestamp_run+'_E'+best_epoch+'_CH'), timestamp_img)
                if show_plots:
                    plt.show()
                else:
                    plt.close()
            
            
            ## plot histogram with statistical information
            if hasInnerActiveRegions23:
                fig = plt.figure()
                plt.title('Model='+timestamp_run+', E='+best_epoch+', img='+timestamp_img+', AR, 2/3 rsun')
                plt.hist(hmi_fake_only_23_only_ar.flatten(), log=True, label='fake',alpha=0.5,bins=300,color='blue')
                plt.hist(hmi_target_only_23_only_ar.flatten(), log=True, label='target',alpha=0.5,bins=300,color='darkgreen')
                plt.legend()
                plt.annotate('skewness_ar_target='+str(np.round(skewness_ar_target,2)),(0.15,0.85),xycoords='figure fraction', fontfamily='monospace')
                plt.annotate('skewness_ar_fake  ='+str(np.round(skewness_ar_fake,2)), (0.15,0.80),xycoords='figure fraction', fontfamily='monospace')
                plt.annotate('kurtosis_ar_target='+str(np.round(kurtosis_ar_target,2)), (0.15,0.75),xycoords='figure fraction', fontfamily='monospace')
                plt.annotate('kurtosis_ar_fake  ='+str(np.round(kurtosis_ar_fake,2)), (0.15,0.70),xycoords='figure fraction', fontfamily='monospace')

                if save_result_histograms:
                    plotter.save_figure(fig, os.path.join(dir_export_histograms, timestamp_run+'_E'+best_epoch+'_AR'), timestamp_img)
                if show_plots:
                    plt.show()
                else:
                    plt.close()
            
        except FileNotFoundError:
            print('Warning file(s) for ' + timestamp_img, ' not found.')
        
        
        
