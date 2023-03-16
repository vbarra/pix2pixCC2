#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:16:42 2022

@author: madannehl

Torch dataset class for solar data.

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SolarDataset(Dataset):
        
    def __init__(self, opt, relevant_timestamp_path):
        super(SolarDataset, self).__init__()
        self.opt = opt    
        self.relevant_timestamp_path = relevant_timestamp_path

        self.preprocess()
        
    def preprocess(self):
        self.aia_paths = []
        self.hmi_paths = []
        self.timestamps = []
        
        self.line = []
        with open(self.relevant_timestamp_path) as f:
            idx = 0 
            for line in f.readlines():
                date_time = line.strip().split("_") # yyyy_mm_dd_hhmm
                year_str = date_time[0]
                month_str = date_time[1]
                day_str =  date_time[2]
                time_str = date_time[3]                          
                
                ### Prepare input paths ###
                AIA_paths = {}
                for channel in self.opt.input_channel_combination:
                    c_str = str(channel).zfill(4)
                    AIA_paths[c_str] = os.path.join(os.path.expanduser(self.opt.data_root_dir), year_str,"AIA_"+c_str,month_str,day_str,"AIA"+year_str+month_str+day_str+"_"+time_str+"_"+c_str+".npz")
                self.aia_paths.append(AIA_paths)
                
                ### Prepare target paths ###
                HMI_paths = {}                  
                HMI_paths['bx'] = os.path.join(os.path.expanduser(self.opt.data_root_dir), year_str,"HMI_Bx",month_str,day_str,"HMI"+year_str+month_str+day_str+"_"+time_str+"_bx.npz")
                HMI_paths['by'] = os.path.join(os.path.expanduser(self.opt.data_root_dir), year_str,"HMI_By",month_str,day_str,"HMI"+year_str+month_str+day_str+"_"+time_str+"_by.npz")
                HMI_paths['bz'] = os.path.join(os.path.expanduser(self.opt.data_root_dir), year_str,"HMI_Bz",month_str,day_str,"HMI"+year_str+month_str+day_str+"_"+time_str+"_bz.npz")
                self.hmi_paths.append(HMI_paths)
                
                ### Additional information ###
                self.timestamps.append(year_str+month_str+day_str+'_'+time_str)
                    
                
                idx += 1
            self.dataset_length = idx
            
        # optional: check if all files of the dataset exist so that __get__item will never fail
        if self.opt.check_dataset:
            for aia in self.aia_paths:
                for path in aia.values():
                    if not os.path.isfile(path):
                        raise FileNotFoundError("Dataset verification failed: " + path)
            for hmi in self.hmi_paths:
                for path in hmi.values():
                    if not os.path.isfile(path):
                        raise FileNotFoundError("Dataset verification failed: " + path)
              
            print("Successfully verified accessibility of the whole dataset.")
        else:
            print('Skipped dataset check.')
                    
            
    def __getitem__(self, index):
        
        ### Load input ###
        AIA_imgs = []
        for c_idx, channel in enumerate(self.opt.input_channel_combination):
            c_str = str(channel).zfill(4)
            aia_np = np.load(self.aia_paths[index][c_str])['x']
            
            if self.opt.logscale_input[c_idx] == True:
                aia_np[np.isnan(aia_np)] = 0.1
                aia_np[aia_np == 0] = 0.1
                aia_np = np.log10(aia_np)
            else:
                aia_np[np.isnan(aia_np)] = 0
            
            if self.opt.sqrt_input[c_idx] == True:
                aia_np = np.sqrt(aia_np)
                
            ## clip and normalize input if wished ##
            UpIA = np.float(self.opt.saturation_upper_limit_input[c_idx])
            LoIA = np.float(self.opt.saturation_lower_limit_input[c_idx])
            if self.opt.saturation_clip_input[c_idx] == True:
                aia_np = (np.clip(aia_np, LoIA, UpIA)-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
            else:
                aia_np = (aia_np-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
            
            AIA_imgs.append(aia_np)
            
        AIA_composed = np.stack(AIA_imgs, axis=0) 
                
        ## convert input to tensor ##
        AIA_tensor = torch.tensor(AIA_composed, dtype=torch.float32)      


        ### Load target ###
        HMI_composed = np.stack([
            np.load(self.hmi_paths[index]["bx"])['x'], #Bx
            np.load(self.hmi_paths[index]["by"])['x'], #By
            np.load(self.hmi_paths[index]["bz"])['x']  #Bz
        ], axis=0)
        
        if self.opt.logscale_target == True:
            HMI_composed[np.isnan(HMI_composed)] = 0.1
            HMI_composed[HMI_composed == 0] = 0.1
            HMI_composed = np.log10(HMI_composed)
        else:
            HMI_composed[np.isnan(HMI_composed)] = 0
            
        ## clip and normalize target if wished ##
        UpIB = np.float(self.opt.saturation_upper_limit_target)
        LoIB = np.float(self.opt.saturation_lower_limit_target)
        if self.opt.saturation_clip_target == True:
            HMI_composed = (np.clip(HMI_composed, LoIB, UpIB)-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
        else:
            HMI_composed = (HMI_composed-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
        
        ## convert target to tensor ##
        HMI_tensor = torch.tensor(HMI_composed, dtype=torch.float32)
        
        
        ### Additional information and return ###
        timestamp = self.timestamps[index]
        
        return (
            AIA_tensor,   # multichannel input
            HMI_tensor,   # multichannel output
            timestamp     # just additional information
        ) 



    def __len__(self):
        return self.dataset_length
