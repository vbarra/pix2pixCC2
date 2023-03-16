"""
Utility programs of the pix2pixCC model

"""

#==============================================================================

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from functools import partial

from aia_hmi_sunpy_plotter import AIA_HMI_Plotter

#==============================================================================
# [1] True or False grid

def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid



#==============================================================================
# [2] Set the Normalization method for the input layer

def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d,affine=True)

    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=False)

    return layer



#==============================================================================
# [3] Set the Padding method for the input layer

def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d

    elif type == 'replication':
        layer = nn.ReplicationPad2d

    elif type == 'zero':
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError("Padding type {} is not valid."
                                  " Please choose among ['reflection', 'replication', 'zero']".format(type))

    return layer



#==============================================================================
# [4] Save or Report the model results 

class Manager(object):
    def __init__(self, opt, train_writer=None):
        self.opt = opt
        self.dtype = opt.data_type
        self.train_writer = train_writer
        
        self.plotter = AIA_HMI_Plotter(opt.data_size, opt.show_plots)
        
    #--------------------------------------------------------------------------      

    def report_loss(self, package):
        print("Epoch: {} D_loss: {:.{prec}}  G_loss: {:.{prec}}".
              format(package['Epoch'],
                      package['D_loss'], package['G_loss'], prec=4))
        
    def log_loss(self, package):
        self.train_writer.add_scalar('G_loss', package['G_loss'], package['Epoch'])
        self.train_writer.add_scalar('D_loss', package['D_loss'], package['Epoch'])

    
    #--------------------------------------------------------------------------
    def adjust_dynamic_range(self, data, drange_in, drange_out):
        if drange_in != drange_out:
            if self.dtype == 32:
                scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                            np.float32(drange_in[1]) - np.float32(drange_in[0]))
                bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            elif self.dtype == 16:
                scale = (np.float16(drange_out[1]) - np.float16(drange_out[0])) / (
                            np.float16(drange_in[1]) - np.float16(drange_in[0]))
                bias = (np.float16(drange_out[0]) - np.float16(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    
    #--------------------------------------------------------------------------
    def tensor2image(self, image_tensor):
        np_image = image_tensor[0].squeeze().cpu().float().numpy()
        if len(np_image.shape) == 3:
            np_image = np.transpose(np_image, (1, 2, 0))  # HWC
        else:
            pass

        np_image = self.adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image
    
    
    #--------------------------------------------------------------------------
    
    
    
    def save_image(self, image_tensor, path):
        Image.fromarray(self.tensor2image(image_tensor)).save(path, self.opt.image_mode)
      
    def save_aia_hmi(self, path, aia_tensor, hmi_real_tensor, hmi_fake_tensor, timestamps, epoch=None):
        
        ### only plotting the first record of the batch ### 
        aia_np =            aia_tensor[0].cpu().float().numpy()
        hmi_real_np =  hmi_real_tensor[0].cpu().float().numpy()
        hmi_fake_np =  hmi_fake_tensor[0].cpu().float().numpy()
        timestamp =         timestamps[0]

        ### undo target scaling
        UpIB = np.float(self.opt.saturation_upper_limit_target)
        LoIB = np.float(self.opt.saturation_lower_limit_target)
        hmi_real_np = hmi_real_np * ((UpIB - LoIB)/2) + (UpIB+ LoIB)/2 
        hmi_fake_np = hmi_fake_np * ((UpIB - LoIB)/2) + (UpIB+ LoIB)/2 
        
        ###
        if epoch is None:
            suptitle = 'Testing result for ' + timestamp
        else:
            suptitle = 'Timestamp '+timestamp+', current progress after Epoch '+ str(epoch) 
            
            
        fig = self.plotter.plot_aia_hmi_all_in_one(self.opt.input_channel_combination, aia_np, hmi_real_np, hmi_fake_np, suptitle)
        fig.savefig(path)
        
    def save_hmi_npz(self, path, hmi_tensor):
        hmi_np =  hmi_tensor.cpu().float().numpy()
        
        # undo target scaling
        UpIB = np.float(self.opt.saturation_upper_limit_target)
        LoIB = np.float(self.opt.saturation_lower_limit_target)
        hmi_np = hmi_np * ((UpIB - LoIB)/2) + (UpIB+ LoIB)/2 

        np.savez(path, x=hmi_np[0], y=hmi_np[1], z=hmi_np[2])
    
    #--------------------------------------------------------------------------        
    def save(self, package, img_dir=None, image=False, model=False, suffix=''):
        if image:
            # path_real = os.path.join(self.opt.image_dir, str(package['current_step']) + '_' + 'real.png')
            # path_fake = os.path.join(self.opt.image_dir, str(package['current_step']) + '_' + 'fake.png')
            path = os.path.join(img_dir, "E"+str(package['Epoch']) + suffix +'_' + 'all.png')
            #self.save_image(package['target_tensor'], path_real)
            #self.save_image(package['input_tensor'], path_real)
            #self.save_image(package['generated_tensor'], path_fake)
            self.save_aia_hmi(
                path,
                package['input_tensor'],
                package['target_tensor'],
                package['generated_tensor'],
                package['timestamps'],
                epoch=package['Epoch'])

        elif model:
            path_D = os.path.join(self.opt.model_dir, "E"+str(package['Epoch']) + suffix +'_'+ 'D.pt')
            path_G = os.path.join(self.opt.model_dir, "E"+str(package['Epoch']) + suffix +'_'+'G.pt')
            torch.save(package['D_state_dict'], path_D)
            torch.save(package['G_state_dict'], path_G)

    
    #--------------------------------------------------------------------------
    def __call__(self, package):

        epoch = package['Epoch']
        
        if self.opt.report_freq and epoch % self.opt.report_freq == 0:
            self.report_loss(package)
        
        if self.opt.log_freq and epoch % self.opt.log_freq == 0:
            self.log_loss(package)

        if self.opt.save_freq and epoch % self.opt.save_freq == 0:
            self.save(package, model=True)
            
        if self.opt.plot_freq and epoch % self.opt.plot_freq == 0:
            self.save(package, img_dir=self.opt.image_dir_train ,image=True)

    
    #--------------------------------------------------------------------------
    
    

#==============================================================================
# Set the initial conditions of weights

def weights_init(module):
    if isinstance(module, nn.Conv2d):
        module.weight.detach().normal_(0.0, 0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)
        
        
#==============================================================================
