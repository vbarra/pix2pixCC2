# -*- coding: utf-8 -*-
"""
Options for the pix2pixCC model

"""

#==============================================================================

import os
import jsonargparse


#==============================================================================

class Option(object):
    def __init__(self, config_path, start_time_str):
        self.start_time_str = start_time_str
        self.parser = jsonargparse.ArgumentParser(default_config_files=[config_path])

        self.parser.add_argument('--gpu_ids', type=str, default="0", help='gpu number. If -1, use cpu')

        #----------------------------------------------------------------------
        # data setting
        
        self.parser.add_argument('--dataset_name', type=str, default='pix2pixCC', help='dataset directory name')
        self.parser.add_argument('--data_format_input', type=str, default='tif', help="Input data extension. [tif, tiff, npy, fits, fts, fit]")
        self.parser.add_argument('--data_format_target', type=str, default='tif', help="Target data extension. [tif, tiff, npy, fits, fts, fit]")
        
        self.parser.add_argument('--input_channel_combination', nargs="+", default=[171, 193, 304], help="Which wavelengths to consider as input data")
        self.parser.add_argument('--data_root_dir', type=str, default='~/git/solari2i/data/solar-data/', help="Target data extension. [tif, tiff, npy, fits, fts, fit]")
        self.parser.add_argument('--check_dataset', type=bool, default=False, help="Checks if all given records of the dataset exist on the filesystem.")
        
        self.parser.add_argument('--input_ch', type=int, default=-1, help="The number of channels of input data, overriden in parse method.")
        self.parser.add_argument('--target_ch', type=int, default=3, help="The number of channels of target data")
        
        self.parser.add_argument('--data_size', type=int, default=512, help='image size of the input and target data')
        
        self.parser.add_argument('--logscale_input', type=list, default=[False, False, False], help='use logarithmic scales to the input data sets')
        self.parser.add_argument('--logscale_target', type=bool, default=False, help="use logarithmic scales to the target data sets")
        self.parser.add_argument('--sqrt_input', type=list, default=[False, False, False], help='use square root on the input channel images')
        self.parser.add_argument('--saturation_lower_limit_input', type=list, default=[-1, -1, -1], help="Saturation value (lower limit) of input")
        self.parser.add_argument('--saturation_upper_limit_input', type=list, default=[1, 1, 1], help="Saturation value (upper limit) of input")
        self.parser.add_argument('--saturation_lower_limit_target', type=float, default=-1, help="Saturation value (lower limit) of target")
        self.parser.add_argument('--saturation_upper_limit_target', type=float, default=1, help="Saturation value (upper limit) of target")
        self.parser.add_argument('--saturation_clip_input', type=list, default=[False, False, False], help="Saturation clip for input data")
        self.parser.add_argument('--saturation_clip_target', type=bool, default=False, help="Saturation clip for target data")


        #----------------------------------------------------------------------
        # network setting
        
        self.parser.add_argument('--batch_size', type=int, default=1, help='the number of batch_size')
        
        self.parser.add_argument('--n_downsample', type=int, default=4, help='how many times you want to downsample input data in Generator')
        self.parser.add_argument('--n_residual', type=int, default=9, help='the number of residual blocks in Generator')
        self.parser.add_argument('--trans_conv', type=bool, default=True, help='using transposed convolutions in Generator')

        self.parser.add_argument('--n_D', type=int, default=1, help='how many Discriminators in differet scales you want to use')
        self.parser.add_argument('--n_CC', type=int, default=4, help='how many downsample output data to compute CC values')
            
        self.parser.add_argument('--n_gf', type=int, default=64, help='The number of channels in the first convolutional layer of the Generator')
        self.parser.add_argument('--n_df', type=int, default=64, help='The number of channels in the first convolutional layer of the Discriminator')

        self.parser.add_argument('--ks_gf', type=int, default=7, help='Kernel size of first layer of the generator, only odd numbers, e.g. 1,3,5,7')
        self.parser.add_argument('--ks_gd', type=int, default=5, help='Kernel size of the downsample layers of the generator,  only odd numbers, e.g. 1,3,5,7')

        self.parser.add_argument('--data_type', type=int, default=32, help='float dtype [16, 32]')
        self.parser.add_argument('--n_workers', type=int, default=1, help='how many threads you want to use')
        
        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d', help='[BatchNorm2d, InstanceNorm2d]')
        self.parser.add_argument('--padding_type', type=str, default='replication',  help='[reflection, replication, zero]')
        
        #----------------------------------------------------------------------
        # data augmentation
        
        self.parser.add_argument('--padding_size', type=int, default=0, help='padding size for random crop')
        self.parser.add_argument('--max_rotation_angle', type=int, default=0, help='rotation angle in degrees')
        
        #----------------------------------------------------------------------
        # checking option
        
        self.parser.add_argument('--report_freq', type=int, default=10, help='Frequency in epochs for printing loss information to console during training, 0=off')
        self.parser.add_argument('--log_freq', type=int, default=100, help='Frequency in epochs for saving loss information to tensorboardx during training, 0=off')
        self.parser.add_argument('--save_freq', type=int, default=10000, help='Frequency in epochs for saving the model to disk, 0=off')
        self.parser.add_argument('--plot_freq', type=int, default=100, help='Frequency in epochs for generating and saving plots during training, 0=off')
        self.parser.add_argument('--val_freq', type=int, default=0, help='Frequency in epochs for evaluation of the model with testing dataset, 0=off')
        self.parser.add_argument('--save_scale', type=float, default=1)
        self.parser.add_argument('--display_scale', type=float, default=1)
        self.parser.add_argument('--image_mode', type=str, default='png', help='extension for saving image')
        self.parser.add_argument('--show_plots', type=bool, default=False, help='Open the interactive plots that are generated during training and testing')

        #----------------------------------------------------------------------

#==============================================================================
        ### Train Options ###
        
        #----------------------------------------------------------------------
        # directory path for training
        
        self.parser.add_argument('--input_dir_train', type=str, default='./datasets/Train/Input', help='directory path of the input files for the model training')
        self.parser.add_argument('--target_dir_train', type=str, default='./datasets/Train/Target', help='directory path of the target files for the model training')
        self.parser.add_argument('--relevant_timestamp_path_train', type=str, default='relevant_timestamps_train.txt', help='file that contains all relevant timestamps for training')
        
        #----------------------------------------------------------------------
        # Train setting
        
        self.parser.add_argument('--is_train', type=bool, default=True, help='train flag')
        self.parser.add_argument('--n_epochs', type=int, default=150, help='how many epochs you want to train')
        self.parser.add_argument('--latest_iter', type=int, default=0, help='Resume iteration')
        self.parser.add_argument('--no_shuffle', action='store_true', default=False, help='if you want to shuffle the order')
        
        #----------------------------------------------------------------------
        # hyperparameters 
        
        self.parser.add_argument('--lambda_LSGAN', type=float, default=2.0, help='weight for LSGAN loss')
        self.parser.add_argument('--lambda_FM', type=float, default=10.0, help='weight for Feature Matching loss')
        self.parser.add_argument('--lambda_CC', type=float, default=5.0, help='weight for CC loss')
        self.parser.add_argument('--lambda_SSIM', type=float, default=0.0, help='weight for structural similarity loss')
        
        self.parser.add_argument('--ch_balance', type=float, default=1, help='Set channel balance of input and target data')
        self.parser.add_argument('--ccc', type=bool, default=True, help='using Concordance Correlation Coefficient values (False -> using Pearson CC values)')
        
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--lr', type=float, default=0.0002)

        self.parser.add_argument('--patience', type=int, default=-1, help='Number of epochs that no improvement is tolerated before stopping the training, -1=Off')
        #----------------------------------------------------------------------
        
        
#==============================================================================
        ### Test Options ###
        
        #----------------------------------------------------------------------
        # directory path for test
        self.parser.add_argument('--input_dir_test', type=str, default='./datasets/Test/Input', help='directory path of the input files for the model test')
        self.parser.add_argument('--relevant_timestamp_path_test', type=str, default='relevant_timestamps_test.txt', help='file that contains all relevant timestamps for testing')
        
        #----------------------------------------------------------------------
        # test setting
        
        self.parser.add_argument('--iteration', type=int, default=-1, help='if you want to generate from input for the specific iteration')
        self.parser.add_argument('--n_test_exports', type=int, default=1, help='how much plots and/or npz files will be generated by the testing script for each file of saved weights, 0=ALL')    
        self.parser.add_argument('--export_plots', type=bool, default=True, help='whether to export the plots to compare input, target and generated data')
        self.parser.add_argument('--export_npz_fake', type=bool, default=False, help='whether to export the generated HMI images as npz files')
        self.parser.add_argument('--export_npz_target', type=bool, default=False, help='whether to export the target HMI images as npz files')
        self.parser.add_argument('--skip_existing_folders', type=bool, default=True, help='whether to skip existing folders when exporting images or npz files during testing')
        self.parser.add_argument('--skip_best_checkpoints', type=bool, default=True, help='whether to skip the models that are saved as E*_best_G/D.pt during testing')
        #----------------------------------------------------------------------
        
        
#==============================================================================


    def parse(self):
        opt = self.parser.parse_args()
        opt.format = 'png'  # extension for checking image 
        opt.flip = False
                    
        #--------------------------------
        if opt.data_type == 16:
            opt.eps = 1e-4
        elif opt.data_type == 32:
            opt.eps = 1e-8
            
        opt.input_ch = len(opt.input_channel_combination)
        
        #--------------------------------        
        # check if each multi channel option fits the number of input channels
        if opt.input_ch != len(opt.logscale_input):
            raise RuntimeError ("Option logscale_input must be a list with " + str(opt.input_ch) + " booleans (one value for each input channel).")
        if opt.input_ch != len(opt.saturation_clip_input):
            raise RuntimeError ("Option saturation_clip_input must be a list with " + str(opt.input_ch) + " booleans (one value for each input channel).")        
        if opt.input_ch != len(opt.saturation_lower_limit_input):
            raise RuntimeError ("Option saturation_lower_limit_input must be a list with " + str(opt.input_ch) + " floats (one value for each input channel).")        
        if opt.input_ch != len(opt.saturation_upper_limit_input):
            raise RuntimeError ("Option saturation_upper_limit_input must be a list with " + str(opt.input_ch) + " floats (one value for each input channel).")        
        
        #--------------------------------        
        opt.result_dir = os.path.join('./results', opt.dataset_name, self.start_time_str)
        
        opt.image_dir_train = os.path.join(opt.result_dir, 'Image', 'Train')
        opt.image_dir_test = os.path.join(opt.result_dir, 'Image', 'Test')
        opt.model_dir = os.path.join(opt.result_dir, 'Model')
        
        os.makedirs(opt.image_dir_train, exist_ok=True)
        os.makedirs(opt.image_dir_test, exist_ok=True)
        os.makedirs(opt.model_dir, exist_ok=True)

        
        #--------------------------------
        return opt
